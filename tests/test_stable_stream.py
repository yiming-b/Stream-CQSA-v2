"""
Tests for the stable Stream-CQSA forward path and its scheduler.

The CPU torch fallback exercises the merge arithmetic independently of the CUDA
build, so these run anywhere. CUDA-specific tests are skipped when the extension
is unavailable or does not cover the requested head dim / dtype.
"""

from __future__ import annotations

import math

import pytest
import torch

from stream_cqsa.reference import sdpa_reference
from stream_cqsa.stable_stream import (
    StableAccumulator,
    TraceRecorder,
    build_tasks,
    choose_parallelism,
    estimate_task_bytes,
    local_stats_flash,
    local_stats_torch,
    stream_cqsa_forward,
)

CUDA = torch.cuda.is_available()


def _qkv(b, h, n, d, *, std=1.0, seed=0, device="cpu", dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return tuple(
        (torch.randn(b, h, n, d, generator=g) * std).to(device=device, dtype=dtype)
        for _ in range(3)
    )


# ---------------------------------------------------------------------------
# Accumulator arithmetic
# ---------------------------------------------------------------------------


def test_accumulator_merge_order_does_not_change_result_much():
    torch.manual_seed(0)
    B, H, N, D, L = 1, 2, 32, 8, 12
    parts = []
    for s in range(4):
        g = torch.Generator().manual_seed(s)
        idx = torch.randperm(N, generator=g)[:L]
        # Token-major layout, matching the FlashAttention I/O contract.
        out = torch.randn(B, L, H, D, generator=g)
        lse = torch.randn(B, L, H, generator=g) * 5
        parts.append((idx, out, lse))

    def run(order):
        a = StableAccumulator(B, H, N, D, device=torch.device("cpu"))
        for i in order:
            a.merge_lse(*parts[i])
        return a.output()

    assert torch.allclose(run([0, 1, 2, 3]), run([3, 1, 0, 2]), atol=1e-5, rtol=1e-5)


def test_accumulator_handles_empty_rows_without_nan():
    """A fully-masked local row arrives as lse = -inf and must contribute nothing."""
    B, H, N, D, L = 1, 1, 8, 4, 4
    a = StableAccumulator(B, H, N, D, device=torch.device("cpu"))
    idx = torch.arange(L)
    out = torch.randn(B, L, H, D)
    lse = torch.full((B, L, H), float("-inf"))
    a.merge_lse(idx, out, lse)
    assert torch.isfinite(a.output()).all()
    assert torch.count_nonzero(a.output()) == 0
    assert a.untouched_tokens() == N


@pytest.mark.parametrize("sentinel", [float("-inf"), float("inf"), float("nan")])
def test_empty_local_row_never_erases_an_existing_contribution(sentinel):
    """
    Regression test. An all-masked local row carries no information, but the
    inner kernels disagree on how to say so: the torch fallback returns -inf,
    FlashAttention returns **+inf**. Treating +inf as a real maximum makes
    old_scale = exp(old_m - inf) = 0, which erases everything merged so far --
    observed as 586 of 2048 tokens going empty under causal masking.
    """
    B, H, N, D, L = 1, 1, 4, 4, 4
    a = StableAccumulator(B, H, N, D, device=torch.device("cpu"))
    idx = torch.arange(L)

    real = torch.ones(B, L, H, D)
    a.merge_lse(idx, real, torch.zeros(B, L, H))
    before = a.output().clone()

    # An empty subproblem must be a no-op, whichever sentinel it uses.
    junk = torch.full((B, L, H, D), float(sentinel))
    a.merge_lse(idx, junk, torch.full((B, L, H), float(sentinel)))

    after = a.output()
    assert torch.isfinite(after).all(), f"{sentinel} produced non-finite output"
    assert torch.allclose(after, before), f"{sentinel} changed an existing contribution"
    assert a.untouched_tokens() == 0


def test_accumulator_never_overflows_on_huge_lse():
    B, H, N, D, L = 1, 1, 4, 4, 4
    a = StableAccumulator(B, H, N, D, device=torch.device("cpu"))
    idx = torch.arange(L)
    out = torch.ones(B, L, H, D)
    a.merge_lse(idx, out, torch.full((B, L, H), 5000.0))
    a.merge_lse(idx, 2 * out, torch.full((B, L, H), 5001.0))
    res = a.output()
    assert torch.isfinite(res).all()
    # Second contribution is e^1 times heavier: (1 + 2e)/(1 + e).
    expected = (1.0 + 2.0 * math.e) / (1.0 + math.e)
    assert torch.allclose(res, torch.full_like(res, expected), atol=1e-5)


# ---------------------------------------------------------------------------
# Forward correctness (CPU fallback)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [128, 129, 512, 513])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("itr", [1, 2])
def test_forward_matches_sdpa_cpu(n, causal, itr):
    q, k, v = _qkv(1, 2, n, 32, seed=n + itr)
    out, info = stream_cqsa_forward(
        q, k, v, itr=itr, causal=causal, inner=local_stats_torch
    )
    ref = sdpa_reference(q, k, v, causal=causal)
    assert info["untouched_tokens"] == 0
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), (out - ref).abs().max().item()


@pytest.mark.parametrize("std", [1.0, 6.0, 12.0, 30.0])
@pytest.mark.parametrize("causal", [False, True])
def test_forward_stable_where_unshifted_merge_fails(std, causal):
    """
    Regression test for the silent-zero bug. The unshifted contract
    (den = exp(lse)) overflows fp32 for std >= 6 here; the stable merge must
    stay accurate.
    """
    q, k, v = _qkv(1, 1, 512, 64, std=std, seed=1)
    out, _ = stream_cqsa_forward(q, k, v, itr=1, causal=causal, inner=local_stats_torch)
    ref = sdpa_reference(q, k, v, causal=causal)
    assert torch.isfinite(out).all()
    rel = ((out - ref).abs().max() / ref.abs().max()).item()
    assert rel < 1e-4, rel


def test_unshifted_merge_really_does_fail_here():
    """Guards that the test above is exercising a real failure, not a strawman."""
    q, k, v = _qkv(1, 1, 512, 64, std=12.0, seed=1)
    scale = 64 ** -0.5
    lse = torch.logsumexp(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
    assert lse.max().item() > 88.7
    den = torch.exp(lse)
    assert torch.isinf(den).any()
    # ...and nan_to_num turns that inf into a silent zero.
    assert (torch.nan_to_num(den, posinf=0.0) == 0).any()


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def test_build_tasks_covers_every_token():
    N, itr = 513, 2
    tasks = build_tasks(N, itr, B=1, H=2, D=64, itemsize=2, pin=False)
    assert len(tasks) == 7 ** itr
    seen = torch.zeros(N, dtype=torch.bool)
    for t in tasks:
        seen[t.token_ids] = True
        assert t.local_size == t.token_ids.numel()
        assert t.estimated_mem_gib > 0
    assert bool(seen.all())


def test_cqs_block_summaries_match_bruteforce():
    """
    The kernel's O(1) tile test is only sound if these summaries are exact:
    OR must include every bit present in the block, AND only bits present in
    *all* of them.
    """
    from stream_cqsa.interface import cqs_block_summaries

    g = torch.Generator().manual_seed(0)
    for L in (1, 63, 64, 65, 879, 1000):
        bits = torch.randint(0, 8, (L,), generator=g, dtype=torch.int64)
        blk = 64
        blk_or, blk_and = cqs_block_summaries(bits, blk)
        nblk = (L + blk - 1) // blk
        assert blk_or.numel() == nblk and blk_and.numel() == nblk
        for b in range(nblk):
            chunk = bits[b * blk : min((b + 1) * blk, L)]
            want_or, want_and = 0, -1
            for x in chunk.tolist():
                want_or |= x
                want_and &= x
            assert int(blk_or[b]) == want_or, (L, b)
            assert int(blk_and[b]) == want_and, (L, b)


def test_cqs_block_summaries_imply_tile_clear_test():
    """
    The property the kernel relies on: if (or_row & or_col) == 0 then no pair in
    that tile is masked, so the tile can skip masking entirely.
    """
    from stream_cqsa.interface import cqs_block_summaries
    from stream_cqsa.reference import group_bits_for_path

    _, bits_np = group_bits_for_path(2048, (4,), sorted_gather=True)
    bits = torch.from_numpy(bits_np)
    blk = 64
    blk_or, _ = cqs_block_summaries(bits, blk)
    nblk = blk_or.numel()
    pair = torch.bitwise_and(bits[:, None], bits[None, :]).ne(0)

    clear_tiles = 0
    for r in range(nblk):
        for c in range(nblk):
            if int(blk_or[r] & blk_or[c]) == 0:
                clear_tiles += 1
                rs, re = r * blk, min((r + 1) * blk, bits.numel())
                cs, ce = c * blk, min((c + 1) * blk, bits.numel())
                assert not bool(pair[rs:re, cs:ce].any()), (r, c)
    # The optimisation is pointless if it never fires.
    assert clear_tiles > 0.5 * nblk * nblk, clear_tiles / (nblk * nblk)


def test_estimate_task_bytes_scales_linearly():
    a = estimate_task_bytes(1000, 1, 8, 128, 2)
    b = estimate_task_bytes(2000, 1, 8, 128, 2)
    assert b == pytest.approx(2 * a, rel=1e-6)


def test_trace_records_expected_schema():
    q, k, v = _qkv(1, 1, 128, 32, seed=5)
    tr = TraceRecorder(enabled=True, device=torch.device("cpu"))
    _, info = stream_cqsa_forward(
        q, k, v, itr=1, causal=False, inner=local_stats_torch, trace=tr
    )
    assert tr.rows
    for row in tr.rows:
        assert set(row) == set(TraceRecorder.FIELDS)
        assert row["end_time_s"] >= row["start_time_s"]
    stages = {r["stage"] for r in tr.rows}
    assert {"gather", "compute", "merge"} <= stages
    assert info["stage_totals_ms"]


def test_trace_jsonl_roundtrip(tmp_path):
    import json

    q, k, v = _qkv(1, 1, 128, 32, seed=6)
    tr = TraceRecorder(enabled=True, device=torch.device("cpu"))
    stream_cqsa_forward(q, k, v, itr=1, inner=local_stats_torch, trace=tr)
    p = tmp_path / "trace.jsonl"
    tr.write_jsonl(p)
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert len(rows) == len(tr.rows)
    assert rows[0]["run_id"] == tr.run_id


# ---------------------------------------------------------------------------
# CUDA path
# ---------------------------------------------------------------------------


def _flash_supports(D, dtype, causal):
    if not CUDA:
        return False
    try:
        q, k, v = (torch.randn(1, 64, 1, D, device="cuda", dtype=dtype) for _ in range(3))
        bits = torch.zeros(64, dtype=torch.int64, device="cuda")
        local_stats_flash(q, k, v, bits, causal=causal, scale=D ** -0.5)
        return True
    except RuntimeError:
        return False


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_cuda_forward_matches_sdpa(d, causal):
    if not _flash_supports(d, torch.float16, causal):
        pytest.skip(f"cqsa_cuda build lacks fp16 hdim{d} causal={causal}")
    n = 2048
    q, k, v = _qkv(1, 4, n, d, seed=2, device="cuda", dtype=torch.float16)
    out, info = stream_cqsa_forward(q, k, v, itr=1, causal=causal)
    ref = sdpa_reference(q.float(), k.float(), v.float(), causal=causal)
    assert info["untouched_tokens"] == 0
    rel = ((out - ref).abs().max() / ref.abs().max()).item()
    assert rel < 2e-2, rel


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("n_par", [1, 4, 8])
@pytest.mark.parametrize("itr", [1, 2])
def test_cuda_multistream_merge_is_not_racy(n_par, itr):
    """
    Regression test for a real race: the merge is a read-modify-write of the
    shared accumulator, so running it on the worker streams let concurrent
    subproblems interleave on the same token rows. That produced relative error
    1.0 at N=16384, itr=2, 8 streams while every single-stream test passed.

    Needs enough tokens that subproblems genuinely overlap in time; small
    shapes finish too fast to interleave and hide the bug.
    """
    if not _flash_supports(64, torch.float16, False):
        pytest.skip("cqsa_cuda build lacks fp16 hdim64")
    n = 16384
    q, k, v = _qkv(1, 8, n, 64, seed=4, device="cuda", dtype=torch.float16)
    out, info = stream_cqsa_forward(q, k, v, itr=itr, causal=False, max_parallel=n_par)
    ref = sdpa_reference(q.float(), k.float(), v.float(), causal=False)
    assert info["untouched_tokens"] == 0
    rel = ((out - ref).abs().max() / ref.abs().max()).item()
    assert rel < 2e-2, f"n_par={n_par} itr={itr} rel_err={rel}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_choose_parallelism_respects_budget():
    tasks = build_tasks(4096, 1, B=1, H=8, D=64, itemsize=2, pin=False)
    n = choose_parallelism(tasks, device=torch.device("cuda"), safety=0.5)
    assert 1 <= n <= 8
    # A budget of essentially zero must still yield at least one in-flight task.
    tight = choose_parallelism(
        tasks, device=torch.device("cuda"), safety=0.5, reserve_gib=10_000.0
    )
    assert tight == 1


# ---------------------------------------------------------------------------
# Decomposition planning
# ---------------------------------------------------------------------------


def test_estimate_peak_bytes_floor_is_itr_independent():
    """
    Only the transient shrinks with itr; the O(N.H.D) floor does not. This is
    why itr=3 measured the same peak as itr=2.
    """
    from stream_cqsa.stable_stream import estimate_peak_bytes

    kw = dict(B=1, H=8, D=64, itemsize=2)
    peaks = [estimate_peak_bytes(32768, i, **kw) for i in (1, 2, 3, 6)]
    assert peaks == sorted(peaks, reverse=True), peaks
    floor = (3 * 2 + 4 + 4) * 32768 * 8 * 64
    assert peaks[-1] > floor            # never below the floor
    assert peaks[-1] < 1.05 * floor     # and converging onto it


def test_estimate_peak_bytes_scales_linearly_in_N():
    """Floor and transient are both O(N), so the ratio is scale-invariant --
    measured at 1.36x for itr=1 across N = 32768..131072."""
    from stream_cqsa.stable_stream import estimate_peak_bytes

    kw = dict(B=1, H=8, D=64, itemsize=2)
    a = estimate_peak_bytes(32768, 1, **kw)
    b = estimate_peak_bytes(131072, 1, **kw)
    assert b == pytest.approx(4 * a, rel=1e-3)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_plan_decomposition_prefers_monolithic_when_it_fits():
    from stream_cqsa.stable_stream import plan_decomposition

    itr, why = plan_decomposition(
        4096, B=1, H=8, D=64, itemsize=2, device=torch.device("cuda")
    )
    assert itr == 0, why
    assert "monolithic" in why


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_plan_decomposition_decomposes_under_pressure():
    """Reserving almost all memory must push it off the monolithic path."""
    from stream_cqsa.stable_stream import plan_decomposition

    free_gib = torch.cuda.mem_get_info(torch.device("cuda"))[0] / (1 << 30)
    itr, why = plan_decomposition(
        131072, B=1, H=8, D=64, itemsize=2, device=torch.device("cuda"),
        reserve_gib=free_gib - 0.5,
    )
    assert itr >= 1, why


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_auto_itr_matches_flash_attention():
    """itr='auto' must return the same answer whichever branch it picks."""
    from stream_cqsa.reference import sdpa_reference

    q, k, v = _qkv(1, 4, 2048, 64, seed=9, device="cuda", dtype=torch.float16)
    out, info = stream_cqsa_forward(q, k, v, itr="auto", causal=False)
    ref = sdpa_reference(q.float(), k.float(), v.float(), causal=False)
    rel = ((out - ref).abs().max() / ref.abs().max()).item()
    assert rel < 2e-2, (rel, info["plan_reason"])
    assert info["itr"] >= 0


# ---------------------------------------------------------------------------
# fp32 statistics path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_fp32_out_returns_float32():
    from stream_cqsa.interface import flash_attn_func_cqs_group_bits

    L, H, D = 512, 4, 64
    q, k, v = (torch.randn(1, L, H, D, device="cuda", dtype=torch.float16) for _ in range(3))
    bits = torch.zeros(L, dtype=torch.int64, device="cuda")
    out = flash_attn_func_cqs_group_bits(q, k, v, bits, softmax_scale=D ** -0.5,
                                         fp32_out=True)
    assert out.dtype == torch.float32
    assert out.shape == (1, L, H, D)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_fp32_out_is_more_accurate_than_fp16_out():
    """
    The whole point: the kernel holds acc_o in fp32 registers and the stock
    epilogue rounds it to the input dtype. That rounding measured ~83% of a
    subproblem's error, so returning fp32 should cut the error several-fold.
    """
    from stream_cqsa.reference import group_bits_for_path
    from stream_cqsa.stable_stream import local_stats_flash, local_stats_torch

    N, H, D = 8192, 4, 64
    scale = D ** -0.5
    ids_np, bits_np = group_bits_for_path(N, (0,), sorted_gather=True)
    bits = torch.as_tensor(bits_np, device="cuda", dtype=torch.long)
    L = len(ids_np)
    g = torch.Generator(device="cpu").manual_seed(0)
    q, k, v = (torch.randn(1, L, H, D, generator=g).to("cuda", torch.float16)
               for _ in range(3))

    # fp32 ground truth for this exact local problem
    o32, _ = local_stats_torch(q.float(), k.float(), v.float(), bits,
                               causal=False, scale=scale)
    den = o32.abs().max()

    o_fp16, _ = local_stats_flash(q, k, v, bits, causal=False, scale=scale,
                                  fp32_out=False)
    o_fp32, _ = local_stats_flash(q, k, v, bits, causal=False, scale=scale,
                                  fp32_out=True)
    e16 = ((o_fp16 - o32).abs().max() / den).item()
    e32 = ((o_fp32 - o32).abs().max() / den).item()
    assert e32 < e16 / 2.0, f"fp32 path not materially better: {e32:.3e} vs {e16:.3e}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("causal", [False, True])
def test_fp32_out_end_to_end_still_exact(causal):
    from stream_cqsa.reference import sdpa_reference

    q, k, v = _qkv(1, 4, 4096, 64, seed=3, device="cuda", dtype=torch.float16)
    out, info = stream_cqsa_forward(q, k, v, itr=1, causal=causal)
    ref = sdpa_reference(q.float(), k.float(), v.float(), causal=causal)
    assert info["untouched_tokens"] == 0
    rel = ((out - ref).abs().max() / ref.abs().max()).item()
    assert rel < 2e-2, rel


# ---------------------------------------------------------------------------
# Segmented input
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("n", [4096, 16384])
def test_segmented_input_matches_gather(n, causal):
    """
    Segmented input reaches the same numbers by a completely different data
    path: no gather at all, the kernel reads the original Q/K/V in place via a
    per-tile base pointer offset. Agreement with the gather path to fp16
    roundoff is the real correctness signal for the block map.
    """
    q, k, v = _qkv(1, 8, n, 64, seed=5, device="cuda", dtype=torch.float16)
    a, ia = stream_cqsa_forward(q, k, v, itr=1, causal=causal, segmented=False)
    b, ib = stream_cqsa_forward(q, k, v, itr=1, causal=causal, segmented=True)
    assert ib["segmented"] is True and ia["segmented"] is False
    assert ib["untouched_tokens"] == 0
    rel = ((a - b).abs().max() / a.abs().max()).item()
    assert rel < 1e-3, rel


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_segmented_falls_back_when_unsupported():
    """CPU / unsorted gather have no block map, so segmentation must disable
    itself rather than produce wrong indices."""
    q, k, v = _qkv(1, 2, 2048, 64, seed=6, device="cuda", dtype=torch.float16)
    _, info = stream_cqsa_forward(q, k, v, itr=1, segmented=True, sorted_gather=False)
    assert info["segmented"] is False


# ---------------------------------------------------------------------------
# CUDA backward
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("itr", [1, 2])
def test_cuda_backward_matches_autograd(itr, causal):
    """
    The CUDA backward is exact because every subproblem gets the GLOBAL lse:
    p = exp(s - lse) <= 1, so the standard softmax backward applies on the
    retained pair set and the per-subproblem gradients simply add.
    """
    from stream_cqsa.stable_stream import stream_cqsa_backward

    n, d = 2048, 64
    g = torch.Generator().manual_seed(n + itr)
    qf, kf, vf = (torch.randn(1, 4, n, d, generator=g) for _ in range(3))
    dof = torch.randn(1, 4, n, d, generator=g)

    qa, ka, va = (t.clone().cuda().float().requires_grad_(True) for t in (qf, kf, vf))
    sdpa_reference(qa, ka, va, causal=causal).backward(dof.cuda().float())

    q, k, v = (t.cuda().half() for t in (qf, kf, vf))
    out, info = stream_cqsa_forward(q, k, v, itr=itr, causal=causal)
    dq, dk, dv = stream_cqsa_backward(
        q, k, v, dof.cuda().half(), out, info["lse"], itr=itr, causal=causal
    )
    for got, want, name in ((dq, qa.grad, "dQ"), (dk, ka.grad, "dK"), (dv, va.grad, "dV")):
        rel = ((got - want).abs().max() / want.abs().max().clamp_min(1e-30)).item()
        assert rel < 5e-3, f"{name} rel={rel}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_forward_exposes_global_lse():
    """The backward needs it, and it must be finite where exp(lse) is not."""
    q, k, v = _qkv(1, 2, 1024, 64, std=12.0, seed=1, device="cuda", dtype=torch.float16)
    _, info = stream_cqsa_forward(q, k, v, itr=1)
    lse = info["lse"]
    assert lse.shape == (1, 2, 1024)
    assert torch.isfinite(lse).all()
    assert torch.isinf(torch.exp(lse.double())).any() or lse.max() > 88.7


# ---------------------------------------------------------------------------
# Shared chunk pool
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("slots", [5, 6, 7])
def test_shared_chunks_matches_independent_loading(slots):
    """
    Sharing must not change the answer. `slots=5` is the important case: it is
    the first size that forces eviction, which is where the pool's correctness
    actually gets exercised.
    """
    n, H, D = 32768, 8, 64
    q, k, v = _qkv(1, H, n, D, seed=11, dtype=torch.float16)
    ref, _ = stream_cqsa_forward(q.cuda(), k.cuda(), v.cuda(), itr=1)
    out, info = stream_cqsa_forward(
        q, k, v, itr=1, stream_from_host=True, max_parallel=2,
        shared_chunks=True, chunk_pool_slots=slots,
    )
    assert info["shared_chunks"] is True
    assert info["untouched_tokens"] == 0
    rel = ((out.cpu() - ref.cpu()).abs().max() / ref.abs().max().cpu()).item()
    assert rel < 5e-3, f"slots={slots} rel={rel} (evictions={info['evictions']})"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_shared_chunks_actually_saves_transfers():
    """Each of the c chunks should move once, not once per subsequence."""
    n, H, D = 32768, 8, 64
    q, k, v = _qkv(1, H, n, D, seed=12, dtype=torch.float16)
    _, info = stream_cqsa_forward(
        q, k, v, itr=1, stream_from_host=True, max_parallel=2,
        shared_chunks=True, chunk_pool_slots=7,
    )
    # 7 subsequences x 3 chunks each = 21 loads, but only 7 distinct chunks.
    assert info["chunk_loads_requested"] == 21
    assert info["chunk_transfers"] == 7
    assert info["transfer_saving_pct"] > 60


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_shared_chunks_rejects_unsupported_configs():
    q, k, v = _qkv(1, 4, 4096, 64, seed=13, dtype=torch.float16)
    with pytest.raises(ValueError, match="stream_from_host"):
        stream_cqsa_forward(q.cuda(), k.cuda(), v.cuda(), itr=1, shared_chunks=True)
    with pytest.raises(ValueError, match="itr=1"):
        stream_cqsa_forward(q, k, v, itr=2, stream_from_host=True, shared_chunks=True)
