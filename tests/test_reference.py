"""
Correctness tests for the Stream-CQSA reference semantics.

These are the semantic ground truth the CUDA/streaming path must match:
  * every target attention pair is retained exactly once (exactness)
  * the stable reference forward reproduces monolithic SDPA
  * the stable merge survives score magnitudes that overflow exp()

Run:
    source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/env.sh
    pytest packages/stream-cqsa/tests/test_reference.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stream_cqsa.reference import (
    all_paths,
    build_path_state,
    chunk_layout,
    check_pair_coverage,
    dense_keep_mask_for_path,
    group_bits_for_path,
    quorum_chunks,
    sdpa_reference,
    stream_cqsa_forward_reference,
)

# Deliberately includes lengths that are not multiples of 7, and lengths
# smaller than the chunk count, which is where index arithmetic tends to break.
COVERAGE_N = [8, 15, 29, 49, 127]


# ---------------------------------------------------------------------------
# Decomposition structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 6, 7, 8, 29, 127])
def test_chunk_layout_partitions_exactly(n):
    sizes, starts, ends = chunk_layout(n)
    assert sum(sizes) == n
    assert starts[0] == 0
    assert ends[-1] == n
    for i in range(1, 7):
        assert starts[i] == ends[i - 1]
    # First `remainder` chunks absorb the remainder, so sizes differ by <= 1.
    assert max(sizes) - min(sizes) <= 1


def test_quorum_chunks_match_paper_table():
    expected = {
        0: [0, 1, 3],
        1: [1, 2, 4],
        2: [2, 3, 5],
        3: [3, 4, 6],
        4: [4, 5, 0],
        5: [5, 6, 1],
        6: [6, 0, 2],
    }
    for owner, chunks in expected.items():
        assert quorum_chunks(owner) == chunks


def test_offdiagonal_chunk_pairs_covered_once():
    """The cyclic difference set (0,1,3) mod 7 covers each ordered off-diagonal
    chunk pair exactly once."""
    seen = {}
    for owner in range(7):
        chunks = quorum_chunks(owner)
        for a in chunks:
            for b in chunks:
                if a == b:
                    continue
                seen[(a, b)] = seen.get((a, b), 0) + 1
    assert len(seen) == 7 * 6
    assert set(seen.values()) == {1}


@pytest.mark.parametrize("n", [29, 127])
@pytest.mark.parametrize("itr", [1, 2])
def test_path_state_token_ids_are_unique(n, itr):
    for path in all_paths(itr):
        token_ids, _, _ = build_path_state(n, path)
        assert len(set(token_ids.tolist())) == len(token_ids)


@pytest.mark.parametrize("n", [29, 127])
def test_group_bits_mask_owner_diagonal_only(n):
    """Owner chunk keeps its diagonal block; other gathered chunks lose theirs."""
    for owner in range(7):
        token_ids, bits = group_bits_for_path(n, (owner,))
        _, labels, trace = build_path_state(n, (owner,))
        lab = labels[0]
        for chunk_id in trace[0]["chunks"]:
            idx = np.nonzero(lab == chunk_id)[0]
            if idx.size == 0:
                continue
            same = np.bitwise_and(bits[idx][:, None], bits[idx][None, :]) == 0
            if chunk_id == owner:
                assert same.all(), f"owner diagonal {chunk_id} must be kept"
            else:
                assert not same.any(), f"non-owner diagonal {chunk_id} must be masked"


# ---------------------------------------------------------------------------
# Exactness: pair coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", COVERAGE_N)
@pytest.mark.parametrize("causal", [False, True])
def test_pair_coverage_itr1(n, causal):
    check_pair_coverage(n, 1, causal=causal)


@pytest.mark.parametrize("n", [15, 29, 49])
@pytest.mark.parametrize("causal", [False, True])
def test_pair_coverage_itr2(n, causal):
    check_pair_coverage(n, 2, causal=causal)


@pytest.mark.parametrize("causal", [False, True])
def test_pair_coverage_itr3(causal):
    check_pair_coverage(29, 3, causal=causal)


def test_causal_uses_global_token_ids_not_local_positions():
    """
    Regression guard for the subtlest bug in the design: inside a gathered
    subsequence, chunk 3 can sit directly after chunk 1, so a local lower
    triangle is NOT the causal mask. If causality were applied on local
    positions, coverage would break.
    """
    n = 29
    # Owners 0..3 gather (i, i+1, i+3) without wrapping, so their gathered order
    # happens to be globally ascending and local order == global order. Only the
    # wrapping owners 4, 5, 6 actually distinguish the two rules.
    owner = 4
    token_ids, keep = dense_keep_mask_for_path(n, (owner,), causal=True)
    L = len(token_ids)
    assert not np.all(np.diff(token_ids) > 0), "owner must produce non-monotonic token ids"

    _, keep_noncausal = dense_keep_mask_for_path(n, (owner,), causal=False)
    wrong = keep_noncausal & np.tril(np.ones((L, L), dtype=bool))

    assert not np.array_equal(keep, wrong), (
        "local-position causality coincidentally matched; test is not exercising the bug"
    )

    # The correct mask is global-id ordering.
    ids = token_ids
    assert np.array_equal(keep, keep_noncausal & (ids[None, :] <= ids[:, None]))

    # And the wrong rule really would break exactness, not just reorder work.
    counts_wrong = np.zeros((n, n), dtype=np.int64)
    for o in range(7):
        tids, kp = dense_keep_mask_for_path(n, (o,), causal=False)
        kp = kp & np.tril(np.ones((len(tids), len(tids)), dtype=bool))
        rows, cols = np.nonzero(kp)
        np.add.at(counts_wrong, (tids[rows], tids[cols]), 1)
    target = (np.arange(n)[None, :] <= np.arange(n)[:, None]).astype(np.int64)
    assert not np.array_equal(counts_wrong, target), (
        "local-position causality must fail the coverage invariant"
    )


# ---------------------------------------------------------------------------
# Sorted gather: makes local order == global order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [15, 29, 127, 1024])
@pytest.mark.parametrize("itr", [1, 2, 3])
def test_sorted_gather_token_ids_are_monotonic(n, itr):
    """Each level concatenates contiguous runs of an already-ascending sequence."""
    for path in all_paths(itr):
        ids, _, _ = build_path_state(n, path, sorted_gather=True)
        if len(ids) > 1:
            assert np.all(np.diff(ids) > 0), (path, ids)


@pytest.mark.parametrize("n", COVERAGE_N)
@pytest.mark.parametrize("itr", [1, 2])
@pytest.mark.parametrize("causal", [False, True])
def test_sorted_gather_is_exact(n, itr, causal):
    """The reordered decomposition must still retain every pair exactly once."""
    check_pair_coverage(n, itr, causal=causal, sorted_gather=True)


@pytest.mark.parametrize("n", [29, 127])
@pytest.mark.parametrize("itr", [1, 2])
def test_sorted_gather_makes_local_triangle_exact(n, itr):
    """
    The payoff: with sorted gather a plain local lower-triangular mask is
    *identical* to the global-token-id causal mask, so the stock FlashAttention
    causal kernel is exactly right with no kernel change.
    """
    for path in all_paths(itr):
        ids, keep_causal = dense_keep_mask_for_path(
            n, path, causal=True, sorted_gather=True
        )
        _, keep_full = dense_keep_mask_for_path(
            n, path, causal=False, sorted_gather=True
        )
        L = len(ids)
        local_tri = keep_full & np.tril(np.ones((L, L), dtype=bool))
        assert np.array_equal(keep_causal, local_tri), path


def test_sorted_gather_changes_leaf_problems_at_itr2():
    """
    Documents that this is a genuinely different decomposition at itr>=2, not a
    relabelling: level-2 chunk boundaries land on different tokens.
    """
    a = {tuple(build_path_state(127, p, sorted_gather=False)[0].tolist())
         for p in all_paths(2)}
    b = {tuple(build_path_state(127, p, sorted_gather=True)[0].tolist())
         for p in all_paths(2)}
    assert a != b
    # ...whereas at itr=1 only the local permutation differs.
    for p in all_paths(1):
        x, _, _ = build_path_state(127, p, sorted_gather=False)
        y, _, _ = build_path_state(127, p, sorted_gather=True)
        assert set(x.tolist()) == set(y.tolist())


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("itr", [1, 2])
def test_sorted_gather_forward_matches_sdpa(causal, itr):
    q, k, v = _qkv(1, 2, 129, 32, device="cpu", seed=11)
    got = stream_cqsa_forward_reference(
        q, k, v, itr=itr, causal=causal, sorted_gather=True
    )
    want = sdpa_reference(q, k, v, causal=causal)
    assert torch.allclose(got, want, atol=2e-5, rtol=2e-5), (got - want).abs().max().item()


# ---------------------------------------------------------------------------
# Numerics: reference forward vs monolithic SDPA
# ---------------------------------------------------------------------------


def _qkv(b, h, n, d, *, device, dtype=torch.float32, seed=0, std=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda: (torch.randn(b, h, n, d, generator=g, dtype=torch.float32) * std).to(
        device=device, dtype=dtype
    )
    return mk(), mk(), mk()


@pytest.mark.parametrize("n", [15, 29, 128, 129])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("itr", [1, 2])
def test_forward_reference_matches_sdpa(n, causal, itr):
    q, k, v = _qkv(1, 2, n, 32, device="cpu", seed=n + itr)
    got = stream_cqsa_forward_reference(q, k, v, itr=itr, causal=causal)
    want = sdpa_reference(q, k, v, causal=causal)
    assert torch.allclose(got, want, atol=2e-5, rtol=2e-5), (
        (got - want).abs().max().item()
    )


@pytest.mark.parametrize("causal", [False, True])
def test_two_pass_merge_matches_streaming_merge(causal):
    q, k, v = _qkv(1, 2, 51, 32, device="cpu", seed=7)
    a = stream_cqsa_forward_reference(q, k, v, itr=2, causal=causal, two_pass=False)
    b = stream_cqsa_forward_reference(q, k, v, itr=2, causal=causal, two_pass=True)
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("std", [1.0, 6.0, 12.0])
@pytest.mark.parametrize("causal", [False, True])
def test_forward_reference_stable_under_large_scores(std, causal):
    """
    The stable merge must hold up where naive ``exp(lse)`` accumulation
    overflows fp32. With std=12 and D=32 the row logsumexp lands far above
    log(3.4e38) ~= 88.7, so any implementation that materialises exp(lse)
    returns inf/NaN (or silently zero after nan_to_num).
    """
    q, k, v = _qkv(1, 1, 29, 32, device="cpu", seed=3, std=std)
    got = stream_cqsa_forward_reference(q, k, v, itr=1, causal=causal)
    want = sdpa_reference(q, k, v, causal=causal)
    assert torch.isfinite(got).all()
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-4), (got - want).abs().max().item()


def test_naive_exp_lse_accumulation_actually_overflows():
    """
    Pins the failure mode the stable merge exists to avoid, so the design note
    is backed by a test rather than an assertion.
    """
    q, k, v = _qkv(1, 1, 29, 32, device="cpu", seed=3, std=12.0)
    scale = 32 ** -0.5
    scores = torch.einsum("bhld,bhmd->bhlm", q, k) * scale
    lse = torch.logsumexp(scores, dim=-1)
    assert lse.max().item() > 88.7, lse.max().item()
    assert torch.isinf(torch.exp(lse)).any(), "expected fp32 overflow in exp(lse)"


# ---------------------------------------------------------------------------
# Degenerate shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 7])
def test_tiny_sequences(n):
    q, k, v = _qkv(1, 1, n, 16, device="cpu", seed=n)
    got = stream_cqsa_forward_reference(q, k, v, itr=1, causal=True)
    want = sdpa_reference(q, k, v, causal=True)
    assert torch.allclose(got, want, atol=2e-5, rtol=2e-5)


# ---------------------------------------------------------------------------
# Alternative difference sets
# ---------------------------------------------------------------------------

# Perfect (Singer) difference sets: every non-zero residue mod c occurs exactly
# once as a difference of two members, which forces c = l^2 - l + 1. That lambda=1
# property is what makes each off-diagonal chunk pair covered exactly once.
DIFFERENCE_SETS = [
    (7, (0, 1, 3)),
    (13, (0, 1, 3, 9)),
    (21, (0, 1, 4, 14, 16)),
    (31, (0, 1, 3, 8, 12, 18)),
    (57, (0, 1, 3, 13, 32, 36, 43, 52)),
]


@pytest.mark.parametrize("c,iset", DIFFERENCE_SETS)
def test_is_perfect_difference_set(c, iset):
    """The construction is only exact if lambda == 1. Check it directly."""
    diffs = sorted((a - b) % c for a in iset for b in iset if a != b)
    assert diffs == list(range(1, c)), (c, iset)
    assert c == len(iset) ** 2 - len(iset) + 1


@pytest.mark.parametrize("c,iset", DIFFERENCE_SETS)
@pytest.mark.parametrize("causal", [False, True])
def test_alternative_difference_sets_are_exact(c, iset, causal):
    """Every target pair retained exactly once, at N not a multiple of c."""
    check_pair_coverage(
        c * 4 + 1, 1, causal=causal, c=c, sorted_gather=True, interest_set=iset
    )


@pytest.mark.parametrize("c,iset", DIFFERENCE_SETS)
def test_alternative_sets_keep_sorted_gather_monotonic(c, iset):
    """Sorted gather must still give global order == local order for any c."""
    n = c * 7 + 3
    for owner in range(c):
        ids, _, _ = build_path_state(n, (owner,), c, iset, sorted_gather=True)
        if len(ids) > 1:
            assert np.all(np.diff(ids) > 0), (c, owner)


# ---------------------------------------------------------------------------
# Block-aligned chunks (prerequisite for segmented input)
# ---------------------------------------------------------------------------

ALIGN = 128


@pytest.mark.parametrize("n", [1000, 2048, 4096, 12000, 131072])
@pytest.mark.parametrize("c", [7, 13])
def test_aligned_chunk_layout_is_a_partition(n, c):
    sizes, starts, ends = chunk_layout(n, c, ALIGN)
    assert sum(sizes) == n
    assert ends[-1] == n
    for i in range(1, c):
        assert starts[i] == ends[i - 1]
    # Every start is block-aligned, and only the LAST chunk may be ragged --
    # that is what stops a kernel tile from straddling a run boundary.
    assert all(st % ALIGN == 0 for st in starts)
    assert all(sz % ALIGN == 0 for sz in sizes[:-1])


@pytest.mark.parametrize("n", [2048, 4096])
@pytest.mark.parametrize("causal", [False, True])
def test_aligned_chunks_are_exact(n, causal):
    """
    Exactness must not depend on chunk sizes. The coverage argument is
    combinatorial on chunk *indices*, so any contiguous partition into c chunks
    works -- the default layout already produces unequal chunks whenever c does
    not divide n.
    """
    check_pair_coverage(n, 1, causal=causal, sorted_gather=True, align=ALIGN)


@pytest.mark.parametrize("n", [4096, 131072])
def test_aligned_gather_runs_start_on_block_boundaries(n):
    """
    The property the segmented kernel depends on: under sorted gather, every
    run of the gathered subsequence begins at a local offset that is a multiple
    of the tile size, so a tile never spans two runs and the gather reduces to a
    per-tile base pointer offset.
    """
    sizes, starts, ends = chunk_layout(n, 7, ALIGN)
    for owner in range(7):
        _, _, trace = build_path_state(n, (owner,), sorted_gather=True, align=ALIGN)
        offset = 0
        for chunk_id in trace[0]["chunks"]:
            assert offset % ALIGN == 0, (owner, chunk_id, offset)
            offset += sizes[chunk_id]


def test_aligned_chunks_still_monotonic_under_sorted_gather(n=131072):
    for owner in range(7):
        ids, _, _ = build_path_state(n, (owner,), sorted_gather=True, align=ALIGN)
        assert np.all(np.diff(ids) > 0)


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


def _grads_from_autograd(q, k, v, dout, causal):
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    sdpa_reference(q, k, v, causal=causal).backward(dout)
    return q.grad, k.grad, v.grad


@pytest.mark.parametrize("n,itr", [(129, 1), (129, 2), (512, 1)])
@pytest.mark.parametrize("causal", [False, True])
def test_backward_matches_autograd(n, itr, causal):
    from stream_cqsa.reference import stream_cqsa_backward_reference

    g = torch.Generator().manual_seed(n + itr)
    q, k, v = (torch.randn(1, 2, n, 32, generator=g) for _ in range(3))
    dout = torch.randn(1, 2, n, 32, generator=g)
    gq, gk, gv = _grads_from_autograd(q, k, v, dout, causal)
    dq, dk, dv = stream_cqsa_backward_reference(q, k, v, dout, itr=itr, causal=causal)
    for got, want, name in ((dq, gq, "dQ"), (dk, gk, "dK"), (dv, gv, "dV")):
        rel = ((got - want).abs().max() / want.abs().max().clamp_min(1e-30)).item()
        assert rel < 1e-5, f"{name} rel={rel}"


@pytest.mark.parametrize("std", [12.0, 30.0])
@pytest.mark.parametrize("causal", [False, True])
def test_backward_stable_where_unshifted_form_overflows(std, causal):
    """
    The shipped backward consumes dNum = dO/Den and dDen = -sum(dO*Num)/Den^2,
    inheriting the forward's exp(lse) overflow. Passing the GLOBAL lse instead
    keeps every weight exp(s - lse) <= 1, so the gradients stay exact at score
    magnitudes where Den is inf.
    """
    from stream_cqsa.reference import stream_cqsa_backward_reference

    g = torch.Generator().manual_seed(3)
    q, k, v = ((torch.randn(1, 2, 256, 32, generator=g) * std) for _ in range(3))
    dout = torch.randn(1, 2, 256, 32, generator=g)

    lse = torch.logsumexp(torch.matmul(q, k.transpose(-2, -1)) * 32 ** -0.5, dim=-1)
    assert torch.isinf(torch.exp(lse)).any(), "test must exercise the overflow regime"

    gq, gk, gv = _grads_from_autograd(q, k, v, dout, causal)
    dq, dk, dv = stream_cqsa_backward_reference(q, k, v, dout, itr=1, causal=causal)
    for got, want in ((dq, gq), (dk, gk), (dv, gv)):
        assert torch.isfinite(got).all()
        rel = ((got - want).abs().max() / want.abs().max().clamp_min(1e-30)).item()
        # Looser than the unit-scale test on purpose. At these magnitudes the
        # softmax is nearly one-hot, so the gradient is a difference of large
        # near-equal terms and fp32 cancellation dominates (~1e-4). The point
        # here is finite-and-correct versus the shipped form's exact 0 / NaN,
        # i.e. relative error 1.0 -- which 2e-3 still separates cleanly.
        assert rel < 2e-3, rel


def test_unshifted_backward_really_would_overflow():
    """Pins that the guard above is exercising a real failure, not a strawman."""
    g = torch.Generator().manual_seed(3)
    q, k, v = ((torch.randn(1, 2, 256, 32, generator=g) * 12.0) for _ in range(3))
    dout = torch.randn(1, 2, 256, 32, generator=g)
    scale = 32 ** -0.5
    lse = torch.logsumexp(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
    out = sdpa_reference(q, k, v, causal=False)

    den = torch.exp(lse)                       # Den in the shipped formulation
    assert torch.isinf(den).any()
    d_num = dout / den[..., None]              # -> 0
    d_den = -(dout * out).sum(-1) / den ** 2   # -> 0 or NaN
    assert (d_num == 0).any() or torch.isnan(d_num).any()
    assert (d_den == 0).any() or torch.isnan(d_den).any()
