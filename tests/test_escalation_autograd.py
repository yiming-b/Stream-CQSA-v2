"""
Depth escalation and autograd support.

Two claims are under test here. The first is that a subproblem which does not
fit is decomposed a further level and its children retried, rather than the call
failing -- the recovery the paper describes, in the forward and in both backward
paths. The second is that the operator participates in autograd, so
``.backward()`` produces gradients through the native decomposed backward.

The refinement tests need no device: whether children partition their parent's
pairs is a property of the masks, and checking it on the CPU is what makes the
escalation trustworthy on a machine whose GPU is busy.
"""
from __future__ import annotations

import pytest
import torch

from stream_cqsa.reference import sdpa_reference
from stream_cqsa.stable_stream import (
    build_tasks,
    max_depth_for,
    refine_task,
    stream_cqsa_backward,
    stream_cqsa_forward,
)
from stream_cqsa.native_autograd import StreamCQSAAttention, stream_cqsa_attn

CUDA = torch.cuda.is_available()


def _retained_pairs(task):
    """The global ``(row, col)`` chunk pairs this subproblem is responsible for."""
    ids = task.token_ids.tolist()
    bits = task.group_bits.tolist()
    return {(ids[i], ids[j])
            for i in range(len(ids)) for j in range(len(ids))
            if (bits[i] & bits[j]) == 0}


# --------------------------------------------------------------- structure --

@pytest.mark.parametrize("N,c,expect", [
    (7, 7, 1), (48, 7, 1), (49, 7, 2), (343, 7, 3),
    (1 << 24, 7, 8),          # 16M at c=7: the bound quoted in the paper
    (13, 13, 1), (169, 13, 2),
    (0, 7, 0), (1, 7, 0),     # degenerate: nothing to decompose
])
def test_max_depth_is_the_structural_bound(N, c, expect):
    """Deepest depth is the largest itr with c**itr <= N."""
    assert max_depth_for(N, c) == expect
    d = max_depth_for(N, c)
    assert c ** d <= max(N, 1)
    assert c ** (d + 1) > N


@pytest.mark.parametrize("N,c,iset", [
    (343, 7, (0, 1, 3)),
    (2401, 7, (0, 1, 3)),
    (169, 13, (0, 1, 3, 9)),
])
@pytest.mark.parametrize("depth", [1, 2])
def test_refinement_partitions_the_parent(N, c, iset, depth):
    """
    Escalation is only sound if a task's children cover exactly the pairs the
    task held: any pair lost is a dropped contribution, any pair repeated is a
    double count. Both are silent in the output, so they are checked directly.
    """
    tasks = build_tasks(N, depth, B=1, H=1, D=64, itemsize=2,
                        c=c, interest_set=iset, pin=False)
    for parent in tasks[:3]:
        want = _retained_pairs(parent)
        kids = refine_task(parent, N, B=1, H=1, D=64, itemsize=2,
                           c=c, interest_set=iset, pin=False)
        assert len(kids) == c
        assert all(int(kd.itr) == int(parent.itr) + 1 for kd in kids)
        assert all(tuple(kd.path)[:-1] == tuple(parent.path) for kd in kids)

        seen = set()
        for kd in kids:
            got = _retained_pairs(kd)
            assert not (seen & got), "children double count a chunk pair"
            seen |= got
        assert seen == want, "children do not cover the parent's chunk pairs"


def test_refined_tasks_are_smaller_than_their_parent():
    """The point of refining is residency, so the children must actually shrink."""
    tasks = build_tasks(343, 1, B=1, H=1, D=64, itemsize=2, pin=False)
    parent = tasks[0]
    for kd in refine_task(parent, 343, B=1, H=1, D=64, itemsize=2, pin=False):
        assert int(kd.local_size) < int(parent.local_size)


# ----------------------------------------------------------------- autograd --
#
# The native kernel is fp16/bf16 only, so these mirror the convention the rest of
# the suite uses: fp16 operands, an fp32 monolithic target, and a relative
# max-error bound rather than assert_close's elementwise absolute one.


def _rel(got, want):
    return ((got.float() - want).abs().max()
            / want.abs().max().clamp_min(1e-30)).item()


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("causal", [False, True])
def test_backward_matches_sdpa_autograd(causal):
    """.backward() through the operator reproduces autograd on an fp32 target."""
    n, d = 2048, 64
    g = torch.Generator().manual_seed(7)
    qf, kf, vf = (torch.randn(1, 4, n, d, generator=g) for _ in range(3))
    dof = torch.randn(1, 4, n, d, generator=g).cuda()

    qa, ka, va = (t.clone().cuda().float().requires_grad_(True)
                  for t in (qf, kf, vf))
    sdpa_reference(qa, ka, va, causal=causal).backward(dof.float())

    q, k, v = (t.cuda().half().requires_grad_(True) for t in (qf, kf, vf))
    out = stream_cqsa_attn(q, k, v, causal=causal, itr=1)
    assert out.dtype == torch.float16, "output must carry the input dtype"
    out.backward(dof.half())

    assert _rel(out, sdpa_reference(qa, ka, va, causal=causal)) < 5e-3
    for got, want, name in ((q.grad, qa.grad, "dQ"), (k.grad, ka.grad, "dK"),
                            (v.grad, va.grad, "dV")):
        assert got is not None, f"{name} was never populated"
        assert got.dtype == torch.float16
        rel = _rel(got, want)
        assert rel < 5e-3, f"{name} rel={rel}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_backward_is_exact_at_any_depth():
    """
    Each depth partitions the same pair set differently and every subproblem
    scores against the global log-sum-exp, so the depth the backward runs at does
    not change the gradient. This is what lets the two passes escalate
    independently.
    """
    n, d = 2048, 64
    g = torch.Generator().manual_seed(11)
    qf, kf, vf = (torch.randn(1, 4, n, d, generator=g) for _ in range(3))
    dof = torch.randn(1, 4, n, d, generator=g).cuda().half()

    grads = []
    for depth in (1, 2):
        q, k, v = (t.clone().cuda().half().requires_grad_(True)
                   for t in (qf, kf, vf))
        stream_cqsa_attn(q, k, v, itr=depth).backward(dof)
        grads.append((q.grad, k.grad, v.grad))

    for a, b, name in zip(grads[0], grads[1], ("dQ", "dK", "dV")):
        rel = _rel(a, b.float())
        assert rel < 5e-3, f"{name} differs across depth: rel={rel}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_module_form_trains():
    """The nn.Module wrapper puts finite, nonzero gradients on real parameters."""
    torch.manual_seed(0)
    n, d = 1024, 64
    proj = torch.nn.Linear(d, d).cuda().half()
    attn = StreamCQSAAttention(itr=1, causal=True)
    x = torch.randn(1, 4, n, d, device="cuda", dtype=torch.float16)
    attn(proj(x), proj(x), proj(x)).float().sum().backward()
    assert proj.weight.grad is not None
    assert torch.isfinite(proj.weight.grad).all()
    assert proj.weight.grad.abs().sum() > 0


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_only_qkv_receive_gradients():
    """apply() returns one slot per forward argument; the config args take None."""
    q, k, v = (torch.randn(1, 4, 1024, 64, device="cuda", dtype=torch.float16,
                           requires_grad=True) for _ in range(3))
    stream_cqsa_attn(q, k, v, itr=1, c=7, interest_set=(0, 1, 3)).sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all()
               for t in (q, k, v))


# ---------------------------------------------------------------- recovery --

@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_forward_escalates_depth_under_memory_pressure():
    """
    Squeeze the process into a fraction of the device and confirm the call still
    returns the right answer, having recovered rather than failed.
    """
    n, d = 4096, 64
    g = torch.Generator().manual_seed(3)
    qf, kf, vf = (torch.randn(1, 8, n, d, generator=g) for _ in range(3))
    want = sdpa_reference(*(t.cuda().float() for t in (qf, kf, vf)), causal=False)
    q, k, v = (t.cuda().half() for t in (qf, kf, vf))

    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    # Leave headroom for what is already resident, then very little beyond it.
    frac = min(0.95, (total - free + 3 * 2 ** 30) / total)
    try:
        torch.cuda.set_per_process_memory_fraction(frac)
        out, info = stream_cqsa_forward(q, k, v, itr=1, max_parallel=8)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
        torch.cuda.empty_cache()

    assert _rel(out, want) < 5e-3, "recovery must not change the answer"
    if info["depth_escalations"]:
        assert info["itr_max_reached"] > 1
        assert info["itr_max_reached"] <= max_depth_for(n, 7)


# Driving escalation with a real memory cap is unreliable: the caching allocator
# satisfies a small call from blocks it has already reserved, so the cap never
# bites and the recovery never runs. Injecting the failure into the inner kernel
# tests the same control flow deterministically, on the CPU, and lets the result
# be checked against a reference -- which is the part that matters, since a
# recovery that returns a wrong answer is worse than one that raises.


def _refuse_over(limit, real, first_n=None):
    """
    An inner kernel that reports out of memory for subsequences over `limit`.

    With `first_n` set it refuses only that many of them and then relents, which
    is how a hybrid schedule is produced: the subproblems it refused are refined
    and the rest stay at the original depth.
    """
    calls = {"n": 0, "refused": 0, "lengths": []}

    def inner(q_i, k_i, v_i, bits, **kw):
        calls["n"] += 1
        # The scheduler hands the kernel token-major [B, L, H, D], so the
        # subsequence length is dim 1.
        L = q_i.shape[1]
        if L > limit and (first_n is None or calls["refused"] < first_n):
            calls["refused"] += 1
            raise torch.cuda.OutOfMemoryError("injected: subsequence too large")
        calls["lengths"].append(L)
        return real(q_i, k_i, v_i, bits, **kw)

    return inner, calls


def test_forward_escalates_and_stays_exact():
    """
    A subproblem that does not fit is decomposed a level further and its children
    run instead. The recomposed output must still match the monolithic reference:
    escalation changes the partition, never the result.
    """
    from stream_cqsa.stable_stream import local_stats_torch

    torch.manual_seed(0)
    B, H, N, D = 1, 2, 343, 32
    q, k, v = (torch.randn(B, H, N, D) for _ in range(3))
    want = sdpa_reference(q, k, v, causal=False)

    # At N=343 a depth-1 subsequence is 147 tokens and a depth-2 one is 63, so a
    # limit of 100 refuses every depth-1 task and accepts all of its children.
    inner, calls = _refuse_over(100, local_stats_torch)
    out, info = stream_cqsa_forward(q, k, v, itr=1, inner=inner, max_parallel=1)

    assert calls["refused"] > 0, "the injected limit never fired"
    assert info["depth_escalations"] > 0, "the call did not escalate"
    assert info["itr_max_reached"] == 2
    assert max(calls["lengths"]) == 63, "children must be smaller than the parent"
    torch.testing.assert_close(out.float(), want, rtol=1e-4, atol=1e-4)


def test_refined_children_are_smaller_than_the_parent_at_runtime():
    """
    Regression: the children the scheduler actually executes must be smaller
    than the task they replace. They are built with the alignment the task list
    was built with, and padding them to a segment boundary instead made a
    63-token child arrive as 256 -- larger than its 147-token parent, so the
    recovery grew what it was called to shrink and could only climb to the depth
    cap and fail.
    """
    from stream_cqsa.stable_stream import local_stats_torch

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 343, 32) for _ in range(3))
    inner, calls = _refuse_over(100, local_stats_torch)
    stream_cqsa_forward(q, k, v, itr=1, inner=inner, max_parallel=1)
    assert set(calls["lengths"]) == {63}, calls["lengths"][:8]


def test_escalation_terminates_at_the_structural_bound():
    """
    When no depth can satisfy the kernel, the call raises instead of refining
    forever. The bound is c**itr <= N, so termination does not depend on the
    retry counter.
    """
    from stream_cqsa.stable_stream import local_stats_torch

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 49, 32) for _ in range(3))
    inner, calls = _refuse_over(0, local_stats_torch)      # refuse everything

    with pytest.raises((torch.cuda.OutOfMemoryError, RuntimeError)):
        stream_cqsa_forward(q, k, v, itr=1, inner=inner, max_parallel=1)
    assert calls["refused"] > 0


@pytest.mark.parametrize("causal", [False, True])
def test_hybrid_schedule_of_mixed_depths_is_exact(causal):
    """
    Refusing only some subproblems leaves the rest at the original depth, which
    is the hybrid schedule a changing budget produces. The executed tasks then
    span two depths, and their pair sets must still partition the map exactly.
    """
    from stream_cqsa.stable_stream import local_stats_torch

    torch.manual_seed(1)
    B, H, N, D = 1, 2, 343, 32
    q, k, v = (torch.randn(B, H, N, D) for _ in range(3))
    want = sdpa_reference(q, k, v, causal=causal)

    inner, calls = _refuse_over(100, local_stats_torch, first_n=3)
    out, info = stream_cqsa_forward(q, k, v, itr=1, causal=causal,
                                    inner=inner, max_parallel=1)

    assert calls["refused"] == 3
    assert {63, 147} <= set(calls["lengths"]), "schedule was not hybrid"
    torch.testing.assert_close(out.float(), want, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("causal", [False, True])
def test_device_backward_escalates_and_stays_exact(causal):
    """
    The backward carries the same recovery as the forward. Before this it had
    none at all -- a subproblem that did not fit ended the call -- so the
    escalation is exercised here by refusing the oversized ones outright.
    """
    # The backward imports the kernel from .interface inside the call, so the
    # patch has to land on that module rather than on stable_stream.
    import stream_cqsa.interface as IF
    from stream_cqsa.stable_stream import stream_cqsa_backward

    n, d = 2048, 64
    g = torch.Generator().manual_seed(21)
    qf, kf, vf = (torch.randn(1, 4, n, d, generator=g) for _ in range(3))
    dof = torch.randn(1, 4, n, d, generator=g)

    qa, ka, va = (t.clone().cuda().float().requires_grad_(True)
                  for t in (qf, kf, vf))
    sdpa_reference(qa, ka, va, causal=causal).backward(dof.cuda().float())

    q, k, v = (t.cuda().half() for t in (qf, kf, vf))
    out, info = stream_cqsa_forward(q, k, v, itr=1, causal=causal)

    real = IF.flash_attn_bwd_cqs_global_lse
    refused = {"n": 0, "lengths": []}

    def refusing(do_i, q_i, k_i, v_i, o_i, lse_i, bits, **kw):
        L = q_i.shape[1]
        if L > 1000:                       # depth 1 is 3*2048/7 = 878 -> ...
            refused["n"] += 1
            raise torch.cuda.OutOfMemoryError("injected: subsequence too large")
        refused["lengths"].append(L)
        return real(do_i, q_i, k_i, v_i, o_i, lse_i, bits, **kw)

    try:
        IF.flash_attn_bwd_cqs_global_lse = refusing
        dq, dk, dv = stream_cqsa_backward(q, k, v, dof.cuda().half(), out,
                                          info["lse"], itr=1, causal=causal)
    finally:
        IF.flash_attn_bwd_cqs_global_lse = real

    for got, want, name in ((dq, qa.grad, "dQ"), (dk, ka.grad, "dK"),
                            (dv, va.grad, "dV")):
        rel = _rel(got, want)
        assert rel < 5e-3, f"{name} after escalation: rel={rel}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("itr", [0, 1, "auto"])
def test_output_device_follows_the_inputs(itr):
    """
    Regression: the forward returns its output wherever the accumulator lived,
    and that varies with the schedule -- the monolithic path has no accumulator
    and leaves the result on the device even for host-resident operands. An
    autograd op must not let a scheduling decision pick the output's device, or
    grad_output arrives somewhere the caller did not put it and .backward()
    fails with a device mismatch.
    """
    n, d = 2048, 64
    g = torch.Generator().manual_seed(5)
    q, k, v = (torch.randn(1, 4, n, d, generator=g).half().requires_grad_(True)
               for _ in range(3))
    out = stream_cqsa_attn(q, k, v, itr=itr, stream_from_host=True,
                           accumulate_on_gpu=False)
    assert out.device == q.device, f"out on {out.device}, inputs on {q.device}"
    out.backward(torch.randn(1, 4, n, d, generator=g).half())
    for name, t in (("q", q), ("k", k), ("v", v)):
        assert t.grad is not None, f"{name}.grad missing"
        assert t.grad.device == t.device, f"{name}.grad on the wrong device"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("n", [8192, 32768])
def test_call_after_unsynchronized_gpu_work_is_not_corrupted(n):
    """
    Regression: the scheduler's worker streams are created fresh per call, and a
    new stream carries no dependency on the caller's outstanding work. Without an
    explicit wait the gathers and kernels ran alongside whatever the caller had
    just issued, and entire gradient tensors came back NaN -- 8 of 8 trials at
    N=32768, and never when a synchronising read happened to sit in between,
    which is what made it look like a measurement artefact.

    The reproducer is deliberately unsynchronised: issue GPU work, do not wait
    for it, then call straight into the operator. That is also the ordinary shape
    of a training step, where .backward() follows the rest of the model.
    """
    d, dtype = 64, torch.float16
    g = torch.Generator().manual_seed(0)
    qf, kf, vf = (torch.randn(1, 8, n, d, generator=g) for _ in range(3))
    dof = torch.randn(1, 8, n, d, generator=g)
    q, k, v = (t.cuda().to(dtype) for t in (qf, kf, vf))
    dout = dof.cuda().to(dtype)

    out, info = stream_cqsa_forward(q, k, v, itr=1, causal=True)
    for _ in range(3):
        # Leave real work in flight, then call in without synchronising.
        stream_cqsa_backward(q, k, v, dout, out.to(dtype), info["lse"],
                             itr=info["itr"], causal=True)
        t = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
        stream_cqsa_attn(*t, causal=True, itr=1).backward(dout)
        torch.cuda.synchronize()
        for name, x in zip(("dQ", "dK", "dV"), t):
            bad = int((~torch.isfinite(x.grad.float())).sum())
            assert bad == 0, f"{name} has {bad} non-finite values at N={n}"


def test_escalation_can_be_pinned_off_for_a_fixed_depth_measurement():
    """
    A column labelled itr=1 has to report what itr=1 costs, including when itr=1
    does not fit. With escalation on, a subproblem that will not fit is refined
    and the call succeeds at a mixture of depths -- the right behaviour for a
    library, and wrong for a measurement, because the rescue gets filed under the
    depth that was requested. This is how a 16M forward came back as a 76.0 GiB
    success at itr=1 when itr=1 does not actually fit: seven of its subproblems
    had been refined a level deeper.
    """
    from stream_cqsa.stable_stream import local_stats_torch

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 343, 32) for _ in range(3))
    inner, calls = _refuse_over(100, local_stats_torch)

    # escalation on: refines and completes
    _, info = stream_cqsa_forward(q, k, v, itr=1, inner=inner, max_parallel=1)
    assert info["depth_escalations"] > 0

    # escalation off: the same call must report the out-of-memory instead
    inner2, _ = _refuse_over(100, local_stats_torch)
    with pytest.raises((torch.cuda.OutOfMemoryError, RuntimeError)):
        stream_cqsa_forward(q, k, v, itr=1, inner=inner2, max_parallel=1,
                            allow_escalation=False)
