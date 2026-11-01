"""
v2.2.4 regression: every quorum size pinned through ``stream_cqsa.attention(..., c=)`` computes exact attention.

In v2.1.0-v2.2.3 ``attention(..., itr=1, c=73)`` pinned ``c`` but kept the plan's interest set ``(0, 1, 3)``
(the c=7 set), so the engine ran 73 subproblems of 3 chunks each, most chunk pairs were never gathered, and the
result had a norm-relative error of about 1.3 against flash attention, in a third of the time. Every c other
than 7 was wrong (c=31: 0.72, c=57: 1.1, c=91: 1.4, c=133: 1.8); c=7 was correct only because its set is the
default. Now a pinned ``c`` takes ``QUORUM_SETS[c]`` and the task builders refuse any pair that is not a perfect
difference set (``stable_stream.check_quorum_set``).

Rows: every c in ``QUORUM_SETS`` at N=65536, itr=1, fp16, causal, B=1 H=4 D=64, classic engine
(``kernel="cuda"``), forward and backward against torch SDPA on the same inputs. The call is pinned (``itr=1``)
so the monolithic path cannot be what is compared, and the info is asserted to report ``c`` subproblems with
``QUORUM_SETS[c]``. Shared GPU: correctness only, < 2 GiB per row.
"""
import pytest
import torch
import torch.nn.functional as F

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
B, H, D, N = 1, 4, 64, 65536
# norm-relative error vs SDPA fp16 on the same inputs: 1.3e-4..2.4e-4 (out) / 3.5e-4 (grads) for every c
# (A100, v11 module); the v2.2.3 failure is ~1e0
TOL_OUT, TOL_GRAD = 1e-3, 1.5e-3


def _quorum_sets():
    from stream_cqsa.autoconfig import QUORUM_SETS
    return QUORUM_SETS


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _inputs():
    g = torch.Generator(device="cuda").manual_seed(N)
    q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, generator=g) for _ in range(3))
    dout = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, generator=g)
    return q, k, v, dout


def _reference(q, k, v, dout):
    qr, kr, vr = (t.clone().requires_grad_(True) for t in (q, k, v))
    out = F.scaled_dot_product_attention(qr, kr, vr, is_causal=True)
    grads = torch.autograd.grad(out, (qr, kr, vr), dout)
    return out.detach(), [g_.detach() for g_ in grads]


@cuda
@pytest.mark.parametrize("c", sorted(_quorum_sets()), ids=lambda c: f"c{c}")
def test_pinned_quorum_set_is_exact(c):
    from stream_cqsa import attention
    QUORUM_SETS = _quorum_sets()
    q, k, v, dout = _inputs()
    ref, ref_grads = _reference(q, k, v, dout)
    qs, ks, vs = (t.clone().requires_grad_(True) for t in (q, k, v))
    out, info = attention(qs, ks, vs, is_causal=True, kernel="cuda", itr=1, c=c, bwd_itr=1, return_info=True)
    # the decomposition must have run (an unpinned call that fits would be SDPA against itself)
    assert info["engine"] == "classic" and info.get("itr") == 1 and info.get("c") == c, info
    assert tuple(info.get("interest_set", QUORUM_SETS[c])) == QUORUM_SETS[c], info
    grads = torch.autograd.grad(out, (qs, ks, vs), dout)
    e_out = _rel(out, ref)
    e_grad = max(_rel(g, r) for g, r in zip(grads, ref_grads))
    assert e_out < TOL_OUT, f"c={c}: output rel error {e_out:.2e} (v2.2.3 bug: ~1e0)"
    assert e_grad < TOL_GRAD, f"c={c}: gradient rel error {e_grad:.2e} (v2.2.3 bug: ~1e0)"


@cuda
def test_pinned_c_without_interest_set_reports_its_own_set():
    """The forward info of a pinned c=73 reports 73 subproblems on (0, 1, 3, 7, 15, 31, 36, 54, 63)."""
    from stream_cqsa import attention
    q, k, v, _ = _inputs()
    with torch.no_grad():
        out, info = attention(q, k, v, is_causal=True, kernel="cuda", itr=1, c=73, return_info=True)
    assert info["n_subproblems"] == 73 and tuple(info["interest_set"]) == _quorum_sets()[73], info


def test_task_builders_refuse_a_mismatched_pair():
    from stream_cqsa.stable_stream import build_tasks, check_quorum_set
    for c, iset in ((73, (0, 1, 3)), (7, (0, 1, 2)), (7, (0, 1, 1)), (13, (0, 1, 3))):
        with pytest.raises(ValueError, match="perfect difference set"):
            check_quorum_set(c, iset)
        with pytest.raises(ValueError, match="perfect difference set"):
            build_tasks(8192, 1, B=1, H=1, D=64, itemsize=2, c=c, interest_set=iset)


def test_every_quorum_set_passes_the_guard():
    from stream_cqsa.stable_stream import check_quorum_set
    for c, iset in _quorum_sets().items():
        check_quorum_set(c, iset)


def test_attention_rejects_a_mismatched_or_unknown_pin():
    from stream_cqsa import attention
    q, k, v = (torch.randn(1, 2, 8192, 64, dtype=torch.float16) for _ in range(3))
    hw = {"cuda:0": "40GiB", "host": "64GiB"}
    with pytest.raises(ValueError, match="perfect difference set"):
        attention(q, k, v, is_causal=True, hardware=hw, itr=1, c=73, interest_set=(0, 1, 3))
    with pytest.raises(ValueError, match="not a known quorum size"):
        attention(q, k, v, is_causal=True, hardware=hw, itr=1, c=11)
