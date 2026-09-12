"""Host-device streamed backward (Algorithm algo:bwd, global-lse form).

The backward loop gathers `idx`, computes on the subsequence, and scatters back
with index_add -- no step needs the full N-token tensors -- so it streams from
host memory exactly as the forward does. These tests check that streaming
changes only *where* tensors live, never the result.
"""
import pytest
import torch

from stream_cqsa.stable_stream import stream_cqsa_forward, stream_cqsa_backward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _setup(n=2048, b=1, h=2, d=64, dtype=torch.float16, itr=1, causal=False):
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=dtype) for _ in range(3))
    dout = torch.randn(b, h, n, d, device="cuda", dtype=dtype)
    out, info = stream_cqsa_forward(q, k, v, itr=itr, causal=causal)
    lse = info["lse"] if isinstance(info, dict) else info.lse
    return q, k, v, dout, out.to(dtype), lse


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("itr", [1, 2])
@pytest.mark.parametrize("causal", [False, True])
def test_host_matches_device(dtype, itr, causal):
    q, k, v, dout, out, lse = _setup(dtype=dtype, itr=itr, causal=causal)
    kw = dict(itr=itr, causal=causal)
    ref = stream_cqsa_backward(q, k, v, dout, out, lse, **kw)
    got = stream_cqsa_backward(
        *(t.cpu() for t in (q, k, v, dout, out, lse)), stream_from_host=True, **kw
    )
    # Only the fp32 index_add ordering differs, so this is far tighter than the
    # dtype's own error floor (3e-4 fp16 / 2.4e-3 bf16 against float64).
    for g, r in zip(got, ref):
        num = (g.cpu().double() - r.cpu().double()).norm()
        assert num / r.cpu().double().norm() < 1e-4


def test_host_uses_less_device_memory():
    """The point of the path: device peak must not scale with N."""
    q, k, v, dout, out, lse = _setup(n=8192)
    kw = dict(itr=2, causal=True)

    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    stream_cqsa_backward(q, k, v, dout, out, lse, **kw)
    torch.cuda.synchronize()
    dev_peak = torch.cuda.max_memory_allocated()

    hosted = tuple(t.cpu() for t in (q, k, v, dout, out, lse))
    del q, k, v, dout, out, lse
    torch.cuda.empty_cache()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    stream_cqsa_backward(*hosted, stream_from_host=True, **kw)
    torch.cuda.synchronize()
    host_peak = torch.cuda.max_memory_allocated()

    assert host_peak < dev_peak / 2, f"host {host_peak} vs device {dev_peak}"


def test_cpu_inputs_without_flag_are_rejected():
    """Silently falling back would hide an OOM-recovery path that was asked for."""
    t = torch.randn(1, 2, 256, 64, dtype=torch.float16)
    lse = torch.randn(1, 2, 256)
    with pytest.raises(ValueError, match="stream_from_host"):
        stream_cqsa_backward(t, t, t, t, t, lse, itr=1)


@pytest.mark.parametrize("causal", [False, True])
def test_itr0_returns_lse_and_backward_works(causal):
    """
    itr=0 (no decomposition) must still hand back the global lse.

    itr="auto" selects itr=0 below the OOM boundary, so a caller following the
    documented forward-then-backward flow would otherwise hit a KeyError purely
    because the planner declined to decompose.
    """
    torch.manual_seed(0)
    b, h, n, d = 1, 2, 1024, 64
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(3))
    dout = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)

    out, info = stream_cqsa_forward(q, k, v, itr=0, causal=causal)
    assert "lse" in info, "itr=0 must return the global lse"
    assert tuple(info["lse"].shape) == (b, h, n)

    dq, dk, dv = stream_cqsa_backward(
        q, k, v, dout, out.to(torch.float16), info["lse"], itr=0, causal=causal
    )
    import math
    qd, kd, vd = (t.double().requires_grad_() for t in (q, k, v))
    sc = qd @ kd.transpose(-1, -2) / math.sqrt(d)
    if causal:
        sc = sc.masked_fill(
            torch.triu(torch.ones(n, n, device="cuda", dtype=torch.bool), 1),
            float("-inf"))
    ref = sc.softmax(-1) @ vd
    rq, rk, rv = torch.autograd.grad(ref, [qd, kd, vd], dout.double())
    for got, want in ((dq, rq), (dk, rk), (dv, rv)):
        assert (got.double() - want).norm() / want.norm() < 7e-4
