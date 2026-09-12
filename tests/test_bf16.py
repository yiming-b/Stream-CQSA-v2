"""bf16 support: forward and backward, both head dims, causal and not.

Guards the `common` kernel set. Error tolerances are set from measured values
(bf16 is ~8x fp16 because it carries 8 fewer mantissa bits); they check that
accuracy is bounded by the dtype rather than by a defect in the CQS path.
"""
import math

import pytest
import torch

from stream_cqsa.stable_stream import stream_cqsa_forward, stream_cqsa_backward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# measured on A100/sm80 at N=4096: fp16 fwd 1.9e-4 / bwd 3.1e-4,
# bf16 fwd 1.5e-3 / bwd 2.5e-3. Tolerances are ~2x headroom over those.
TOL = {torch.float16: (4e-4, 7e-4), torch.bfloat16: (3e-3, 5e-3)}


def _ref(q, k, v, causal):
    qd, kd, vd = (t.double().requires_grad_() for t in (q, k, v))
    s = qd @ kd.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        n = q.shape[-2]
        m = torch.triu(torch.ones(n, n, device=q.device, dtype=torch.bool), 1)
        s = s.masked_fill(m, float("-inf"))
    return s.softmax(-1) @ vd, (qd, kd, vd)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_dtype_fwd_bwd(dtype, head_dim, causal):
    torch.manual_seed(0)
    b, h, n = 1, 2, 2048
    q, k, v = (torch.randn(b, h, n, head_dim, device="cuda", dtype=dtype) for _ in range(3))

    out, info = stream_cqsa_forward(q, k, v, itr=1, causal=causal)
    lse = info["lse"] if isinstance(info, dict) else info.lse

    ref, (qd, kd, vd) = _ref(q, k, v, causal)
    ftol, btol = TOL[dtype]
    assert (out.double() - ref).norm() / ref.norm() < ftol

    dout = torch.randn(b, h, n, head_dim, device="cuda", dtype=dtype)
    dq, dk, dv = stream_cqsa_backward(
        q, k, v, dout, out.to(dtype), lse, itr=1, causal=causal
    )
    rq, rk, rv = torch.autograd.grad(ref, [qd, kd, vd], dout.double())
    for got, want in ((dq, rq), (dk, rk), (dv, rv)):
        assert (got.double() - want).norm() / want.norm() < btol
