"""The default backward is the Triton kernel; it must match SDPA's gradients for fp16/bf16, hdim 64/128."""
import pytest, torch, torch.nn.functional as F
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
dev = torch.device("cuda")


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


@cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_default_backward_matches_sdpa(dtype, D, causal):
    pytest.importorskip("triton")
    from stream_cqsa import stream_cqsa_attn
    N, H = 4096, 4
    q, k, v = (torch.randn(1, H, N, D, device=dev, dtype=dtype, requires_grad=True) for _ in range(3))
    dout = torch.randn(1, H, N, D, device=dev, dtype=dtype)
    stream_cqsa_attn(q, k, v, causal=causal, itr=1).backward(dout.float())
    q2, k2, v2 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    F.scaled_dot_product_attention(q2, k2, v2, is_causal=causal).backward(dout.double())
    q3, k3, v3 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    F.scaled_dot_product_attention(q3, k3, v3, is_causal=causal).backward(dout)
    for g, g64, gs in zip((q.grad, k.grad, v.grad), (q2.grad, k2.grad, v2.grad), (q3.grad, k3.grad, v3.grad)):
        assert rel(g, g64) <= 1.5 * rel(gs, g64) + 1e-4
