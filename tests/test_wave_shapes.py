"""Native wave kernel across dtypes and head dims (skips the ones the build lacks)."""
import pytest, torch, torch.nn.functional as F
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
dev = torch.device("cuda")


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


@cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_wave_forward_backward_shapes(dtype, D, causal):
    from stream_cqsa.native_wave import native_available, native_supports, wave_forward, wave_attention
    if not native_available():
        pytest.skip("cqsa_native not built")
    if not native_supports(dtype, D):
        pytest.skip(f"cqsa_native build lacks {dtype} head_dim={D}")
    N, H = 4096, 4
    q, k, v = (torch.randn(1, H, N, D, device=dev, dtype=dtype) for _ in range(3))
    ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), is_causal=causal)
    o_s = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    out, info = wave_forward(q, k, v, causal=causal, itr=1, c=7, interest_set=(0, 1, 3))
    assert rel(out, ref) <= 1.2 * rel(o_s, ref) + 1e-6
    # backward
    qg, kg, vg = (t.clone().requires_grad_(True) for t in (q, k, v))
    dout = torch.randn(1, H, N, D, device=dev, dtype=dtype)
    wave_attention(qg, kg, vg, causal=causal, itr=1, c=7, interest_set=(0, 1, 3)).backward(dout.float())
    q2, k2, v2 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    F.scaled_dot_product_attention(q2, k2, v2, is_causal=causal).backward(dout.double())
    q3, k3, v3 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    F.scaled_dot_product_attention(q3, k3, v3, is_causal=causal).backward(dout)
    for g, g64, gs in zip((qg.grad, kg.grad, vg.grad), (q2.grad, k2.grad, v2.grad), (q3.grad, k3.grad, v3.grad)):
        assert rel(g, g64) <= 1.5 * rel(gs, g64) + 1e-4
