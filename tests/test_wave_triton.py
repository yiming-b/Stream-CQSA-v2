"""The wave engine on the Triton kernel: agrees with the CUDA wave kernel (fp16 rounding) and
with float64; host accumulator and host-resident inputs; c=3 and c=133; non-causal."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

try:
    import triton  # noqa: F401
    HAVE_TRITON = True
except Exception:
    HAVE_TRITON = False


def _rel(a, b):
    b = b.to(a.device)
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _native():
    from stream_cqsa.native_wave import native_available, native_supports
    return native_available() and native_supports(torch.float16, 64)


@pytest.mark.skipif(not HAVE_TRITON, reason="needs triton")
@pytest.mark.parametrize("N,itr,c,causal,cap,host,acc_cpu", [
    (8192, 1, 7, True, None, False, False), (8192, 1, 7, False, None, False, False), (8192, 1, 7, True, 2, False, False),
    (8192, 2, 7, True, 5, False, False), (8192, 1, 3, True, None, False, False), (8192, 2, 3, False, None, False, False),
    (8192, 1, 7, True, 2, False, True), (8192, 1, 7, True, None, True, True), (4096, 1, 7, True, 3, True, False)])
def test_triton_wave_vs_fp64_and_cuda(N, itr, c, causal, cap, host, acc_cpu):
    from stream_cqsa.native_wave import wave_forward
    from stream_cqsa.autoconfig import QUORUM_SETS
    dev = torch.device("cuda"); H, D = 8, 64
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, H, N, D, dtype=torch.float16, generator=g) for _ in range(3))
    qd, kd, vd = (t.to(dev) for t in (q, k, v))
    ref = F.scaled_dot_product_attention(qd.double(), kd.double(), vd.double(), is_causal=causal)
    e_sdpa = _rel(F.scaled_dot_product_attention(qd, kd, vd, is_causal=causal), ref)
    src = (q, k, v) if host else (qd, kd, vd)
    out_t, info_t = wave_forward(*src, causal=causal, itr=itr, c=c, interest_set=QUORUM_SETS[c], kernel="triton",
                                 max_wave_subproblems=cap, accumulate_on_gpu=not acc_cpu)
    assert info_t["kernel"] == "triton"
    assert out_t.device.type == ("cpu" if acc_cpu else "cuda")
    assert _rel(out_t, ref) <= 1.2 * e_sdpa + 1e-6
    if _native():
        out_c, info_c = wave_forward(qd, kd, vd, causal=causal, itr=itr, c=c, interest_set=QUORUM_SETS[c], kernel="cuda")
        assert _rel(out_t, out_c) < 4e-4                                     # two kernels, fp16 rounding
        assert (info_t["lse"].to(dev) - info_c["lse"]).abs().max().item() < 1e-4


@pytest.mark.skipif(not HAVE_TRITON, reason="needs triton")
def test_triton_wave_c133_and_deterministic():
    from stream_cqsa.native_wave import wave_forward
    from stream_cqsa.autoconfig import QUORUM_SETS, is_perfect_difference_set
    assert is_perfect_difference_set(133, QUORUM_SETS[133]) and is_perfect_difference_set(3, QUORUM_SETS[3])
    dev = torch.device("cuda"); H, D = 8, 64; N = 16384
    g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(1, H, N, D, dtype=torch.float16, generator=g).to(dev) for _ in range(3))
    o1, i1 = wave_forward(q, k, v, causal=True, itr=1, c=133, interest_set=QUORUM_SETS[133], kernel="triton")
    o2, _ = wave_forward(q, k, v, causal=True, itr=1, c=133, interest_set=QUORUM_SETS[133], kernel="triton")
    assert i1["n_subproblems"] == 133 and torch.equal(o1, o2)
    o7, _ = wave_forward(q, k, v, causal=True, itr=1, c=7, interest_set=QUORUM_SETS[7], kernel="triton")
    assert _rel(o1, o7) < 4e-4


@pytest.mark.skipif(not HAVE_TRITON, reason="needs triton")
def test_api_kernel_wave_triton():
    from stream_cqsa.api import attention
    dev = torch.device("cuda"); H, D = 8, 64; N = 8192
    g = torch.Generator().manual_seed(2)
    q, k, v = (torch.randn(1, H, N, D, dtype=torch.float16, generator=g).to(dev) for _ in range(3))
    ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), is_causal=True)
    e_sdpa = _rel(F.scaled_dot_product_attention(q, k, v, is_causal=True), ref)
    for acc in (True, False):
        o = attention(q, k, v, is_causal=True, itr=1, kernel="wave-triton", accumulate_on_gpu=acc)
        assert o.device == q.device and _rel(o, ref) <= 1.2 * e_sdpa + 1e-6
