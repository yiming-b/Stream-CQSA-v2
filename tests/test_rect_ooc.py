"""Rectangular OOC rival (R2 fixed / R3 adaptive): exact against float64, both traversals, uneven N,
causal and non-causal, under a small device budget."""
import pytest, torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
try:
    import flash_attn  # noqa: F401
    HAVE_FA = True
except Exception:
    HAVE_FA = False


def _rel(a, b):
    a = a.double().cpu(); b = b.double().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def _ref(q, k, v, dout, causal):
    qd, kd, vd = (t.cuda().double().requires_grad_(True) for t in (q, k, v))
    o = F.scaled_dot_product_attention(qd, kd, vd, is_causal=causal)
    dq, dk, dv = torch.autograd.grad(o, [qd, kd, vd], dout.cuda().double())
    return o.detach(), dq, dk, dv


@pytest.mark.skipif(not HAVE_FA, reason="needs flash-attn")
@pytest.mark.parametrize("N,causal,schedule,tiles", [
    (4096, True, "fixed", (1024, 1024)), (4096, False, "fixed", (1024, 2048)), (6000, True, "fixed", (1024, 1024)),
    (6000, True, "adaptive", (1536, 1024)), (4096, False, "adaptive", (1024, 1024)), (6000, True, "adaptive", None)])
def test_rect_ooc_exact(N, causal, schedule, tiles):
    from stream_cqsa.baselines.rect_ooc import RectOOC
    H, D = 8, 64
    g = torch.Generator().manual_seed(0)
    q, k, v, dout = (torch.randn(1, H, N, D, dtype=torch.float16, generator=g) for _ in range(4))
    o_ref, dq_ref, dk_ref, dv_ref = (t.cpu() for t in _ref(q, k, v, dout, causal))
    # FA-2's own error is the yardstick
    o_fa = F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=causal).cpu()
    e_fa = _rel(o_fa, o_ref)
    torch.cuda.empty_cache()
    kw = dict(tile_q=tiles[0], tile_k=tiles[1]) if tiles else {}
    r = RectOOC(budget_bytes=2 << 30, schedule=schedule, **kw)
    out, lse, st = r.forward(q, k, v, causal=causal)
    assert out.shape == q.shape and not out.is_cuda and st.n_pairs >= 1 and st.h2d_bytes > 0
    assert _rel(out, o_ref) <= 1.2 * e_fa + 1e-6, (_rel(out, o_ref), e_fa)
    assert st.peak_alloc_bytes - st.base_alloc_bytes <= r.budget and not st.budget_breached
    dq, dk, dv, sb = r.backward(q, k, v, out, dout, lse, causal=causal)
    qd = q.cuda().requires_grad_(True); kd = k.cuda().requires_grad_(True); vd = v.cuda().requires_grad_(True)
    o16 = F.scaled_dot_product_attention(qd, kd, vd, is_causal=causal)
    g16 = [t.cpu() for t in torch.autograd.grad(o16, [qd, kd, vd], dout.cuda())]
    for got, ref, fa in zip((dq, dk, dv), (dq_ref, dk_ref, dv_ref), g16):
        assert _rel(got, ref) <= 1.5 * _rel(fa, ref) + 1e-4, (_rel(got, ref), _rel(fa, ref))
    assert sb.traversal == ("kv_outer" if schedule == "adaptive" else "q_outer")
    assert sb.n_pairs >= 1 and not sb.budget_breached


@pytest.mark.skipif(not HAVE_FA, reason="needs flash-attn")
def test_rect_ooc_tile_plan_and_cache():
    from stream_cqsa.baselines.rect_ooc import plan_tiles, RectOOC
    Tq, Tk = plan_tiles(1 << 20, 8, 64, 2, 10 << 30, direction="fwd", double=False)
    assert Tq >= 1024 and Tk >= 1024 and Tq % 1024 == 0
    Tq2, _ = plan_tiles(1 << 20, 8, 64, 2, 10 << 30, direction="bwd", double=True)
    assert Tq2 <= Tq                                   # the backward's pair costs more
    r = RectOOC(budget_bytes=2 << 30, schedule="adaptive")
    H, D = 8, 64; g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(1, H, 4096, D, dtype=torch.float16, generator=g) for _ in range(3))
    _, _, s1 = r.forward(q, k, v); _, _, s2 = r.forward(q, k, v)
    assert s1.probe_s > 0 and not s1.probe_cached and s2.probe_cached and s2.probe_s == 0.0
