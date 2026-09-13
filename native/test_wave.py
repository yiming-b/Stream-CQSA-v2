"""
Exactness tests for the native wave kernel + engine (next/native).

    python next/native/test_wave.py            (needs a GPU and the cqsa_native build)
"""
import os, sys, time, math
import torch
import torch.nn.functional as F
os.environ.setdefault("CQSA_CUDA_MODULE", "cqsa_cuda")
os.environ.setdefault("CQSA_CUDA_MODULE_NONCAUSAL", "cqsa_cuda_nc")
from stream_cqsa.native_wave import wave_forward, wave_backward, wave_attention, native_ext, wave_tasks, build_wave_tables, block_base_relative, SEG_ALIGN
from stream_cqsa.stable_stream import stream_cqsa_forward
from stream_cqsa.autoconfig import QUORUM_SETS
import stream_cqsa.interface as I

dev = torch.device("cuda")
ext = native_ext()
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        fails += 1


def ref64(q, k, v, causal):
    qf, kf, vf = (t.double() for t in (q, k, v))
    s = qf @ kf.transpose(-1, -2) * (q.shape[-1] ** -0.5)
    if causal:
        N = q.shape[-2]
        s = s.masked_fill(torch.ones(N, N, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    p = torch.softmax(s, -1)
    return p @ vf, torch.logsumexp(s, -1)


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


torch.manual_seed(0)
# ---- 1. kernel alone: a single subproblem through fwd_wave (W=1) equals the v11 kernel bit for bit
N, H, D = 8192, 4, 64
q, k, v = (torch.randn(1, H, N, D, device=dev, dtype=torch.float16) for _ in range(3))
for causal in (True, False):
    tasks = wave_tasks(N, 1, 7, (0, 1, 3), H=H, D=D, itemsize=2)
    t = tasks[2]
    idx = t.token_ids.to(dev)
    q_i, k_i, v_i = (x.index_select(2, idx).transpose(1, 2) for x in (q, k, v))     # [1, L, H, D]
    o_ref = I.flash_attn_func_cqs_group_bits(q_i, k_i, v_i, t.group_bits.to(dev), causal=causal, softmax_scale=D ** -0.5,
                                             cqs_blk_or=t.extra["blk_or"].to(dev), cqs_blk_and=t.extra["blk_and"].to(dev),
                                             fp32_out=True, return_attn_probs=True)
    o_ref, lse_ref = o_ref[0], o_ref[1]
    tb = build_wave_tables([t], dev, uniform=causal)
    L = int(t.local_size)
    o_w, lse_w = ext.fwd_wave(q_i[0].contiguous(), k_i[0].contiguous(), v_i[0].contiguous(), tb.cu, tb.max_L, tb.total,
                              tb.bits, tb.blk_or, tb.blk_and, tb.blk_cu, None, None, 0, D ** -0.5, causal,
                              uniform_S=tb.S, uniform_W=tb.W) if not causal else (None, None)
    if causal:
        # uniform layout: the packed wave is [W*S, H, D]; pad the single subproblem to S rows
        pad = lambda x: torch.cat([x[0], x[0][:1].expand(tb.S - L, -1, -1)], 0).contiguous()
        o_w, lse_w = ext.fwd_wave(pad(q_i), pad(k_i), pad(v_i), tb.cu, tb.max_L, tb.total, tb.bits, tb.blk_or, tb.blk_and,
                                  tb.blk_cu, None, None, 0, D ** -0.5, causal, uniform_S=tb.S, uniform_W=tb.W)
        o_w, lse_w = o_w[:L], lse_w[0, :, :L]
        check(f"kernel W=1 packed == v11 kernel (causal={causal})", torch.equal(o_w, o_ref[0]) and torch.equal(lse_w, lse_ref[0]),
              f"max|d|={ (o_w - o_ref[0]).abs().max().item():.2e}")
    else:   # non-causal is served by the v9 kernel in the old build (different loop), so only causal must be bit-identical
        check(f"kernel W=1 packed ~= v9 kernel (causal={causal})", rel(o_w, o_ref[0]) < 1e-5 and (lse_w - lse_ref[0]).abs().max().item() < 1e-4,
              f"rel={rel(o_w, o_ref[0]):.1e} lse max|d|={(lse_w - lse_ref[0]).abs().max().item():.1e}")
    # gather-free (block map into the original) == packed
    qt, kt, vt = (x[0].transpose(0, 1) for x in (q, k, v))
    bb = block_base_relative(tb, lambda g: g).to(dev)
    o_g, lse_g = ext.fwd_wave(qt, kt, vt, tb.cu, tb.max_L, tb.total, tb.bits, tb.blk_or, tb.blk_and, tb.blk_cu, bb, tb.bb_cu,
                              SEG_ALIGN, D ** -0.5, causal, uniform_S=tb.S, uniform_W=tb.W)
    if causal:
        o_g, lse_g = o_g[:L], lse_g[0, :, :L]
    check(f"kernel W=1 gather-free == packed (causal={causal})", torch.equal(o_g, o_w) and torch.equal(lse_g, lse_w))

# ---- 2. engine: wave_forward vs fp64 and vs the previous engine, several configs
for (N, B, itr, c, causal) in [(4096, 2, 1, 7, True), (4096, 1, 1, 7, False), (8192, 1, 2, 7, True), (8192, 1, 1, 13, True),
                                (6000, 1, 1, 7, True), (16384, 1, 1, 31, False)]:
    q, k, v = (torch.randn(B, H, N, D, device=dev, dtype=torch.float16) for _ in range(3))
    iset = QUORUM_SETS[c]
    out, info = wave_forward(q, k, v, causal=causal, itr=itr, c=c, interest_set=iset)
    o64, l64 = ref64(q, k, v, causal)
    e_w = rel(out, o64)
    o_fa = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    e_fa = rel(o_fa, o64)
    old, _ = stream_cqsa_forward(q, k, v, itr=itr, causal=causal, c=c, interest_set=iset, allow_escalation=False)
    e_old = rel(old, o64)
    lse_err = (info["lse"].double() - l64).abs().max().item()
    check(f"wave_forward N={N} B={B} itr={itr} c={c} causal={causal}", e_w <= 1.2 * e_fa and lse_err < 1e-2,
          f"rel err wave {e_w:.2e} | FA-2 {e_fa:.2e} | old engine {e_old:.2e} | lse max|d| {lse_err:.1e} | waves {info['wave_sizes']}")
    # several waves (cap 2 subproblems per wave) must give the same result up to merge order
    out2, info2 = wave_forward(q, k, v, causal=causal, itr=itr, c=c, interest_set=iset, max_wave_subproblems=2)
    check(f"  wave cap=2 ({info2['n_waves']} waves) agrees", rel(out2, out) < 1e-6, f"rel {rel(out2, out):.1e}")
    out3, _ = wave_forward(q, k, v, causal=causal, itr=itr, c=c, interest_set=iset)
    check(f"  deterministic (bit-identical rerun)", torch.equal(out3, out))

# ---- 3. host-resident inputs through the chunk pool
N = 8192
q, k, v = (torch.randn(1, H, N, D, dtype=torch.float16) for _ in range(3))
out_h, info_h = wave_forward(q, k, v, causal=True, itr=1, c=7, interest_set=(0, 1, 3))
out_d, _ = wave_forward(q.to(dev), k.to(dev), v.to(dev), causal=True, itr=1, c=7, interest_set=(0, 1, 3))
check(f"host-resident (pool slots {info_h['pool_slots']}, waves {info_h['wave_sizes']}) == device-resident", rel(out_h, out_d) < 1e-6, f"rel {rel(out_h, out_d):.1e}")
out_h2, info_h2 = wave_forward(q, k, v, causal=True, itr=1, c=7, interest_set=(0, 1, 3), pool_slots=4)
check(f"host-resident small pool (slots 4, waves {info_h2['wave_sizes']})", rel(out_h2, out_d) < 1e-6, f"rel {rel(out_h2, out_d):.1e}")

# ---- 4. backward vs fp64 autograd
for (N, itr, c, causal) in [(2048, 1, 7, True), (2048, 1, 7, False), (4096, 2, 7, True)]:
    q, k, v = (torch.randn(1, H, N, D, device=dev, dtype=torch.float16, requires_grad=True) for _ in range(3))
    dout = torch.randn(1, H, N, D, device=dev, dtype=torch.float16)
    iset = QUORUM_SETS[c]
    out = wave_attention(q, k, v, causal=causal, itr=itr, c=c, interest_set=iset)
    out.backward(dout.float())
    g_w = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v): t.grad = None
    q64, k64, v64 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    o64, _ = ref64(q64, k64, v64, causal)
    o64.backward(dout.double())
    g_64 = [t.grad for t in (q64, k64, v64)]
    # FA-2 (SDPA) gradients as the yardstick
    q2, k2, v2 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    F.scaled_dot_product_attention(q2, k2, v2, is_causal=causal).backward(dout)
    g_fa = [t.grad for t in (q2, k2, v2)]
    errs = [rel(a, b) for a, b in zip(g_w, g_64)]
    errs_fa = [rel(a, b) for a, b in zip(g_fa, g_64)]
    check(f"wave backward N={N} itr={itr} causal={causal}", all(e <= 1.5 * f + 1e-4 for e, f in zip(errs, errs_fa)),
          "dq/dk/dv rel err wave " + "/".join(f"{e:.1e}" for e in errs) + "  FA-2 " + "/".join(f"{e:.1e}" for e in errs_fa))
    # two waves
    _, info_f = wave_forward(q.detach(), k.detach(), v.detach(), causal=causal, itr=itr, c=c, interest_set=iset)
    (dq2, dk2, dv2), _ = wave_backward(q.detach(), k.detach(), v.detach(), out.detach(), dout, info_f["lse"],
                                       causal=causal, itr=itr, c=c, interest_set=iset, max_wave_subproblems=3)
    check(f"  backward wave cap=3 agrees", max(rel(a, b) for a, b in zip((dq2, dk2, dv2), g_w)) < 2e-3,
          f"rel {max(rel(a, b) for a, b in zip((dq2, dk2, dv2), g_w)):.1e}")

print(f"\n{fails} failures", flush=True)
sys.exit(1 if fails else 0)
