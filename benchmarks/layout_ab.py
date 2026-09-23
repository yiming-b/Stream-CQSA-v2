"""
Layout A/B for the v11 CQS forward kernel, plus backward and small-L splits, in ONE process on ONE GPU.

Answers docs/ultimate/00_map.md sec 7.2 items 1-2:
  (a) is v11's 1.40x-vs-FA-2 (classic layout, align=1, L=56175) versus 0.98x (wave layout,
      align=128, L=56064) a LAYOUT effect?  Same binary, both task layouts, FA-2 arm, one job.
  (b) backward: how much of the 2.0x is dead-tile overhead (real pattern vs all-zero bits)
      versus live-tile slowdown (all-zero bits vs FA-2's plain backward at the same L)?
  (c) small L (c=73 at N=131072, and c=7 at N=16384): real vs zero pattern vs FA-2.

FA-2 arm: flash_attn is NOT installed in this env, so the FA-2 arm is torch SDPA with the
FLASH_ATTENTION backend forced (torch.nn.attention.sdpa_kernel), i.e. torch's own build of the
FlashAttention-2 kernels. The kernel names seen by the profiler are recorded to prove it.

Timing: 1 warm-up + 5 timed runs with CUDA events, median (min also stored).
Output: results/ultimate/layout_ab.json and a printed table.

Run (see slurm/layout_ab.slurm):
    source next/env_next.sh; PYTHONPATH=...:next/native
    CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9
    python -P -u benchmarks/layout_ab.py
"""
import os, sys, json, socket, subprocess, statistics, time, traceback

os.environ.setdefault("CQSA_CUDA_MODULE", "cqsa_cuda_next_v11")
os.environ.setdefault("CQSA_CUDA_MODULE_NONCAUSAL", "cqsa_cuda_next_v9")
os.environ["CQSA_BACKWARD"] = "cuda"          # flash_attn_bwd_cqs_global_lse -> the CUDA CQS backward, never Triton
os.environ.setdefault("CQSA_CPU_THREADS", "8")

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.set_num_threads(int(os.environ["CQSA_CPU_THREADS"]))

from stream_cqsa.interface import (flash_attn_func_cqs_group_bits, flash_attn_bwd_cqs_global_lse,
                                   cqs_block_summaries, CQS_BLK_SIZE, CQSA_CUDA_MODULE, cqsa_cuda)
from stream_cqsa.stable_stream import build_tasks
from stream_cqsa.autoconfig import QUORUM_SETS

assert cqsa_cuda is not None, f"{CQSA_CUDA_MODULE} did not import"

OUT_DIR = os.environ.get("LAYOUT_AB_OUT", "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v2/results/ultimate")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSON = os.path.join(OUT_DIR, "layout_ab.json")

H, D = 8, 64
DT = torch.float16
SCALE = D ** -0.5
DEV = torch.device("cuda")
ITERS, WARM = 5, 1
SEG_ALIGN = 128

# ---------------------------------------------------------------------------- helpers
def gpu_info():
    info = dict(torch_device=torch.cuda.get_device_name(0), torch=torch.__version__,
                node=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
                slurm_node=os.environ.get("SLURMD_NODENAME"), cuda_module=CQSA_CUDA_MODULE)
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        info["nvidia_smi"] = r.stdout.strip()
    except Exception as e:  # pragma: no cover
        info["nvidia_smi"] = f"unavailable: {e!r}"
    return info


def timed(fn, it=ITERS, wu=WARM):
    """1 warm-up + `it` timed runs with CUDA events; returns (median_ms, min_ms, all)."""
    for _ in range(wu):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts), min(ts), ts


def to_dev(t):
    return t.to(DEV, non_blocking=False).contiguous()


def tile_census(blk_or, blk_and, L, blk=64):
    """Lower-triangular 64x64 tile census from the summaries: masked / clear / mixed fractions."""
    o = blk_or.cpu().numpy().astype(np.int64); a = blk_and.cpu().numpy().astype(np.int64)
    nb = (L + blk - 1) // blk
    o, a = o[:nb], a[:nb]
    tri = np.tril(np.ones((nb, nb), dtype=bool))
    masked = (np.bitwise_and.outer(a, a) != 0) & tri
    clear = (np.bitwise_and.outer(o, o) == 0) & tri
    mixed = tri & ~masked & ~clear
    n = int(tri.sum())
    return dict(n_tiles64=n, masked=int(masked.sum()) / n, clear=int(clear.sum()) / n, mixed=int(mixed.sum()) / n)


def v11_fwd(q, k, v, bits, bo, ba, block_base=None, seg_align=0, want_lse=False):
    r = flash_attn_func_cqs_group_bits(q, k, v, bits, softmax_scale=SCALE, causal=True,
                                       cqs_blk_or=bo, cqs_blk_and=ba, cqs_blk_size=CQS_BLK_SIZE,
                                       fp32_out=True, return_attn_probs=want_lse,
                                       cqs_block_base=block_base, cqs_seg_align=seg_align)
    return r if want_lse else r


_SDPA_NAMES = None


def sdpa_flash_fwd_ms(L, record_names=False):
    """FA-2 arm: torch SDPA with the FlashAttention backend forced. Plain causal call at length L."""
    global _SDPA_NAMES
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, H, L, D, device=DEV, dtype=DT) for _ in range(3))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=SCALE)
        med, mn, ts = timed(fn)
        if record_names and _SDPA_NAMES is None:
            try:
                from torch.profiler import profile, ProfilerActivity
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    fn(); torch.cuda.synchronize()
                _SDPA_NAMES = sorted({e.name for e in prof.events() if "flash" in e.name.lower() or "attention" in e.name.lower()})
            except Exception as e:  # noqa
                _SDPA_NAMES = [f"profiler unavailable: {type(e).__name__}: {e}"]
    del q, k, v
    return med, mn


def sdpa_flash_bwd_ms(L):
    """Backward-only time of torch's FlashAttention-2 at length L (autograd.grad with retain_graph)."""
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, H, L, D, device=DEV, dtype=DT, requires_grad=True) for _ in range(3))
    do = torch.randn(1, H, L, D, device=DEV, dtype=DT)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=SCALE)
        fwd_med, fwd_min, _ = timed(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=SCALE))
        bwd_med, bwd_min, _ = timed(lambda: torch.autograd.grad(o, (q, k, v), do, retain_graph=True))
    del q, k, v, do, o
    return dict(fwd_ms=fwd_med, fwd_min_ms=fwd_min, bwd_ms=bwd_med, bwd_min_ms=bwd_min)


def fp64_rowblock_check(out32, q, k, v, bits, r0, nrows=128):
    """max |out32[rows] - fp64 reference| over rows [r0, r0+nrows), per-head loop. q/k/v [1,L,H,D]."""
    L = q.shape[1]
    r1 = min(L, r0 + nrows)
    rows = torch.arange(r0, r1, device=DEV)
    cols = torch.arange(L, device=DEV)
    keep = (cols[None, :] <= rows[:, None]) & ((bits[rows][:, None] & bits[None, :]) == 0)
    maxabs, maxref = 0.0, 0.0
    for h in range(H):
        qh = q[0, r0:r1, h].double(); kh = k[0, :, h].double(); vh = v[0, :, h].double()
        s = (qh @ kh.T) * SCALE
        s = s.masked_fill(~keep, float("-inf"))
        p = torch.softmax(s, dim=-1)
        ref = p @ vh
        got = out32[0, r0:r1, h].double()
        maxabs = max(maxabs, (got - ref).abs().max().item())
        maxref = max(maxref, ref.abs().max().item())
        del s, p, ref, got
    return dict(rows=[int(r0), int(r1)], max_abs=maxabs, max_abs_ref=maxref)


def first_boundary(bits_cpu):
    nz = np.nonzero(bits_cpu.numpy())[0]
    return int(nz[0]) if nz.size else 0


# ---------------------------------------------------------------------------- native wave ext
NATIVE = None
NATIVE_ERR = None
try:
    from stream_cqsa.native_wave import native_ext, build_wave_tables, block_base_for
    NATIVE = native_ext()
except Exception as e:  # noqa
    NATIVE_ERR = f"{type(e).__name__}: {e}"


def wave_pack_index(tasks, S):
    """Row index into the token-major [N,H,D] tensor for the uniform [W*S] packing (padding aliases the last real row)."""
    parts = []
    for t in tasks:
        ids = t.token_ids
        L = int(ids.numel())
        parts.append(ids if L == S else torch.cat([ids, ids[-1:].expand(S - L)]))
    return torch.cat(parts).to(DEV)


def wave_call(tb, q3, k3, v3, bb=None):
    """fwd_wave, uniform layout. q3/k3/v3 [R,H,D]. Returns (out_acc [W*S,H,D] fp32, lse [W,H,S])."""
    return NATIVE.fwd_wave(q3, k3, v3, tb.cu, tb.max_L, tb.total, tb.bits, tb.blk_or, tb.blk_and, tb.blk_cu,
                           bb, tb.bb_cu if bb is not None else None, SEG_ALIGN if bb is not None else 0,
                           SCALE, True, uniform_S=tb.S, uniform_W=tb.W)


# ============================================================================ (a) layout A/B forward
def part_a(res):
    res["a_layout_fwd"] = []
    sdpa_cache = {}

    def sdpa(L):
        if L not in sdpa_cache:
            sdpa_cache[L] = sdpa_flash_fwd_ms(L, record_names=True)
        return sdpa_cache[L]

    for N in (131072, 524288):
        torch.manual_seed(0)
        qt, kt, vt = (torch.randn(1, N, H, D, device=DEV, dtype=DT) for _ in range(3))   # token-major, in place
        for layout, seg_align in (("classic_align1", 0), ("aligned_128", SEG_ALIGN)):
            tasks = build_tasks(N, 1, B=1, H=H, D=D, itemsize=2, sorted_gather=True, pin=False,
                                c=7, interest_set=(0, 1, 3), seg_align=seg_align)
            for t in tasks:
                L = int(t.local_size)
                idx = t.token_ids.to(DEV)
                q_i, k_i, v_i = (x.index_select(1, idx).contiguous() for x in (qt, kt, vt))
                bits = to_dev(t.group_bits); bo = to_dev(t.extra["blk_or"]); ba = to_dev(t.extra["blk_and"])
                zero = torch.zeros_like(bits); zo, za = (to_dev(x) for x in cqs_block_summaries(zero, CQS_BLK_SIZE))
                row = dict(N=N, layout=layout, seg_align=seg_align, path=list(t.path), L=L, L_mod_128=L % 128,
                           census=tile_census(bo, ba, L))
                m, mn, _ = timed(lambda: v11_fwd(q_i, k_i, v_i, bits, bo, ba)); row["v11_real_ms"], row["v11_real_min_ms"] = m, mn
                m, mn, _ = timed(lambda: v11_fwd(q_i, k_i, v_i, zero, zo, za)); row["v11_zero_ms"], row["v11_zero_min_ms"] = m, mn
                s = sdpa(L); row["sdpa_flash_ms"], row["sdpa_flash_min_ms"] = s
                row["ratio_real_vs_sdpa"] = row["v11_real_ms"] / row["sdpa_flash_ms"]
                row["ratio_zero_vs_sdpa"] = row["v11_zero_ms"] / row["sdpa_flash_ms"]
                out_g = v11_fwd(q_i, k_i, v_i, bits, bo, ba)                      # fp32 [1,L,H,D], gathered
                if t.path[0] in (0, 3):
                    r0 = max(0, first_boundary(t.group_bits) - 64)
                    row["fp64_check_gathered"] = fp64_rowblock_check(out_g, q_i, k_i, v_i, bits, r0)
                if seg_align:
                    seg_base = to_dev(t.extra["seg_base"])
                    q_s, k_s, v_s = (x.as_strided((1, L, H, D), x.stride()) for x in (qt, kt, vt))
                    m, mn, _ = timed(lambda: v11_fwd(q_s, k_s, v_s, bits, bo, ba, block_base=seg_base, seg_align=SEG_ALIGN))
                    row["v11_inplace_ms"], row["v11_inplace_min_ms"] = m, mn
                    out_ip = v11_fwd(q_s, k_s, v_s, bits, bo, ba, block_base=seg_base, seg_align=SEG_ALIGN)
                    row["maxdiff_inplace_vs_gathered"] = (out_ip - out_g).abs().max().item()
                    del out_ip
                    if NATIVE is not None:
                        try:
                            tb = build_wave_tables([t], DEV, uniform=True)
                            pidx = wave_pack_index([t], tb.S)
                            qp, kp, vp = (x[0].index_select(0, pidx).contiguous() for x in (qt, kt, vt))
                            m, mn, _ = timed(lambda: wave_call(tb, qp, kp, vp)); row["wave_W1_packed_ms"], row["wave_W1_packed_min_ms"] = m, mn
                            out_w = wave_call(tb, qp, kp, vp)[0][:L]
                            row["maxdiff_wave_packed_vs_v11_gathered"] = (out_w - out_g[0]).abs().max().item()
                            bb = block_base_for(tb, lambda g: g).to(DEV)
                            q3, k3, v3 = (x[0] for x in (qt, kt, vt))
                            m, mn, _ = timed(lambda: wave_call(tb, q3, k3, v3, bb)); row["wave_W1_inplace_ms"], row["wave_W1_inplace_min_ms"] = m, mn
                            out_wi = wave_call(tb, q3, k3, v3, bb)[0][:L]
                            row["maxdiff_wave_inplace_vs_wave_packed"] = (out_wi - out_w).abs().max().item()
                            row["wave_S"] = int(tb.S)
                            del out_w, out_wi, qp, kp, vp, tb
                        except Exception as e:  # noqa
                            row["wave_error"] = f"{type(e).__name__}: {e}"
                            traceback.print_exc()
                del out_g, q_i, k_i, v_i
                res["a_layout_fwd"].append(row)
                print(f"[a] N={N} {layout:14s} path={t.path} L={L:6d} (L%128={L%128:3d}) "
                      f"v11 real {row['v11_real_ms']:8.2f}  zero {row['v11_zero_ms']:8.2f}  sdpaF {row['sdpa_flash_ms']:8.2f}  "
                      f"real/sdpa {row['ratio_real_vs_sdpa']:.3f}  zero/sdpa {row['ratio_zero_vs_sdpa']:.3f}"
                      + (f"  inplace {row['v11_inplace_ms']:8.2f}" if 'v11_inplace_ms' in row else "")
                      + (f"  waveW1 {row['wave_W1_packed_ms']:8.2f}/{row['wave_W1_inplace_ms']:8.2f}" if 'wave_W1_packed_ms' in row else "")
                      + f"  mixed {row['census']['mixed']*100:.2f}% masked {row['census']['masked']*100:.1f}%", flush=True)
                torch.cuda.empty_cache()
        del qt, kt, vt
        torch.cuda.empty_cache()
    res["sdpa_flash_kernel_names"] = _SDPA_NAMES
    # summary: mean over the 7 tasks per (N, layout)
    res["a_summary"] = []
    for N in (131072, 524288):
        for layout in ("classic_align1", "aligned_128"):
            rows = [r for r in res["a_layout_fwd"] if r["N"] == N and r["layout"] == layout]
            s = dict(N=N, layout=layout, n_tasks=len(rows), L_min=min(r["L"] for r in rows), L_max=max(r["L"] for r in rows))
            for key in ("v11_real_ms", "v11_zero_ms", "sdpa_flash_ms", "v11_inplace_ms", "wave_W1_packed_ms", "wave_W1_inplace_ms"):
                vals = [r[key] for r in rows if key in r]
                if vals:
                    s[key + "_mean"] = sum(vals) / len(vals)
            s["real_over_sdpa_mean"] = sum(r["ratio_real_vs_sdpa"] for r in rows) / len(rows)
            s["zero_over_sdpa_mean"] = sum(r["ratio_zero_vs_sdpa"] for r in rows) / len(rows)
            s["mixed_frac_mean"] = sum(r["census"]["mixed"] for r in rows) / len(rows)
            res["a_summary"].append(s)


# ============================================================================ (b) backward split
def part_b(res):
    res["b_backward"] = []
    for N in (32768, 65536, 131072):
        tasks = build_tasks(N, 1, B=1, H=H, D=D, itemsize=2, sorted_gather=True, pin=False, c=7, interest_set=(0, 1, 3), seg_align=0)
        t = tasks[0]
        L = int(t.local_size)
        torch.manual_seed(0)
        q, k, v, do = (torch.randn(1, L, H, D, device=DEV, dtype=DT) for _ in range(4))
        bits = to_dev(t.group_bits); bo = to_dev(t.extra["blk_or"]); ba = to_dev(t.extra["blk_and"])
        zero = torch.zeros_like(bits); zo, za = (to_dev(x) for x in cqs_block_summaries(zero, CQS_BLK_SIZE))
        row = dict(N=N, L=L, layout="classic_align1", path=list(t.path), census=tile_census(bo, ba, L))
        for name, b_, o_, a_ in (("real", bits, bo, ba), ("zero", zero, zo, za)):
            out, lse, _ = v11_fwd(q, k, v, b_, o_, a_, want_lse=True)          # out fp32 [1,L,H,D], lse [1,H,L]
            o16 = out.to(DT); lse = lse.contiguous()
            m, mn, _ = timed(lambda: v11_fwd(q, k, v, b_, o_, a_)); row[f"v11_fwd_{name}_ms"] = m
            fn = lambda: flash_attn_bwd_cqs_global_lse(do, q, k, v, o16, lse, b_, softmax_scale=SCALE, causal=True,
                                                       cqs_blk_or=o_, cqs_blk_and=a_, cqs_blk_size=CQS_BLK_SIZE)
            m, mn, _ = timed(fn); row[f"cuda_bwd_{name}_ms"], row[f"cuda_bwd_{name}_min_ms"] = m, mn
            if name == "real":
                g1 = [x.clone() for x in fn()]; g2 = fn()
                row["bwd_rerun_maxdiff_dq_dk_dv"] = [(a - b).abs().max().item() for a, b in zip(g1, g2)]
                del g1, g2
            del out, o16, lse
        s = sdpa_flash_bwd_ms(L)
        row["sdpa_flash_fwd_ms"], row["sdpa_flash_bwd_ms"] = s["fwd_ms"], s["bwd_ms"]
        row["sdpa_flash_bwd_min_ms"] = s["bwd_min_ms"]
        row["bwd_real_over_sdpa"] = row["cuda_bwd_real_ms"] / row["sdpa_flash_bwd_ms"]
        row["bwd_zero_over_sdpa"] = row["cuda_bwd_zero_ms"] / row["sdpa_flash_bwd_ms"]
        row["bwd_dead_tile_overhead_ms"] = row["cuda_bwd_real_ms"] - row["cuda_bwd_zero_ms"]
        row["bwd_live_slowdown_ms"] = row["cuda_bwd_zero_ms"] - row["sdpa_flash_bwd_ms"]
        row["fwd_real_over_sdpa"] = row["v11_fwd_real_ms"] / row["sdpa_flash_fwd_ms"]
        row["fwd_zero_over_sdpa"] = row["v11_fwd_zero_ms"] / row["sdpa_flash_fwd_ms"]
        res["b_backward"].append(row)
        print(f"[b] L={L:6d}: CUDA bwd real {row['cuda_bwd_real_ms']:8.2f}  zero {row['cuda_bwd_zero_ms']:8.2f}  sdpaF bwd {row['sdpa_flash_bwd_ms']:8.2f}  "
              f"real/sdpa {row['bwd_real_over_sdpa']:.2f}  zero/sdpa {row['bwd_zero_over_sdpa']:.2f}  "
              f"| fwd real {row['v11_fwd_real_ms']:7.2f} zero {row['v11_fwd_zero_ms']:7.2f} sdpaF {row['sdpa_flash_fwd_ms']:7.2f}", flush=True)
        del q, k, v, do
        torch.cuda.empty_cache()


# ============================================================================ (c) small-L forward
def part_c(res):
    res["c_small_L"] = {}
    # c=73, itr=1 at N=131072 (QUORUM_SETS[73] has 9 members -> L = 9N/73 ~ 16.2K, not 3N/73)
    N, c = 131072, 73
    iset = QUORUM_SETS[c]
    torch.manual_seed(0)
    qt, kt, vt = (torch.randn(1, N, H, D, device=DEV, dtype=DT) for _ in range(3))
    tasks = build_tasks(N, 1, B=1, H=H, D=D, itemsize=2, sorted_gather=True, pin=False, c=c, interest_set=iset, seg_align=0)
    per = []
    sdpa_cache = {}
    for t in tasks:
        L = int(t.local_size)
        idx = t.token_ids.to(DEV)
        q_i, k_i, v_i = (x.index_select(1, idx).contiguous() for x in (qt, kt, vt))
        bits = to_dev(t.group_bits); bo = to_dev(t.extra["blk_or"]); ba = to_dev(t.extra["blk_and"])
        zero = torch.zeros_like(bits); zo, za = (to_dev(x) for x in cqs_block_summaries(zero, CQS_BLK_SIZE))
        r = dict(path=list(t.path), L=L, census=tile_census(bo, ba, L))
        r["v11_real_ms"] = timed(lambda: v11_fwd(q_i, k_i, v_i, bits, bo, ba))[0]
        r["v11_zero_ms"] = timed(lambda: v11_fwd(q_i, k_i, v_i, zero, zo, za))[0]
        if L not in sdpa_cache:
            sdpa_cache[L] = sdpa_flash_fwd_ms(L)[0]
        r["sdpa_flash_ms"] = sdpa_cache[L]
        if t.path[0] == 0:
            out_g = v11_fwd(q_i, k_i, v_i, bits, bo, ba)
            r["fp64_check_gathered"] = fp64_rowblock_check(out_g, q_i, k_i, v_i, bits, max(0, first_boundary(t.group_bits) - 64))
            del out_g
        per.append(r)
        del q_i, k_i, v_i
    tot = lambda key: sum(r[key] for r in per)
    summ = dict(N=N, c=c, interest_set=list(iset), n_tasks=len(per), L_min=min(r["L"] for r in per), L_max=max(r["L"] for r in per),
                sum_v11_real_ms=tot("v11_real_ms"), sum_v11_zero_ms=tot("v11_zero_ms"), sum_sdpa_flash_ms=tot("sdpa_flash_ms"),
                mixed_frac_mean=sum(r["census"]["mixed"] for r in per) / len(per),
                masked_frac_mean=sum(r["census"]["masked"] for r in per) / len(per), per_task=per)
    summ["real_over_sdpa"] = summ["sum_v11_real_ms"] / summ["sum_sdpa_flash_ms"]
    summ["zero_over_sdpa"] = summ["sum_v11_zero_ms"] / summ["sum_sdpa_flash_ms"]
    summ["real_over_zero"] = summ["sum_v11_real_ms"] / summ["sum_v11_zero_ms"]
    # monolithic FA-2 (SDPA flash) on the full N for the end-to-end reference
    summ["sdpa_flash_monolithic_N_ms"] = sdpa_flash_fwd_ms(N)[0]
    # the wave kernel on all 73 subproblems in ONE launch (aligned tasks, uniform layout), if importable
    if NATIVE is not None:
        try:
            atasks = build_tasks(N, 1, B=1, H=H, D=D, itemsize=2, sorted_gather=True, pin=False, c=c, interest_set=iset, seg_align=SEG_ALIGN)
            tb = build_wave_tables(atasks, DEV, uniform=True)
            pidx = wave_pack_index(atasks, tb.S)
            qp, kp, vp = (x[0].index_select(0, pidx).contiguous() for x in (qt, kt, vt))
            summ["wave_W73_packed_ms"] = timed(lambda: wave_call(tb, qp, kp, vp))[0]
            bb = block_base_for(tb, lambda g: g).to(DEV)
            q3, k3, v3 = (x[0] for x in (qt, kt, vt))
            summ["wave_W73_inplace_ms"] = timed(lambda: wave_call(tb, q3, k3, v3, bb))[0]
            summ["wave_S"] = int(tb.S); summ["wave_W"] = int(tb.W)
            summ["aligned_L_min"] = min(int(t.local_size) for t in atasks); summ["aligned_L_max"] = max(int(t.local_size) for t in atasks)
            del qp, kp, vp, tb
        except Exception as e:  # noqa
            summ["wave_error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
    res["c_small_L"]["c73_N131072"] = summ
    print(f"[c] c=73 N=131072 ({len(per)} tasks, L {summ['L_min']}-{summ['L_max']}): sum v11 real {summ['sum_v11_real_ms']:.1f}  zero {summ['sum_v11_zero_ms']:.1f}  "
          f"sdpaF {summ['sum_sdpa_flash_ms']:.1f} ms  real/sdpa {summ['real_over_sdpa']:.2f} zero/sdpa {summ['zero_over_sdpa']:.2f} real/zero {summ['real_over_zero']:.2f}  "
          f"mixed {summ['mixed_frac_mean']*100:.2f}%  mono sdpaF {summ['sdpa_flash_monolithic_N_ms']:.1f} ms"
          + (f"  wave W=73 {summ['wave_W73_packed_ms']:.1f}/{summ['wave_W73_inplace_ms']:.1f} ms" if 'wave_W73_packed_ms' in summ else ""), flush=True)
    del qt, kt, vt
    torch.cuda.empty_cache()
    # c=7 at N=16384 (L=7023), the map's 2.9x row
    N = 16384
    torch.manual_seed(0)
    tasks = build_tasks(N, 1, B=1, H=H, D=D, itemsize=2, sorted_gather=True, pin=False, c=7, interest_set=(0, 1, 3), seg_align=0)
    t = tasks[0]; L = int(t.local_size)
    q_i, k_i, v_i = (torch.randn(1, L, H, D, device=DEV, dtype=DT) for _ in range(3))
    bits = to_dev(t.group_bits); bo = to_dev(t.extra["blk_or"]); ba = to_dev(t.extra["blk_and"])
    zero = torch.zeros_like(bits); zo, za = (to_dev(x) for x in cqs_block_summaries(zero, CQS_BLK_SIZE))
    r = dict(N=N, c=7, L=L, path=list(t.path), census=tile_census(bo, ba, L))
    r["v11_real_ms"] = timed(lambda: v11_fwd(q_i, k_i, v_i, bits, bo, ba))[0]
    r["v11_zero_ms"] = timed(lambda: v11_fwd(q_i, k_i, v_i, zero, zo, za))[0]
    r["sdpa_flash_ms"] = sdpa_flash_fwd_ms(L)[0]
    r["real_over_sdpa"] = r["v11_real_ms"] / r["sdpa_flash_ms"]; r["zero_over_sdpa"] = r["v11_zero_ms"] / r["sdpa_flash_ms"]
    res["c_small_L"]["c7_N16384"] = r
    print(f"[c] c=7 N=16384 L={L}: v11 real {r['v11_real_ms']:.3f} zero {r['v11_zero_ms']:.3f} sdpaF {r['sdpa_flash_ms']:.3f} ms  "
          f"real/sdpa {r['real_over_sdpa']:.2f} zero/sdpa {r['zero_over_sdpa']:.2f}", flush=True)


# ============================================================================ table + main
def print_table(res):
    print("\n==================== LAYOUT A/B (forward, causal, fp16, H=8 D=64, c=7 itr=1; ms, median of 5) ====================")
    print("FA-2 arm = torch SDPA with SDPBackend.FLASH_ATTENTION forced (flash_attn not installed). Kernels seen:", res.get("sdpa_flash_kernel_names"))
    print(f"{'N':>7} {'layout':>15} {'path':>5} {'L':>7} {'L%128':>5} | {'v11 real':>9} {'v11 zero':>9} {'sdpaF':>9} | {'real/F':>6} {'zero/F':>6} | "
          f"{'inplace':>8} {'waveW1':>8} {'waveIP':>8} | {'mixed%':>6} {'masked%':>7}")
    for r in res["a_layout_fwd"]:
        print(f"{r['N']:>7} {r['layout']:>15} {r['path'][0]:>5} {r['L']:>7} {r['L_mod_128']:>5} | {r['v11_real_ms']:9.2f} {r['v11_zero_ms']:9.2f} {r['sdpa_flash_ms']:9.2f} | "
              f"{r['ratio_real_vs_sdpa']:6.3f} {r['ratio_zero_vs_sdpa']:6.3f} | "
              f"{r.get('v11_inplace_ms', float('nan')):8.2f} {r.get('wave_W1_packed_ms', float('nan')):8.2f} {r.get('wave_W1_inplace_ms', float('nan')):8.2f} | "
              f"{r['census']['mixed']*100:6.2f} {r['census']['masked']*100:7.1f}")
    print("\n-- per-(N, layout) means over the 7 tasks --")
    for s in res["a_summary"]:
        print(f"  N={s['N']:>7} {s['layout']:>15}: v11 real {s['v11_real_ms_mean']:8.2f}  zero {s['v11_zero_ms_mean']:8.2f}  sdpaF {s['sdpa_flash_ms_mean']:8.2f}  "
              f"real/F {s['real_over_sdpa_mean']:.3f}  zero/F {s['zero_over_sdpa_mean']:.3f}  mixed {s['mixed_frac_mean']*100:.2f}%"
              + (f"  inplace {s['v11_inplace_ms_mean']:8.2f}" if 'v11_inplace_ms_mean' in s else "")
              + (f"  waveW1 {s['wave_W1_packed_ms_mean']:8.2f}/{s['wave_W1_inplace_ms_mean']:8.2f}" if 'wave_W1_packed_ms_mean' in s else ""))
    print("\n-- bit-identity --")
    for r in res["a_layout_fwd"]:
        bits_ = {k_: v_ for k_, v_ in r.items() if k_.startswith("maxdiff") or k_ == "fp64_check_gathered"}
        if bits_:
            print(f"  N={r['N']} {r['layout']} path={r['path']}: {bits_}")
    print("\n==================== BACKWARD SPLIT (classic layout path (0,), causal; ms) ====================")
    print(f"{'L':>7} | {'bwd real':>9} {'bwd zero':>9} {'sdpaF bwd':>9} | {'real/F':>6} {'zero/F':>6} | {'dead-tile ms':>12} {'live slow ms':>12} | {'fwd real':>8} {'fwd zero':>8} {'sdpaF':>8}")
    for r in res["b_backward"]:
        print(f"{r['L']:>7} | {r['cuda_bwd_real_ms']:9.2f} {r['cuda_bwd_zero_ms']:9.2f} {r['sdpa_flash_bwd_ms']:9.2f} | {r['bwd_real_over_sdpa']:6.2f} {r['bwd_zero_over_sdpa']:6.2f} | "
              f"{r['bwd_dead_tile_overhead_ms']:12.2f} {r['bwd_live_slowdown_ms']:12.2f} | {r['v11_fwd_real_ms']:8.2f} {r['v11_fwd_zero_ms']:8.2f} {r['sdpa_flash_fwd_ms']:8.2f}"
              f"   rerun maxdiff dq/dk/dv {r['bwd_rerun_maxdiff_dq_dk_dv']}")
    print("\n==================== SMALL L ====================")
    s = res["c_small_L"]["c73_N131072"]
    print(f"c=73 N=131072: {s['n_tasks']} tasks L {s['L_min']}-{s['L_max']}; sum over tasks: v11 real {s['sum_v11_real_ms']:.1f}  zero {s['sum_v11_zero_ms']:.1f}  sdpaF {s['sum_sdpa_flash_ms']:.1f} ms; "
          f"real/F {s['real_over_sdpa']:.2f}  zero/F {s['zero_over_sdpa']:.2f}  real/zero {s['real_over_zero']:.2f}; mixed {s['mixed_frac_mean']*100:.2f}% masked {s['masked_frac_mean']*100:.1f}%; "
          f"monolithic sdpaF at N: {s['sdpa_flash_monolithic_N_ms']:.1f} ms"
          + (f"; wave W=73 one launch packed/inplace {s['wave_W73_packed_ms']:.1f}/{s['wave_W73_inplace_ms']:.1f} ms (S={s['wave_S']})" if 'wave_W73_packed_ms' in s else f"; wave: {s.get('wave_error', NATIVE_ERR)}"))
    r = res["c_small_L"]["c7_N16384"]
    print(f"c=7 N=16384 L={r['L']}: v11 real {r['v11_real_ms']:.3f}  zero {r['v11_zero_ms']:.3f}  sdpaF {r['sdpa_flash_ms']:.3f} ms; real/F {r['real_over_sdpa']:.2f} zero/F {r['zero_over_sdpa']:.2f}; "
          f"mixed {r['census']['mixed']*100:.2f}% masked {r['census']['masked']*100:.1f}%")


def main():
    t0 = time.time()
    res = dict(info=gpu_info(), fa2_arm="torch SDPA with SDPBackend.FLASH_ATTENTION forced (flash_attn not installed in CQSA311)",
               timing="1 warm-up + 5 timed runs, CUDA events, median (min also stored)", H=H, D=D, dtype="float16", causal=True,
               native_ext=(None if NATIVE is None else getattr(NATIVE, "__file__", "cqsa_native")), native_import_error=NATIVE_ERR)
    print(json.dumps(res["info"]), flush=True)
    print("native wave ext:", res["native_ext"], NATIVE_ERR or "", flush=True)
    for part in (part_a, part_b, part_c):
        try:
            part(res)
        except Exception as e:  # noqa
            res[f"error_{part.__name__}"] = traceback.format_exc()
            print(f"!! {part.__name__} failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        json.dump(res, open(OUT_JSON, "w"), indent=1)
    res["elapsed_s"] = time.time() - t0
    json.dump(res, open(OUT_JSON, "w"), indent=1)
    try:
        print_table(res)
    except Exception:  # noqa
        traceback.print_exc()
    print(f"\nwrote {OUT_JSON}  ({res['elapsed_s']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
