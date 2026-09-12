"""
Clean-node kernel baseline: what track 1 has to beat.

Four kernels at the SAME subproblem shape L = 3N/7, so decomposition cost is
excluded and only kernel efficiency is compared:

  fa2        upstream flash-attn 2.8.3 (the real baseline)
  cqsa_plain the shipped modified binary with CQS disabled (cqsa_cuda.fwd)
  cqsa_zero  the CQS entry point with an all-zero mask  (cost of carrying)
  cqsa_mask  the CQS entry point with the real mask     (cost of masking)
  sdpa       torch SDPA (default backend) at the same shape

Both causal and non-causal, N in {16K, 32K, 64K, 128K}. Also the monolithic
fa2 at full N, so 7 * cqsa_mask / fa2_mono gives the decomposition overhead.

    sbatch next/slurm/run1.slurm next/bench/kernel_baseline.py
"""
import torch, torch.nn.functional as F, json, sys, os
from stream_cqsa.interface import flash_attn_func, flash_attn_func_cqs_group_bits
from stream_cqsa.reference import group_bits_for_path
try:
    from flash_attn import flash_attn_func as fa2_func
    HAVE_FA2 = True
except Exception as e:
    HAVE_FA2 = False; print("upstream flash_attn unavailable:", e)

def cuda_ms(fn, iters=10, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters

H, D, dtype = 8, 64, torch.float16
scale = D ** -0.5
print(f"GPU {torch.cuda.get_device_name(0)}  H={H} D={D} fp16  torch {torch.__version__}")
rows = []
for causal in (False, True):
    print(f"\n=== causal={causal} ===")
    print(f"{'N':>7} {'L':>6} {'fa2':>8} {'cqsa_plain':>10} {'cqsa_zero':>10} {'cqsa_mask':>10} {'sdpa':>8} | {'mask/fa2':>8} {'fa2_mono':>9} {'7mask/mono':>10}")
    for N in (16384, 32768, 65536, 131072):
        ids_np, bits_np = group_bits_for_path(N, (0,), sorted_gather=True)
        L = int(ids_np.shape[0])
        bits = torch.as_tensor(bits_np, device="cuda", dtype=torch.int64)
        zero = torch.zeros_like(bits)
        q, k, v = (torch.randn(1, L, H, D, device="cuda", dtype=dtype) for _ in range(3))
        r = dict(N=N, L=L, causal=causal)
        r["fa2"] = cuda_ms(lambda: fa2_func(q, k, v, softmax_scale=scale, causal=causal)) if HAVE_FA2 else float("nan")
        r["cqsa_plain"] = cuda_ms(lambda: flash_attn_func(q, k, v, softmax_scale=scale, causal=causal))
        r["cqsa_zero"] = cuda_ms(lambda: flash_attn_func_cqs_group_bits(q, k, v, zero, softmax_scale=scale, causal=causal))
        r["cqsa_mask"] = cuda_ms(lambda: flash_attn_func_cqs_group_bits(q, k, v, bits, softmax_scale=scale, causal=causal))
        qb, kb, vb = (t.transpose(1, 2) for t in (q, k, v))            # SDPA wants [B,H,L,D]
        r["sdpa"] = cuda_ms(lambda: F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal, scale=scale))
        del q, k, v, qb, kb, vb; torch.cuda.empty_cache()
        qf, kf, vf = (torch.randn(1, N, H, D, device="cuda", dtype=dtype) for _ in range(3))
        r["fa2_mono"] = cuda_ms(lambda: fa2_func(qf, kf, vf, softmax_scale=scale, causal=causal), iters=5) if HAVE_FA2 else float("nan")
        del qf, kf, vf; torch.cuda.empty_cache()
        rows.append(r)
        print(f"{N:>7} {L:>6} {r['fa2']:8.2f} {r['cqsa_plain']:10.2f} {r['cqsa_zero']:10.2f} {r['cqsa_mask']:10.2f} {r['sdpa']:8.2f} | "
              f"{r['cqsa_mask']/r['fa2']:8.2f} {r['fa2_mono']:9.2f} {7*r['cqsa_mask']/r['fa2_mono']:10.2f}", flush=True)
out = os.path.join(os.path.dirname(__file__), "..", "logs", "kernel_baseline.json")
json.dump(rows, open(out, "w"), indent=1)
print(f"\nwrote {out}")
