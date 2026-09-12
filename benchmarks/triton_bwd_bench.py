"""Triton backward vs CUDA backward, clean node: kernel-level at L=3N/7 (N=16K..131K) and engine-level (256K, 1M)."""
import torch, time, gc, os, json
from stream_cqsa.triton_kernel import cqs_attention_backward
from stream_cqsa.interface import flash_attn_bwd_cqs_global_lse, cqs_block_summaries
from stream_cqsa.stable_stream import local_stats_flash, stream_cqsa_forward, stream_cqsa_backward
from stream_cqsa.reference import group_bits_for_path
from flash_attn import flash_attn_func
def ms(fn, it=5, wu=2):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record(); [fn() for _ in range(it)]; e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / it
H, D = 8, 64; scale = D ** -0.5; res = dict(kernel=[], engine=[])
print(torch.cuda.get_device_name(0), flush=True)
for causal in (True, False):
    for N in (16384, 32768, 65536, 131072):
        ids, bits_np = group_bits_for_path(N, (0,), sorted_gather=True); L = len(bits_np)
        q, k, v, do = (torch.randn(1, L, H, D, device="cuda", dtype=torch.float16) for _ in range(4))
        b = torch.as_tensor(bits_np, device="cuda"); bo, ba = (t.cuda() for t in cqs_block_summaries(b))
        out, lse = local_stats_flash(q, k, v, b, causal=causal, scale=scale, blk_or=bo, blk_and=ba); lse = lse.transpose(1, 2).contiguous(); o16 = out.to(q.dtype)
        # FA-2 monolithic backward on the same L (fwd+bwd via autograd, minus fwd)
        qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
        def fa2_bwd():
            o = flash_attn_func(qq, kk, vv, softmax_scale=scale, causal=causal); o.backward(do)
        t_fa2 = ms(fa2_bwd) - ms(lambda: flash_attn_func(qq, kk, vv, softmax_scale=scale, causal=causal))
        r = dict(N=N, L=L, causal=causal, fa2_bwd=t_fa2,
                 cuda=ms(lambda: flash_attn_bwd_cqs_global_lse(do, q, k, v, o16, lse, b, softmax_scale=scale, causal=causal, cqs_blk_or=bo, cqs_blk_and=ba)),
                 triton=ms(lambda: cqs_attention_backward(do, q, k, v, lse, b, causal=causal, scale=scale, out=o16, blk_or=bo, blk_and=ba)))
        for bm, bn in ((64, 64), (128, 64), (64, 128), (128, 128)):
            for nw in (4, 8):
                try: r[f"tri_{bm}x{bn}_w{nw}"] = ms(lambda: cqs_attention_backward(do, q, k, v, lse, b, causal=causal, scale=scale, out=o16, blk_or=bo, blk_and=ba, block_m=bm, block_n=bn, num_warps=nw))
                except Exception as e: r[f"tri_{bm}x{bn}_w{nw}"] = None
        best = min((v_ for k_, v_ in r.items() if k_.startswith("tri_") and v_), default=r["triton"])
        res["kernel"].append(r)
        print(f"N={N:>7} L={L:>6} causal={int(causal)}: FA-2 bwd {t_fa2:7.2f}  CUDA CQS bwd {r['cuda']:7.2f}  Triton CQS bwd {r['triton']:7.2f} (best tile {best:7.2f})  triton/cuda {r['triton']/r['cuda']:.2f}", flush=True)
        del q, k, v, do, qq, kk, vv; gc.collect(); torch.cuda.empty_cache()
for N in (262144, 1048576):
    torch.manual_seed(0); q, k, v, do = (torch.randn(1, H, N, D, device="cuda", dtype=torch.float16) for _ in range(4))
    out, info = stream_cqsa_forward(q, k, v, itr=1, causal=True); o16 = out.to(q.dtype)
    for name in ("cuda", "triton"):
        if name == "triton": os.environ["CQSA_BACKWARD"] = "triton"
        else: os.environ.pop("CQSA_BACKWARD", None)
        ts = []
        for rep in range(3):
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); t0 = time.perf_counter()
            g = stream_cqsa_backward(q, k, v, do, o16, info["lse"], itr=1, causal=True, allow_escalation=False); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        if name == "cuda": ref = [x.float().cpu() for x in g]
        err = max(((x.float().cpu() - r_).norm() / r_.norm()).item() for x, r_ in zip(g, ref))
        res["engine"].append(dict(N=N, backend=name, s=min(ts[1:]), rel_vs_cuda=err))
        print(f"engine backward N={N:>8} itr=1 acc=gpu {name:6s}: {min(ts[1:]):7.2f} s  (max rel vs cuda {err:.1e})", flush=True)
    os.environ.pop("CQSA_BACKWARD", None); del q, k, v, do, out; gc.collect(); torch.cuda.empty_cache()
json.dump(res, open("/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/logs/triton_bwd_bench.json", "w"), indent=1)
