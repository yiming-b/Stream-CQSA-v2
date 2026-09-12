"""Triton CQS kernel vs native v11 (CUDA) vs FlashAttention-2, clean node.
1. block-size sweep for the Triton kernel at the itr=1 subproblem shape (L=3N/7, N=131072)
2. kernel-level ms/call at N=16K..131K, causal and non-causal, real bits
3. engine end to end (N=256K, 1M) with inner=triton vs inner=native, acc=GPU and acc=CPU/host
    sbatch next/slurm/run1_test.slurm next/bench/triton_bench.py
"""
import json, torch, time, gc, itertools
from stream_cqsa.triton_kernel import cqs_attention_forward, triton_inner
from stream_cqsa.stable_stream import local_stats_flash, stream_cqsa_forward
from stream_cqsa.interface import cqs_block_summaries
from stream_cqsa.reference import group_bits_for_path
from flash_attn import flash_attn_func
def ms(fn, it=10, wu=3):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record(); [fn() for _ in range(it)]; e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / it
H, D = 8, 64; scale = D ** -0.5; res = dict(sweep=[], kernel=[], engine=[])
print(torch.cuda.get_device_name(0), flush=True)
# 1. sweep
ids, bits_np = group_bits_for_path(131072, (0,), sorted_gather=True); L = len(bits_np)
torch.manual_seed(0); q, k, v = (torch.randn(1, L, H, D, device="cuda", dtype=torch.float16) for _ in range(3))
b = torch.as_tensor(bits_np, device="cuda"); bo, ba = (t.cuda() for t in cqs_block_summaries(b))
best = None
for bm, bn, nw, ns in itertools.product((64, 128), (64, 128), (4, 8), (2, 3, 4)):
    try:
        t = ms(lambda: cqs_attention_forward(q, k, v, b, causal=True, scale=scale, blk_or=bo, blk_and=ba, block_m=bm, block_n=bn, num_warps=nw, num_stages=ns), it=5)
        res["sweep"].append(dict(bm=bm, bn=bn, warps=nw, stages=ns, ms=t)); print(f"sweep BM={bm} BN={bn} warps={nw} stages={ns}: {t:.2f} ms", flush=True)
        if best is None or t < best[0]: best = (t, dict(block_m=bm, block_n=bn, num_warps=nw, num_stages=ns))
    except Exception as e:
        print(f"sweep BM={bm} BN={bn} warps={nw} stages={ns}: {type(e).__name__}", flush=True)
print("best:", best, flush=True); cfg = best[1]; res["best"] = cfg
# 2. kernel-level
for causal in (True, False):
    for N in (16384, 32768, 65536, 131072):
        ids, bits_np = group_bits_for_path(N, (0,), sorted_gather=True); L = len(bits_np)
        q, k, v = (torch.randn(1, L, H, D, device="cuda", dtype=torch.float16) for _ in range(3))
        b = torch.as_tensor(bits_np, device="cuda"); bo, ba = (t.cuda() for t in cqs_block_summaries(b))
        r = dict(N=N, L=L, causal=causal,
                 fa2=ms(lambda: flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)),
                 triton_plain=ms(lambda: cqs_attention_forward(q, k, v, None, causal=causal, scale=scale, **cfg)),
                 triton_cqs=ms(lambda: cqs_attention_forward(q, k, v, b, causal=causal, scale=scale, blk_or=bo, blk_and=ba, **cfg)),
                 native_cqs=ms(lambda: local_stats_flash(q, k, v, b, causal=causal, scale=scale, blk_or=bo, blk_and=ba)))
        res["kernel"].append(r)
        print(f"N={N:>7} L={L:>6} causal={int(causal)}: fa2 {r['fa2']:7.2f}  triton plain {r['triton_plain']:7.2f}  triton CQS {r['triton_cqs']:7.2f}  native CQS {r['native_cqs']:7.2f}   triton/native {r['triton_cqs']/r['native_cqs']:.2f}", flush=True)
        del q, k, v; gc.collect(); torch.cuda.empty_cache()
# 3. engine
import functools
tri = functools.partial(triton_inner, **{})   # default block sizes are set below via a wrapper
def triton_best(q_i, k_i, v_i, bits, *, causal, scale, blk_or=None, blk_and=None, blk_size=64, block_base=None, **_):
    if block_base is not None: raise ValueError("segmented")
    out, lse = cqs_attention_forward(q_i, k_i, v_i, bits, causal=causal, scale=scale, blk_or=blk_or, blk_and=blk_and, **cfg)
    return out, lse.transpose(1, 2)
for N in (262144, 1048576):
    torch.manual_seed(0); qh, kh, vh = (torch.randn(1, H, N, D, dtype=torch.float16).pin_memory() for _ in range(3))
    for label, kw in [("acc=gpu device inputs", dict(itr=1, max_parallel=2)),
                      ("acc=cpu host inputs", dict(itr=1, max_parallel=1, stream_from_host=True, low_memory=True))]:
        for name, inner in (("native", local_stats_flash), ("triton", triton_best)):
            if kw.get("stream_from_host"): q, k, v = qh, kh, vh
            else: q, k, v = (t.cuda() for t in (qh, kh, vh))
            ts = []
            for rep in range(3):
                gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); t0 = time.perf_counter()
                out, info = stream_cqsa_forward(q, k, v, causal=True, inner=inner, allow_escalation=False, shared_chunks=(name == "native" and kw.get("stream_from_host", False)), **kw)
                torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
            if name == "native": ref = out.float().cpu()
            err = ((out.float().cpu() - ref).norm() / ref.norm()).item()
            res["engine"].append(dict(N=N, config=label, inner=name, s=min(ts[1:]), rel_vs_native=err))
            print(f"engine N={N:>8} {label:22s} inner={name:6s}: {min(ts[1:]):7.2f} s  (vs native engine rel {err:.1e})", flush=True)
            del out; gc.collect(); torch.cuda.empty_cache()
json.dump(res, open("/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/logs/triton_bench.json", "w"), indent=1)
