"""Why did the shipped arm of e2e_compare measure 33.6 s at 1M (vs 15.6 s in pipeline_baseline)?
Same node, same process: time the shipped engine at 1M with e2e's exact kwargs, pinned and unpinned inputs,
and with the e2e timeit wrapper (reset_peak_memory_stats + empty_cache per rep)."""
import torch, time, gc, sys
import stream_cqsa.stable_stream as ss
from stream_cqsa.stable_stream import stream_cqsa_forward
print("engine:", ss.__file__, flush=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1048576
B, H, D = 1, 8, 64
kw = dict(itr=1, causal=True, stream_from_host=True, low_memory=True, allow_escalation=False, max_parallel=1)
torch.manual_seed(0)
q, k, v = (torch.randn(B, H, N, D, dtype=torch.float16) for _ in range(3))
for label, pin in (("unpinned", False), ("pinned", True)):
    qq, kk, vv = ((t.pin_memory() if pin else t) for t in (q, k, v))
    for rep in range(3):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.perf_counter(); out, info = stream_cqsa_forward(qq, kk, vv, **kw); torch.cuda.synchronize()
        print(f"{label} rep{rep}: {time.perf_counter()-t0:.2f}s  n_par={info.get('n_parallel')}  stages={ {k2: round(v2) for k2, v2 in info.get('stage_totals_ms', {}).items()} }", flush=True)
        del out
