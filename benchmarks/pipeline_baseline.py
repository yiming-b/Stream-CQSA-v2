"""
Clean-node baseline for track 2: how much does n_par buy, per accumulator
placement, and where does the time go?

    sbatch next/slurm/run1.slurm next/bench/pipeline_baseline.py

Forward only, itr=1, causal, Q/K/V host-resident (the paper's configuration).
For n_par=1 the stage breakdown is recorded so the serial sum of stages can be
compared against wall clock -- the gap between them is what pipelining can
recover.
"""
import torch, time, gc, json, os
from stream_cqsa.stable_stream import stream_cqsa_forward, TraceRecorder

B, H, D = 1, 8, 64
print(f"GPU {torch.cuda.get_device_name(0)}  H={H} D={D} itr=1 causal fp16, Q/K/V on host")
rows = []
for N in (65536, 262144, 1048576):
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, N, D, dtype=torch.float16) for _ in range(3))
    print(f"\n=== N={N} ===")
    print(f"{'acc':>4} {'n_par':>6} {'wall ms':>10} {'stages':>60}")
    for acc_gpu in (True, False):
        for n_par in (1, 2, 4):
            ts = []; stages = {}
            for rep in range(3):
                gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
                tr = TraceRecorder(enabled=(rep == 0 and n_par == 1))
                t0 = time.perf_counter()
                out, info = stream_cqsa_forward(q, k, v, itr=1, causal=True, stream_from_host=True,
                                                low_memory=not acc_gpu, allow_escalation=False,
                                                max_parallel=n_par, trace=tr)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
                if tr.enabled:
                    stages = {k2: round(v2, 1) for k2, v2 in info.get("stage_totals_ms", {}).items()}
                del out, info
            wall = sorted(ts)[1]
            r = dict(N=N, acc="GPU" if acc_gpu else "CPU", n_par=n_par, wall_ms=wall, stages=stages)
            rows.append(r)
            st = " ".join(f"{k2}={v2:.0f}" for k2, v2 in stages.items() if k2 != "wait") if stages else ""
            print(f"{r['acc']:>4} {n_par:>6} {wall:10.1f} {st:>60}", flush=True)
    del q, k, v
out = os.path.join(os.path.dirname(__file__), "..", "logs", "pipeline_baseline.json")
json.dump(rows, open(out, "w"), indent=1)
print(f"\nwrote {out}")
