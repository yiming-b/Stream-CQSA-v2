"""
Single GPU, small N, large c: how much does running several subproblems
concurrently buy, and what does the execution order look like?

For c in (7, 31, 73) at N=131072 (subproblems of 3N/7, 6N/31, 9N/73 tokens; all
c^1 subproblems fit on the device at once), sweep max_parallel = 1, 2, 4, 8, 16
(device-resident inputs, device accumulator) and record wall time + peak memory.
The execution-order timeline (one row per CUDA stream) is produced by
parallel_timeline.py from a Chrome trace: torch.profiler's Python events do not
expose the stream id (every kernel reports stream 0), the exported trace does.

    sbatch next/slurm/run1_test.slurm next/bench/parallel_subproblems.py
Outputs: next/logs/parallel_subproblems.json, parallel_subproblems_time.png
"""
import gc, json, time, os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stream_cqsa.stable_stream import stream_cqsa_forward
from stream_cqsa.autoconfig import QUORUM_SETS
OUT = "results/parallel_subproblems"
N, H, D = 131072, 8, 64
torch.manual_seed(0)
q, k, v = (torch.randn(1, H, N, D, device="cuda", dtype=torch.float16) for _ in range(3))
print(torch.cuda.get_device_name(0), flush=True)

def run(c, n_par, reps=3):
    kw = dict(itr=1, causal=True, c=c, interest_set=QUORUM_SETS[c], max_parallel=n_par, allow_escalation=False)
    ts = []
    for _ in range(reps + 1):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.perf_counter(); out, info = stream_cqsa_forward(q, k, v, **kw); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return dict(c=c, n_par=n_par, tasks=info["n_subproblems"], L=int(N * len(QUORUM_SETS[c]) / c), ms=min(ts[1:]) * 1e3,
                peak_gib=torch.cuda.max_memory_allocated() / 2**30, n_parallel_used=info["n_parallel"])

res = dict(N=N, sweep=[], timeline={})
for c in (7, 31, 73):
    for n_par in (1, 2, 4, 8, 16):
        if n_par > c ** 1: continue
        r = run(c, n_par); res["sweep"].append(r)
        print(f"c={c:2d} ({r['tasks']:2d} subproblems of L={r['L']}): n_par={n_par:2d} -> {r['ms']:7.1f} ms, peak {r['peak_gib']:.2f} GiB", flush=True)
# monolithic reference on the same GPU
from flash_attn import flash_attn_func
qq, kk, vv = (t.transpose(1, 2) for t in (q, k, v))
for _ in range(3): flash_attn_func(qq, kk, vv, causal=True)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(5): flash_attn_func(qq, kk, vv, causal=True)
torch.cuda.synchronize(); res["mono_ms"] = (time.perf_counter() - t0) / 5 * 1e3
print(f"monolithic FlashAttention-2: {res['mono_ms']:.1f} ms", flush=True)

# ---- plot 1: time vs n_par
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for c in (7, 31, 73):
    rows = [r for r in res["sweep"] if r["c"] == c]
    ax[0].plot([r["n_par"] for r in rows], [r["ms"] for r in rows], "o-", label=f"c={c} ({rows[0]['tasks']} subproblems, L={rows[0]['L']})")
    ax[1].plot([r["n_par"] for r in rows], [r["peak_gib"] for r in rows], "o-", label=f"c={c}")
ax[0].axhline(res["mono_ms"], color="k", ls="--", label="monolithic FA-2")
ax[0].set_xscale("log", base=2); ax[0].set_xlabel("subproblems in flight (max_parallel)"); ax[0].set_ylabel("forward wall time (ms)")
ax[0].set_title(f"N={N}, one A100, itr=1, acc=GPU"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].set_xscale("log", base=2); ax[1].set_xlabel("subproblems in flight"); ax[1].set_ylabel("peak device memory (GiB)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/parallel_subproblems_time.png", dpi=110)

json.dump(res, open(f"{OUT}/parallel_subproblems.json", "w"), indent=1)
print("wrote plot", flush=True)
