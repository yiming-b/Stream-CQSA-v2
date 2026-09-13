"""Execution-order timeline of the subproblems on one GPU, per CUDA stream, from a Chrome trace.
c=73 at N=131072 (73 subproblems of L=16K), max_parallel = 1, 4, 16.  -> next/logs/parallel_subproblems_timeline.png"""
import json, os, torch, gc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torch.profiler import profile, ProfilerActivity
from stream_cqsa.stable_stream import stream_cqsa_forward
from stream_cqsa.autoconfig import QUORUM_SETS
OUT = "results/parallel_subproblems"
N, H, D, c = 131072, 8, 64, 73
torch.manual_seed(0); q, k, v = (torch.randn(1, H, N, D, device="cuda", dtype=torch.float16) for _ in range(3))
configs = [1, 4, 16]
fig, axes = plt.subplots(len(configs), 1, figsize=(14, 3.0 * len(configs)))
summary = {}
for ax, n_par in zip(axes, configs):
    kw = dict(itr=1, causal=True, c=c, interest_set=QUORUM_SETS[c], max_parallel=n_par, allow_escalation=False)
    stream_cqsa_forward(q, k, v, **kw); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        stream_cqsa_forward(q, k, v, **kw); torch.cuda.synchronize()
    path = f"{OUT}/trace_c{c}_np{n_par}.json"; prof.export_chrome_trace(path)
    tr = json.load(open(path))["traceEvents"]
    ev = [e for e in tr if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "stream" in e.get("args", {})]
    t0 = min(e["ts"] for e in ev); t1 = max(e["ts"] + e["dur"] for e in ev)
    attn = sorted([e for e in ev if "flash_fwd_kernel" in e["name"] or "cqs_attn_fwd" in e["name"]], key=lambda e: e["args"].get("correlation", 0))
    order = {id(e): i for i, e in enumerate(attn)}
    streams = sorted({e["args"]["stream"] for e in ev})
    cmap = plt.get_cmap("tab20")
    for yi, sid in enumerate(streams):
        for e in ev:
            if e["args"]["stream"] != sid: continue
            x0 = (e["ts"] - t0) / 1e3; w = e["dur"] / 1e3
            if id(e) in order:
                j = order[id(e)]; ax.barh(yi, w, left=x0, height=0.8, color=cmap(j % 20), edgecolor="black", lw=0.3)
                if w > 1.0: ax.text(x0 + w / 2, yi, str(j), ha="center", va="center", fontsize=6)
            else:
                ax.barh(yi, max(w, 0.05), left=x0, height=0.45, color="gray", alpha=0.6)
    ax.set_yticks(range(len(streams))); ax.set_yticklabels([f"stream {s}" for s in streams], fontsize=8)
    ax.set_title(f"c=73: 73 subproblems of L={int(N*9/73)} tokens, N=131072, max_parallel={n_par}  |  device span {(t1-t0)/1e3:.0f} ms  |  "
                 f"colored bars = attention kernels (number = launch order), gray = gather / merge / copies", fontsize=9)
    ax.set_xlabel("ms"); ax.grid(axis="x", alpha=0.3)
    summary[n_par] = dict(device_ms=(t1 - t0) / 1e3, streams=streams, attn_kernels=len(attn))
    print(n_par, summary[n_par], flush=True)
    os.remove(path)
fig.tight_layout(); fig.savefig(f"{OUT}/parallel_subproblems_timeline.png", dpi=110); json.dump(summary, open(f"{OUT}/parallel_timeline.json", "w"), indent=1)
print("wrote timeline", flush=True)
