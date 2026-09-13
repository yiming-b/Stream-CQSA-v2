"""
Native wave kernel vs the previous engine (one kernel per subproblem, streams) vs monolithic FA-2.

    sbatch next/native/run_test.slurm            (test_wave.py then this)
Outputs: next/logs/wave_bench.json, wave_bench.png, wave_timeline.png
"""
import os, gc, json, time
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.environ.setdefault("CQSA_CUDA_MODULE", "cqsa_cuda"); os.environ.setdefault("CQSA_CUDA_MODULE_NONCAUSAL", "cqsa_cuda_nc")
from stream_cqsa.native_wave import wave_forward, wave_backward, native_ext, wave_tasks, build_wave_tables, block_base_relative, SEG_ALIGN
from stream_cqsa.stable_stream import stream_cqsa_forward, stream_cqsa_backward
from stream_cqsa.autoconfig import QUORUM_SETS
import stream_cqsa.interface as I
from flash_attn import flash_attn_func
OUT = "results/native"
dev = torch.device("cuda"); ext = native_ext()
print(torch.cuda.get_device_name(0), flush=True)
res = dict(device=torch.cuda.get_device_name(0))

def timed(fn, reps=3, warm=1):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3, torch.cuda.max_memory_allocated() / 2**30

# ---- 0. kernel alone on the c=7 subproblem (L=3N/7 of N=131072): did the refactor cost the single-launch path anything?
N, H, D = 131072, 8, 64
torch.manual_seed(0)
q, k, v = (torch.randn(1, H, N, D, device=dev, dtype=torch.float16) for _ in range(3))
tasks = wave_tasks(N, 1, 7, (0, 1, 3), H=H, D=D, itemsize=2)
t = tasks[3]; idx = t.token_ids.to(dev)
q_i, k_i, v_i = (x.index_select(2, idx).transpose(1, 2).contiguous() for x in (q, k, v))
bits, bo, ba = t.group_bits.to(dev), t.extra["blk_or"].to(dev), t.extra["blk_and"].to(dev)
import importlib
v11 = importlib.import_module("cqsa_cuda")
def run_mod(mod):
    return mod.fwd_cqs_group_bits(q_i, k_i, v_i, None, None, 0.0, D ** -0.5, True, -1, -1, 0.0, False, None, bits, bo, ba, 64, None, None, None)
tb = build_wave_tables([t], dev, uniform=True)
qt, kt, vt = (x[0].transpose(0, 1) for x in (q, k, v)); bb = block_base_relative(tb, lambda g: g).to(dev)
L = int(t.local_size); pad = lambda x: torch.cat([x[0], x[0][:1].expand(tb.S - L, -1, -1)], 0).contiguous()
qp, kp, vp = pad(q_i), pad(k_i), pad(v_i)
res["kernel_L"] = int(t.local_size)
res["kernel_ms"] = dict(
    v11=timed(lambda: run_mod(v11), reps=10)[0],
    native_single=timed(lambda: run_mod(ext), reps=10)[0],
    native_wave_W1_packed=timed(lambda: ext.fwd_wave(qp, kp, vp, tb.cu, tb.max_L, tb.total, tb.bits, tb.blk_or, tb.blk_and, tb.blk_cu, None, None, 0, D ** -0.5, True, uniform_S=tb.S, uniform_W=tb.W), reps=10)[0],
    native_wave_W1_inplace=timed(lambda: ext.fwd_wave(qt, kt, vt, tb.cu, tb.max_L, tb.total, tb.bits, tb.blk_or, tb.blk_and, tb.blk_cu, bb, tb.bb_cu, SEG_ALIGN, D ** -0.5, True, uniform_S=tb.S, uniform_W=tb.W), reps=10)[0],
    fa2=timed(lambda: flash_attn_func(q_i, k_i, v_i, causal=True), reps=10)[0])
del qp, kp, vp
print("kernel alone (L=%d causal):" % t.local_size, {k_: round(v_, 2) for k_, v_ in res["kernel_ms"].items()}, flush=True)
del q_i, k_i, v_i

# ---- 1. N=131072: c in (7, 31, 73), engine comparison (device-resident inputs)
res["n131k"] = []
qq, kk, vv = (x.transpose(1, 2) for x in (q, k, v))
mono_ms, _ = timed(lambda: flash_attn_func(qq, kk, vv, causal=True), reps=5)
res["mono_ms_131k"] = mono_ms
print(f"monolithic FA-2: {mono_ms:.1f} ms", flush=True)
for c in (7, 31, 73):
    iset = QUORUM_SETS[c]
    row = dict(c=c, L=int(N * len(iset) / c))
    for n_par in (1, 2):
        ms, pk = timed(lambda: stream_cqsa_forward(q, k, v, itr=1, causal=True, c=c, interest_set=iset, max_parallel=n_par, allow_escalation=False))
        row[f"old_npar{n_par}_ms"] = ms; row[f"old_npar{n_par}_gib"] = pk
    for cap in (1, 2, 4, None):
        ms, pk = timed(lambda: wave_forward(q, k, v, causal=True, itr=1, c=c, interest_set=iset, max_wave_subproblems=cap))
        row[f"wave_cap{cap}_ms"] = ms; row[f"wave_cap{cap}_gib"] = pk
    o_w, _ = wave_forward(q, k, v, causal=True, itr=1, c=c, interest_set=iset)
    o_o, _ = stream_cqsa_forward(q, k, v, itr=1, causal=True, c=c, interest_set=iset, allow_escalation=False)
    row["rel_diff_wave_vs_old"] = ((o_w - o_o).norm() / o_o.norm()).item()
    res["n131k"].append(row); print(row, flush=True)
    del o_w, o_o

# ---- 2. backward at N=131072, c=7 itr=1
dout = torch.randn_like(q)
out_w, info_w = wave_forward(q, k, v, causal=True, itr=1, c=7, interest_set=(0, 1, 3))
ms_wb, pk_wb = timed(lambda: wave_backward(q, k, v, out_w, dout, info_w["lse"], causal=True, itr=1, c=7, interest_set=(0, 1, 3)))
ms_ob, pk_ob = timed(lambda: stream_cqsa_backward(q, k, v, dout, out_w, info_w["lse"], itr=1, causal=True, c=7, interest_set=(0, 1, 3), accumulate_on_gpu=True, allow_escalation=False))
res["bwd_131k_c7"] = dict(wave_ms=ms_wb, wave_gib=pk_wb, old_ms=ms_ob, old_gib=pk_ob)
qq2, kk2, vv2 = (x.transpose(1, 2).detach().clone().requires_grad_(True) for x in (q, k, v))
def fa_bwd():
    o = flash_attn_func(qq2, kk2, vv2, causal=True); o.backward(dout.transpose(1, 2))
res["bwd_131k_c7"]["fa2_fwd_bwd_ms"] = timed(fa_bwd)[0]
print("backward 131k c=7:", res["bwd_131k_c7"], flush=True)
del out_w, dout, qq2, kk2, vv2

# ---- 3. N=1M device-resident: c=7 / c=31 itr=1, one wave vs old engine n_par=2 vs FA-2
del q, k, v, qq, kk, vv; gc.collect(); torch.cuda.empty_cache()
N = 1 << 20
q, k, v = (torch.randn(1, H, N, D, device=dev, dtype=torch.float16) for _ in range(3))
qq, kk, vv = (x.transpose(1, 2) for x in (q, k, v))
res["n1m"] = dict(mono_ms=timed(lambda: flash_attn_func(qq, kk, vv, causal=True), reps=2)[0])
for c in (7, 31):
    iset = QUORUM_SETS[c]
    ms, pk = timed(lambda: wave_forward(q, k, v, causal=True, itr=1, c=c, interest_set=iset), reps=2)
    res["n1m"][f"wave_c{c}_ms"] = ms; res["n1m"][f"wave_c{c}_gib"] = pk
    _, info = wave_forward(q, k, v, causal=True, itr=1, c=c, interest_set=iset); res["n1m"][f"wave_c{c}_waves"] = info["wave_sizes"]
    ms, pk = timed(lambda: stream_cqsa_forward(q, k, v, itr=1, causal=True, c=c, interest_set=iset, max_parallel=2, allow_escalation=False), reps=2)
    res["n1m"][f"old_c{c}_ms"] = ms; res["n1m"][f"old_c{c}_gib"] = pk
    print(f"N=1M c={c}:", {k_: v_ for k_, v_ in res["n1m"].items() if f"c{c}" in k_}, flush=True)
# host-resident 1M, c=7: wave (chunk pool) vs old engine streaming from host with a device accumulator
qh, kh, vh = (x.cpu() for x in (q, k, v))
del q, k, v, qq, kk, vv; gc.collect(); torch.cuda.empty_cache()
ms, pk = timed(lambda: wave_forward(qh, kh, vh, causal=True, itr=1, c=7, interest_set=(0, 1, 3)), reps=2)
_, info = wave_forward(qh, kh, vh, causal=True, itr=1, c=7, interest_set=(0, 1, 3))
res["n1m"]["host_wave_c7_ms"] = ms; res["n1m"]["host_wave_c7_gib"] = pk; res["n1m"]["host_wave_c7_plan"] = dict(slots=info["pool_slots"], waves=info["wave_sizes"])
ms, pk = timed(lambda: stream_cqsa_forward(qh, kh, vh, itr=1, causal=True, c=7, interest_set=(0, 1, 3), max_parallel=2, allow_escalation=False, stream_from_host=True), reps=2)
res["n1m"]["host_old_c7_ms"] = ms; res["n1m"]["host_old_c7_gib"] = pk
print("N=1M host-resident:", {k_: v_ for k_, v_ in res["n1m"].items() if "host" in k_}, flush=True)
json.dump(res, open(f"{OUT}/wave_bench.json", "w"), indent=1)

# ---- plot: N=131072 engines
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
labels = ["old n_par=1", "old n_par=2", "wave cap 1", "wave cap 2", "wave cap 4", "wave (all)"]
keys = ["old_npar1", "old_npar2", "wave_cap1", "wave_cap2", "wave_cap4", "wave_capNone"]
x = range(len(labels)); w = 0.27
for i, row in enumerate(res["n131k"]):
    ax[0].bar([xx + (i - 1) * w for xx in x], [row[f"{k_}_ms"] for k_ in keys], w, label=f"c={row['c']} ({row['c']} subproblems, L={row['L']})")
    ax[1].bar([xx + (i - 1) * w for xx in x], [row[f"{k_}_gib"] for k_ in keys], w, label=f"c={row['c']}")
ax[0].axhline(res["mono_ms_131k"], color="k", ls="--", label="monolithic FA-2"); ax[0].set_ylabel("forward wall time (ms)")
ax[0].set_title("N=131072, one A100, itr=1, device-resident inputs"); ax[1].set_ylabel("peak device memory (GiB)")
for a in ax: a.set_xticks(list(x)); a.set_xticklabels(labels, rotation=20, fontsize=8); a.legend(fontsize=8); a.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/wave_bench.png", dpi=110)

# ---- timeline: one launch per wave (c=73) from a Chrome trace
from torch.profiler import profile, ProfilerActivity
N = 131072
q, k, v = (torch.randn(1, H, N, D, device=dev, dtype=torch.float16) for _ in range(3))
fig, axes = plt.subplots(2, 1, figsize=(14, 5.5))
for ax_, (label, fn) in zip(axes, [("previous engine, max_parallel=2 (73 kernels)", lambda: stream_cqsa_forward(q, k, v, itr=1, causal=True, c=73, interest_set=QUORUM_SETS[73], max_parallel=2, allow_escalation=False)),
                                   ("native wave kernel: 73 subproblems in ONE launch + one merge", lambda: wave_forward(q, k, v, causal=True, itr=1, c=73, interest_set=QUORUM_SETS[73]))]):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        fn(); torch.cuda.synchronize()
    path = f"{OUT}/trace_wave.json"; prof.export_chrome_trace(path); tr = json.load(open(path))["traceEvents"]; os.remove(path)
    ev = [e for e in tr if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "stream" in e.get("args", {})]
    t0 = min(e["ts"] for e in ev); t1 = max(e["ts"] + e["dur"] for e in ev)
    streams = sorted({e["args"]["stream"] for e in ev}); cmap = plt.get_cmap("tab20")
    j = 0
    for yi, sid in enumerate(streams):
        for e in sorted([e for e in ev if e["args"]["stream"] == sid], key=lambda e: e["ts"]):
            x0 = (e["ts"] - t0) / 1e3; wdt = e["dur"] / 1e3
            if "flash_fwd_kernel" in e["name"]:
                ax_.barh(yi, wdt, left=x0, height=0.8, color=cmap(j % 20), edgecolor="black", lw=0.3); j += 1
            elif "wave_merge" in e["name"]:
                ax_.barh(yi, max(wdt, 0.2), left=x0, height=0.8, color="red", edgecolor="black", lw=0.3)
            else:
                ax_.barh(yi, max(wdt, 0.05), left=x0, height=0.45, color="gray", alpha=0.6)
    ax_.set_yticks(range(len(streams))); ax_.set_yticklabels([f"stream {s}" for s in streams], fontsize=8)
    ax_.set_title(f"{label}  |  device span {(t1 - t0) / 1e3:.0f} ms  |  colored = attention kernel, red = wave merge, gray = other", fontsize=9)
    ax_.set_xlabel("ms"); ax_.grid(axis="x", alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/wave_timeline.png", dpi=110)
print("done", flush=True)
