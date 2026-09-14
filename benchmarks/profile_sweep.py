"""
Profile Stream-CQSA configurations one parameter at a time (A100-80GB, forward pass).

Fixed: B=1, H=8, D=64, fp16, causal, classic engine + the CUDA extension (cqsa_cuda), device-resident
Q/K/V, two subproblems in flight, no escalation. Each configuration runs in its own
subprocess (flushed GPU), 1 warm-up + `--reps` timed repetitions, peak memory reset per rep.

  sweep c    : c in {7,13,21,31,57,73,91,133}, N=512K, itr=1, acc=GPU
  sweep N x4 : N in {64K..2M}, c=7, (itr, acc) in {1,2} x {GPU, CPU}

    python benchmarks/profile_sweep.py run --out results/profile_sweep     (the sweeps; JSON per config)
    python benchmarks/profile_sweep.py fit --out results/profile_sweep     (fits + plot)
"""
import argparse, gc, json, os, subprocess, sys, time
import numpy as np
import torch
os.environ.setdefault("CQSA_BACKWARD", "cuda")

B, H, D = 1, 8, 64
N_LIST = [1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21]
C_LIST = [7, 13, 21, 31, 57, 73, 91, 133]
N_C = 1 << 19


def configs():
    cfgs = [dict(sweep="c", c=c, N=N_C, itr=1, acc="gpu") for c in C_LIST]
    for itr in (1, 2):
        for acc in ("gpu", "cpu"):
            cfgs += [dict(sweep=f"N_itr{itr}_{acc}", c=7, N=N, itr=itr, acc=acc) for N in N_LIST]
    return cfgs


def worker(cfg, reps, warmup):
    from stream_cqsa.stable_stream import stream_cqsa_forward
    from stream_cqsa.autoconfig import QUORUM_SETS
    dev = torch.device("cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(B, H, cfg["N"], D, generator=g, device=dev, dtype=torch.float16) for _ in range(3))
    kw = dict(itr=cfg["itr"], causal=True, c=cfg["c"], interest_set=QUORUM_SETS[cfg["c"]], allow_escalation=False,
              max_parallel=2, low_memory=(cfg["acc"] == "cpu"), accumulate_on_gpu=(cfg["acc"] == "gpu"))
    base = torch.cuda.memory_allocated()
    times, peaks, info = [], [], {}
    for i in range(warmup + reps):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, info = stream_cqsa_forward(q, k, v, **kw)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        pk = torch.cuda.max_memory_allocated()
        del out
        if i >= warmup:
            times.append(dt); peaks.append(pk)
    return dict(cfg, times_s=times, peak_gib=[p / 2**30 for p in peaks], workspace_gib=[(p - base) / 2**30 for p in peaks],
                inputs_gib=base / 2**30, n_subproblems=info.get("n_subproblems"), n_parallel=info.get("n_parallel"),
                stage_ms=info.get("stage_totals_ms"))


def run(out_dir, reps, warmup, only=None):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); done.add((r["sweep"], r["c"], r["N"], r["itr"], r["acc"]))
    for cfg in configs():
        if only and cfg["sweep"] not in only: continue
        key = (cfg["sweep"], cfg["c"], cfg["N"], cfg["itr"], cfg["acc"])
        if key in done: continue
        w = json.dumps(dict(cfg, reps=reps, warmup=warmup))
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), "worker", w], stdout=subprocess.PIPE, text=True)
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        r = json.loads(line[7:]) if line else dict(cfg, status="error", rc=proc.returncode)
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        if "times_s" in r:
            print(f"{cfg['sweep']:12s} c={cfg['c']:3d} N={cfg['N']:8d} itr={cfg['itr']} acc={cfg['acc']}: "
                  f"{np.mean(r['times_s']):8.3f} s +- {np.std(r['times_s']):.3f}  peak {np.mean(r['peak_gib']):.2f} GiB  ({r['n_subproblems']} subproblems)", flush=True)
        else:
            print(f"{cfg}: ERROR rc={r.get('rc')}", flush=True)


def fit(out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = [json.loads(l) for l in open(os.path.join(out_dir, "results.jsonl")) if "times_s" in l]
    fits = {}
    sweeps = ["N_itr1_gpu", "N_itr1_cpu", "N_itr2_gpu", "N_itr2_cpu"]
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    # c sweep
    rc = sorted([r for r in rows if r["sweep"] == "c"], key=lambda r: r["c"])
    if rc:
        cs = [r["c"] for r in rc]; t = [np.mean(r["times_s"]) for r in rc]; te = [np.std(r["times_s"]) for r in rc]
        m = [np.mean(r["peak_gib"]) for r in rc]
        axes[0, 0].errorbar(cs, t, yerr=te, fmt="o-"); axes[0, 0].set_xscale("log"); axes[0, 0].set_xticks(cs); axes[0, 0].set_xticklabels(cs)
        axes[0, 0].set_title(f"vary c (N={N_C}, itr=1, acc=GPU)"); axes[0, 0].set_xlabel("c"); axes[0, 0].set_ylabel("forward time (s)"); axes[0, 0].grid(alpha=0.3)
        axes[1, 0].plot(cs, m, "s-"); axes[1, 0].set_xscale("log"); axes[1, 0].set_xticks(cs); axes[1, 0].set_xticklabels(cs)
        axes[1, 0].set_xlabel("c"); axes[1, 0].set_ylabel("peak device memory (GiB)"); axes[1, 0].grid(alpha=0.3)
        fits["c"] = [dict(c=c_, time_s=t_, time_std=e_, peak_gib=m_) for c_, t_, e_, m_ in zip(cs, t, te, m)]
    for j, sw in enumerate(sweeps, start=1):
        rs = sorted([r for r in rows if r["sweep"] == sw], key=lambda r: r["N"])
        if not rs: continue
        N = np.array([r["N"] for r in rs], float); t = np.array([np.mean(r["times_s"]) for r in rs]); te = np.array([np.std(r["times_s"]) for r in rs])
        m = np.array([np.mean(r["peak_gib"]) for r in rs])
        x = N / 1e6
        pt = np.polyfit(x, t, 2); pm = np.polyfit(x, m, 1)
        r2 = lambda y, yh: 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)
        r2t, r2m = r2(t, np.polyval(pt, x)), r2(m, np.polyval(pm, x))
        fits[sw] = dict(N=N.tolist(), time_s=t.tolist(), time_std=te.tolist(), peak_gib=m.tolist(),
                        time_fit="t[s] = %.4g*(N/1e6)^2 + %.4g*(N/1e6) + %.4g" % tuple(pt), time_r2=r2t,
                        mem_fit="peak[GiB] = %.4g*(N/1e6) + %.4g" % tuple(pm), mem_r2=r2m)
        xx = np.linspace(x.min(), x.max(), 200)
        lab = sw.replace("N_", "").replace("_", ", acc=").replace("itr", "itr=")
        ax = axes[0, j]; ax.errorbar(x, t, yerr=te, fmt="o", label="measured"); ax.plot(xx, np.polyval(pt, xx), "-", label=f"quadratic fit, R2={r2t:.4f}")
        ax.set_title(f"vary N (c=7, {lab})"); ax.set_xlabel("N (M tokens)"); ax.set_ylabel("forward time (s)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax = axes[1, j]; ax.plot(x, m, "s", label="measured"); ax.plot(xx, np.polyval(pm, xx), "-", label=f"linear fit, R2={r2m:.4f}")
        ax.set_xlabel("N (M tokens)"); ax.set_ylabel("peak device memory (GiB)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Stream-CQSA forward, A100-80GB, fp16 causal, B=1 H=8 D=64, v11 CUDA kernel, 2 in flight; mean +- std of 5 runs")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "profile_sweep.png"), dpi=110)
    json.dump(fits, open(os.path.join(out_dir, "fits.json"), "w"), indent=1)
    for sw, f_ in fits.items():
        if sw == "c":
            print("c sweep:", ", ".join(f"c={r['c']}: {r['time_s']:.2f}s/{r['peak_gib']:.2f}GiB" for r in f_))
        else:
            print(f"{sw}: {f_['time_fit']} (R2 {f_['time_r2']:.4f});  {f_['mem_fit']} (R2 {f_['mem_r2']:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["run", "fit", "worker"]); ap.add_argument("arg", nargs="?")
    ap.add_argument("--out", default="next/logs/profile_sweep"); ap.add_argument("--reps", type=int, default=5); ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if a.mode == "worker":
        w = json.loads(a.arg); print("RESULT " + json.dumps(worker(w, w["reps"], w["warmup"])), flush=True)
    elif a.mode == "run":
        run(a.out, a.reps, a.warmup, only=[s for s in a.only.split(",") if s])
    else:
        fit(a.out)
