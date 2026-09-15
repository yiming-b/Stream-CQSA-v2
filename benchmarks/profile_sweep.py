"""
Profile Stream-CQSA configurations one parameter at a time (A100-80GB, forward pass).

Fixed: B=1, H=8, D=64, fp16, causal; kernel in {classic engine + CUDA extension, native wave kernel,
classic engine + Triton kernel}; device-resident
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


KERNELS = ("cuda", "wave", "triton")     # classic engine + CUDA extension; native wave kernel; classic engine + Triton


def configs():
    cfgs = []
    for kern in KERNELS:
        cfgs += [dict(sweep="c", kernel=kern, c=c, N=N_C, itr=1, acc="gpu") for c in C_LIST]
        for itr in (1, 2):
            for acc in ("gpu", "cpu"):
                cfgs += [dict(sweep=f"N_itr{itr}_{acc}", kernel=kern, c=7, N=N, itr=itr, acc=acc) for N in N_LIST]
    return cfgs


def worker(cfg, reps, warmup):
    kern = cfg.get("kernel", "cuda")
    if kern == "triton":
        os.environ["CQSA_FORWARD"] = "triton"          # read by the engine at import / call time
    from stream_cqsa.stable_stream import stream_cqsa_forward
    from stream_cqsa.autoconfig import QUORUM_SETS
    if kern == "wave":
        from stream_cqsa.native_wave import wave_forward
    dev = torch.device("cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(B, H, cfg["N"], D, generator=g, device=dev, dtype=torch.float16) for _ in range(3))
    if kern == "wave":
        kw = dict(itr=cfg["itr"], causal=True, c=cfg["c"], interest_set=QUORUM_SETS[cfg["c"]], accumulate_on_gpu=(cfg["acc"] == "gpu"))
        run_fwd = lambda: wave_forward(q, k, v, **kw)
    else:
        kw = dict(itr=cfg["itr"], causal=True, c=cfg["c"], interest_set=QUORUM_SETS[cfg["c"]], allow_escalation=False,
                  max_parallel=2, low_memory=(cfg["acc"] == "cpu"), accumulate_on_gpu=(cfg["acc"] == "gpu"))
        run_fwd = lambda: stream_cqsa_forward(q, k, v, **kw)
    base = torch.cuda.memory_allocated()
    times, peaks, info = [], [], {}
    for i in range(warmup + reps):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, info = run_fwd()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        pk = torch.cuda.max_memory_allocated()
        del out; info.pop("lse", None)
        if i >= warmup:
            times.append(dt); peaks.append(pk)
    return dict(cfg, kernel=kern, times_s=times, peak_gib=[p / 2**30 for p in peaks], workspace_gib=[(p - base) / 2**30 for p in peaks],
                inputs_gib=base / 2**30, n_subproblems=info.get("n_subproblems") or (sum(info["wave_sizes"]) if "wave_sizes" in info else None),
                n_parallel=info.get("n_parallel"), n_waves=len(info["wave_sizes"]) if "wave_sizes" in info else None,
                stage_ms=info.get("stage_totals_ms"))


def run(out_dir, reps, warmup, only=None, kernels=None):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); done.add((r["sweep"], r.get("kernel", "cuda"), r["c"], r["N"], r["itr"], r["acc"]))
    for cfg in configs():
        if only and cfg["sweep"] not in only: continue
        if kernels and cfg["kernel"] not in kernels: continue
        key = (cfg["sweep"], cfg["kernel"], cfg["c"], cfg["N"], cfg["itr"], cfg["acc"])
        if key in done: continue
        w = json.dumps(dict(cfg, reps=reps, warmup=warmup))
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), "worker", w], stdout=subprocess.PIPE, text=True)
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        r = json.loads(line[7:]) if line else dict(cfg, status="error", rc=proc.returncode)
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        if "times_s" in r:
            print(f"{cfg['sweep']:12s} {cfg['kernel']:6s} c={cfg['c']:3d} N={cfg['N']:8d} itr={cfg['itr']} acc={cfg['acc']}: "
                  f"{np.mean(r['times_s']):8.3f} s +- {np.std(r['times_s']):.3f}  peak {np.mean(r['peak_gib']):.2f} GiB  ({r['n_subproblems']} subproblems)", flush=True)
        else:
            print(f"{cfg}: ERROR rc={r.get('rc')}", flush=True)


def fit(out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = [json.loads(l) for l in open(os.path.join(out_dir, "results.jsonl")) if "times_s" in l]
    for r in rows: r.setdefault("kernel", "cuda")
    KL = {"cuda": "classic engine, CUDA kernel", "wave": "native wave kernel", "triton": "classic engine, Triton kernel"}
    CO = {"cuda": "C0", "wave": "C1", "triton": "C2"}
    r2 = lambda y, yh: 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)
    fits = {}
    sweeps = ["N_itr1_gpu", "N_itr1_cpu", "N_itr2_gpu", "N_itr2_cpu"]
    fig, axes = plt.subplots(2, 5, figsize=(24, 8.5))
    for kern in KERNELS:
        rc = sorted([r for r in rows if r["sweep"] == "c" and r["kernel"] == kern], key=lambda r: r["c"])
        if rc:
            cs = [r["c"] for r in rc]; t = [np.mean(r["times_s"]) for r in rc]; te = [np.std(r["times_s"]) for r in rc]
            m = [np.mean(r["peak_gib"]) for r in rc]
            axes[0, 0].errorbar(cs, t, yerr=te, fmt="o-", color=CO[kern], label=KL[kern])
            axes[1, 0].plot(cs, m, "s-", color=CO[kern], label=KL[kern])
            fits[f"c/{kern}"] = [dict(c=c_, time_s=t_, time_std=e_, peak_gib=m_) for c_, t_, e_, m_ in zip(cs, t, te, m)]
    for ax in axes[:, 0]:
        ax.set_xscale("log"); ax.set_xticks(C_LIST); ax.set_xticklabels(C_LIST); ax.set_xlabel("c"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0, 0].set_title(f"vary c (N={N_C}, itr=1, acc=GPU)"); axes[0, 0].set_ylabel("forward time (s)"); axes[1, 0].set_ylabel("peak device memory (GiB)")
    for j, sw in enumerate(sweeps, start=1):
        lab = sw.replace("N_", "").replace("_", ", acc=").replace("itr", "itr=")
        axes[0, j].set_title(f"vary N (c=7, {lab})")
        for kern in KERNELS:
            rs = sorted([r for r in rows if r["sweep"] == sw and r["kernel"] == kern], key=lambda r: r["N"])
            if len(rs) < 3: continue
            N = np.array([r["N"] for r in rs], float); t = np.array([np.mean(r["times_s"]) for r in rs]); te = np.array([np.std(r["times_s"]) for r in rs])
            m = np.array([np.mean(r["peak_gib"]) for r in rs]); x = N / 1e6
            pt = np.polyfit(x, t, 2); pm = np.polyfit(x, m, 1)
            r2t, r2m = r2(t, np.polyval(pt, x)), r2(m, np.polyval(pm, x))
            fits[f"{sw}/{kern}"] = dict(N=N.tolist(), time_s=t.tolist(), time_std=te.tolist(), peak_gib=m.tolist(),
                                        time_fit="t[s] = %.4g*(N/1e6)^2 + %.4g*(N/1e6) + %.4g" % tuple(pt), time_r2=r2t,
                                        mem_fit="peak[GiB] = %.4g*(N/1e6) + %.4g" % tuple(pm), mem_r2=r2m)
            xx = np.linspace(x.min(), x.max(), 200)
            axes[0, j].errorbar(x, t, yerr=te, fmt="o", color=CO[kern]); axes[0, j].plot(xx, np.polyval(pt, xx), "-", color=CO[kern], label=f"{KL[kern]}: quadratic, R2={r2t:.4f}")
            axes[1, j].plot(x, m, "s", color=CO[kern]); axes[1, j].plot(xx, np.polyval(pm, xx), "-", color=CO[kern], label=f"{KL[kern]}: linear, R2={r2m:.4f}")
        for ax, yl in ((axes[0, j], "forward time (s)"), (axes[1, j], "peak device memory (GiB)")):
            ax.set_xlabel("N (M tokens)"); ax.set_ylabel(yl); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("Stream-CQSA forward, A100-80GB, fp16 causal, B=1 H=8 D=64; classic engine (CUDA / Triton kernel, 2 in flight) and native wave kernel; mean +- std of 5 runs")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "profile_sweep.png"), dpi=110)
    json.dump(fits, open(os.path.join(out_dir, "fits.json"), "w"), indent=1)
    for key, f_ in fits.items():
        if key.startswith("c/"):
            print(f"{key}:", ", ".join(f"c={r['c']}: {r['time_s']:.2f}s/{r['peak_gib']:.2f}GiB" for r in f_))
        else:
            print(f"{key}: {f_['time_fit']} (R2 {f_['time_r2']:.4f});  {f_['mem_fit']} (R2 {f_['mem_r2']:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["run", "fit", "worker"]); ap.add_argument("arg", nargs="?")
    ap.add_argument("--out", default="next/logs/profile_sweep"); ap.add_argument("--reps", type=int, default=5); ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--only", default="", help="comma list of sweeps to run")
    ap.add_argument("--kernels", default=",".join(KERNELS), help="comma list of kernels to run")
    a = ap.parse_args()
    if a.mode == "worker":
        w = json.loads(a.arg); print("RESULT " + json.dumps(worker(w, w["reps"], w["warmup"])), flush=True)
    elif a.mode == "run":
        run(a.out, a.reps, a.warmup, only=[s for s in a.only.split(",") if s], kernels=[s for s in a.kernels.split(",") if s])
    else:
        fit(a.out)
