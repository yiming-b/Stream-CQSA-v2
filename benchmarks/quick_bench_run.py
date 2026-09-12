"""Clean-node run of the devkit: quick_bench sweep + native-kernel compare + calibration + autotune.
    sbatch next/slurm/run1_test.slurm next/bench/quick_bench_run.py --N 65536 262144 1048576 --budget 70
"""
import argparse, json, os, sys, torch
from stream_cqsa.devkit import quick_bench, compare_kernels, default_configs, Config
from stream_cqsa.autoconfig import detect_hardware, calibrate, plan, autotune, CostModel
from stream_cqsa.stable_stream import local_stats_flash
ap = argparse.ArgumentParser(); ap.add_argument("--N", type=int, nargs="+", default=[65536, 262144, 1048576])
ap.add_argument("--budget", type=float, default=70.0); ap.add_argument("--out", default="quick_bench_clean.json")
a = ap.parse_args()
hw = detect_hardware(); print(hw.summary(), flush=True)
res = dict(hardware=hw.summary(), kernel=os.environ.get("CQSA_CUDA_MODULE"), calib=None, bench=[], compare=[], plans=[])
cm = calibrate(hw, N=131072); res["calib"] = json.loads(cm.to_json())
for N in a.N:
    print(f"\n##### compare_kernels: native kernel in Stream-CQSA vs flash-attn monolithic, N={N}", flush=True)
    for cfg in (Config(itr=1, acc="gpu", n_par=2), Config(itr=1, acc="cpu", stream_from_host=True, n_par=1), Config(itr=2, acc="gpu", n_par=2)):
        rep = compare_kernels(local_stats_flash, N=N, cfg=cfg, reps=2, acc_rows=256)
        res["compare"].append(dict(cfg=cfg.name(), **rep.as_dict()))
    print(f"\n##### quick_bench N={N} budget={a.budget} GiB", flush=True)
    r = quick_bench(N=N, budget_gib=a.budget, configs=default_configs(max_itr=2, n_pars=(1, 2, 4)), reps=2, acc_rows=256)
    res["bench"].append(r)
    p_model = plan(N=N, hardware=hw, model=cm)
    print(f"planner (calibrated model) says: {p_model.name()} est {p_model.est_time_s:.2f}s {p_model.est_peak_gib:.1f}GiB -- {p_model.reason[:160]}")
    res["plans"].append(dict(N=N, plan=p_model.name(), est=p_model.est_time_s, reason=p_model.reason, measured_pick=(r["pick"] or {}).get("config")))
    json.dump(res, open(f"/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/logs/{a.out}", "w"), indent=1, default=str)
# a memory-constrained planning example at the largest N: budget = 40% of the device
small = detect_hardware(budget_fraction=0.4)
for N in a.N:
    p = plan(N=N, hardware=small, model=cm); print(f"budget {small.devices[0].budget_bytes/2**30:.0f} GiB, N={N}: {p.name()} est {p.est_time_s:.1f}s {p.est_peak_gib:.1f}GiB")
print("\n##### autotune (measured pick) at N=262144, 40% budget")
p = autotune(N=262144, hardware=small); print("autotune ->", p.name(), p.reason)
