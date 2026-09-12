"""Clean-node check of the (c, interest_set) axis at N=1M: time and peak memory per quorum set at itr=1
(and c=7/13 at itr=2), acc=GPU, host-resident inputs; the planner's model prediction alongside.
    sbatch next/slurm/run1_test.slurm next/bench/quorum_axis.py
"""
import json, torch, gc
from stream_cqsa.stable_stream import stream_cqsa_forward
from stream_cqsa.devkit import measure, reference_rows, sample_rows, accuracy_vs_fp64
from stream_cqsa.autoconfig import QUORUM_SETS, CostModel, calibrate, detect_hardware
N = 1048576; torch.manual_seed(0)
q, k, v = (torch.randn(1, 8, N, 64, dtype=torch.float16).pin_memory() for _ in range(3))
rows = sample_rows(N, 128); ref = reference_rows(q, k, v, rows, causal=True, scale=64 ** -0.5)
cm = calibrate(detect_hardware(), N=131072)
out_rows = []
for c, s in QUORUM_SETS.items():
    for itr in (1, 2):
        if c ** itr > 200: continue
        l = len(s)
        try:
            out, perf = measure(lambda: stream_cqsa_forward(q, k, v, itr=itr, causal=True, c=c, interest_set=s, stream_from_host=True,
                                                            allow_escalation=False, max_parallel=2, shared_chunks=(itr == 1))[0], device="cuda", reps=2)
            err = accuracy_vs_fp64(out, q, k, v, causal=True, scale=64 ** -0.5, rows=rows, ref_rows=ref)["rel_fro"]
            pred = cm.cqsa_time(N, 1, 8, 64, itr, True, "gpu", True, 2, c=c, l=l)
            r = dict(c=c, l=l, itr=itr, tasks=c ** itr, L=int(N * (l / c) ** itr), s=perf["s"], peak_gib=perf["peak_gib"], rel_err=err, model_s=pred)
            print(f"c={c:2d} l={l} itr={itr}: tasks={c**itr:4d} L={r['L']:7d}  {perf['s']:7.2f} s (model {pred:6.2f})  peak {perf['peak_gib']:.2f} GiB  err {err:.1e}", flush=True)
            out_rows.append(r); del out
        except Exception as e:
            print(f"c={c} itr={itr}: {type(e).__name__}: {str(e)[:100]}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
json.dump(dict(N=N, calib=json.loads(cm.to_json()), rows=out_rows), open("/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/logs/quorum_axis_1M.json", "w"), indent=1)
