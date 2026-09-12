"""
Correctness and strong scaling of the multi-device forward.

    sbatch next/slurm/run4.slurm next/distributed/test_dist.py [N ...]

Every rank runs the distributed forward. Rank 0 additionally runs the
single-device engine on the full problem as the reference and reports the
relative error, so the cross-rank merge is checked against the thing it claims
to reproduce. Timing: distributed wall vs single-device wall on rank 0.
"""
import os, sys, time, json
import torch, torch.distributed as dist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dist_forward import dist_stream_cqsa_forward, dist_stream_cqsa_backward
from stream_cqsa.stable_stream import stream_cqsa_forward, stream_cqsa_backward

import traceback
# torchrun hides child tracebacks unless an error handler is installed; print
# our own to stdout so a failure is diagnosable from the slurm log.
def _excepthook(t, v, tb):
    print(f"[rank {os.environ.get('RANK','?')}] UNCAUGHT {t.__name__}: {v}", flush=True)
    traceback.print_exception(t, v, tb, file=sys.stdout); sys.stdout.flush()
sys.excepthook = _excepthook
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
from datetime import timedelta
# Rank 0 runs the single-device reference between collectives; at large N that
# is minutes, longer than NCCL's default 10-minute watchdog allows for the
# other ranks waiting at the next barrier.
dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank), timeout=timedelta(hours=2))
def _log(msg):
    print(f"[rank {os.environ.get('RANK','?')} {time.strftime('%H:%M:%S')}] {msg}", flush=True)
rank, world = dist.get_rank(), dist.get_world_size()
print(f"[rank {rank}/{world}] device {torch.cuda.current_device()} {torch.cuda.get_device_name()}", flush=True)
B, H, D = 1, 8, 64
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("N", type=int, nargs="*", default=[65536, 262144, 1048576])
ap.add_argument("--itr", type=int, nargs="+", default=[1])
ap.add_argument("--npar", type=int, default=1)
ap.add_argument("--bwd", action="store_true", help="also time and check the distributed backward")
ap.add_argument("--acc", choices=("cpu", "gpu"), default="cpu",
                help="where the per-rank accumulator lives; gpu is faster when the fp32 output fits")
args = ap.parse_args()
Ns = args.N
# Engine settings: the recommended acc=CPU configuration from the Track 2 sweep
# (shared_chunks on). Identical for the distributed and the single-device arm.
kw = dict(causal=True, stream_from_host=True, low_memory=(args.acc == "cpu"), allow_escalation=False,
          shared_chunks=True, max_parallel=args.npar)
# Warm-up: CUDA context, NCCL communicator, cuBLAS/kernel module load, pinned
# pools. Without it the first measured N pays ~1 s of one-time cost.
_q, _k, _v = (torch.randn(B, H, 16384, D, dtype=torch.float16) for _ in range(3))
dist_stream_cqsa_forward(_q, _k, _v, itr=1, **kw)
if rank == 0:
    stream_cqsa_forward(_q, _k, _v, itr=1, **kw)
del _q, _k, _v; torch.cuda.synchronize(); dist.barrier()
rows = []
for itr in args.itr:
  for N in Ns:
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, N, D, dtype=torch.float16) for _ in range(3))   # host-resident, same on all ranks
    # shared_chunks (contiguous chunk DMA) is an itr=1 feature in the engine.
    kw["shared_chunks"] = (itr == 1)
    _log(f"fwd N={N} itr={itr}: entering distributed forward")
    dist.barrier(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_d, info = dist_stream_cqsa_forward(q, k, v, itr=itr, **kw)
    dist.barrier(); torch.cuda.synchronize()
    t_dist = time.perf_counter() - t0
    if rank == 0:
        t0 = time.perf_counter()
        out_1, info1 = stream_cqsa_forward(q, k, v, itr=itr, **kw)
        torch.cuda.synchronize()
        t_single = time.perf_counter() - t0
        rel = ((out_d.float() - out_1.float()).norm() / out_1.float().norm()).item()
        lse_rel = ((info["lse"].float() - info1["lse"].float()).abs().max()).item()
        r = dict(N=N, itr=itr, world=world, npar=args.npar, acc=args.acc, t_single_s=t_single, t_dist_s=t_dist,
                 speedup=t_single / t_dist, rel_err=rel, lse_max_abs_err=lse_rel,
                 t_local_s=info["t_local_s"], t_merge_s=info["t_merge_s"],
                 tasks_per_rank=len(info["tasks_mine"]))
        rows.append(r)
        json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs",
                  f"dist_test_w{world}_itr{'-'.join(map(str, args.itr))}_{args.acc}{'_bwd' if args.bwd else ''}.json"), "w"), indent=1)
        print(f"N={N:>8} itr={itr} world={world}  single {t_single:7.2f}s  dist {t_dist:7.2f}s  "
              f"speedup {t_single/t_dist:4.2f}x  (local {info['t_local_s']:.2f}s + merge {info['t_merge_s']:.2f}s)  "
              f"rel.err {rel:.2e}  lse.err {lse_rel:.2e}  tasks/rank {len(info['tasks_mine'])}", flush=True)
    if args.bwd:
        # Every rank takes part in the distributed backward; only rank 0 runs
        # the single-device reference afterwards.
        torch.manual_seed(1)
        dout = torch.randn_like(q)
        bkw = dict(causal=True, stream_from_host=True, accumulate_on_gpu=(args.acc == "gpu"),
                   allow_escalation=False, max_parallel=args.npar)
        lse_d = info["lse"].float().cpu(); out_dc = out_d.float().cpu()
        _log(f"bwd N={N} itr={itr}: entering distributed backward")
        dist.barrier(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        (dq_d, dk_d, dv_d), binfo = dist_stream_cqsa_backward(
            q, k, v, dout, out_dc.to(q.dtype), lse_d, itr=itr, **bkw)
        dist.barrier(); torch.cuda.synchronize()
        t_bd = time.perf_counter() - t0
        _log(f"bwd N={N} itr={itr}: distributed backward done in {t_bd:.1f}s (local {binfo['t_local_s']:.1f}s, merge {binfo['t_merge_s']:.1f}s)")
        if rank == 0:
            t0 = time.perf_counter()
            dq1, dk1, dv1 = stream_cqsa_backward(
                q, k, v, dout, out_1.float().cpu().to(q.dtype), info1["lse"].float().cpu(), itr=itr, **bkw)
            torch.cuda.synchronize()
            t_b1 = time.perf_counter() - t0
            errs = [((a.float().cpu() - b.float().cpu()).norm() / b.float().cpu().norm()).item()
                    for a, b in ((dq_d, dq1), (dk_d, dk1), (dv_d, dv1))]
            r.update(bwd_t_single_s=t_b1, bwd_t_dist_s=t_bd, bwd_speedup=t_b1 / t_bd,
                     bwd_t_local_s=binfo["t_local_s"], bwd_t_merge_s=binfo["t_merge_s"],
                     bwd_rel_err=dict(dq=errs[0], dk=errs[1], dv=errs[2]))
            print(f"        bwd itr={itr}  single {t_b1:7.2f}s  dist {t_bd:7.2f}s  speedup {t_b1/t_bd:4.2f}x  "
                  f"(local {binfo['t_local_s']:.2f}s + merge {binfo['t_merge_s']:.2f}s)  "
                  f"rel.err dq {errs[0]:.2e} dk {errs[1]:.2e} dv {errs[2]:.2e}", flush=True)
            del dq1, dk1, dv1
            rows[-1] = r
        del dout, dq_d, dk_d, dv_d
    del q, k, v, out_d
    if rank == 0: del out_1, info1
    torch.cuda.empty_cache()
if rank == 0:
    out = os.path.join(os.path.dirname(__file__), "..", "logs", f"dist_test_w{world}_itr{'-'.join(map(str, args.itr))}_{args.acc}{'_bwd' if args.bwd else ''}.json")
    json.dump(rows, open(out, "w"), indent=1); print("wrote", out)
dist.barrier(); dist.destroy_process_group()
