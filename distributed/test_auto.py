"""auto_attention inside a torch.distributed group: the planner should pick the multi-device
configuration for 2x80GB at N=1M-2M, run it, and match the single-device result.
    sbatch next/slurm/run2_test.slurm next/distributed/test_auto.py 1048576 2097152
"""
import os, sys, time, torch, torch.distributed as dist
from datetime import timedelta
local_rank = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank), timeout=timedelta(hours=1))
rank, world = dist.get_rank(), dist.get_world_size()
from stream_cqsa.autoconfig import auto_attention, hardware_from_dict, detect_hardware
from stream_cqsa.stable_stream import stream_cqsa_forward
hw = hardware_from_dict({f"cuda:{i}": "72GiB" for i in range(world)} | {"host": "150GiB", "link_gbs": 200})
for N in [int(x) for x in sys.argv[1:]] or [1048576]:
    torch.manual_seed(0); q, k, v = (torch.randn(1, 8, N, 64, dtype=torch.float16) for _ in range(3))
    dist.barrier(); t0 = time.perf_counter()
    out, p = auto_attention(q, k, v, causal=True, hardware=hw, verbose=(rank == 0), allow_escalation=False)
    dist.barrier(); torch.cuda.synchronize(); t_auto = time.perf_counter() - t0
    if rank == 0:
        t0 = time.perf_counter(); ref, _ = stream_cqsa_forward(q, k, v, itr=1, causal=True, stream_from_host=True, low_memory=True, allow_escalation=False, shared_chunks=True)
        torch.cuda.synchronize(); t_single = time.perf_counter() - t0
        err = ((out.float().cpu() - ref.float().cpu()).norm() / ref.float().cpu().norm()).item()
        print(f"N={N}: auto -> {p.name()} in {t_auto:.2f}s (planner est {p.est_time_s:.1f}s); single-device itr=1 acc=cpu {t_single:.2f}s; rel.err {err:.2e}", flush=True)
    del out
dist.barrier(); dist.destroy_process_group()
