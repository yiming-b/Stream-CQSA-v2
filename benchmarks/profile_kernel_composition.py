"""
What kernels actually run for one subproblem, plain vs CQS-masked?

The inner-kernel bench shows the CQS entry point is 1.15x plain FA even with
an all-zero mask. Is that cost INSIDE the attention kernel, or in extra
kernels around it (fp32 acc materialisation, mask prep, conversions)?
torch.profiler answers that without needing ncu permissions.

    python next/bench/profile_kernel_composition.py [N]
"""
import sys, torch
from torch.profiler import profile, ProfilerActivity
from stream_cqsa.interface import flash_attn_func, flash_attn_func_cqs_group_bits
from stream_cqsa.reference import group_bits_for_path

N = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
H, D, dtype = 8, 64, torch.float16
scale = D ** -0.5
ids_np, bits_np = group_bits_for_path(N, (0,), sorted_gather=True)
L = int(ids_np.shape[0])
bits = torch.as_tensor(bits_np, device="cuda", dtype=torch.int64)
zero_bits = torch.zeros_like(bits)
q, k, v = (torch.randn(1, L, H, D, device="cuda", dtype=dtype) for _ in range(3))
print(f"GPU {torch.cuda.get_device_name(0)}  N={N} -> L={L}  H={H} D={D}\n")

def run(label, fn):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(5): fn()
        torch.cuda.synchronize()
    ev = [e for e in p.key_averages() if e.device_time_total > 0]
    ev.sort(key=lambda e: -e.device_time_total)
    tot = sum(e.device_time_total for e in ev) / 5 / 1000
    print(f"== {label}: {tot:.3f} ms device time per call, {len(ev)} distinct kernels")
    for e in ev[:8]:
        print(f"   {e.device_time_total/5/1000:8.3f} ms  x{e.count//5:<3} {e.key[:90]}")
    print()

run("fa_plain",      lambda: flash_attn_func(q, k, v, softmax_scale=scale))
run("cqs zero-mask", lambda: flash_attn_func_cqs_group_bits(q, k, v, zero_bits, softmax_scale=scale))
run("cqs real-mask", lambda: flash_attn_func_cqs_group_bits(q, k, v, bits, softmax_scale=scale))
