"""
Stream-CQSA the simple way: the one-call API on one GPU, with progress output.

    sbatch slurm/quickstart.slurm          (from the repo root; output in results/quickstart/)

Shows: the dry run (estimate), the monolithic path (bit-identical to SDPA), the
decomposed path under a memory cap (verbose progress), host-resident inputs,
autograd, and patch_sdpa on a small module that only knows F.scaled_dot_product_attention.
"""
import time, torch, torch.nn.functional as F
import stream_cqsa
from stream_cqsa import attention, estimate, patched_sdpa

dev = torch.device("cuda")
H, D = 8, 64
total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"{torch.cuda.get_device_name(0)}, {total_gib:.0f} GiB\n", flush=True)


def timed(fn):
    torch.cuda.synchronize(); t0 = time.perf_counter(); out = fn(); torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


# 1. what would it cost?  (no computation)
for N in (1 << 17, 1 << 20, 1 << 22, 1 << 24):
    estimate(N, B=1, H=H, D=D, causal=True)
    print()

# 2. below the boundary: the call IS the monolithic kernel
N = 1 << 17
q, k, v = (torch.randn(1, H, N, D, device=dev, dtype=torch.float16) for _ in range(3))
out, t = timed(lambda: attention(q, k, v, is_causal=True, verbose=True))
ref, t_ref = timed(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
print(f"N={N}: attention {t*1e3:.1f} ms, SDPA {t_ref*1e3:.1f} ms, identical={torch.equal(out, ref)}\n", flush=True)

# 3. above the boundary (simulated with a memory cap): exact decomposition with progress
N = 1 << 20
CAP = 6.0
torch.cuda.set_per_process_memory_fraction(CAP / total_gib)
hw = {"cuda:0": f"{CAP * 0.8:.1f}GiB", "host": "200GiB"}
q, k, v = (torch.randn(1, H, N, D, device=dev, dtype=torch.float16) for _ in range(3))
out, t = timed(lambda: attention(q, k, v, is_causal=True, verbose=True, hardware=hw))
print(f"N={N} under a {CAP:.0f} GiB cap: {t:.2f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
torch.cuda.set_per_process_memory_fraction(1.0)
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(f"  result on {out.device} (the caller's Q/K/V fill the capped device, so it is handed back in host memory); "
      f"rel. difference to the monolithic fp16 kernel: {rel(out.to(dev), ref):.1e}\n", flush=True)
del q, k, v, out, ref; torch.cuda.empty_cache()

# 4. host-resident inputs, larger N, the same call
N = 1 << 22
qh, kh, vh = (torch.randn(1, H, N, D, dtype=torch.float16) for _ in range(3))     # in host memory
torch.cuda.set_per_process_memory_fraction(16.0 / total_gib); torch.cuda.reset_peak_memory_stats()
out, t = timed(lambda: attention(qh, kh, vh, is_causal=True, verbose=True, hardware={"cuda:0": "12GiB", "host": "200GiB"}))
print(f"N={N} host-resident under a 16 GiB cap: {t:.1f} s, device peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
      f"output on {out.device} {out.dtype}\n", flush=True)
torch.cuda.set_per_process_memory_fraction(1.0)
del qh, kh, vh, out; torch.cuda.empty_cache()

# 5. autograd: the same call inside a graph
N = 1 << 18
q, k, v = (torch.randn(1, H, N, D, device=dev, dtype=torch.float16, requires_grad=True) for _ in range(3))
torch.cuda.set_per_process_memory_fraction(4.0 / total_gib)
out = attention(q, k, v, is_causal=True, verbose=True, hardware={"cuda:0": "3GiB", "host": "200GiB"})
out.float().sum().backward()
torch.cuda.set_per_process_memory_fraction(1.0)
q2, k2, v2 = (t_.detach().clone().requires_grad_(True) for t_ in (q, k, v))
F.scaled_dot_product_attention(q2, k2, v2, is_causal=True).float().sum().backward()
print(f"N={N} autograd under a 4 GiB cap: dq/dk/dv rel. difference to SDPA's gradients "
      f"{rel(q.grad, q2.grad):.1e} / {rel(k.grad, k2.grad):.1e} / {rel(v.grad, v2.grad):.1e}\n", flush=True)
del q, k, v, q2, k2, v2, out; torch.cuda.empty_cache()


# 6. an existing module that only knows F.scaled_dot_product_attention
class Block(torch.nn.Module):
    def forward(self, x):                       # x: [B, N, 3*H*D]
        B_, N_, _ = x.shape
        qkv = x.view(B_, N_, 3, H, D).permute(2, 0, 3, 1, 4)
        return F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)


N = 1 << 20
x = torch.randn(1, N, 3 * H * D, device=dev, dtype=torch.float16)
torch.cuda.set_per_process_memory_fraction(6.0 / total_gib)
with patched_sdpa(min_tokens=65536):
    y, t = timed(lambda: Block()(x))
torch.cuda.set_per_process_memory_fraction(1.0)
print(f"patched F.scaled_dot_product_attention inside a module, N={N} under a 6 GiB cap: {t:.2f} s, out {tuple(y.shape)}", flush=True)
print("\ndone")
