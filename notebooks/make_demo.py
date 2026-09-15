"""
Generate and execute the Stream-CQSA v2 feature demo notebook.

    python next/notebooks/make_demo.py            # writes + executes stream_cqsa_v2_demo.ipynb
    python next/notebooks/make_demo.py --no-exec  # write only

A memory cap (torch.cuda.set_per_process_memory_fraction) simulates a smaller
device so every recovery path runs on one 40 GB A100 in a few minutes.
"""
import argparse, os, sys, time
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stream_cqsa_v2_demo.ipynb")

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Stream-CQSA v2 — every feature in one notebook

Stream-CQSA is **exact out-of-memory recovery for attention**: when a call does not fit,
the pair set is decomposed over a cyclic quorum set into independent subproblems that run
one at a time (or a few at a time, or on several devices) and are recomposed exactly.
It defines no attention rule of its own; it reproduces whatever kernel it wraps.

This notebook exercises every feature of the v2 package on a single GPU. A **memory cap**
(`torch.cuda.set_per_process_memory_fraction`) plays the role of a smaller device so the
recovery paths trigger at sizes that run in seconds.

Sections: 0 install and call · 1 setup · 2 the OOM boundary · 3 the engine's knobs (`itr`, `acc`, host residency,
concurrency, quorum sets) · 4 autograd with independent forward/backward depths ·
5 automatic configuration from a hardware description · 6 the developer kit (exactness +
performance of any inner kernel) · 7 adapters: automatic conversion of FlexAttention kernels ·
8 the classic CUDA kernel vs FlashAttention-2 · 9 the two engines on the two
kernels (classic / wave × CUDA / Triton, and how `attention()` chooses) · 10 multi-device.
""")

md("""
## 0. Install and call

```bash
pip install stream-cqsa            # Triton kernels: any recent NVIDIA GPU, no compiler, no torch/CUDA matching
stream-cqsa-doctor                 # what loaded, and how far this machine can go
```

Optional, for the fastest kernels: the two prebuilt CUDA wheels for your python / torch / CUDA / GPU from the
[release page](https://github.com/yiming-b/Stream-CQSA-v2/releases) (`stream_cqsa` with the classic CUDA
kernel, `stream_cqsa_native` with the wave kernel). The package uses whatever is installed.

Then it is one call with the signature of `torch.nn.functional.scaled_dot_product_attention`: when the
call fits, it *is* the normal kernel; when it does not, it is decomposed exactly, on the fastest
configuration the planner finds for this machine, and recomposed. Nothing to configure.
""")
code("""
import torch, stream_cqsa
print("stream_cqsa", stream_cqsa.__version__, "| kernels:", stream_cqsa.kernels_available())
q, k, v = (torch.randn(1, 8, 262_144, 64, device="cuda", dtype=torch.float16) for _ in range(3))
out = stream_cqsa.attention(q, k, v, is_causal=True)                 # drop-in for F.scaled_dot_product_attention
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
print(tuple(out.shape), out.dtype, "| max |out - SDPA| =", (out.float() - ref.float()).abs().max().item())
""")
code("""
# The same call when the device is too small: describe the budget (or let it be detected), and watch it plan.
out_small, plan = stream_cqsa.attention(q, k, v, is_causal=True, hardware={"cuda:0": "1GiB", "host": "64GiB"}, verbose=True, return_plan=True)
print("plan:", plan.name(), "| max |out - SDPA| =", (out_small.float() - ref.float()).abs().max().item())
# Two more one-liners: route an existing model's attention through it, and ask before running a huge one.
stream_cqsa.patch_sdpa(); stream_cqsa.unpatch_sdpa()                 # F.scaled_dot_product_attention -> stream_cqsa.attention
_ = stream_cqsa.estimate(N=16_777_216, B=1, H=8, D=64, dtype=torch.float16, causal=True)   # prints the plan table
del q, k, v, out, ref, out_small
""")

md("## 1. Setup")
code("""
import os, time, gc, math, json
import torch, torch.nn.functional as F
# Production kernels: the classic CUDA kernel (a causal and a non-causal build, csrc/ and csrc_nc/), the wave
# kernel (native/, cqsa_native) and the Triton kernels; the package finds whichever are installed.
import stream_cqsa
from stream_cqsa.stable_stream import stream_cqsa_forward, stream_cqsa_backward, TraceRecorder
from stream_cqsa.native_autograd import stream_cqsa_attn, StreamCQSAAttention
from stream_cqsa.oom_fallback import attention_oom_safe
from stream_cqsa.autoconfig import detect_hardware, hardware_from_dict, plan, calibrate, autotune, auto_attention, QUORUM_SETS
from stream_cqsa.devkit import compare_kernels, quick_bench, Config, run_config, measure, reference_rows, sample_rows, accuracy_vs_fp64
import stream_cqsa.interface as I
dev = torch.device("cuda")
print(torch.cuda.get_device_name(0), "|", torch.__version__)
print("classic CUDA kernel: causal build", "loaded" if I.cqsa_cuda is not None else "missing", "| non-causal build", "loaded" if I.cqsa_cuda_noncausal is not None else "missing")
B, H, D = 1, 8, 64
def make_qkv(N, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(B, H, N, D, generator=g, dtype=torch.float32).to(torch.float16).to(device) for _ in range(3))
def fp64_error(out, q, k, v, causal=True, rows=128):
    r = sample_rows(q.shape[2], rows); ref = reference_rows(q, k, v, r, causal=causal, scale=D**-0.5)
    return accuracy_vs_fp64(out, q, k, v, causal=causal, scale=D**-0.5, rows=r, ref_rows=ref)["rel_fro"]
def cap(gib):
    \"\"\"Simulate a device with `gib` GiB: the caching allocator refuses to grow past this fraction.\"\"\"
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(min(1.0, gib / total))
    print(f"memory cap: {gib:.1f} GiB of {total:.1f} GiB")
""")

md("""
## 2. The OOM boundary, and the drop-in recovery

`attention_oom_safe` runs the normal SDPA path and falls back to Stream-CQSA only when
it raises out-of-memory. Under a 2 GiB cap a 512K-token call cannot hold Q/K/V + workspace on the
device; the fallback streams them from the host (`release_inputs=True` lets it move device inputs
out), tries each depth with the accumulator on the device and then in host memory, and returns the
exact result. Deep depths carry a Python-side setup cost (the task list is built per subproblem,
~0.25 s each at 512K, so itr=3 = 343 tasks costs a minute before the first kernel runs); the planner
and `auto_attention` avoid that by preferring a larger quorum set at a lower depth.
""")
code("""
N = 524_288
q, k, v = make_qkv(N)                      # host-resident (1.5 GiB fp16)
cap(2.0)                                   # a 2 GiB device: Q/K/V + output + workspace do not fit
held = []
try:
    for t in (q, k, v):
        held.append(t.to(dev))
    out = F.scaled_dot_product_attention(*held, is_causal=True)
    print("SDPA fit?!")
except torch.cuda.OutOfMemoryError as e:
    print("SDPA under the cap: OutOfMemoryError ->", str(e)[:60], "...")
held.clear(); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
out = attention_oom_safe(q, k, v, causal=True, release_inputs=True)   # same signature and dtype as SDPA
torch.cuda.synchronize()
print(f"attention_oom_safe: {time.perf_counter()-t0:.1f} s, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, "
      f"out {tuple(out.shape)} {out.dtype}, rel.err vs float64 {fp64_error(out, q, k, v):.1e}")
del out
""")

md("""
## 3. The engine's knobs

`stream_cqsa_forward(q, k, v, itr=, c=, interest_set=, low_memory=, stream_from_host=, max_parallel=, shared_chunks=)`
returns the fp32 output and an `info` dict (depth used, subproblem count, per-stage timings when traced).

* `itr` — decomposition depth: `c**itr` subproblems of `N·(l/c)**itr` tokens (`"auto"` = plan from free memory; 0 = monolithic).
* `(c, interest_set)` — the cyclic quorum set. Larger `c` at a lower depth gives the same subproblem size
  as a smaller `c` at a higher depth with less pair work; every perfect difference set in `QUORUM_SETS` is valid.
* `low_memory=True` (acc=CPU) — the fp32 accumulator lives in host memory; `stream_from_host=True` — Q/K/V too.
* `max_parallel` — subproblems in flight; `shared_chunks=True` — contiguous chunk DMA instead of a row gather (itr=1).
""")
code("""
cap(40.0)
N = 262_144
q, k, v = make_qkv(N)
rows = sample_rows(N, 128); ref = reference_rows(q, k, v, rows, causal=True, scale=D**-0.5)
print(f"{'configuration':>58} {'time s':>7} {'peak GiB':>9} {'subproblems':>11} {'rel.err':>8}")
for label, kw in [
    ("itr=1 acc=GPU, inputs on device",                dict(itr=1)),
    ("itr=1 acc=GPU n_par=4",                           dict(itr=1, max_parallel=4)),
    ("itr=2 acc=GPU",                                   dict(itr=2, max_parallel=2)),
    ("c=13 itr=1 acc=GPU (13 subproblems of 4N/13)",    dict(itr=1, c=13, interest_set=QUORUM_SETS[13], max_parallel=2)),
    ("c=31 itr=1 acc=GPU (31 subproblems of 6N/31)",    dict(itr=1, c=31, interest_set=QUORUM_SETS[31], max_parallel=2)),
    ("itr=1 acc=CPU, Q/K/V on host, shared_chunks",     dict(itr=1, low_memory=True, stream_from_host=True, shared_chunks=True)),
    ("itr='auto' (planner: monolithic fits here)",      dict(itr="auto")),
]:
    on_host = kw.get("stream_from_host", False)
    qq, kk, vv = (q, k, v) if on_host else (q.to(dev), k.to(dev), v.to(dev))
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter(); out, info = stream_cqsa_forward(qq, kk, vv, causal=True, allow_escalation=False, **kw); torch.cuda.synchronize()
    err = accuracy_vs_fp64(out, q, k, v, causal=True, scale=D**-0.5, rows=rows, ref_rows=ref)["rel_fro"]
    print(f"{label:>58} {time.perf_counter()-t0:7.3f} {torch.cuda.max_memory_allocated()/2**30:9.2f} {info.get('n_subproblems', 0):11d} {err:8.1e}")
    del out, qq, kk, vv
""")
code("""
# Per-stage timings: trace one call (host accumulator, so all stages appear)
tr = TraceRecorder(enabled=True, device=dev)
out, info = stream_cqsa_forward(q, k, v, itr=1, causal=True, low_memory=True, stream_from_host=True, shared_chunks=True, trace=tr)
print({s: f"{ms:.0f} ms" for s, ms in info["stage_totals_ms"].items()}, "| itr", info["itr"], "| subproblems", info["n_subproblems"])
del out
""")

md("""
## 4. Autograd, with independent forward and backward depths

`stream_cqsa_attn` is a `torch.autograd.Function`. The backward decomposes on its own:
`bwd_itr="auto"` (default) plans the backward's depth from free memory with the backward's own
memory model, `"fwd"` reuses the forward's, an int pins it. Under a cap the two differ.
""")
code("""
cap(6.0)
N = 262_144
q, k, v = (t.to(dev).requires_grad_(True) for t in make_qkv(N))
for fwd_itr, bwd_itr in [(1, "auto"), (1, "fwd"), (2, 1)]:
    q.grad = k.grad = v.grad = None
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = stream_cqsa_attn(q, k, v, causal=True, itr=fwd_itr, bwd_itr=bwd_itr)
    out.float().sum().backward(); torch.cuda.synchronize()
    print(f"fwd itr={fwd_itr} bwd_itr={bwd_itr!r:6}: {time.perf_counter()-t0:6.2f} s, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, "
          f"|dq| {q.grad.float().norm():.3f} |dk| {k.grad.float().norm():.3f} |dv| {v.grad.float().norm():.3f}")
# explicit backward with its own planning
o, info = stream_cqsa_forward(q.detach(), k.detach(), v.detach(), itr=1, causal=True)
binfo = {}
dq, dk, dv = stream_cqsa_backward(q.detach(), k.detach(), v.detach(), torch.ones_like(q).detach(), o.to(q.dtype), info["lse"], causal=True, bwd_info=binfo)
print("explicit backward itr='auto':", binfo.get("plan_reason"))
del out, o, dq, dk, dv, info, binfo; q.grad = k.grad = v.grad = None   # drop device tensors (lse) before the next cap: a
gc.collect(); torch.cuda.empty_cache()                                    # live tensor pins its whole cached segment
cap(40.0)
""")

md("""
## 5. Automatic configuration from a hardware description

`plan()` enumerates monolithic / every `(c, itr)` / accumulator placement / host residency / concurrency /
device count, predicts memory with the engine's estimators and time with a calibratable cost model,
keeps what fits the budget and applies one rule: *use the budget unless a configuration is both faster
and smaller* (the Pareto frontier of time and memory, fastest point). Describe the hardware as a dict
of budgets, or detect it.
""")
code("""
hw = detect_hardware(); print("detected:", hw.summary())
for spec in [{"cuda:0": "40GiB", "host": "256GiB"},
             {"cuda:0": "3GiB", "host": "256GiB"},
             {"cuda:0": "80GiB", "cuda:1": "80GiB", "cuda:2": "80GiB", "cuda:3": "80GiB", "host": "500GiB", "link_gbs": 200}]:
    h = hardware_from_dict(spec)
    for N_, d in [(262_144, "fwd"), (1_048_576, "fwd"), (1_048_576, "bwd"), (16_777_216, "fwd")]:
        p = plan(N=N_, hardware=h, direction=d)
        print(f"{str(spec)[:52]:52s} N={N_:>9} {d}: {p.name():55s} est {p.est_time_s:8.1f} s, {p.est_peak_gib:5.1f} GiB/device")
""")
code("""
# Calibrate the cost model on this GPU (~1 min), then let auto_attention plan and run under a cap.
cm = calibrate(hw, N=65536)
# The planner plans for the 2 GiB device described below. The process cap is 3 GiB because a 1 GiB
# allocator segment cached earlier in this notebook stays pinned by PyTorch's cuBLAS workspace (8 MiB);
# a real 2 GiB device would never have created that segment.
cap(3.0)
q, k, v = make_qkv(524_288)
out, p = auto_attention(q, k, v, causal=True, hardware=hardware_from_dict({"cuda:0": "2GiB", "host": "256GiB"}), model=cm, verbose=True, allow_escalation=False)
torch.cuda.synchronize()
print(f"-> ran {p.name()}: peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, rel.err {fp64_error(out, q, k, v):.1e}")
del out; cap(40.0)
""")
code("""
# autotune: measure the top candidates instead of trusting the model (use when the call will be repeated)
p = autotune(N=131_072, hardware=hardware_from_dict({"cuda:0": "40GiB", "host": "256GiB"}), max_candidates=4, verbose=True)
print("autotune ->", p.name())
""")

md("""
## 6. Developer kit: any inner kernel vs its monolithic call

`compare_kernels(inner_fn, mono_fn)` plugs a kernel into the framework and reports whether the decomposed
result matches the monolithic one (bit-identical / exact within rounding / NOT exact), the error of both
against float64, and time + peak memory of both. `quick_bench` sweeps configurations on one input set.
""")
code("""
from stream_cqsa.stable_stream import local_stats_flash, local_stats_torch
rep = compare_kernels(local_stats_flash, N=131072, itr=1)          # the native kernel vs flash-attn
rep = compare_kernels(local_stats_torch, N=16384, itr=2)           # the dense fp32 torch fallback (differential test)
""")
code("""
res = quick_bench(N=131072, budget_gib=8.0, configs=[Config(mode="mono"), Config(mode="mono", mono_backend="sdpa"),
      Config(itr=1, acc="gpu", n_par=2), Config(itr=1, c=13, interest_set=QUORUM_SETS[13], acc="gpu", n_par=2),
      Config(itr=1, acc="cpu", stream_from_host=True), Config(itr=2, acc="gpu", n_par=4)], reps=2, acc_rows=128)
""")

md("""
## 7. Adapters: automatic conversion of a monolithic kernel

Any attention that torch FlexAttention can express becomes a Stream-CQSA inner kernel via `flex_inner`:
the CQS pair set becomes a block mask, `return_lse` supplies the row statistics, and — the one thing a
conversion must get right — the engine hands the kernel the gather index so position-dependent
`score_mod`s see **global** positions. ALiBi and a sliding window come out exact; the local-index
mistake is caught.
""")
code("""
from stream_cqsa.adapters import flex_inner, dense_inner, sdpa_masked
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
fa = torch.compile(flex_attention, dynamic=False)
def alibi(score, b, h, q_idx, kv_idx): return score - 0.05 * (q_idx - kv_idx).abs()
mono_alibi = lambda q, k, v: fa(q, k, v, score_mod=alibi, scale=D**-0.5)
print("ALiBi, positions remapped (correct):");      compare_kernels(flex_inner(score_mod=alibi), mono_fn=mono_alibi, N=16384, itr=1, causal=False, reference=None)
print("\\nALiBi, local positions (the bug):");        compare_kernels(flex_inner(score_mod=alibi, global_positions=False), mono_fn=mono_alibi, N=16384, itr=1, causal=False, reference=None)
def window(b, h, q_idx, kv_idx): return (q_idx - kv_idx) <= 2048
def mono_win(q, k, v):
    bm = create_block_mask(lambda b, h, qi, ki: (ki <= qi) & window(b, h, qi, ki), None, None, q.shape[2], q.shape[2], device=q.device)
    return fa(q, k, v, block_mask=bm, scale=D**-0.5)
print("\\nsliding window 2048, causal:");             compare_kernels(flex_inner(extra_mask_mod=window), mono_fn=mono_win, N=16384, itr=1, causal=True, reference=None)
print("\\nSDPA with a dense mask and no lse (second lse pass):"); compare_kernels(dense_inner(sdpa_masked, returns_lse=False), N=8192, itr=1)
""")

md("""
## 8. The classic CUDA kernel vs FlashAttention-2

The classic CUDA kernel is FlashAttention-2 with a per-tile CQS verdict that costs a register AND, a
straight-line steady loop over live tiles only, and compile-time CQS on/off. With CQS off it is as fast as
FlashAttention-2; with CQS on it skips the fully masked tiles of a subproblem. Causal and non-causal calls
use two builds of it (`docs/kernel_technical_note.md`). The wave kernel of section 9 runs several such
subproblems per launch.
""")
code("""
from stream_cqsa.interface import flash_attn_func_cqs_group_bits, flash_attn_func, cqs_block_summaries
from stream_cqsa.reference import group_bits_for_path
from flash_attn import flash_attn_func as fa2
def ms(fn, it=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record(); [fn() for _ in range(it)]; e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / it
ids, bits_np = group_bits_for_path(131072, (0,), sorted_gather=True); L = len(bits_np)
qq, kk, vv = (torch.randn(1, L, H, D, device=dev, dtype=torch.float16) for _ in range(3))
bits = torch.as_tensor(bits_np, device=dev); bo, ba = (t.cuda() for t in cqs_block_summaries(bits))
print(f"one itr=1 subproblem, L={L} (3N/7 of N=131072), causal, ms per call:")
print(f"  FlashAttention-2 monolithic on L      : {ms(lambda: fa2(qq, kk, vv, causal=True)):.2f}")
print(f"  classic CUDA kernel, CQS off (plain)  : {ms(lambda: flash_attn_func(qq, kk, vv, causal=True)):.2f}")
print(f"  classic CUDA kernel, CQS on (subproblem): {ms(lambda: flash_attn_func_cqs_group_bits(qq, kk, vv, bits, causal=True, cqs_blk_or=bo, cqs_blk_and=ba)):.2f}  (22% of tiles are skipped as fully masked)")
del qq, kk, vv
""")

md("""
## 9. Two engines, two kernels: classic / wave × CUDA / Triton

Since v2.2 every forward can run on either **engine** and either **kernel**:

* the **classic engine** runs one subproblem per launch (two in flight); the **wave engine** packs several
  subproblems into one batched launch (per-subproblem tables, one deterministic merge per wave), so its
  cost is flat in the subproblem count;
* the **CUDA kernel** (the compiled extensions; the wave kernel is `cqsa_native`) or the **Triton kernel**
  (no build; 11-16% behind CUDA at head dim 64).

`attention(..., kernel=)` picks: `"cuda"` / `"triton"` = classic engine, `"wave-cuda"` / `"wave-triton"` = wave
engine, `"auto"` = the classic engine unless the decomposition has >= 20 subproblems (c >= 21 at itr=1, or
itr=2), where the wave engine measured faster; host-accumulator calls (`accumulate_on_gpu=False`, the paper's
acc=CPU) stay on the classic engine under `auto` and are available on both engines explicitly. The full
comparison is `results/profile_sweep/` in the README.
""")
code("""
from stream_cqsa import attention, kernels_available
from stream_cqsa.api import _pick_wave
from stream_cqsa.native_wave import wave_forward
print("kernels in this environment:", kernels_available())
cap(40.0)
N = 262_144
q, k, v = make_qkv(N); rows = sample_rows(N, 128); ref = reference_rows(q, k, v, rows, causal=True, scale=D**-0.5)
qd, kd, vd = (t.to(dev) for t in (q, k, v))
print(f"{'engine / kernel / configuration':>52} {'time s':>7} {'peak GiB':>9} {'rel.err':>8}")
for label, kernel, kw in [
    ("classic, CUDA, c=7 itr=1",                    "cuda",        dict(itr=1)),
    ("classic, Triton, c=7 itr=1",                  "triton",      dict(itr=1)),
    ("wave, CUDA, c=7 itr=1",                       "wave-cuda",   dict(itr=1)),
    ("wave, Triton, c=7 itr=1",                     "wave-triton", dict(itr=1)),
    ("classic, CUDA, c=31 itr=1 (31 subproblems)",  "cuda",        dict(itr=1, c=31, interest_set=QUORUM_SETS[31])),
    ("wave, CUDA, c=31 itr=1 (31 subproblems)",     "wave-cuda",   dict(itr=1, c=31, interest_set=QUORUM_SETS[31])),
    ("wave, Triton, c=31 itr=1",                    "wave-triton", dict(itr=1, c=31, interest_set=QUORUM_SETS[31])),
    ("classic, CUDA, c=7 itr=1, acc=CPU",           "cuda",        dict(itr=1, low_memory=True, accumulate_on_gpu=False)),
    ("wave, CUDA, c=7 itr=1, acc=CPU",              "wave-cuda",   dict(itr=1, accumulate_on_gpu=False)),
    ("wave, Triton, c=7 itr=1, acc=CPU",            "wave-triton", dict(itr=1, accumulate_on_gpu=False)),
    ("auto (classic here: 7 subproblems)",          "auto",        dict(itr=1)),
    ("auto (wave here: 49 subproblems)",            "auto",        dict(itr=2)),
]:
    attention(qd, kd, vd, is_causal=True, kernel=kernel, **kw)                     # warm-up (Triton compiles on first use)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter(); out = attention(qd, kd, vd, is_causal=True, kernel=kernel, **kw); torch.cuda.synchronize()
    err = accuracy_vs_fp64(out.to(dev), q, k, v, causal=True, scale=D**-0.5, rows=rows, ref_rows=ref)["rel_fro"]
    print(f"{label:>52} {time.perf_counter()-t0:7.3f} {torch.cuda.max_memory_allocated()/2**30:9.2f} {err:8.1e}")
    del out
""")
code("""
# What `kernel="auto"` decides, by subproblem count (the wave engine from 20 subproblems; acc=CPU stays classic)
for c_, itr_, acc_ in [(7, 1, True), (13, 1, True), (21, 1, True), (7, 2, True), (133, 1, True), (133, 1, False)]:
    kw = dict(c=c_, itr=itr_) | ({} if acc_ else dict(accumulate_on_gpu=False))
    print(f"c={c_:3d} itr={itr_} acc={'GPU' if acc_ else 'CPU'}: {c_**itr_:4d} subproblems ->", "wave engine" if _pick_wave("auto", kw, True, qd, "fwd") else "classic engine")
""")
code("""
# The wave engine directly: waves are planned under a packed-token budget (default min(2M, max(512K, N)));
# `kernel=` selects the compute kernel, `accumulate_on_gpu=False` merges each wave into host accumulators.
for kern in ("cuda", "triton"):
    for budget in (None, 128 * 1024):
        out, info = wave_forward(qd, kd, vd, causal=True, itr=1, c=13, interest_set=QUORUM_SETS[13], kernel=kern, max_wave_tokens=budget)
        print(f"wave_forward kernel={info['kernel']:6s} budget={info['max_wave_tokens']:>7d} tokens -> {info['n_waves']} wave(s) of {info['wave_sizes']} subproblems, "
              f"rel.err {accuracy_vs_fp64(out, q, k, v, causal=True, scale=D**-0.5, rows=rows, ref_rows=ref)['rel_fro']:.1e}")
        del out
# The classic engine on the Triton kernel, without the API: CQSA_FORWARD=triton is read per call
os.environ["CQSA_FORWARD"] = "triton"
out, info = stream_cqsa_forward(qd, kd, vd, itr=1, causal=True, allow_escalation=False)
print(f"classic engine on Triton: {info['n_subproblems']} subproblems, rel.err {accuracy_vs_fp64(out, q, k, v, causal=True, scale=D**-0.5, rows=rows, ref_rows=ref)['rel_fro']:.1e}")
os.environ.pop("CQSA_FORWARD"); del out, qd, kd, vd
""")

md("""
## 10. Multi-device

`stream_cqsa.distributed.dist_stream_cqsa_forward / _backward` shard the `c**itr` subproblems round-robin
over the ranks of a `torch.distributed` group (NCCL), run the unmodified engine per rank and recompose with
the engine's own max-shifted merge across ranks (all_reduce), so the result is exact. Launch with
`python -m torch.distributed.run --nproc_per_node=4 script.py`. Measured on A100-80GB nodes (exact at every point):

| GPUs | N | itr | forward speedup vs 1 GPU |
|---|---|---|---|
| 2 | 2M | 1 / 2 | 1.58x / 1.76x |
| 4 | 2M | 1 / 2 | 2.99x / 3.28x |
| 4 | 4M | 1 / 2 | 3.23x / 3.54x |
| 8 (2 nodes) | 4M | 2 | 5.10x |
| 2 | 1M | 1 | backward 1.47x |

The planner includes the device count as an axis: for 4x80 GiB it chooses the distributed configuration from
~1M tokens up, and never for 2 devices (the 4/7 shard bound does not beat one monolithic call).
""")
code("""
import inspect, stream_cqsa.distributed as dd
print(inspect.getsource(dd.dist_stream_cqsa_forward).split(chr(34) * 3)[1].strip()[:900])
""")
md("""
---
*Results, logs and the kernel technical note: `results/`, `docs/`. Everything shown here is
checked against float64 on sampled rows or bit-for-bit against the monolithic kernel.*
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
ap = argparse.ArgumentParser(); ap.add_argument("--no-exec", action="store_true"); ap.add_argument("--out", default=OUT)
a = ap.parse_args()
if not a.no_exec:
    from nbclient import NotebookClient
    t0 = time.time()
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": HERE}}).execute()
    print(f"executed in {time.time() - t0:.0f} s")
nbf.write(nb, a.out); print("wrote", a.out)
