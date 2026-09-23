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
# Stream-CQSA v2 — one function, and what is under it

Stream-CQSA is **exact out-of-memory recovery for attention**: when a call does not fit,
the pair set is decomposed over a cyclic quorum set into independent subproblems that run
one at a time (or a few at a time, or on several devices) and are recomposed exactly.
It defines no attention rule of its own; it reproduces whatever kernel it wraps.

**Part A is all a user needs: one function, `stream_cqsa.attention`, with the signature of
`torch.nn.functional.scaled_dot_product_attention`.** It runs the normal kernel when the call fits and the exact
decomposition when it does not, plans the configuration, recovers from out-of-memory, supports autograd, and takes
host or device inputs. Part B shows the engine layer underneath, for kernel and systems developers.

A **memory cap** (`torch.cuda.set_per_process_memory_fraction`) plays the role of a smaller device so the
recovery paths trigger at sizes that run in seconds.

Part A: 0 install and call · 1 setup · 2 when it does not fit · 3 optional arguments · 4 training ·
5 estimate and calibrate · 6 engines and kernels.
Part B: the engine and its trace · 7 developer kit · 8 adapters · 9 the kernel vs FlashAttention-2 · 10 multi-device.
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

On a shared or small GPU, creating the inputs is the one step outside the package's control: create
them in host memory (or hand them to `stream_cqsa.place_inputs`, which moves what fits and leaves the
rest pinned in host memory), and `attention` streams whatever is on the host. `verbose=True` prints one
line per call stating every automatic choice: monolithic or decomposed (itr, c, subproblems), where
Q/K/V are, where the accumulator is, the engine and kernel, and where the output goes.
""")
code("""
import torch, stream_cqsa
print("stream_cqsa", stream_cqsa.__version__, "| kernels:", stream_cqsa.kernels_available())
q, k, v = (torch.randn(1, 8, 262_144, 64, dtype=torch.float16) for _ in range(3))   # created in host memory: safe on any GPU
q, k, v = stream_cqsa.place_inputs(q, k, v, verbose=True)                         # onto the GPU if they fit, else stay on the host
out = stream_cqsa.attention(q, k, v, is_causal=True, verbose=True)                 # drop-in for F.scaled_dot_product_attention
ref = torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True)
print(tuple(out.shape), out.dtype, "| max |out - SDPA| =", (out.float().cuda() - ref.float()).abs().max().item())
""")
code("""
# The same call when the device is too small: describe the budget (or let it be detected), and watch it plan.
out_small, plan = stream_cqsa.attention(q, k, v, is_causal=True, hardware={"cuda:0": "1GiB", "host": "64GiB"}, verbose=True, return_plan=True)
print("plan:", plan.name(), "| max |out - SDPA| =", (out_small.float().cuda() - ref.float()).abs().max().item())
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
from stream_cqsa.autoconfig import QUORUM_SETS                                       # the registered quorum sets (c = 3 ... 133)
# engine layer, used in Part B only:
from stream_cqsa.stable_stream import stream_cqsa_forward, TraceRecorder
from stream_cqsa.autoconfig import hardware_from_dict, autotune
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
## 2. When it does not fit

Nothing changes in the call. Here the device is described as 2 GiB (`hardware=`; without it the free memory is
detected): SDPA cannot hold a 512K-token call there, `attention` plans a decomposition, streams the host inputs
and returns the exact result. (The process cap is 3 GiB because allocator segments cached earlier in this
notebook stay pinned by PyTorch's cuBLAS workspace; the plan is for the 2 GiB description.)
""")
code("""
SMALL = {"cuda:0": "2GiB", "host": "128GiB"}
N = 524_288
q, k, v = make_qkv(N)                      # host-resident (1.5 GiB fp16)
cap(2.0)
held = []
try:
    for t in (q, k, v):
        held.append(t.to(dev))
    out = F.scaled_dot_product_attention(*held, is_causal=True)
    print("SDPA fit?!")
except torch.cuda.OutOfMemoryError as e:
    print("SDPA on a 2 GiB device: OutOfMemoryError ->", str(e)[:60], "...")
held.clear(); cap(3.0); torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
out, info = stream_cqsa.attention(q, k, v, is_causal=True, hardware=SMALL, return_info=True)
torch.cuda.synchronize()
print(f"stream_cqsa.attention: {time.perf_counter()-t0:.1f} s, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, "
      f"out {tuple(out.shape)} {out.dtype} on {out.device}, rel.err vs float64 {fp64_error(out, q, k, v):.1e}")
print("what ran:", {k_: info.get(k_) for k_ in ("engine", "kernel", "itr", "c", "n_subproblems", "oom_retries")})
del out
""")

md("""
### Inside a model: Q/K/V are already on the GPU

In an end-to-end LLM the attention inputs are activations that already sit on the device, and they may leave
too little room for the attention itself. The call is still the same. When nothing fits next to Q/K/V,
`attention` releases them to pinned host memory (the tensors you hold are rebound, so the device memory is
really freed), streams from there, puts the **output on the GPU first**, and then brings Q/K/V back if room
remains; otherwise they stay valid host tensors. `evict_inputs=` / `restore_inputs=` (`"auto"`, `True`,
`False`) force or forbid either step. Here the device budget is described as 1.2 GiB while Q/K/V alone are
0.75 GiB.
""")
code("""
cap(40.0)                                                     # lift the previous section's cap: the restore step needs real room
N = 262_144
q, k, v = (t.to(dev) for t in make_qkv(N))                    # device-resident, as inside a model (0.75 GiB)
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
out, info = stream_cqsa.attention(q, k, v, is_causal=True, hardware={"cuda:0": "1.2GiB", "host": "64GiB"},
                                  return_info=True, verbose=True)
print()
print("evicted:", info["inputs_evicted"], "| released on the GPU:", info["evict_released_gib"], "GiB",
      "| output on", out.device, "| Q/K/V restored:", info["inputs_restored"], "-> q on", q.device)
print(f"max |out - SDPA| relative: {((out.float()-ref.float()).norm()/ref.float().norm()).item():.1e}")
del out, ref, q, k, v
""")

md("""
## 3. Optional arguments of the same function

Everything is a keyword of `attention`; none is needed. Pinning a setting turns that part of the planner off.

| argument | meaning |
|---|---|
| `itr=` | decomposition depth: `c**itr` subproblems of `N*(l/c)**itr` tokens (default: planned; the monolithic call when it fits) |
| `c=, interest_set=` | the cyclic quorum set, any entry of `stream_cqsa.autoconfig.QUORUM_SETS` (c = 3 ... 133) |
| `accumulate_on_gpu=False` (with `low_memory=True`) | fp32 accumulator in host memory (the paper's acc=CPU) |
| host tensors as inputs | Q/K/V streamed from host memory; device tensors stay on the device when a plan fits |
| `max_parallel=` | subproblems in flight |
| `kernel=` | `"auto"`, `"cuda"`, `"triton"`, `"wave-cuda"`, `"wave-triton"` (section 6) |
| `bwd_itr=` | depth of the backward: `"auto"` (planned on its own), `"fwd"`, or an int |
| `evict_inputs=`, `restore_inputs=` | device-resident Q/K/V: release them to host memory for the computation / bring them back after the output (`"auto"`, `True`, `False`) |
| `hardware=` | a budget description `{"cuda:0": "10GiB", "host": "64GiB"}` instead of detection |
| `verbose=True`, `return_plan=True`, `return_info=True`, `plan_only=True` | see what it decides and what ran |
""")
code("""
cap(40.0)
N = 262_144
q, k, v = make_qkv(N)
rows = sample_rows(N, 128); ref = reference_rows(q, k, v, rows, causal=True, scale=D**-0.5)
print(f"{'attention(q, k, v, is_causal=True, ...)':>58} {'time s':>7} {'peak GiB':>9} {'subproblems':>11} {'rel.err':>8}")
for label, host, kw in [
    ("(nothing: planned; monolithic fits here)",         False, dict()),
    ("itr=1",                                            False, dict(itr=1)),
    ("itr=1, max_parallel=4",                            False, dict(itr=1, max_parallel=4)),
    ("itr=2",                                            False, dict(itr=2)),
    ("c=13, interest_set=QUORUM_SETS[13], itr=1",        False, dict(itr=1, c=13, interest_set=QUORUM_SETS[13])),
    ("c=31, interest_set=QUORUM_SETS[31], itr=1",        False, dict(itr=1, c=31, interest_set=QUORUM_SETS[31])),
    ("itr=1, accumulate_on_gpu=False, host inputs",      True,  dict(itr=1, low_memory=True, accumulate_on_gpu=False)),
]:
    qq, kk, vv = (q, k, v) if host else (q.to(dev), k.to(dev), v.to(dev))
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter(); out, info = stream_cqsa.attention(qq, kk, vv, is_causal=True, return_info=True, **kw); torch.cuda.synchronize()
    err = accuracy_vs_fp64(out.to(dev), q, k, v, causal=True, scale=D**-0.5, rows=rows, ref_rows=ref)["rel_fro"]
    print(f"{label:>58} {time.perf_counter()-t0:7.3f} {torch.cuda.max_memory_allocated()/2**30:9.2f} {info.get('n_subproblems', 1):11d} {err:8.1e}")
    del out, qq, kk, vv
""")

md("""
## 4. Training: gradients flow through the same call

When the inputs require gradients, `attention` is differentiable. The backward decomposes on its own
(`bwd_itr="auto"`, planned from free memory with the backward's memory model), independently of the forward.
""")
code("""
cap(6.0)
N = 262_144
q, k, v = (t.to(dev).requires_grad_(True) for t in make_qkv(N))
for fwd_itr, bwd_itr in [(1, "auto"), (1, "fwd"), (2, 1)]:
    q.grad = k.grad = v.grad = None
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = stream_cqsa.attention(q, k, v, is_causal=True, itr=fwd_itr, bwd_itr=bwd_itr)
    out.float().sum().backward(); torch.cuda.synchronize()
    print(f"attention(..., itr={fwd_itr}, bwd_itr={bwd_itr!r:6}): {time.perf_counter()-t0:6.2f} s, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, "
          f"|dq| {q.grad.float().norm():.3f} |dk| {k.grad.float().norm():.3f} |dv| {v.grad.float().norm():.3f}")
del out; q.grad = k.grad = v.grad = None; del q, k, v
gc.collect(); torch.cuda.empty_cache()
cap(40.0)
""")

md("""
## 5. Asking before running, and fitting the cost model to this GPU

`estimate` is the dry run: does the monolithic call fit, what would be chosen, how much device and host memory,
how long. `calibrate` times a few kernels once per GPU model (about a minute) and saves the fitted cost model
to `~/.cache/stream_cqsa`, which every later `attention` / `estimate` call on that GPU uses.
""")
code("""
_ = stream_cqsa.estimate(N=1_048_576, hardware={"cuda:0": "3GiB", "host": "256GiB"})    # a small device
_ = stream_cqsa.calibrate(N=65536)                                                       # fit the cost model to this GPU
_ = stream_cqsa.estimate(N=16_777_216)                                                   # this machine, now with the fitted model
""")

md("""
## 6. Two engines, two kernels, one argument

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
---
# Part B. Under the hood (for kernel and systems developers)

Nothing below is needed to use the package. These are the engine-level interfaces `attention` is built on:
the streaming engine and its per-stage trace, the measuring planner, the developer kit for plugging in
another inner kernel, the adapters, the kernel itself against FlashAttention-2, and multi-device execution.
""")
code("""
# The engine under attention(): per-stage timings of one host-accumulator call, and the measuring planner
q, k, v = make_qkv(262_144)
tr = TraceRecorder(enabled=True, device=dev)
out, info = stream_cqsa_forward(q, k, v, itr=1, causal=True, low_memory=True, stream_from_host=True, shared_chunks=True, trace=tr)
print({s_: f"{ms:.0f} ms" for s_, ms in info["stage_totals_ms"].items()}, "| itr", info["itr"], "| subproblems", info["n_subproblems"])
del out
p = autotune(N=131_072, hardware=hardware_from_dict({"cuda:0": "40GiB", "host": "256GiB"}), max_candidates=4, verbose=True)
print("autotune ->", p.name())
""")

md("""
## 7. Developer kit: any inner kernel vs its monolithic call

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
## 8. Adapters: automatic conversion of a monolithic kernel

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
## 9. The classic CUDA kernel vs FlashAttention-2

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
