# Stream-CQSA v2

Exact out-of-memory recovery for attention. When an attention call does not fit
in device memory, Stream-CQSA decomposes its pair set over a cyclic quorum set
into independent subproblems, runs them one at a time (or a few at a time, or
on several devices), and recomposes the exact result. It defines no attention
rule of its own: it reproduces whatever kernel it wraps.

Paper (v1): [arXiv:2604.20819](https://arxiv.org/abs/2604.20819) ·
v1 repo: [yiming-b/Stream-CQSA](https://github.com/yiming-b/Stream-CQSA)

## How it works

![Stream-CQSA: data movement in the forward and backward](docs/stream_cqsa_demo.gif)

Q/K/V live in host memory in 7 chunks. Each of the 7 subproblems gathers 3 chunks
(its owner chunk and two quorum partners) to the device, runs the CQS kernel on
them, and hands back a partial result: `(out_i, lse_i)` in the forward, which are
merged into a host accumulator with the same max-shifted arithmetic FlashAttention
uses inside one kernel; `(dq_i, dk_i, dv_i)` in the backward, computed against
the *global* log-sum-exp and scatter-added into host gradient buffers. Only one
subproblem's inputs and partial result are on the device at a time, and every
kept query–key pair is counted exactly once, so the result is exact.

## What is new in v2

| area | v2 |
|---|---|
| **native kernel** | second-generation forward kernel: 19% faster than v1 on the real workload (L=131K causal, A100), and with CQS off it now equals FlashAttention-2 (`docs/kernel_technical_note.md`) |
| **engine** | pipelined host accumulator (d2h + merge overlap the next kernel), contiguous-chunk DMA (`shared_chunks`), measured concurrency policy; 1.44–1.86x faster end to end than v1 on one A100 |
| **multi-device** | `stream_cqsa.distributed`: exact sharding of the subproblems over a `torch.distributed` group; 3.2–3.5x on 4 GPUs, 5.1x on 8 GPUs / 2 nodes |
| **automatic configuration** | `stream_cqsa.autoconfig`: monolithic vs decomposed, depth `itr`, quorum set `(c, interest_set)`, accumulator placement, host residency, concurrency and device count chosen from a hardware description and a calibratable cost model |
| **independent fwd/bwd** | the backward plans its own decomposition depth (`bwd_itr="auto"`) |
| **developer kit** | `stream_cqsa.devkit`: plug any inner kernel into the framework and get exactness (bit-identical / within rounding / not) and performance against its monolithic call; `quick_bench` sweeps |
| **Triton kernels (no build)** | `stream_cqsa.triton_kernel`: forward AND backward CQS kernels in Triton with the CUDA kernels' contract, selected automatically when no extension is compiled (`CQSA_BACKWARD=triton` forces the backward). Forward: 0.75–0.91x the CUDA kernel at L=56K (faster), within 13% at 899K; non-causal it beats FlashAttention-2 itself on the same L. Backward: 0.77x the CUDA CQS backward at L=56K, engine backward at 1M 29 s vs 39 s. Engine forward at 1M: 1.15x the CUDA path (`docs/LOG.md`, Phase 4) |
| **native wave kernel** (`native/`, `stream_cqsa.native_wave`) | a CQS kernel that runs *several subproblems per launch*: the wave is one batched launch with per-subproblem group bits, tile summaries and a block map that reads the original Q/K/V in place (no gather); one deterministic merge kernel recomposes the wave. 9-30% faster than the stream-based engine at N=131K (more subproblems, more gain), bit-identical to the v11 kernel per subproblem; forward + backward (`docs/LOG.md`, Phase 6) |
| **adapters** | `stream_cqsa.adapters`: automatic conversion of FlexAttention-expressible kernels (ALiBi, windows, soft-cap, document masks) into inner kernels, with global-position remapping |

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126     # match your CUDA
CQSA_SKIP_EXT=1 pip install --no-build-isolation git+ssh://git@github.com/yiming-b/Stream-CQSA-v2.git   # no compilation
python -m stream_cqsa.doctor                                            # what this machine can run, and how far
```

The package runs with **no compilation**: the CQS forward and backward kernels are also
implemented in Triton, and the engine uses them whenever no extension is present
(`pip install triton` if your torch did not bring it). `pip install flash-attn` is optional
and gives the monolithic fast path its own kernel.

While this repository is private the `git+ssh` line above (any account with access) is the
install; once it is public, `git+https://...` works as well, and the release page can carry
wheels (`.github/workflows/wheels.yml` builds a pure-Python wheel and CUDA-extension wheels
for common python/torch/CUDA pairs on every `v*` tag):
`pip install https://github.com/yiming-b/Stream-CQSA-v2/releases/download/v2.1.0/stream_cqsa-2.1.0-py3-none-any.whl`.
Publishing to PyPI (`pip install stream-cqsa`) is a separate step: the `pypi` job in the
workflow uploads the pure wheel and the sdist when the repository variable `PYPI_PUBLISH`
is `true` and the project is registered on PyPI with trusted publishing for this workflow.

With the CUDA extension, from a checkout: `pip install -e . --no-build-isolation`
(40-75 min of nvcc; `CQSA_KERNEL_SET=common` covers fp16/bf16 and head dims 64/128).

## Use

```python
import torch, stream_cqsa

q, k, v = (torch.randn(1, 8, 4_000_000, 64, dtype=torch.float16) for _ in range(3))   # host or device
out = stream_cqsa.attention(q, k, v, is_causal=True)          # SDPA's signature; exact, always fits
```

That is the whole API for most uses. Below the memory boundary the call *is* the
monolithic kernel; above it the planner chooses the decomposition from the free memory
and the call runs exactly, on the best kernel available. Gradients flow when the inputs
require them. To route an existing model without touching it:

```python
stream_cqsa.patch_sdpa()          # F.scaled_dot_product_attention now falls back to Stream-CQSA above 64K tokens
with stream_cqsa.patched_sdpa():  # or scoped
    model(...)
```

Seeing what it does, and what it would cost:

```python
out = stream_cqsa.attention(q, k, v, is_causal=True, verbose=True)   # or CQSA_VERBOSE=1
#  Stream-CQSA: forward of N=4.0M tokens ... decomposed over c=31 at depth itr=1: 31 subproblems on cuda:0 | ... | expected ~2m10s
#  Stream-CQSA: 100%|██████████| 31/31 [02:05<00:00,  4.05s/subproblem]
#  Stream-CQSA: done in 2m06s (31 subproblems)
stream_cqsa.estimate(16_777_216, H=8, D=64)     # dry run: monolithic fits?, chosen configuration, device/host memory, time
stream_cqsa.calibrate()                         # ~1 min once per GPU model; saved to ~/.cache/stream_cqsa and used from then on
```

Explicit control (all optional keyword overrides of `attention`, or the functions
themselves): `stream_cqsa_forward(q, k, v, itr=, c=, interest_set=, low_memory=,
stream_from_host=, max_parallel=, verbose=)`, `stream_cqsa_backward(..., itr="auto")`,
`stream_cqsa_attn` (autograd), `native_wave.wave_forward/wave_attention` (the wave kernel),
`plan(...)`, `autotune(...)`, `devkit.compare_kernels(inner_fn, mono_fn)`,
`adapters.flex_inner(score_mod, extra_mask_mod)`, `distributed.dist_stream_cqsa_forward/_backward`
under `torch.distributed`. `attention(..., kernel="wave"|"cuda"|"triton")` pins the kernel.

`sbatch slurm/quickstart.slurm` runs all of the above on one GPU (`benchmarks/quickstart.py`;
output in `results/quickstart/`): the dry run, the monolithic path, the decomposed path under
a memory cap with progress, host-resident inputs, autograd, and a patched module.

Notebooks (executed, outputs included): `notebooks/stream_cqsa_v2_demo.ipynb` runs every
feature on one GPU; `notebooks/oom_boundary_demo.ipynb` sweeps N explicitly under a memory cap
and shows the baseline matching Stream-CQSA below the boundary and OOM-ing above it while
Stream-CQSA continues, exact.

## Layout

    stream_cqsa/      the package (engine, planner, devkit, adapters, distributed, autograd)
    csrc/, csrc_nc/   the two kernel source trees (FlashAttention-2 + CQS; v11 and v9)
    native/           the multi-subproblem "wave" kernel (cqsa_native): sources, build, tests, benchmark
    tests/            pytest suite (engine, planner, devkit, adapters)
    benchmarks/       kernel A/B, pipeline A/B, end-to-end, quick bench, ncu driver, quorum axis
    distributed/      multi-device tests (2/4/8 GPUs)
    slurm/            job templates, incl. slurm/paper/ (the paper's experiments re-run with v2)
    notebooks/        the demo notebook and its generator
    docs/             kernel technical note, design notes (kernel suite / automatic conversion),
                      the full engineering log (LOG.md), per-step kernel diffs
    results/          every measurement referenced in the docs (JSON + slurm logs)

## Headline measurements (A100-SXM4-80GB, fp16, B=1 H=8 D=64, causal)

| N | FlashAttention-2 (device-resident) | Stream-CQSA v1 | Stream-CQSA v2 | v2 / v1 |
|---|---|---|---|---|
| 256K | 0.36 s | 2.17 s | 1.17 s | 1.86x |
| 1M | 5.96 s | 15.51 s | 9.49 s | 1.63x |
| 2M | 24.18 s | 50.97 s | 35.31 s | 1.44x |

Both Stream-CQSA arms stream Q/K/V from host memory with a host accumulator (the
recovery configuration); outputs are identical to v1 and, against float64, more
accurate than the monolithic fp16 call (2.8e-4 vs 5.5e-4 at 1M). Kernel alone on
the real subproblem: 30.2 → 24.4 ms (FA-2: 17.5 ms). Details and every other
number: `docs/LOG.md`, `results/`.

## License

BSD-3-Clause (see `LICENSE`); vendored FlashAttention-2 and CUTLASS notices in `third_party/`.
