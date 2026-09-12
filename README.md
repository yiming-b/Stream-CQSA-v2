# Stream-CQSA v2

Exact out-of-memory recovery for attention. When an attention call does not fit
in device memory, Stream-CQSA decomposes its pair set over a cyclic quorum set
into independent subproblems, runs them one at a time (or a few at a time, or
on several devices), and recomposes the exact result. It defines no attention
rule of its own: it reproduces whatever kernel it wraps.

Paper (v1): [arXiv:2604.20819](https://arxiv.org/abs/2604.20819) ·
v1 repo: [yiming-b/Stream-CQSA](https://github.com/yiming-b/Stream-CQSA)

## What is new in v2

| area | v2 |
|---|---|
| **native kernel** | second-generation forward kernel: 19% faster than v1 on the real workload (L=131K causal, A100), and with CQS off it now equals FlashAttention-2 (`docs/kernel_technical_note.md`) |
| **engine** | pipelined host accumulator (d2h + merge overlap the next kernel), contiguous-chunk DMA (`shared_chunks`), measured concurrency policy; 1.44–1.86x faster end to end than v1 on one A100 |
| **multi-device** | `stream_cqsa.distributed`: exact sharding of the subproblems over a `torch.distributed` group; 3.2–3.5x on 4 GPUs, 5.1x on 8 GPUs / 2 nodes |
| **automatic configuration** | `stream_cqsa.autoconfig`: monolithic vs decomposed, depth `itr`, quorum set `(c, interest_set)`, accumulator placement, host residency, concurrency and device count chosen from a hardware description and a calibratable cost model |
| **independent fwd/bwd** | the backward plans its own decomposition depth (`bwd_itr="auto"`) |
| **developer kit** | `stream_cqsa.devkit`: plug any inner kernel into the framework and get exactness (bit-identical / within rounding / not) and performance against its monolithic call; `quick_bench` sweeps |
| **Triton kernel (no build)** | `stream_cqsa.triton_kernel`: the CQS forward kernel in Triton with the same contract as the CUDA one; selected automatically when no extension is compiled. Faster than the CUDA kernel below L≈56K, at parity there, ~1.4x slower end to end at 1M (`docs/LOG.md`, Phase 4) |
| **adapters** | `stream_cqsa.adapters`: automatic conversion of FlexAttention-expressible kernels (ALiBi, windows, soft-cap, document masks) into inner kernels, with global-position remapping |

## Install (from source; the extension is compiled for your GPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130   # match your CUDA
pip install ninja flash-attn                                            # flash-attn: the monolithic baseline
git clone https://github.com/yiming-b/Stream-CQSA-v2.git && cd Stream-CQSA-v2
CQSA_KERNEL_SET=common pip install -e . --no-build-isolation             # fp16+bf16, head dims 64/128, sm80
```

**No build at all:** `pip install -e . --no-build-isolation --no-deps` without running the extension build (or simply importing the package from a checkout) still works: the engine falls back to the Triton kernel for the forward (Triton ships with torch). The CUDA build is only needed for the fastest forward and for the backward.

`setup.py` builds two extensions: `cqsa_cuda` (from `csrc/`, the v11 forward,
used for causal calls) and `cqsa_cuda_nc` (from `csrc_nc/`, the v9 forward, used
for non-causal calls — v11's non-causal CQS-on instantiation is mis-compiled by
ptxas, see the technical note). `CQSA_KERNEL_SET=a100_fp16_hdim64_128` builds
fp16 only in ~40 min on 8 cores. Requires `csrc/cutlass` (vendored).

## Use

```python
import torch
from stream_cqsa import attention_oom_safe, stream_cqsa_attn, auto_attention

q, k, v = (torch.randn(1, 8, 4_000_000, 64, dtype=torch.float16) for _ in range(3))   # host or device

out = attention_oom_safe(q, k, v, causal=True)            # SDPA first; Stream-CQSA only on OOM
out = stream_cqsa_attn(q, k, v, causal=True)              # autograd; backward plans its own depth
out, plan = auto_attention(q, k, v, causal=True,          # plan from a hardware budget, then run
                           hardware={"cuda:0": "40GiB", "host": "256GiB"})
```

Explicit control: `stream_cqsa_forward(q, k, v, itr=, c=, interest_set=, low_memory=,
stream_from_host=, max_parallel=, shared_chunks=)` and `stream_cqsa_backward(...,
itr="auto")`. Planner: `plan(N, B, H, D, dtype, causal, hardware, direction="fwd"|"bwd")`,
`calibrate()`, `autotune()`. Devkit: `compare_kernels(inner_fn, mono_fn)`,
`quick_bench()`. Adapters: `flex_inner(score_mod, extra_mask_mod)`. Multi-device:
`distributed.dist_stream_cqsa_forward/_backward` under `torch.distributed`.

The notebook `notebooks/stream_cqsa_v2_demo.ipynb` runs every feature on one GPU,
using a memory cap to simulate a smaller device.

## Layout

    stream_cqsa/      the package (engine, planner, devkit, adapters, distributed, autograd)
    csrc/, csrc_nc/   the two kernel source trees (FlashAttention-2 + CQS; v11 and v9)
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
