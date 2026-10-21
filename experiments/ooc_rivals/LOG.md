# OOC rivals study: log

## 2026-09-15

- Rival implemented: `stream_cqsa/baselines/rect_ooc.py` (R2 fixed / R3 adaptive; forward on
  `_flash_attn_forward` with fp32 log-sum-exp merge, backward on `_flash_attn_backward` with the global
  lse and O; independent q/kv tiles with exact causal sub-pairs; pinned event-guarded staging, double
  buffering and kv-outer backward for R3; byte / kernel / phase / memory accounting). Tests
  `tests/test_rect_ooc.py`: 7 passed (fp64 exactness within FA-2's error, both schedules, uneven N,
  non-causal, budget compliance, probe cache). A pinned-buffer reuse race (the next window's host write
  overtook the previous asynchronous H2D) was the one bug found on the way: buffers are now event-guarded.
- `stream_cqsa.attention(..., return_info=True)` added so the harness logs the automatic mode's engine,
  kernel, depth, subproblem count and retries.
- Harness `experiments/ooc_rivals/bench.py` (worker per measurement, allocator cap = budget, planner given
  the same budget, cold + 3 warm timings, float64 sampled-row forward check, gradient check against the
  first finished method's gradients per N (FA-2 where it fits)). Smoke on della-vis1 at 64K / 2 GiB: all
  four methods, both directions, ok (Stream-CQSA auto chose the monolithic call there).
- Grid submitted (gpu-short, `gpu80&pcie`, 96G host): jobs 13952418-13952425 = budgets {10, 20, 40, 80}
  GiB x {fwd, bwd}; N in {256K, 512K, 1M, 2M, 4M}, methods fa2 / rect_ooc / rect_ooc_adaptive / cqsa_auto.

## 2026-09-23

### Grid results (harness v1, jobs 13952418-25, A100 80GB PCIe) and what was wrong with them

All eight grid jobs finished. Two defects in `bench.py run_once` and one host-memory failure:

- **FA-2 forward timed twice.** The fa2 forward arm called `flash_attn_func` twice inside the timed region
  (the result and the "kernel_only" measurement) and did the head-major -> token-major host transpose of
  Q/K/V there. FA-2 forward rows were ~2x inflated (1M: 17.0 s transfer-inclusive vs 8.0 s kernel-only).
- **Layouts.** Every arm received head-major `[B,H,N,D]` host inputs. FA-2 needs `[B,N,H,D]`, the
  rectangular rival packs to token-major (its `pack_host_s` phase), Stream-CQSA's streaming engines gather
  token-major (they warn and copy), and the cqsa backward arm also cloned Q/K/V inside the timed region.
  The transposes are small (rect: 0.54 s of 132 s at 4M, <0.5%) but they are not the method's work.
- **4M backward rows died of host OOM** at `--mem=96G` (MaxRSS 100.66 GB): the correctness check made three
  full fp64 copies per gradient (`rel()`: 48 GiB transient at 4M), the cold run's outputs stayed alive
  through the warm runs, the gradient reference (12 GiB) was loaded whole, plus the head-major copies.

### Harness v2 (commit f5ac946)

Rule, documented in `bench.py`: every timed region is host inputs -> host outputs and contains exactly one
execution of the method; input preparation is untimed and gives each method the host layout it consumes
without a copy -- token-major Q/K/V (and dO) for FA-2 and both rivals, token-major Q/K/V and head-major dO
for Stream-CQSA (its backward copies dO to token-major itself either way and forms rowsum(dO*O) on head-major
slices). Values are identical to v1 (same generator stream, re-laid out). Casts needed to meet the output
contract stay timed (the rival's fp32 -> fp16 output cast; Stream-CQSA's inside `attention()`). The FA-2
kernel-only reference is measured after the timed runs, outside them. Host memory: correctness right after
the cold run, outputs freed before the warm runs, chunked fp64 comparisons (256 MiB temporaries), mmap'd
gradient reference, the rival's fp16 output cast after its backward; per-stage VmHWM recorded. Rows carry
`harness: "v2"`; the driver re-measures v1 rows instead of skipping them. Jobs run from a `git archive`
snapshot (`slurm/ooc_rivals/snapshot.sh`) with the kernel .so files frozen in
`/scratch/gpfs/AKEY/yb2807/Stream-CQSA-snapshots/kernels-ooc-rivals-2026-09-23` (MD5SUMS; identical to
the v11 / v9 / cqsa_native builds the v1 grid used).

Host footprint, measured on della-vis1 and extrapolated (1 KiB/token = one fp16 [N,H,D] tensor):
FA-2 forward 4.9 GiB at 1M (generation peak) -> ~20 GiB at 4M (job: 24G). Rect backward 21.7 GiB at 1M
incl. 1.4 GiB pinned; its pinned staging grows with the budget (4M: 2.6 / 5.2 / 10.5 / 20.9 GiB fixed,
up to 3.4 / 6.8 / 13.7 / 27.3 GiB adaptive at 10 / 20 / 40 / 80 GiB). Stream-CQSA backward 19.3 GiB at 1M
(acc=CPU). Largest 4M row estimated at ~99 GiB (rect adaptive at 80 GiB) -> the 4M backward job asks 120G.

### Finding: Stream-CQSA auto's 4M backward at a 10 GiB budget OOMs inside the package (v2.2.3)

v1 row: `OutOfMemoryError ... Tried to allocate 8.00 GiB ... 8.00 GiB is allocated`. The plan is sound:
`cqsa c=133 itr=1 acc=cpu n_par=1 host-resident Q/K/V`, predicted 4.8 GiB device, ~709 s (planner dry run,
`attention(..., plan_only=True)`; runners-up c57/c73/c91 at 5.9-7.5 GiB). The failure is in the autograd
wrapper, not the plan: reproduced at 512K / 1.25 GiB, 512K / 2.5 GiB and 1M / 5-5.9 GiB on della-vis1,
always at `stream_cqsa/native_autograd.py:93` (`out32.to(device=q.device, dtype=q.dtype)`).
`_StreamCQSANative.forward` calls `stream_cqsa_forward` without `out_device="acc"`, so the forward moves
the host accumulator's fp32 output to the device (N*H*D*4 = 8 GiB at 4M); casting that strided device
tensor back to host fp16 allocates a second fp32 copy (8 GiB). 16 GiB > 10 GiB. The same round trip
explains the exactly-8.00 GiB device peaks of the 2M backward rows at 10 and 20 GiB (4 + 4 GiB), and costs
two 8 GiB PCIe transfers per call at 4M. Near the edge it is flaky: at 512K / 2.5 GiB and 1M / 5.9 GiB the
cold run passed and the first warm run failed. The planner's device estimate does not include this O(N)
term. Fix (in `stream_cqsa/`, not done here): pass `out_device="acc"` in the autograd forward, as the
forward-only path of `attention()` already does, and cast on the host. v2 records failed rows with the
plan (`plan_only` after the failure), the OOM site (`error_where`) and the failed repetition.
