# Stream-CQSA next — work log

Three tracks, from the paper's Limitations section:

1. **Kernel** — a more efficient A100 kernel than the shipped CQS-modified FA-2.
2. **Parallel** — run resident subproblems concurrently when it is faster than
   one at a time.
3. **Distributed** — multi-device execution.

Everything here lives under `next/`. Nothing is pushed. Every measured number
below names the job or command that produced it.

---

## 2026-09-11 — reconnaissance

Interactive node della-vis1: A100-PCIE-40GB, 497 MiB in use by one other
process. Usable for smoke runs. Multi-GPU nodes: a100:4 x 79 nodes, a100:2,
a100:8 x 1 (della-l06g12). H100:4/8 also present.

### What the code already has

* `stream_cqsa_forward` already creates `n_par` CUDA streams and round-robins
  tasks over them (`stable_stream.py:1211`, `run_one(task, slot)`). So
  "parallel" is structurally present. The probe trace from the paper work
  (job 13175119) showed tasks executing strictly one after another even with
  7 streams at N=128K. Hypothesis: the acc=CPU d2h is a **blocking** copy
  (`out_i.to("cpu", non_blocking=False)`), so the host waits for each kernel
  before it can launch the next subproblem. That serializes the pipeline
  regardless of stream count. To verify.
* `choose_parallelism()` picks `n_par` from memory, not throughput. The paper
  says one subproblem already saturates the SMs (880 blocks vs 108 SMs at
  N=32K), so concurrent *compute* should not help at large N; overlap of
  gather/h2d/d2h with compute should. Must measure, not assume.
* **No distributed code exists anywhere** in the package. Track 3 is from
  scratch. Subproblems are independent and recompose via a max-shifted merge
  of per-token (m, l, acc), so the natural design is: shard the c**itr tasks
  across ranks, each rank runs its shard with the existing single-device
  engine, then all-reduce the (m, l, acc) statistics with the same merge.
* Kernel: 7 files carry CQS modifications
  (`flash_api.cpp, flash_fwd_kernel.h, flash_bwd_kernel.h, cqsa_kernel.cu,
  flash.h, flash_bwd_launch_template.h, mask.h`). An inner-kernel
  microbenchmark exists at `experiments/kernel/bench_inner_kernel.py`.
  Build is ~33 min (`MAX_JOBS=16`). Prebuilt `.so` dated Aug 12 in
  `packages/stream-cqsa/`. A new kernel must build into `next/kernel/`, not
  clobber that.

### Plan and order

Kernel work is the highest risk (CUDA, long compile), so:
1. Baseline first: measure the shipped kernel vs FA-2 vs SDPA at the *subproblem*
   shape, forward and backward, so kernel efficiency is separated from
   scheduling. This number is what track 1 has to beat and what tracks 2/3 must
   not regress.
2. Track 2 (pure Python, testable on the interactive GPU).
3. Track 3 (needs a 2–4 GPU slurm job).
4. Track 1, targeted at whatever the baseline profile shows is actually slow.

### Baseline 1 — inner kernel at identical shape (`00_bench_inner_kernel_baseline.txt`)

Interactive A100-PCIE-40GB, H=8 D=64 fp16, non-causal. L = 3N/7 is one
subproblem's length.

    N       L      fa_plain  fa_cqs   cqs/plain  no-mask cqs/plain
    8192    3512   0.53 ms   1.69 ms  3.19x      1.37x
    16384   7023   1.57      4.44     2.83x      1.25x
    32768   14044  3.26      6.41     1.97x      1.13x
    65536   28088  12.00     17.46    1.46x      1.15x

Reading. Two separable overheads. **Carrying** the CQS feature with an all-zero
mask costs 1.13-1.37x over plain FA at the same shape. **Masking** costs a
further 1.27x at 64K and 2.35x at 8K, even though only ~0.8-2% of tiles are
mixed and go through the per-element path. And the masked kernel does ~22% LESS
arithmetic than plain (intra-non-owner-chunk tiles are skipped), so relative to
work actually done it is ~1.9x slower at 64K. That gap is track 1's target.
Ideal decomposition overhead is 1.29x; measured 7*cqs/mono is 1.89x at 64K.

### Baseline 2 — does n_par do anything? (`00_npar_sweep_baseline.txt`)

N=65536, itr=1, causal, Q/K/V host-resident, same node.

    acc     n_par=1   2       4       8
    GPU     480.9     359.3   407.1   401.7   ms   -> 1->2 gives 25%, then flat
    CPU     3005.4    2924.5  2921.4  2930.2  ms   -> FLAT

Hypothesis confirmed: with the accumulator on the host, stream count changes
nothing, because `out_i.to("cpu", non_blocking=False)` blocks the host thread
per subproblem. acc=GPU gains 25% from a second stream (transfer/gather overlap)
and nothing beyond, matching the SM-saturation argument.

Also: acc=CPU is 6.2x slower than acc=GPU here (3005 vs 481 ms). The paper
sweep had 4.0x at this N. The absolute numbers on della-vis1 are ~4x slower
than the paper's (669 ms at 64K acc=CPU there) -- shared node, PCIe card,
possibly contended CPU. Use it for correctness and shape-of-result only;
do real timing on slurm.

### Decisions

* Track 2 target is the acc=CPU forward. The backward already has the right
  design (`scatter_pool` worker thread + non-blocking d2h). Port it: pinned
  output slots, non-blocking d2h + event, merge on a worker thread. Expected:
  wall -> max(compute, gather+merge) instead of their sum. Large gain at
  small/mid N, a few percent at 16M where compute is 96%.
* Track 3: shard tasks over ranks; each rank runs the existing engine; combine
  with all-reduce(max m) then reduce-scatter(sum of rescaled l, acc), chunked
  over tokens so the fp32 accumulator never has to fit on one device whole.
  Standard NCCL ops only; exact.
* Track 1: profile before touching CUDA. Suspects: per-element path loads
  int64 group bits from global memory per element; carrying cost of the param
  block/pointer arithmetic on every tile.

### Kernel: composition and registers (`01_kernel_composition_65536.txt`, `01_cuobjdump_*.txt`)

* The CQS forward is ONE kernel, the same
  `flash_fwd_kernel<Flash_fwd_kernel_traits<64,128,128,4,...>>` template as
  plain FA. No extra kernels. All overhead is inside the CUDA code.
* The per-call `Memcpy DtoH (Pageable)` seen in the profile is
  `cqs_block_summaries()` recomputing summaries on the host when the caller
  passes none -- bench-only. `build_tasks()` precomputes and pins them per
  task, so the scheduler path has no per-subproblem host sync.
* Register theory is OUT. Shipped fwd hdim64: REG:255, STACK 304-648 across
  the 22 template instances. **Upstream FA-2 2.8.3 hdim64 fwd: also REG:255,
  STACK 208-584.** Spills are inherent to FA-2's hdim64 (128x128, 4 warps)
  tile config on sm80, not introduced by CQS. (The paper's constant-memory fix
  was for the backward, and the backward symbols do not match my grep -- to
  check separately, not load-bearing now.)
* What the numbers DO support: a mixed tile costs ~40x a normal tile. From the
  bench, masking adds 3.7 ms at 64K with ~0.8% of ~48K tiles mixed, i.e.
  ~10 us per mixed tile against ~0.25 us per normal tile. The per-element
  path does a dependent int64 global load (`cqs_group_bits[col_idx]`) per
  score element inside an unrolled loop. That is the concrete track-1 target:
  stage the tile's column bits into shared memory once (128 x int64 = 1 KB,
  one coalesced load per thread) and test against smem. Second target: fully
  masked tiles still load their K/V tiles into smem; only the GEMMs are
  skipped. Both need a build (~5-8 min for the hdim64-only kernel set).

### Track 2 implemented in `next/pkg` (patch applied, tests below)

Forward, host accumulator: pinned fp32 output slots per stream, non-blocking
d2h + event, merge on one worker thread. The wait on the slot's previous
merge is placed immediately before the d2h copy, not at slot acquisition, so
even at n_par=1 the merge of subproblem i overlaps the compute of i+1. OOM
recovery drains in-flight merges before halving streams. Worker timings are
flushed into the trace after the join (stamp() indexes trace rows by
length, so a concurrent append would misattribute cuda_ms). Env flag
`CQSA_FWD_MERGE_ASYNC=0` restores the shipped behaviour for A/B.

Also added `task_subset` to `stream_cqsa_forward` -- the one hook the
distributed driver (`next/distributed/dist_forward.py`) needs.

**Tests after the patch:** 84 passed in 7.6 s (`02_tests_after_pipeline_patch.txt`)
-- test_stable_stream (72) + test_bwd_host_stream (12), run against next/pkg.

**Queue:** gpu-short had 1,778 pending jobs and my baselines were scheduled
for 01:24 / 03:42. gpu-test QOS has priority 8000 vs 5000 (max 3 jobs/user,
59 min). Baselines resubmitted there: kbase=13761648, pbase=13761649.

### Clean-node kernel baseline (job 13761648, A100-SXM4-80GB, no co-tenants) — `kernel_baseline.json`

CAUSAL, H=8 D=64 fp16, ms per call at the subproblem shape L=3N/7:

    N       L      fa2    cqsa_plain  cqsa_zero  cqsa_mask  sdpa  | mask/fa2  7*mask/fa2_mono
    16384   7023   0.42   0.73        0.92       1.49       0.45  | 3.50      6.28
    32768   14044  1.31   1.99        2.37       3.62       1.41  | 2.76      4.24
    65536   28088  4.58   6.48        7.49       9.69       4.95  | 2.12      3.02
    131072  56175  16.97  23.16       26.54      30.60      18.28 | 1.80      2.39

Non-causal at 131K: fa2 34.0 / plain 40.4 / zero 47.7 / mask 53.4.

**The decomposition of the 1.80x at 131K causal:**
* `cqsa_plain` vs upstream `fa2`: **1.36x**. The modified binary is slower with
  CQS switched off at runtime. This is the largest single term and it is NOT
  masking work -- it is the cost of the code being there.
* enabling CQS with an all-zero mask: a further 1.15x.
* the real mask: a further 1.15x, while doing ~22% less GEMM work.
* sdpa == fa2 to within 5% (SDPA dispatches to flash for these shapes). Good
  sanity check on the harness.

If the CQS kernel ran at fa2 * 1.15, Stream-CQSA at 131K causal would be
7*19.5/89.4 = 1.53x monolithic instead of 2.39x: a 1.56x speedup of the whole
forward at that size. That is the prize for track 1.

Ideal decomposition overhead (7*(3/7)^2) is 1.29x. The measured 2.39x at 131K
means the kernel is 1.85x off ideal.

### Clean-node pipeline baseline (job 13761649, shipped code) — `pipeline_baseline.json`

    N=65536    acc=GPU n_par 1/2/4:  166 / 109 /  86 ms
               acc=CPU n_par 1/2/4:  563 / 582 / 554 ms   stages(n_par=1): merge 283, gather 71, compute 66, d2h 56
    N=262144   acc=GPU:             1068 / 870 / 794 ms
               acc=CPU:             2226 / 2283 / 2297 ms  stages: compute 733, merge 708, gather 286, d2h 197

acc=CPU: flat in n_par, and the serial sum of stages (1924 ms at 256K) is the
wall clock. Perfect overlap would give max(compute 733, merge+gather 994) ~
1.0 s at 256K, i.e. ~2.2x, and ~1.6x at 64K where the host merge dominates.

### Track 2 A/B on the interactive node (`02_pipeline_ab_interactive.txt`)

Pipelined path is **bit-identical** to the shipped one (|diff|max = 0.0 at
every setting), rel. err vs SDPA 2.2e-4, no untouched tokens, lse finite.
Speed on della-vis1 was 1.04-1.19x, but that node's absolute numbers are
4-10x off the clean node (2066 ms at N=16K!) -- CPU contention, and the merge
is CPU work. Clean-node A/B is job 13761849.

Pipeline baseline at N=1M (same job): acc=GPU n_par=4 10757 ms; acc=CPU
15590/15616/15696 ms for n_par=1/2/4; stages compute 10534, merge 2630,
gather 1121, d2h 901. Overlap ceiling ~1.45x there. So the pipelining gain
should fall from ~2.2x (256K) through ~1.45x (1M) toward a few percent at 16M,
where compute is 96% -- exactly the shape the paper's stage figure predicts.

First large-N A/B job (13761849) failed on import: run1_test.slurm sourced
env.sh (packages/) not next/env_next.sh. Templates fixed; resubmitted.
Baselines (kbase/pbase) correctly used the shipped package.

### Track 1 plan, revised by the clean baseline

The decisive number is cqsa_plain/fa2 = 1.36x at 131K causal with CQS OFF at
runtime. Registers are allocated statically for the worst path, and the
forward's Mask object holds 13 registers' worth of CQS state (4 pointers, 5
ints) live across the whole kernel. mask.h documents the fix -- compute the
tile verdict from params (constant memory) and pass it into apply_mask, so
the pointers never become live registers -- "measured 7x on the kernel" --
and the backward already does it (flash_bwd_kernel.h:462-590). The forward
does not. That is change v1: mechanical, low risk, mirrors shipped code.

Build attribution: next/kernel/<variant>/ builds cqsa_cuda_next_<variant>;
`base` is the unmodified source and must reproduce the shipped .so's timing
before any kernel change is credited or blamed.

### Track 3: driver correct on one rank (`02_dist_smoke_1rank.txt`)

`dist_forward.py` (shard tasks round-robin, run the engine on the shard,
all_reduce MAX on lse then SUM on the rescaled output and weights, chunked
over tokens) reproduces the single-device result exactly: rel.err 0.00e+00,
lse max abs err 0.00e+00 at N=16K and 64K, world=1. First attempt failed on
`Tensors must be contiguous`: a sliced [B,H,n,D] view keeps its strides
through the elementwise product. 2-GPU test submitted (job 13762096).

### Kernel variants building

`next/kernel/base` (unmodified) and `next/kernel/v1` (forward register fix:
summaries withheld from Mask, tile verdicts from params via cqs_tile_masked/
cqs_tile_clear, passed into apply_mask -- diff in `next/kernel/v1.diff`, 77
lines, 4 sites in compute_attn_1rowblock, splitkv path untouched). Kernel set
a100_fp16_hdim64_128. Extensions cqsa_cuda_next_{base,v1}; select with
CQSA_CUDA_MODULE. First build attempt failed because setup.py reads the
version from stream_cqsa/__init__.py, which the variant dir lacked.

### Track 2 clean-node A/B, partial (job 13762063, A100-SXM4, no co-tenants)

acc=CPU, itr=1, causal, Q/K/V host-resident. shipped vs pipelined (this patch):

    N=262144   n_par=1  2440.6 -> 1733.0 ms   1.41x
               n_par=2  2764.8 -> 2009.5 ms   1.38x
               n_par=4  3095.1 -> 2960.0 ms   1.05x
    N=1048576  n_par=1 16374.9 -> 14296.5 ms  1.15x     (rest still running)

Two readings. (1) The pipeline works: 1.41x at 256K where the ceiling is
~2.2x. (2) **More streams hurt, in both arms.** One subproblem already fills
the SMs (880 blocks vs 108 SMs at 32K, more at 256K), so concurrent kernels
only contend, and at n_par=4 the host's extra gathers fight the merge worker
for the same 8-thread torch pool. Answer to "execute in parallel when memory
permits": on one device, no -- overlap, not concurrency, is what pays, and
the right n_par for the host-accumulator path is 1 (or 2 for acc=GPU, where
the 1->2 gain was 25% in the baseline and nothing beyond).

Why 1.41x and not 2.2x at n_par=1: the host blocks in slot_done.synchronize()
until compute(i)+d2h(i) land, so gather(i+1) (41 ms/task) does not overlap
compute(i); and merge(i) (101 ms/task on 8 CPU threads, ~2.3 GB/s effective)
is CPU-bound. Next: `shared_chunks=True` replaces the CPU row-gather with
contiguous chunk DMA (no CPU work), and cpu_threads=16 on a 16-core
allocation for the merge.

Backward with `_SCATTER_ASYNC` default ON: 47 tests pass
(`02_tests_bwd_scatter_async.txt`).

Track 3: 2-GPU job 13762096 failed on rank 1 after 14 s with the traceback
swallowed by torchrun. One-rank path is proven, so it is two-rank specific.
Test now installs its own excepthook and passes an explicit device_id to
init_process_group; resubmitted.

### Track 1, v2 planned (in parallel with the v1 build)

Mixed tiles: apply_mask's per-element path does one dependent int64 global
load per score element (`cqs_group_bits[col_idx]`), 128 per thread per tile
across an unrolled loop -- consistent with the measured ~19-40x cost of a
mixed tile. v2 stages the tile's 128 column bits into 1 KB of shared memory
with one coalesced load per thread (kNThreads == kBlockN == 128), placed
between the loop's existing __syncthreads so no new barrier is needed, and
the per-element test reads smem. Launch smem grows by kBlockN*8 bytes for
the non-split kernel only. Semantics unchanged: same bits, same predicate.

### Track 2 — clean-node A/B complete (job 13762063, A100-SXM4-80GB, 0 co-tenants)

`CQSA_FWD_MERGE_ASYNC=1` (pipelined d2h + merge) vs `=0` (shipped blocking path), itr=1 causal, Q/K/V on host, acc=CPU. Output bit-identical to shipped in every arm (checked earlier at 16K/64K).

| N | n_par | shipped ms | pipelined ms | speedup |
|---|---|---|---|---|
| 256K | 1 | 2440.6 | 1733.0 | 1.41x |
| 256K | 2 | 2764.8 | 2009.5 | 1.38x |
| 256K | 4 | 3095.1 | 2960.0 | 1.05x |
| 1M | 1 | 16374.9 | 14296.5 | 1.15x |
| 1M | 2 | 17149.6 | 14768.6 | 1.16x |
| 1M | 4 | 18601.1 | 17038.5 | 1.09x |
| 2M | 1 | 52528.3 | 48862.5 | 1.08x |
| 2M | 2 | 54169.8 | 49724.5 | 1.09x |
| 2M | 4 | 57161.6 | 54723.5 | 1.04x |

Reading: the win is largest where merge+d2h were a large share of the call (256K: 55%) and shrinks as compute dominates (2M). More CUDA streams hurt BOTH arms at every N — on one device, concurrency between subproblems does not pay because a single subproblem already saturates the SMs; only overlapping the host stages with the kernel pays. This answers the user's item 2 for the single-device case: the engine now runs the host-side work of subproblem i concurrently with the kernel of subproblem i+1, and n_par=1 remains the right default for acc=CPU. Remaining lever for acc=CPU is the host merge itself (2.6 s of 14.3 s at 1M) -> `pipeline_opts.py` sweep (cpu_threads, shared_chunks) still to run.

### Track 3 — 2-GPU failure diagnosed (jobs 13762096, 13762378)

Both ranks died at the first kernel call with
`cqsa_cuda ... undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb`.
Cause: `torchrun` on PATH resolves to `/scratch/gpfs/AKEY/yb2807/CQS-DeepSpeed/envs/cqs/bin/torchrun`
(a different conda env with a different torch), because the dev venv has no torchrun entry point.
The ranks therefore ran under the wrong torch and the prebuilt extension (compiled against 2.10.0+cu130) failed its ABI check.
The 1-rank smoke passed only because it was launched with `python test_dist.py` and RANK/WORLD_SIZE set by hand.
Fix: `run2_test.slurm` / `run4.slurm` now use `srun python -m torch.distributed.run --standalone ...`. Resubmitted as job 13762529.

### Track 1 — v2 created (`next/kernel/v2`, diffs in `next/kernel/v2_kernel.diff`, `v2_mask.diff`)

v1 + shared-memory staging of the mixed tile's column group-bits:
- launch: `smem_size += kBlockN * sizeof(int64_t)` (1 KB), non-split kernel only.
- kernel: `sCqsBits` carved after sV; tile verdict (`cqs_ts`) hoisted to the top of both KV loops (CTA-uniform, since it depends only on block summaries); when `cqs_ts == 1` each thread loads one int64 (`kNThreads == kBlockN == 128`) — in the masking loop before the loop's first `__syncthreads`, in the main loop between the first and second barrier — so no new barrier is added. Previous readers are behind the prior iteration's second barrier, so no WAR hazard.
- mask.h: `apply_mask(..., cqs_smem_bits=nullptr, cqs_smem_col_base=0)`; the per-element test becomes `col in range && (row_bits & sBits[col - base]) != 0`, identical to `cqs_should_mask_row_bits` (whose `row_bits == 0` early-out is already implied by `cqs_row_active`).
Build submitted as CPU job 13762514 (`next/kernel/build.slurm`, 32 cores) since the interactive node is compiling base/v1 with 2 cicc at a time.

### Track 1 — tile census of real subproblems (changes the plan)

Counted tile classes from `cqs_block_summaries` for actual `group_bits_for_path(N, path, sorted_gather=True)` at N=131072 (blk 64, causal half only):

| path | L | masked tiles | clear tiles | mixed tiles |
|---|---|---|---|---|
| (0,), (1,), (3,) | 56175 | 22.2% | 77.6% | 0.2% |
| (0,0) | 24075 | 22.1% | 77.4% | 0.5% |
| (0,1) | 24075 | 31.9% | 67.6% | 0.5% |

CQS chunks are contiguous ranges of the original sequence, so a gathered subproblem is 3 contiguous chunks and each token carries one of only 3 (itr=1) or 5 (itr=2) distinct bit patterns. Consequences:
- The per-element mixed-tile path (v2's target) runs on ~0.2% of tiles. v2 is kept for correctness/robustness of the mixed path but cannot move the workload number.
- 22% of causal tiles are fully masked. The shipped kernel already skips both GEMMs on them but still streams K and V through smem and runs both barriers. **v3** skips the V load of a masked tile and the K prefetch for a masked next tile (`next/kernel/v3`, diff `v3_kernel.diff`, 47 lines): the prefetch site evaluates the same `cqs_tile_masked` verdict for `n_block-1` that the next iteration evaluates for itself, so producer and consumer agree by construction; `cp_async_wait<0>` with nothing in flight is a no-op, barriers unchanged. Build: CPU job (see below).
- The remaining, largest gap is structural: `cqsa_plain` (CQS *disabled*) is 1.36x slower than upstream FA-2 at 131K causal. That is what v1 (Mask-object register fix) targets; if v1 is not enough, the next candidates are the `Check_inf=true` softmax path and the fp32 `acc_out` epilogue.

**Bench defect found:** `kernel_baseline.py` (and the first `kernel_ab.py`) called `flash_attn_func_cqs_group_bits` without block summaries, so every timed call recomputed them on the host (`bits.cpu()` sync + numpy reduce + h2d). That is why "mask" measured slower than "zero" despite 22% fewer GEMMs, and why the 16K ratios looked so bad (3.5x). The engine builds summaries once per task, so the honest kernel number passes them in. `kernel_ab.py` now precomputes `cqs_blk_or/and` once per shape, times `plain`/`zero`/`mask` for every variant, and checks every variant bit-for-bit against shipped (`maxdiff_vs_shipped`). The clean-node `kernel_baseline.json` zero/mask columns are therefore superseded by the coming `kernel_ab.json`; its `fa2`, `cqsa_plain`, `sdpa`, `fa2_mono` columns stand.

### Track 2 — knob sweep with the pipelined merge (job 13762659, A100-SXM4-80GB, 48 cpus, 0 co-tenants)

acc=CPU, itr=1 causal, Q/K/V on host; ms = best of 2; stages from the last rep.

| N | n_par | shared_chunks | cpu_threads | ms | stages (ms) |
|---|---|---|---|---|---|
| 256K | 1 | 0 | 8 | 1318.7 | compute 734, merge 706, gather 333, d2h 64 |
| 256K | 1 | 0 | 16 | 1364.2 | merge 643, gather 376 |
| 256K | 1 | 1 | 8 | 1051.2 | compute 667, merge 716, gather 124, d2h 69 |
| 256K | 1 | 1 | 16 | 1133.4 | |
| 256K | 2 | 0 | 8 | 1128.8 | d2h 290 |
| 256K | 2 | 1 | 8 | 1105.8 | |
| 256K | 2 | 1 | 16 | 1009.6 | compute 785, merge 649, d2h 385, gather 98 |
| 1M | 1 | 0 | 8 | 12910.9 | compute 10546, merge 2460, gather 1374, d2h 253 |
| 1M | 1 | 1 | 8 | 11862.6 | compute 10425, merge 2247, gather 498, d2h 252 |
| 1M | 1 | 1 | 16 | 12165.9 | |
| 1M | 2 | 0 | 8 | 11700.1 | compute 14292 (overlapped), d2h 4281 |
| 1M | 2 | 1 | 8 | 11606.1 | |
| 1M | 2 | 1 | 16 | 11662.7 | |

Reading: `shared_chunks=True` (contiguous chunk DMA instead of a CPU row-gather) is the one knob that pays everywhere: gather 333->124 ms at 256K and 1374->498 ms at 1M, 1.25x / 1.09x end to end. More merge threads do nothing (the merge is memory-bound on the host). n_par=2 now overlaps two subproblems' d2h with compute and is 4-10% better than n_par=1 at 1M, 4% at 256K, i.e. concurrency across subproblems finally pays a little once the host stages no longer block, but the win is small and it doubles the device footprint of the in-flight set. Recommendation for acc=CPU: `shared_chunks=True, cpu_threads=8`, `max_parallel=2` when memory permits, else 1. Note the absolute 256K/1M numbers are ~25% faster than the A/B job's on the same GPU model: this job had 48 cpus (the A/B had 16), and the merge/gather are host-bound.

### Track 3 — 2-GPU result (job 13762529, 2x A100-SXM4-80GB, exact)

`dist_stream_cqsa_forward` vs single-device `stream_cqsa_forward`, itr=1 causal, 7 tasks round-robined 4/3 over 2 ranks; rank 0 also computes the single-device reference.

| N | single s | dist s | speedup | local s | merge s | rel.err | lse.err |
|---|---|---|---|---|---|---|---|
| 64K | 0.55 | 1.70 | 0.32x | 1.07 | 0.62 | 5.4e-8 | 9.5e-7 |
| 256K | 2.02 | 3.08 | 0.66x | 3.04 | 0.02 | 5.0e-8 | 9.5e-7 |
| 1M | 13.86 | 9.98 | 1.39x | 9.86 | 0.09 | 4.8e-8 | 9.5e-7 |

Exactness holds (fp32 rounding level; lse to 1e-6). Speed: the cross-rank merge is negligible (0.09 s at 1M), but the local phase is not 2x faster: 4 tasks per rank vs 7 is at best 1.75x, and the first two rows include NCCL/CUDA warm-up. Next: warm-up call, `shared_chunks=True`, itr=2 (49 tasks) for balance when world_size does not divide 7, and 4-GPU strong scaling.

### Track 1 — base/v1 built; where the kernel time actually goes (interactive A100-PCIE-40GB, load ~160, noisy)

- `base` (rebuilt unmodified source, this toolchain) is bit-identical to `shipped` (max|diff| 0.0) and within noise of it in time: the toolchain is not the difference, so variant deltas can be read against `base`.
- `v1` is bit-identical to shipped; interactive timing suggests ~4% at 131K causal (25.75 vs 28.36 plain) but the node is too loaded to trust. Clean-node A/B submitted (job 13763830, `kernel_ab_clean_v1.json`). Register report: still 255 regs; stack profile shifted (STACK:0 appears for 5 fwd instantiations in v1, none in base).
- Synthetic mask patterns at N=131K (L=56175), v1, ms/call, interactive node:

| pattern | causal | non-causal |
|---|---|---|
| plain (CQS off) | 25.70 | 46.90 |
| zero bits (all tiles clear) | 28.47 | 53.69 |
| real path-(0,) bits (22% masked, ~1% mixed) | 30.37 | 56.44 |
| half-self (25% of pairs masked, tiles pure) | 27.94 | 49.06 |
| all-masked (100% tiles masked, no GEMM at all) | 26.40 | 34.00 |

Two conclusions. (1) **A fully-masked tile still costs ~63% of a live tile** (all-masked non-causal 34.0 vs 53.7): the shipped skip removes the GEMMs but the K/V smem traffic and barriers remain, and the kernel is close to memory-bound at this tile shape. v3 (no K/V loads for masked tiles) should recover most of that: expected ~0.22*0.63 = 14% of the kernel on the real workload. (2) **The real pattern is slower than zero bits despite having 22% fewer GEMMs**, so the ~1% of mixed tiles at the chunk boundaries cost roughly 10x a normal tile each under the per-element global-load path; v2's smem staging is relevant after all. (3) `zero` vs `plain` is an 11-14% tax just for enabling CQS: per tile the kernel evaluates two verdicts (`cqs_tile_masked`, `cqs_tile_clear`), each with dependent int64 loads for 2 row blocks and 2 col blocks. The row half is loop-invariant. **v4** hoists the row-block AND/OR summaries out of the KV loop and evaluates only the column half per tile.

### Track 3 — 2-GPU strong scaling with warm-up and shared_chunks (job 13763575, 2x A100-80GB PCIe)

itr=1 causal, acc=CPU, `shared_chunks=True`, n_par=1; 7 tasks split 4/3 (ideal 1.75x).

| N | single s | dist s | speedup | local s | merge s | rel.err |
|---|---|---|---|---|---|---|
| 256K | 1.53 | 1.54 | 1.00x | 1.47 | 0.06 | 5.0e-8 |
| 1M | 15.28 | 10.82 | 1.41x | 10.55 | 0.23 | 4.8e-8 |
| 2M | 54.70 | 34.60 | 1.58x | 34.06 | 0.44 | 4.6e-8 |

Exact at every N; the cross-rank merge is ~1% of the call. The itr=2 leg crashed: the engine's `shared_chunks` supports itr=1 only (a depth-2 subsequence is not a union of whole top-level chunks). Test now sets `shared_chunks=(itr==1)` and writes its JSON after every row; itr=2 resubmitted (job 13763858). 4-GPU job 13763577 still pending on gpu-test.

### Track 1 — clean-node kernel A/B, shipped / base / v1 (job 13763830, A100-80GB PCIe, 0 co-tenants, `kernel_ab_clean_v1.json`)

ms/call at the itr=1 subproblem shape L=3N/7, H=8 D=64 fp16, block summaries precomputed (engine-faithful). `plain` = CQS off, `zero` = CQS on with all-clear tiles, `mask` = the real path-(0,) bits.

| N | causal | fa2 | shipped plain / zero / mask | v1 plain / zero / mask | v1/shipped (mask) | max diff |
|---|---|---|---|---|---|---|
| 16K | yes | 0.42 | 0.72 / 0.80 / 1.37 | 0.69 / 0.78 / 1.33 | 0.975 | 0 |
| 32K | yes | 1.33 | 2.01 / 2.25 / 3.49 | 1.91 / 2.19 / 3.36 | 0.962 | 0 |
| 64K | yes | 4.80 | 6.72 / 7.69 / 9.68 | 6.38 / 7.47 / 9.45 | 0.976 | 0 |
| 131K | yes | 19.60 | 25.55 / 28.42 / 30.87 | 24.23 / 27.28 / 29.76 | 0.964 | 0 |
| 131K | no | 39.56 | 45.69 / 51.79 / 55.48 | 45.36 / 51.42 / 55.87 | 1.007 | 0 |

- `base` == `shipped` within 0.5% at every point and bit-identical: rebuilt-here numbers are directly comparable to the paper's binary.
- `v1` (Mask object no longer carries the summary pointers; verdicts computed from params): 2.5-4% faster on causal, neutral non-causal, bit-identical. Modest, kept.
- Decomposition of the remaining gap at 131K causal: plain/fa2 = 1.30x (structural, CQS off), zero/plain = 1.11x (verdict + Check_inf tax, v4 target), mask/zero = 1.09x at 131K but **1.71x at 16K** and 1.55x at 32K: the chunk-boundary mixed tiles are a fixed count per boundary, so they are a large fraction of small subproblems. v2 (smem column bits) therefore matters most exactly where the deep-itr subproblems live (itr=2/3 at 1M-16M have L = 24K-140K).
- Non-causal `zero` vs `plain` is 1.13x and unchanged by v1, so the tax is not register pressure; it is the per-tile verdict work (v4) and the Check_inf softmax.

### Track 3 — distributed backward implemented (1-rank smoke on the interactive A100-40GB)

- `stream_cqsa_backward` gained `task_subset` (same filter as the forward). A whole-call escalation under a subset keeps only the deeper tasks whose path extends one of the rank's paths, so ranks never double-count after an OOM retry.
- `dist_stream_cqsa_backward(q,k,v,dout,out,lse, itr=..., **engine_kw)` in `next/distributed/dist_forward.py`: each rank runs its shard, gets fp32 partial dq/dk/dv, then one chunked all_reduce(SUM) per gradient. Exact up to fp32 summation order.
- `test_dist.py --bwd` runs and checks it. 1-rank smoke (world=1, N=16K/64K, itr=1 and 2): forward rel.err 0.0; backward dk/dv rel.err 0.0, dq 3-6e-6. The dq residue is the FA-2 backward's own fp32 atomicAdd accumulation of dQ (run-to-run nondeterministic at that level), not the sharding: dk/dv, which the kernel accumulates deterministically, match bit-for-bit.
- 2-GPU fwd+bwd strong-scaling job submitted (see below).

### Track 3 — 2-GPU, itr=2 (job 13763858, 2x A100-SXM4-80GB): balance fixes the 7-task granularity

49 tasks split 25/24 (ideal 1.96x). acc=CPU, n_par=1, no shared_chunks (itr=2).

| N | single s | dist s | speedup | local s | merge s | rel.err |
|---|---|---|---|---|---|---|
| 256K | 3.03 | 2.53 | 1.20x | 2.46 | 0.02 | 9.1e-8 |
| 1M | 21.00 | 12.47 | 1.68x | 12.18 | 0.09 | 8.6e-8 |
| 2M | 64.12 | 36.47 | 1.76x | 35.56 | 0.17 | 8.3e-8 |

At itr=1 the same GPUs gave 1.58x at 2M because 7 tasks cannot split evenly; at itr=2 the shard sizes match and the efficiency reaches 90%. Rule for the multi-device engine: choose the smallest itr whose c^itr tasks divide evenly enough across world_size (for 2 GPUs itr=2; for 4 GPUs itr=2 gives 13/12/12/12; for 7 or 49 GPUs itr=1/2 are perfect). Note the single-device itr=2 call is itself 17-36% slower than itr=1, so the distributed itr=2 numbers should be compared with the single-device itr=1 best (54.7 s at 2M): 1.50x on 2 GPUs against the best single-device configuration.

### Track 3 — 4-GPU strong scaling (job 13763577, 4x A100-SXM4-80GB, 16 cpus for 4 ranks), acc=CPU

| N | itr | tasks/rank | single s | dist s | speedup | local s | merge s | rel.err |
|---|---|---|---|---|---|---|---|---|
| 256K | 1 | 2,2,2,1 | 1.28 | 1.10 | 1.17x | 1.03 | 0.06 | 3.8e-8 |
| 1M | 1 | 2,2,2,1 | 12.56 | 6.31 | 1.99x | 6.20 | 0.07 | 3.5e-8 |
| 2M | 1 | 2,2,2,1 | 45.79 | 18.02 | 2.54x | 17.84 | 0.12 | 3.4e-8 |
| 256K | 2 | 13,12,12,12 | 2.68 | 1.58 | 1.70x | 1.38 | 0.15 | 9.3e-8 |
| 1M | 2 | 13,12,12,12 | 19.15 | 7.08 | 2.70x | 6.84 | 0.07 | 8.7e-8 |
| 2M | 2 | 13,12,12,12 | 60.73 | 19.51 | 3.11x | 18.94 | 0.13 | 8.4e-8 |

Exact at every point; merge <1%. Against the best single-device configuration (itr=1, 45.8 s at 2M) 4 GPUs give 2.35-2.54x. Efficiency is capped by the host: with acc=CPU every rank runs its gather/merge on the same 16 cores (2 tasks/rank took 17.8 s where 2/7 of the single run would be 13.1 s). In the multi-device setting each rank has a whole 80 GB for one shard, so the per-rank accumulator can sit on the device (`low_memory=False`, the paper's acc=GPU mode, 2-3x faster per subproblem). `test_dist.py --acc gpu` added; 4-GPU acc=GPU run submitted with 32 cpus.

Baseline provenance note (Track 1): the `fa2` floor in every kernel table is the editable flash-attn 2.8.3 at `CQS_torch/src/flash-attention`, which is v2.8.3-273 with early CQS edits in mask.h/flash_fwd_kernel.h (runtime-disabled). It is not byte-stock. torch 2.10's own FlashAttention-2 build (SDPA flash backend, stock kernels) timed within 1% of it in `kernel_baseline.json` (sdpa ~ fa2 at every N), so the floor is right; `kernel_ab.py` now also prints the SDPA-flash column so the final table carries a stock reference.

### Track 1 — v2 and v3 built and verified (interactive A100-40GB, node idle at the time: load 12)

Both bit-identical to shipped on every (pattern x causal) at N=16K, 131K path (0,), 131K path (0,1), and N=100003 path (1,) (ragged tail): real bits, zero, all-ones, and a random 0-7 pattern that makes every tile mixed. ms/call, shipped / v2 / v3:

| L | pattern | causal | non-causal |
|---|---|---|---|
| 56175 | real | 31.52 / 30.88 / 33.07 | 57.54 / 55.62 / 60.47 |
| 56175 | zero | 29.78 / 28.98 / 31.46 | 53.89 / 53.23 / 58.41 |
| 56175 | all-masked | 26.89 / 26.76 / 26.59 | 34.49 / 35.44 / 34.45 |
| 56175 | random (all mixed) | 170.1 / 170.8 / 173.3 | 363.3 / 353.2 / 362.8 |
| 24075 | real | 7.03 / 7.04 / 7.52 | 9.75 / 9.90 / 10.91 |

Findings that overturn the earlier assumptions:
1. **v2 (smem column bits) changes nothing**: the all-mixed pattern runs at 5.5x the clear-tile cost with or without it, so the per-element path is bound by its own instruction stream (per-element index math and branches over 64 scores per thread), not by the group-bits global loads. v2 is bit-exact and harmless but not a win; not carried forward.
2. **v3 (no K/V loads for masked tiles) is a loss**: all-masked is unchanged (26.6 vs 26.9), real/zero are 5-8% slower. So a skipped tile's residual cost (63% of a live tile, non-causal) is not the smem traffic either; and v3 adds a second `cqs_tile_masked` evaluation per iteration (for n_block-1), which is what made it slower. The residual is the **tile verdicts themselves**: every iteration performs a chain of dependent int64 loads of the block summaries (2 row blocks + 2 col blocks for AND, again for OR) before it can branch, and in a skipped iteration there is no GEMM to hide that latency behind. A skipped tile still costs ~5 us per CTA-iteration at L=56K; that is the verdict latency, not bandwidth.
3. This also explains the zero-vs-plain tax (11-14%): the verdict's load chain stalls the warp before the QK GEMM issues even on clear tiles.

Therefore the lever is to take the summary loads off the per-tile critical path: v4 already hoists the row half; **v5** will software-pipeline the column half (load the next tile's column summaries one iteration ahead, keep them in registers) so the verdict for tile n is a register AND at the top of its iteration, and drop v3's second evaluation. ncu profile of the all-masked and zero cases to confirm the stall reason before building v5.

### Track 1 — ncu profile of v3 (job 13765375, A100-80GB PCIe; ncu locks SM clock to 1.06 GHz so durations are ~1.4x the free-running ones)

Non-split kernel, L=56175, 439x8 CTAs of 128 threads, 255 regs, 50.2 KB dynamic smem -> 2 CTAs/SM (register- and smem-limited alike, 12.5% occupancy, same as upstream FA-2 hdim64).

| case | duration | instructions | SM active | issue/scheduler | stall | note |
|---|---|---|---|---|---|---|
| all-masked, non-causal | 44.6 ms | 3.23 G | 52% | 0.30 | fixed-latency dep. | 1.54M skipped tiles -> **524 instr per warp per tile** doing nothing but verdicts and barriers |
| zero, non-causal | 69.3 ms | 10.56 G | 98% | 0.34 | fixed-latency dep. | 1715 instr per warp per live tile |
| real, causal | 42.5 ms | 4.89 G | 82% | 0.30 | fixed-latency dep. | mix of the two + mixed tiles |

No spills to local memory in any case (the STACK bytes reported by cuobjdump are the epilogue's stack frame, not loop spills). Memory is nowhere near a limit (DRAM <1%, L2 hit 98%). The kernel is **issue-latency bound**: one instruction per 3.3 cycles per scheduler with only two warps to switch between, and the dominant stall is a fixed-latency dependency, i.e. chains of integer ALU instructions. A skipped tile spends ~520 instructions per warp: two `cqs_tile_masked` verdicts (v3 adds the second) and one `cqs_tile_clear`, each with runtime `blk_size` divisions/modulos (~20 instructions each, dependent) and 2-iteration summary loops. That is the "63% residual" of a masked tile and, on live tiles, the 11-14% zero-vs-plain tax: the same ~300 instructions sit on the critical path before the QK GEMM can issue.

Consequences for the design: (1) v3's load skipping was never going to show (loads were not the bottleneck) and its extra verdict made things worse; (2) v5 — compile-time block size (divisions become shifts) and the column summaries fetched one tile ahead into registers so the verdict is a register AND — attacks exactly the measured stall. v5 is building. (3) The mixed-tile per-element path (5.5x) is likewise instruction-bound (64 scores x index math per thread); a cheaper formulation would test 8 columns per row with one 64-bit AND on a packed per-column bit word, but it is <1% of tiles at large L, so it is deferred.

### Track 1 — upstream FA-2 under ncu (job 13765519, same node class): the CQS kernel executes ~43% more instructions per live tile

| kernel | case | duration (1.06 GHz) | instructions | live tiles | instr / warp / tile | issue cycles/instr |
|---|---|---|---|---|---|---|
| upstream FA-2 (editable 2.8.3) | causal | 22.3 ms | 3.71 G | 772K | **1203** | 5.2 |
| CQS v3, zero bits | non-causal | 69.3 ms | 10.56 G | 1.54M | **1715** | 5.9 |
| CQS v3, all-masked | non-causal | 44.6 ms | 3.23 G | 1.54M skipped | 524 | 6.4 |

Same registers (255), same 2 CTAs/SM, no spills; the difference is instruction count on an issue-bound kernel. Per live tile the CQS-enabled path issues ~510 extra instructions per warp: the tile verdicts (runtime-blk_size divisions, summary loops), the runtime-branched softmax (both `Check_inf` instantiations present), the segmented-address selects, and the `apply_mask` entry test. v5 removes the verdict share; the remaining candidates are the softmax duplication and seg_k/seg_v selects (v6, template flags).

### Track 1 — clean-node A/B, shipped / v1 / v2 / v3 (job 13765300, A100-SXM4-80GB, `kernel_ab_clean_v3.json`)

ms/call, real bits (`mask` column), ratio to shipped; every variant bit-identical to shipped at every point.

| N | causal | fa2 | SDPA-flash | shipped | v1 | v2 | v3 |
|---|---|---|---|---|---|---|---|
| 16K | yes | 0.46 | 0.49 | 1.50 | 1.48 (0.985) | 1.56 (1.034) | 1.61 (1.072) |
| 32K | yes | 1.45 | 1.55 | 3.72 | 3.54 (0.952) | 3.87 (1.038) | 4.07 (1.093) |
| 64K | yes | 5.05 | 5.50 | 9.61 | 9.25 (0.963) | 9.29 (0.968) | 10.07 (1.048) |
| 131K | yes | 18.71 | 18.40 | 30.21 | 29.26 (0.969) | 29.42 (0.974) | 32.15 (1.064) |
| 131K | no | 34.58 | 36.24 | 53.37 | 53.40 (1.001) | 52.22 (0.979) | 57.76 (1.082) |

Confirms the interactive reading on a quiet SXM4 node: v1 is a steady 3-5% causal gain, v2 is within +-3% (its smem staging does not address the mixed-tile cost), v3 is a 5-9% loss (a second per-tile verdict on an issue-bound loop). SDPA-flash (torch's stock FA-2 build) tracks the editable fa2 within 2-5%, so the fa2 floor stands. Kept: v1's change (in v4/v5). Dropped from the line: v3's load skipping stays only in the form v5 gives it for free (the prefetched next-tile verdict).

### Track 3 — 2-GPU forward + backward (job 13765479, 2x A100-SXM4-80GB, acc=CPU, itr=1, 4/3 task split)

| N | fwd single | fwd dist | fwd speedup | bwd single | bwd dist | bwd speedup | bwd local + merge | grad rel.err vs single (dq/dk/dv) |
|---|---|---|---|---|---|---|---|---|
| 64K | 0.42 s | 0.51 s | 0.83x | 0.57 s | 0.79 s | 0.71x | 0.57 + 0.15 | 1.8e-5 / 2.7e-5 / 2.9e-5 |
| 256K | 1.28 | 1.41 | 0.91x | 3.76 | 3.42 | 1.10x | 2.71 + 0.66 | 1.8e-5 / 3.0e-5 / 3.8e-5 |
| 1M | 12.71 | 9.93 | 1.28x | 43.72 | 29.66 | 1.47x | 26.87 + 2.58 | 2.2e-5 / 3.5e-5 / 5.1e-5 |

The distributed backward works end to end. The 2-5e-5 gradient differences are not from the sharding: the test hands each backward its own forward's output rounded to fp16 (the two forwards differ at 5e-8, so a few elements round to the other fp16 neighbour, and the rowsum(dO*O) term inherits that); the 1-rank smoke, where both forwards are identical, matched dk/dv bit-for-bit. The backward's own error against a dense fp32 reference is ~1e-3 (fp16 kernel), so this is 50x below it. The gradient all_reduce (3 x fp32 [B,H,N,D], 2.6 s at 1M for 3 x 128 MB... slow because it runs chunk-by-chunk from host memory; a device-resident accumulator (acc=GPU) makes it a pure NVLink reduce) is the only distributed overhead. Bug fixed on the way: the distributed backward call sat inside the rank-0-only block, so rank 1 skipped it and rank 0 hung in the all_reduce (the "collective timeout" of job 13765008); test restructured so every rank runs the distributed backward and only the reference is rank-0-only.

### Track 1 — ncu: the CQS-disabled path is not instruction-heavy, it issues slower (job 13765554)

| kernel / case | duration | instructions | cycles per issued instr |
|---|---|---|---|
| fa2 non-causal | 44.96 ms | 8.32 G | 4.84 |
| CQS plain (CQS off) non-causal | 54.94 ms | 8.25 G | 5.96 |
| CQS zero bits non-causal | 69.24 ms | 10.56 G | 5.86 |
| fa2 causal | 22.39 ms | 3.71 G | 5.20 |
| CQS plain causal | 29.77 ms | 4.06 G | 6.35 |
| CQS zero bits causal | 37.28 ms | 5.22 G | 6.19 |

So the structural 1.22-1.33x of the plain path is an **issue-efficiency** loss at equal instruction count (5.96 vs 4.84 cycles per instruction), i.e. the loop body schedules worse: the runtime-predicated regions around the V load / QK GEMM / K prefetch, the runtime branch between two `softmax_rescale_o` instantiations, and the segmented-address selects break the straight-line block that FA-2's pipeline relies on (the Explore report's items 8-10, 15, 6). The CQS-enabled tax on top is instruction count (+28%: verdicts), which v5 targets. Plan **v6** = v5 + a compile-time `Is_cqs` kernel flag (BOOL_SWITCH on `params.cqs_enabled` at launch): the CQS-off instantiation becomes upstream's straight-line loop, the CQS-on instantiation keeps one `Check_inf=true` softmax instead of a runtime choice between two, and the fp32 `acc_out` store moves up to right after `normalize_softmax_lse` so `acc_o` dies where upstream's does.

### Track 2 — backward: asynchronous scatter is neutral (job 13765584, A100-SXM4-80GB, `pipeline_ab_bwd_*.json`)

`CQSA_SCATTER_ASYNC=1` vs the shipped synchronous scatter, host accumulator, all operands on host, itr=1 causal:

| N | n_par | shipped ms | async ms | speedup | async vs shipped (dq / dk / dv) | vs fp32 SDPA autograd |
|---|---|---|---|---|---|---|
| 64K | 1 | 366.5 | 391.0 | 0.94x | 6.4e-6 / 0 / 0 | 3.0e-4 / 3.0e-4 / 3.2e-4 |
| 64K | 2 | 267.7 | 272.6 | 0.98x | | |
| 256K | 1 | 3170.4 | 3197.6 | 0.99x | 8.2e-6 / 0 / 0 | |
| 256K | 2 | 2990.0 | 2947.3 | 1.01x | | |
| 1M | 1 | 42110.2 | 42083.6 | 1.00x | 1.1e-5 / 0 / 0 | |

The backward is kernel-bound (the bwd kernel is ~3x the fwd per subproblem) and its host scatter is already small, so overlapping it buys nothing. dk/dv bit-identical; dq differs at 1e-5 from the kernel's fp32 atomicAdd order. Decision: the next/ package keeps the shipped default (`CQSA_SCATTER_ASYNC=0`); the code path stays available behind the flag. n_par=2 helps the backward by 6-27% at 64K-256K (two subproblems' h2d/d2h overlap), consistent with the forward.

### Track 3 — 4-GPU, device-resident accumulator (job 13764432, 4x A100-SXM4-80GB, 32 cpus, `dist_test_w4_itr1-2_gpu.json`)

`--acc gpu` (`low_memory=False`): each rank keeps its fp32 accumulator on its own device; the cross-rank merge is then a pure NVLink all_reduce.

| N | itr | tasks/rank | single s | dist s | speedup | local s | merge s | rel.err |
|---|---|---|---|---|---|---|---|---|
| 1M | 1 | 2,2,2,1 | 12.92 | 5.06 | 2.55x | 4.86 | 0.16 | 3.3e-8 |
| 2M | 1 | 2,2,2,1 | 50.62 | 16.91 | 2.99x | 16.59 | 0.22 | 3.2e-8 |
| 4M | 1 | 2,2,2,1 | 201.15 | 62.27 | 3.23x | 61.61 | 0.42 | 3.1e-8 |
| 1M | 2 | 13,12,12,12 | 19.89 | 6.87 | 2.90x | 6.45 | 0.11 | 8.7e-8 |
| 2M | 2 | 13,12,12,12 | 65.95 | 20.11 | 3.28x | 18.92 | 0.22 | 8.4e-8 |
| 4M | 2 | 13,12,12,12 | 243.36 | 68.77 | 3.54x | 66.66 | 0.43 | 8.3e-8 |

Against the best single-device configuration at each N (itr=1): 4M forward 201 s -> 62 s on 4 GPUs (3.23x; the 2/7 shard bound is 3.5x), and itr=2's 68.8 s is 2.9x. Exact at every point (fp32 rounding level). This is the recommended multi-device configuration: acc=GPU when the shard's fp32 output fits (it is 1/world of the single-device footprint... the accumulator is still full-N per rank in this version, see next), itr chosen so c^itr divides across world_size, `shared_chunks=True` at itr=1.
Remaining inefficiency: each rank still allocates a full-N accumulator and merges full-N chunks; a sharded output (`output="sharded"`, present in the API) plus a reduce_scatter instead of all_reduce would cut per-rank memory and traffic by world_size. Implemented next only if time permits; the correctness path is the same.

### Track 1 — v4 and v5 built and verified (interactive A100-40GB, idle node)

Bit-identical to shipped on all 32 (shape x pattern x causal) cases. ms/call shipped / v1 / v4 / v5:

| L | pattern | causal | non-causal |
|---|---|---|---|
| 56175 | real | 31.64 / 30.05 / 29.92 / **28.66** | 57.10 / 56.57 / **52.93** / 53.82 |
| 56175 | zero | 29.62 / 28.68 / **26.75** / 28.70 | 54.02 / 53.78 / **49.36** / 55.71 |
| 56175 | all-masked | 26.71 / 26.85 / 23.77 / **22.25** | 34.39 / 34.59 / 29.87 / **26.25** |
| 42858 | real | 19.30 / 18.67 / 18.96 / **17.65** | 35.37 / 35.37 / 33.75 / **33.70** |
| 24075 | real | 7.02 / 6.91 / 6.87 / **6.61** | 9.83 / 9.82 / 9.33 / **9.26** |

- v4 (row half hoisted): live tiles 9-10% faster than shipped (`zero`), all-masked 11-13% faster.
- v5 (compile-time block size + column summaries prefetched a tile ahead): all-masked improves further (22.25 / 26.25: a skipped tile is now ~45% of a live one instead of 63%), and the real pattern is the best so far causal (28.66, -9.4% vs shipped), but live tiles regress relative to v4 (`zero` 28.70 vs 26.75): the loop-carried summary struct costs registers in a 255-register loop, and the prefetch runs the general (looping) column path every iteration.
- **v7** (built from v6): fast path in `cqs_col_summary_c` for a full aligned tile (every tile but the ragged last): one 16-byte load per summary array, no loop, no clamps; falls back to the general path otherwise, and returns exactly what it returns. Building. v6 (compile-time CQS flag, single softmax, early fp32 store) building too.
Clean-node A/B for shipped/v1/v4/v5 submitted (job below).

### Regression — package test suite against the next/ engine

`next/pkg` (pipelined merge on, task_subset, scatter default reverted): **278 passed** with the shipped kernel. With `CQSA_CUDA_MODULE=cqsa_cuda_next_v5`: every fp16 test passes (266); the 12 that fail or are deselected are all bf16 parametrisations, because the variants are built with `CQSA_KERNEL_SET=a100_fp16_hdim64_128` (fp16 only, ~35 min per build) while the shipped binary carries fp16+bf16. A release build of the chosen variant must use the shipped kernel set (`common`).

### Track 1 — clean-node A/B, shipped / v1 / v4 / v5 (job 13766639, A100-80GB PCIe, `kernel_ab_clean_v5.json`)

ms/call on the real bits; ratio to shipped in parentheses; all bit-identical.

| N | causal | fa2 | shipped zero / mask | v1 mask | v4 zero / mask | v5 zero / mask |
|---|---|---|---|---|---|---|
| 16K | yes | 0.42 | 0.80 / 1.37 | 1.33 (0.974) | 0.71 / 1.35 (0.989) | 0.77 / 1.35 (0.986) |
| 32K | yes | 1.31 | 2.26 / 3.47 | 3.38 (0.975) | 2.03 / 3.43 (0.988) | 2.19 / 3.37 (0.970) |
| 64K | yes | 4.79 | 7.70 / 9.68 | 9.45 (0.977) | 6.95 / 9.19 (0.949) | 7.51 / 8.87 (0.916) |
| 131K | yes | 19.54 | 28.37 / 30.83 | 29.68 (0.963) | 25.76 / 29.09 (0.944) | 27.75 / **27.87 (0.904)** |
| 131K | no | 39.24 | 52.05 / 55.78 | 55.73 (0.999) | 47.35 / 51.83 (0.929) | 53.26 / 52.04 (0.933) |

v5 is the best on the real workload at the sizes that matter (-8..10% at 64K-131K) and v4 is the best on all-clear tiles (-9% `zero`); v5's live-tile regression relative to v4 is the general (looping) column path executed every iteration for the prefetch. v7 (vectorised, loop-free column path on top of v6) is the candidate that keeps both. At 16K-32K the mixed-tile per-element path dominates and none of the variants move it; that path is the remaining item for deep-itr (small-L) subproblems.

### Track 1 — v6 and v7 built and verified (interactive A100-40GB, idle node): the structural gap is closed, but the non-causal CQS-on instantiation spills

Bit-identical to shipped in every case (CQS on: 4 patterns x 2 causal x 3 shapes; CQS off: 2 x 3). ms/call:

| L | case | shipped | v5 | v6 | v7 | fa2 |
|---|---|---|---|---|---|---|
| 56175 | **plain (CQS off)**, causal | 26.60 | 26.42 | **20.94** | **20.77** | 24.42 |
| 56175 | plain, non-causal | 47.38 | 49.27 | **39.68** | **39.84** | 41.50 |
| 42858 | plain, causal | 15.92 | 15.89 | **12.49** | **12.50** | 12.16 |
| 56175 | real, causal | 31.86 | 28.87 | 28.10 | **27.51** | |
| 56175 | zero, causal | 29.88 | 29.26 | 28.91 | 28.30 | |
| 56175 | all-masked, causal | 26.64 | 22.29 | 21.98 | 21.32 | |
| 56175 | real, **non-causal** | 57.36 | 53.81 | 178.19 | 176.90 | |
| 56175 | zero, non-causal | 54.06 | 55.11 | 211.73 | 210.34 | |

1. **The compile-time `Is_cqs` flag + early fp32 store makes the CQS-off path as fast as upstream FA-2** (20.8 vs 24.4 ms causal on this node, 39.8 vs 41.5 non-causal): the 1.22-1.33x structural gap identified by ncu is gone. That confirms the diagnosis (issue efficiency lost to runtime-predicated regions, the duplicated softmax and the late fp32 store), and it is the answer for anyone using the CQS build as a drop-in FA-2.
2. Causal CQS-on: v7 is the best variant so far, **-13.6% vs shipped on the real pattern** (27.51 vs 31.86), all-masked -20%.
3. **Non-causal CQS-on is 3.3x slower in v6/v7**: `cuobjdump` shows the non-causal `Is_cqs=true` instantiations at STACK 600-664 B (v5: 344-376 B), the causal ones at 432-456 B; ptxas spills the hot loop in exactly those instantiations. Not a semantic issue (outputs are bit-identical), a register-allocation one: making the flag constexpr changed the code enough that the non-causal specialisation, which has a single masking step and therefore a differently shaped loop, no longer fits in 255 registers.
Isolation builds in flight: **v8** = v5 + v7's vector column loads (no template flag, no early store); **v9** = v8 + early fp32 store. If both are spill-free in every instantiation, the template flag is what tips the non-causal allocation and the final kernel can use a three-way mode (off / on / runtime) chosen per instantiation. Clean-node A/B with v5/v6/v7 and an ncu run on v7 submitted.

### Track 1 — clean-node A/B, shipped / v5 / v6 / v7 (job 13767807, A100-SXM4-80GB, `kernel_ab_clean_v7.json`) and ncu of v7 (job 13767808)

| N | causal | fa2 | shipped plain / mask | v5 mask | v6 plain / mask | v7 plain / mask | v7/shipped |
|---|---|---|---|---|---|---|---|
| 64K | yes | 5.05 | 7.14 / 9.51 | 8.75 | 5.12 / 9.05 | 5.12 / 8.54 | **0.899** |
| 131K | yes | 18.66 | 24.47 / 30.23 | 27.10 | 17.22 / 26.46 | 18.99 / 25.82 | **0.854** |
| 131K | no | 34.68 | 41.54 / 53.38 | 49.50 | 33.49 / 176.2 | 33.93 / 175.4 | 3.285 |

- Causal, CQS on: v7 is 14.6% faster than shipped at 131K and 10% at 64K, bit-identical. Plain (CQS off) now equals FA-2 (18.99 vs 18.66; 33.9 vs 34.7 non-causal).
- Non-causal, CQS on: ncu shows the v6/v7 instantiation executes **24.0 G instructions** for the zero pattern where v3/v5 execute 10.6 G, at 10.3 cycles per issued instruction (vs 5.9). No local-memory spill traffic is reported, so the STACK figure was a red herring; the non-causal `Is_cqs=true` specialisation is generating ~2.3x the instructions per tile. The causal specialisation of the same source is fine (4.39 G for zero, vs 5.22 G in v3: -16%). Cause to be isolated with v8/v9 (no template flag).

### Track 3 — two nodes, 8 GPUs (job 13766375, 2x della-l09g nodes, 4x A100-SXM4-80GB each, acc=GPU, itr=2, 49 tasks -> 7/6 per rank)

| N | single s | dist s | speedup | local s | merge s | rel.err |
|---|---|---|---|---|---|---|
| 2M | 55.83 | 11.78 | 4.74x | 10.43 | 0.70 | 6.5e-8 |
| 4M | 206.10 | 40.42 | 5.10x | 34.16 | 4.97 | 6.5e-8 |

Multi-node works unchanged (c10d rendezvous, `slurm/run_multinode.slurm`); exact. The shard bound is 7x (7 of 49 tasks per rank); the gap is the inter-node merge (5 s at 4M: the all_reduce now crosses the InfiniBand fabric for 3 x 4 GiB per chunk sequence) and per-rank task-size variance. A reduce_scatter to a sharded output would cut the merge traffic by 8x.

### Track 1 — v8 / v9 (interactive, idle node): the template flag is what breaks the non-causal instantiation

v8 = v5 + vector column loads; v9 = v8 + early fp32 store. Both bit-identical to shipped in every case, both fine in non-causal. ms/call shipped / v5 / v8 / v9:

| L | case | causal | non-causal |
|---|---|---|---|
| 56175 | real | 31.87 / 28.85 / **27.80** / 27.98 | 57.58 / 53.93 / **52.79** / 52.71 |
| 56175 | zero | 29.92 / 29.24 / 28.51 / 28.53 | 54.18 / 55.86 / 54.85 / 54.75 |
| 56175 | all-masked | 26.93 / 22.29 / 21.62 / 21.49 | 34.29 / 26.26 / 24.83 / 24.92 |
| 42858 | real | 19.43 / 18.19 / **17.60** / 17.71 | 35.74 / 33.26 / 33.02 / 33.13 |
| 56175 | plain | 26.77 / 26.52 / 26.23 / 26.40 | 47.47 / 49.24 / 48.97 / 49.05 |

The vector column loads recover v5's live-tile regression (v8 real = -12.8% vs shipped causal, -8.3% non-causal). The early store alone (v9) does nothing on the CQS-on path and nothing for plain: the plain-path gain in v6/v7 comes from the compile-time flag. Final design **v10**: a three-way `CqsMode` template (0 off, 1 on, 2 runtime) dispatched per instantiation: causal -> 0/1 (v7 behaviour: plain == FA-2, real -14%), non-causal -> 2 (v9 behaviour), so no instantiation is the pathological one. Building.

Regression with `CQSA_CUDA_MODULE=cqsa_cuda_next_v8`: 266 fp16 tests pass, the 4 failures are the bf16 parametrisations of `test_bwd_host_stream` (kernel-set limitation, as above). End-to-end comparison (SDPA / FA-2 / shipped Stream-CQSA / next engine + v8 kernel, N = 256K, 1M, 2M, one A100) submitted as job 13769036.

### Track 1 — clean-node A/B, shipped / v7 / v8 / v9 (job 13768943, A100-SXM4-80GB, `kernel_ab_clean_v9.json`), and SASS sizes

| N | causal | fa2 | shipped mask | v7 mask | v8 mask | v9 mask | best/shipped |
|---|---|---|---|---|---|---|---|
| 64K | yes | 5.06 | 9.55 | 8.31 | 8.48 | 8.50 | 0.871 (v7) |
| 131K | yes | 18.61 | 30.26 | 25.79 | 26.40 | 26.28 | 0.852 (v7) |
| 64K | no | 9.80 | 16.42 | 47.02 | 14.86 | 14.80 | 0.902 (v9) |
| 131K | no | 36.02 | 53.39 | 175.55 | 48.45 | 48.36 | 0.906 (v9) |

v8/v9 are the best non-causal kernels (-9.5%) and within 2% of v7 on causal; v7 is the best causal (-15%) and the only one whose CQS-off path equals FA-2. v10 (three-way mode) combines them; building.

SASS sizes (v7, hdim64, `cuobjdump -sass`): the CQS-off instantiations are 3.7-7.8K instructions, the CQS-on ones 74-78K — twenty times larger, because `apply_mask`'s per-element CQS path (three test modes, 64 scores per thread, fully unrolled) is instantiated in both KV loops. It runs only on mixed tiles, so it is not on the live-tile path, but it is why `zero` (CQS on, all tiles clear) is still +43% over `plain` in v7 even though the per-tile verdict is now a register AND: the remaining difference on live tiles is structural — the `if (!cqs_skip_block)` regions around the V load / QK GEMM and the K prefetch, the `continue`, and the loop-carried prefetch struct — exactly the "issue efficiency" pattern ncu showed for the plain path before v6.

**v11** therefore restructures the steady loop as a loop over *live* tiles: the body is upstream's straight-line block (no predicated regions, no `continue`), and fully masked tiles are skipped by a search for the next live tile that runs while the QK GEMM is in flight and steers the K prefetch. Live tiles are visited in the same order, so the accumulation order (and the result) is unchanged. The v2 smem staging is dropped (measured useless). Building alongside v10.

### End-to-end, one A100-SXM4-80GB (job 13769036, `e2e_compare_v8.json`): SDPA / FA-2 / shipped Stream-CQSA / next Stream-CQSA

B=1 H=8 D=64 fp16 causal, itr=1. FA-2 and SDPA (default backend = flash) hold Q/K/V on the device; both Stream-CQSA arms stream pinned host Q/K/V with the host accumulator (the paper's recovery configuration), n_par=1. "next" = pipelined merge + shared_chunks + v8 kernel. Best of 2 timed reps after a warm-up; peak = `max_memory_allocated`.

| N | FA-2 | SDPA | shipped Stream-CQSA | next Stream-CQSA | next/shipped | rel.err vs FA-2 |
|---|---|---|---|---|---|---|
| 256K | 0.36 s / 1.3 GiB | 0.38 s / 1.3 GiB | 2.05 s / 1.1 GiB | 0.94 s / 1.3 GiB | 2.2x | 2.2e-4 (both) |
| 1M | 5.92 s / 5.0 GiB | 6.27 s / 5.0 GiB | 33.59 s / 4.6 GiB | 13.35 s / 5.3 GiB | 2.5x | 3.0e-4 (both) |
| 2M | 23.93 s / 10.1 GiB | 25.31 s / 10.1 GiB | 86.19 s / 9.2 GiB | 43.26 s / 10.6 GiB | 2.0x | 4.5e-4 (both) |

Caveat under investigation: the shipped arm here is ~2x slower than the same configuration measured earlier by `pipeline_baseline.py` (15.6 s at 1M) and by the A/B's flag-0 arm (16.4 s), while the next arm agrees with its earlier measurements (11.9-14.3 s). The two candidates are pinned inputs (e2e pins Q/K/V; the baselines did not) and the per-rep `reset_peak_memory_stats`/`empty_cache`; a same-node diagnostic (job 13778356) times the shipped engine both ways. Until it lands, read the next/shipped ratio as 1.15-1.4x (the A/B numbers), not 2-2.5x. The accuracy column is unaffected (identical errors, both exact to fp16 rounding).

Diagnostic (job 13778356, A100-80GB PCIe): the shipped engine at 1M takes 19.1-19.2 s pinned and 19.0-20.0 s unpinned, with e2e's exact kwargs and per-rep allocator resets. So neither pinning nor the wrapper explains 33.6 s; the e2e job's node (SXM4, 16 cpus) must have had host-side contention during the shipped arm (its CPU gather and blocking merge are far more sensitive to that than the pipelined engine, which ran normally on the same node minutes later). `e2e_compare.py` now records hostname and 1-minute load average before/after each arm; the table will be re-measured with the final kernel and only accepted if the shipped arm agrees with its 16-19 s baselines.

### Track 1 — v10/v11 first builds hung in ptxas (5+ h on `flash_fwd_hdim128_fp16_sm80`)

The three-way `CQS_MODE_SWITCH` instantiated mode 1 (compile-time CQS on) for every kernel, including the non-causal ones it was designed to avoid; at hdim64 that instantiation is merely 2.3x slower, at hdim128 ptxas never finished it. Fixed: the switch takes `Is_causal` and only ever instantiates modes {0,1} for causal and {0,2} for non-causal kernels. Both builds restarted from clean (07:58).

### Track 1 — v10 and v11 built (08:43 / 08:39) and verified on the interactive A100-40GB (idle)

Bit-identical to shipped in every case (now also a "half" pattern with long masked runs and a ragged tail). ms/call, shipped / v7 / v8 / v10 / v11 (fa2 for plain):

| L | case | causal | non-causal |
|---|---|---|---|
| 56175 | plain (fa2 28.3 / 41.0) | 26.7 / 20.9 / 26.4 / 20.9 / **20.9** | 47.6 / 39.4 / 49.0 / 40.0 / **39.4** |
| 56175 | real | 31.6 / 27.4 / 27.7 / 27.6 / **26.2** | 56.6 / 177 / **52.9** / 192 / 193 |
| 56175 | zero | 29.9 / 28.4 / 28.6 / 28.4 / **27.0** | 54.1 / 210 / **54.6** / 236 / 237 |
| 56175 | all-masked | 26.8 / 21.3 / 21.7 / 21.2 / **19.2** | 34.4 / 33.4 / **24.8** / 33.9 / 29.9 |
| 24075 | real | 7.62 / 6.35 / 6.52 / — / **5.86** | 10.1 / 31.3 / **9.11** / — / 33.3 |
| 42858 | real | 20.1 / 17.5 / 17.6 / 17.3 / **16.5** | 35.7 / 106 / **33.1** / 115 / 115 |

- **v11 is the best causal kernel at every shape: -17% (L=56K), -18% (L=43K), -23% (L=24K) vs shipped**, and its CQS-off path equals FA-2. The live-tile loop pays exactly as predicted (all-masked 19.2 vs 26.8, live tiles 27.0 vs 29.9).
- Non-causal CQS-on is pathological in v10 *and* v11 (3.4x), even though v10's non-causal path is mode 2 = v9's code with a runtime flag, and v9 itself is fine (52.7). So the trigger is not the compile-time flag; it is having the CQS-off instantiation compiled in the same translation unit / the `Cqs_mode` template shape, which changes ptxas's decisions for the non-causal CQS-on kernel (STACK 616-680 B vs ~350 B; ncu on v7 showed 2.3x executed instructions, i.e. a register fragment demoted to local memory, not a spill). Causal instantiations of the same source are unaffected. ncu on v11 submitted to confirm (job 13780907).
- Decision: dispatch by `causal` in the Python interface (`CQSA_CUDA_MODULE` for causal calls, `CQSA_CUDA_MODULE_NONCAUSAL` for non-causal, default = same module): **v11 for causal (the workload) and v9 for non-causal**, both bit-exact. A single-binary fix is a ptxas investigation (`-lineinfo` + source-level ncu) left as follow-up.

Interface dispatch implemented (`next/pkg/stream_cqsa/interface.py`): `CQSA_CUDA_MODULE` serves causal calls, `CQSA_CUDA_MODULE_NONCAUSAL` (default: same) serves non-causal ones; only the two forward entry points that take `causal` dispatch, the backward is unchanged. Verified: with v11/v9 the same process runs causal at 28.8 ms and non-causal at 52.2 ms (interactive node) at L=56K. Test suite with that pairing: 266 fp16 tests pass (bf16: kernel-set limitation). Final end-to-end comparison with v11 submitted (job below), with host load recorded per arm.

ncu on v11 (job 13780907): causal `zero` executes **4.03 G instructions** (v3: 5.22 G; upstream FA-2 causal: 3.71 G) — the live-tile loop brought the CQS-on causal kernel to within 9% of FA-2's instruction count, which is the whole story of its speed. Non-causal `zero`: 23.6 G instructions, L1 hit rate 35% (vs 62% causal), 11.3 cycles per issued instruction: the signature of a register array demoted to local memory in that instantiation (consistent with its STACK 616-680 B). Left for the ptxas follow-up; the interface dispatch sidesteps it.

### Track 1 — FINAL clean-node kernel A/B, shipped / v9 / v10 / v11 (job 13780947, A100-SXM4-80GB, `kernel_ab_clean_v11.json`)

ms/call at the itr=1 subproblem shape (L=3N/7), H=8 D=64 fp16, real path-(0,) bits; every variant bit-identical to shipped.

| N | causal | FA-2 | SDPA-flash | shipped | v9 | v11 | v11/shipped | v11/FA-2 |
|---|---|---|---|---|---|---|---|---|
| 16K | yes | 0.47 | 0.49 | 1.51 | 1.48 | **1.36** | 0.901 | 2.9 |
| 32K | yes | 1.44 | 1.56 | 3.60 | 3.66 | **3.35** | 0.931 | 2.3 |
| 64K | yes | 5.06 | 5.50 | 9.56 | 8.52 | **8.01** | 0.838 | 1.58 |
| 131K | yes | 17.46 | 18.36 | 30.22 | 26.28 | **24.43** | **0.808** | 1.40 |
| 16K | no | 0.77 | 0.81 | 2.53 | **2.47** | (5.48) | 0.976 (v9) | |
| 64K | no | 9.81 | 10.43 | 16.58 | **14.79** | (48.5) | 0.892 (v9) | |
| 131K | no | 34.20 | 36.18 | 53.38 | **48.32** | (183.7) | 0.905 (v9) | |

CQS-off (plain) columns: v11 18.91 / 33.48 vs FA-2 17.46 / 34.20 — the CQS build is no longer slower than upstream when CQS is off (shipped: 23.16 / 40.95).
Summary of the kernel track: on the real causal workload the forward kernel is **19% faster than shipped at L=131K-scale subproblems (16% at 64K)** and now within 1.4x of a monolithic FA-2 call on the same L, of which ~0.78x is the unavoidable work (22% of tiles are skipped); the remaining gap on live tiles is the `Check_inf` softmax, the per-tile `apply_mask` entry with `!Is_even_MN` column masking, and the 10x larger code footprint of the CQS-on instantiation. Non-causal: v9, -10%.

### FINAL end-to-end, one A100-SXM4-80GB (job 13780966, della-l07g7, load 4-7 throughout, `e2e_compare_v11.json`)

B=1 H=8 D=64 fp16 causal, itr=1. FA-2 / SDPA hold Q/K/V on the device; both Stream-CQSA arms stream pinned host Q/K/V with the host accumulator, n_par=1. "next" = pipelined merge + shared_chunks + v11 kernel. Best of 2 timed reps; the shipped arm now agrees with its earlier baselines (15.5 s at 1M), so this table supersedes the first one.

| N | FA-2 | SDPA (flash) | shipped Stream-CQSA | next Stream-CQSA | speedup | next / FA-2 | rel.err vs FA-2 |
|---|---|---|---|---|---|---|---|
| 256K | 0.36 s / 1.3 GiB | 0.39 s | 2.17 s / 1.1 GiB | **1.17 s** / 1.3 GiB | **1.86x** | 3.2x | 2.2e-4 (both) |
| 1M | 5.96 s / 5.0 GiB | 6.31 s | 15.51 s / 4.6 GiB | **9.49 s** / 5.3 GiB | **1.63x** | 1.59x | 3.0e-4 (both) |
| 2M | 24.18 s / 10.1 GiB | 25.66 s | 50.97 s / 9.2 GiB | **35.31 s** / 10.6 GiB | **1.44x** | 1.46x | 4.5e-4 (both) |

Where the 1.63x at 1M comes from (from the stage breakdowns): the pipelined merge/d2h overlap (~1.15x), shared_chunks gather (~1.09x), and the v11 kernel (~1.19x on the compute stage). Stream-CQSA's remaining cost over a monolithic FA-2 call (1.5-1.6x, was 2.1-2.6x) is now mostly the c=7 decomposition itself (7 subproblems of 3N/7 tokens = 1.29x the FLOPs of the monolithic call... 9/7 of the pairs) plus the host stages that cannot overlap. Peak device memory rises by 0.2-0.7 GiB (the shared_chunks device pool). Accuracy identical.

### Open items (not done)
- Non-causal CQS-on instantiation of v10/v11 mis-compiles (served by v9 through the interface dispatch); a single-binary fix needs a ptxas investigation with `-lineinfo`.
- Variants are fp16-only builds; a release build of v11/v9 needs `CQSA_KERNEL_SET=common` (fp16+bf16, ~2x the build time) and the bf16 tests re-run.
- Multi-device: each rank still allocates a full-N accumulator and the merge is an all_reduce; a sharded output with reduce_scatter would divide per-rank memory and traffic by world_size. Backward runs distributed only with `allow_escalation=False` semantics validated (escalation under a shard is implemented but untested).
- The mixed-tile per-element path (5.5x a live tile) is untouched; it matters only for small-L subproblems (N <= 64K).

## Phase 2 (2026-09-12): auto-configuration, developer kit, adapters, kernel technical note

New modules in `next/pkg/stream_cqsa/`:
- `autoconfig.py` — `HardwareSpec` (from `detect_hardware()` or `hardware_from_dict({"cuda:0": "40GiB", "cuda:1": ..., "host": "256GiB", "link_gbs": 200})`), a `CostModel` (pair throughput + per-token host stages + depth factor + distributed balance/merge/fixed terms; defaults from the SXM4 measurements, `calibrate()` refits on the machine in ~1 min), `plan()` enumerating monolithic / itr 1-3 / acc gpu|cpu / host-resident / n_par 1,2,4 / world 1..n_devices, memory from the engine's own estimators, then the rule "Pareto frontier of (time, memory) among feasible, fastest point, ties to less memory"; `autotune()` measures the top candidates instead; `auto_attention(q,k,v, hardware=...)` plans and runs (distributed only inside a matching process group). Heterogeneous machines: plans on the largest identical-device group.
- `devkit.py` — sampled-float64 reference (the paper's), `measure` (time + peak/workspace memory), `Config`/`run_config`, `compare_kernels(inner_fn, mono_fn)` with the verdict bit-identical / exact-within-rounding / NOT exact (relative to the monolithic kernel's own fp64 gap, or a dtype rounding floor when no fp64 reference applies), and `quick_bench()` sweeping configurations with the same rule.
- `adapters.py` — `flex_inner(score_mod, extra_mask_mod)` (FlexAttention → inner kernel; the engine now passes the gather index as `token_ids` and the adapter remaps positions, which is the one thing automatic conversion must do), `dense_inner` (+ second lse pass for kernels without lse).
- `distributed.py` — the multi-device driver moved into the package.
- Engine: `local_stats_flash` accepts `token_ids`/extra kwargs (a regression where the scheduler's full call fell back to the bare call and the shared_chunks path lost `block_base` — wrong results, rel err 1.1-1.45 — was introduced and caught by `quick_bench` within the hour; fixed).

Cost model vs measurements (SXM4, next engine): 1M mono 5.94 (5.96), 1M itr1 cpu/host 8.8 (9.49), 2M 34.3 (35.3), 2M itr2 gpu 54 (66), 4M x4 52 (62), 2M x2 itr2 38 (36.5). Interactive PCIe-40GB calibration: pair_rate 4.98e11/s, kernel_ratio 1.30, depth_factor 1.45 at N=131K (`logs/costmodel_a100_pcie40_interactive.json`).

Planner decisions (default model): 40 GiB budget -> monolithic up to 4M (fits, fastest); 3 GiB budget at 1M -> itr=3 acc=cpu host-resident (2.4 GiB); 80 GiB at 16.7M -> itr=1 acc=cpu host-resident; 4x80 GiB -> distributed from 1M up (est 3.7 s vs 5.9 s mono), monolithic at 256K (fixed collective cost); 2x80 GiB -> monolithic (4/7 shard bound + merge loses to one FA-2 call, consistent with the 2-GPU measurements).

quick_bench on the interactive PCIe-40GB (`logs/quick_bench_{65536,262144}_interactive.json`): monolithic wins whenever it fits (0.039 vs 0.074 s best cqsa at 64K; 0.53 vs 0.68 s at 256K); decomposed results are MORE accurate vs float64 (2.0e-4 vs 3.1e-4, fp32 merge); acc=cpu host is 5-8x slower at these N (host merge dominates), acc=gpu n_par=2-4 is the best decomposed setting.

Adapters verified with compare_kernels (N=16K): Flex plain causal exact at itr 1/2 (2.6e-4 = FA-2's own fp16 error); ALiBi score_mod exact (1.8e-4) with global positions and NOT exact (6.5e-2) without — the check catches it; sliding window 2048 exact (3.8e-4); SDPA+dense mask (no lse) exact via the second pass. Flex inner is 8-10x slower than the native kernel at subproblem sizes.

Tests: `tests/test_devkit_autoconfig.py` (6) + the 266 fp16 engine tests pass with v11/v9.
Docs: `next/docs/kernel_technical_note.md` (kernel design, evidence, measurements, methodology for the paper), `next/docs/design_kernel_suite_and_conversion.md` (why no linear/delta-rule suite; automatic conversion via FlexAttention + KernelSpec resolver design).
Jobs: clean-node devkit run (13791534: calibrate + compare_kernels + quick_bench at 64K/256K/1M + autotune) and 2-GPU `auto_attention` test (13791535).

2-GPU `auto_attention` test (job 13791535, 2x A100-80GB, hardware dict {cuda:0,1: 72GiB, host 150GiB, link 200}): the planner chose the monolithic call at 1M and 2M (fits; 2 devices cannot beat it under the 4/7 shard bound), ran it on rank 0's device, rel.err 3.1e-4 / 4.6e-4 vs the fp32 decomposed reference (fp16 output rounding). Timings on that node were 2.3x the SXM4 e2e numbers for every arm alike (unpinned host inputs copied inside the timed region, first-launch warm-up, node), so they say nothing about the model; the decision is what was under test. A 4-GPU run would exercise the distributed branch (planner picks it from 1M up); left for the next gpu-test slot.

### Clean-node devkit run (job 13791534, A100-SXM4-80GB, 16 cpus, `logs/quick_bench_clean.json`)

Calibration at 131K: pair_rate 6.03e11/s, kernel_ratio 1.50, host gather 173 ns/tok, host merge 1024 ns/tok, d2h 531 ns/tok, device merge 24 ns/tok, itr2/itr1 1.90 (depth_factor 1.46).

Native v11 kernel inside Stream-CQSA vs flash-attn monolithic (`compare_kernels`, same inputs), all verdicts "exact within rounding"; the decomposed result is closer to float64 than the monolithic fp16 call:

| N | config | CQSA vs mono rel | mono vs fp64 | CQSA vs fp64 | time ratio | memory ratio |
|---|---|---|---|---|---|---|
| 64K | itr1 acc=gpu np2 | 2.2e-4 | 2.8e-4 | 1.9e-4 | 2.12x | 2.48x |
| 256K | itr1 acc=gpu np2 | 2.3e-4 | 3.1e-4 | 2.1e-4 | 1.58x | 2.47x |
| 1M | itr1 acc=gpu np2 | 3.1e-4 | 5.5e-4 | 2.8e-4 | **1.35x** | 2.48x |
| 1M | itr1 acc=cpu host np1 | 3.1e-4 | 5.5e-4 | 2.8e-4 | 1.54x | 1.25x |
| 1M | itr2 acc=gpu np2 | 3.7e-4 | 5.5e-4 | 2.3e-4 | 1.41x | 2.02x |

quick_bench sweep (20 configurations x 3 sizes, 70 GiB budget): monolithic FA-2 is the rule's pick at every N (6.12 s at 1M vs the best decomposed 8.03 s = itr1 acc=gpu n_par=4, and 9.25 s for the host-accumulator recovery configuration); the calibrated planner predicted the same pick at all three sizes (est 0.03 / 0.46 / 7.30 s vs measured 0.034 / 0.405 / 6.12 s), and `autotune` at 256K under a 31 GiB budget measured its way to the same answer. The paper's "1.5-1.9x below the boundary" is now 1.31-1.56x with the next engine + v11 on the same hardware class.

## Phase 3 (2026-09-12 afternoon): quorum-set axis, independent backward depth, demo notebook, v2 repo, paper re-runs

- **Quorum-set axis.** `autoconfig.QUORUM_SETS` = the perfect difference sets c=3,7,13,21,31,57,73 (validated); the planner enumerates every (c, itr) with pair work (l^2/c)^itr, masked fraction (l-1)/l^2, c^itr tasks x a per-task overhead (fitted from c=7 vs c=31 in `calibrate`), gathered tokens l^itr N for the host stages. Measured (job 13792976, A100-SXM4-80GB, N=1M, acc=GPU host-resident inputs, n_par=2, all exact):

| c | l | itr | tasks | L | time s | peak GiB |
|---|---|---|---|---|---|---|
| 3 | 2 | 1 | 3 | 699K | 8.56 | 13.9 |
| 7 | 3 | 1 | 7 | 449K | 8.45 | 11.1 |
| 13 | 4 | 1 | 13 | 323K | 8.54 | 9.1 |
| 21 | 5 | 1 | 21 | 250K | 8.52 | 8.0 |
| 31 | 6 | 1 | 31 | 203K | 8.96 | 7.3 |
| 57 | 8 | 1 | 57 | 147K | 9.41 | 7.0 |
| 73 | 9 | 1 | 73 | 129K | 9.68 | 6.9 |
| 7 | 3 | 2 | 49 | 193K | 9.93 | 6.2 |
| 13 | 4 | 2 | 169 | 99K | 11.49 | 6.2 |

  Confirms the premise: for the same subproblem size a larger c at itr=1 beats a smaller c at itr=2 (c=31/itr1 8.96 s vs c=7/itr2 9.93 s at ~7 GiB; c=73/itr1 9.68 s vs c=13/itr2 11.49 s), and at itr=1 time is nearly flat in c while memory falls with it, so the planner's frontier now moves along c before it moves along itr. The model's ordering matches; absolutes are ~25% high with the 131K calibration (fixed overheads), which does not change the choice.
- **Independent forward/backward decomposition.** `stream_cqsa_backward(itr="auto")` plans its own depth from free memory with a backward memory model (`estimate_peak_bytes_bwd`, `plan_decomposition_bwd`); `stream_cqsa_attn(..., bwd_itr="auto"|"fwd"|int)` (default "auto") no longer inherits the forward's depth; the planner's `plan(direction="bwd")` uses the backward estimator and a 3x cost. Under a 3 GiB budget at 1M the forward plans c=13 itr=2 and the backward c=73 itr=1.
- Planners honour a `torch.cuda.set_per_process_memory_fraction` cap (`effective_free_bytes`), so a memory cap is a faithful small-device simulation (used by the demo notebook).
- Engine `local_stats_flash` accepts `token_ids`/extra kwargs (regression fixed). Tests: 8 in `test_devkit_autoconfig.py` + 266 fp16 engine tests pass.
- fp16+bf16 (`common`) builds of v11 and v9 started 14:49 on della-vis1 for the paper re-runs (bf16 rows) and the v2 repo.

- **v2 repository** pushed: https://github.com/yiming-b/Stream-CQSA-v2 (commit 67277a3): package (`stream_cqsa/`, v2.0.0, exports the planner/devkit/adapters), both kernel source trees (`csrc/` = v11, `csrc_nc/` = v9; `setup.py` builds `cqsa_cuda` and `cqsa_cuda_nc`, the interface serves causal/non-causal from them), tests, benchmarks, distributed tests, slurm templates incl. the paper re-run jobs, docs (technical note, design note, LOG, kernel diffs), results (all JSON + key slurm logs). The repo tree passes the suite from its own directory (274 fp16 tests; the 4 bf16 failures are the fp16-only local builds). The demo notebook is added once its execution finishes.
- `common` (fp16+bf16, hdim 64/128) builds of v11 and v9 finished 16:04 (75 min each, concurrent); the FULL suite now passes: **286 passed** (bf16 included). Frozen copies of the earlier fp16-only .so kept in `next/kernel/stable/` for jobs that were already queued.
- Paper experiments re-run with the v2 kernels + engine (`next/slurm/paper/{small,accuracy,mid,large,xl}.slurm`, outputs `outputs/next/paper/results/results.jsonl`, methods sdpa/sdpa_flash/sdpa_mem/flash/cqsa_accgpu/cqsa_acccpu, itr=auto): small = the ladder 8K-2M x {fp16,bf16} x {fwd,bwd} x 6 methods in one job (12 h); accuracy = the 8K many-seed table with the dense fp64 backward reference; mid = 4M fwd+bwd and 8M fwd, all methods; large = 8M bwd + 16.7M fwd (Stream-CQSA only, fp16); xl = 16.7M bwd (acc=CPU). All on gpu-short (24 h cap); queued behind ~1800 jobs.
- Paper re-run, accuracy job (13795508, 5 min, 120 rows, A100-80GB): at N=8192 every method, Stream-CQSA included, has the same forward error vs float64 (fp16 2.69e-4, bf16 2.16e-3) because the planner runs the monolithic call there (itr=0), exactly as in the paper's table; the backward rows carry acc_rel_dq/dk/dv. Results: `outputs/next/paper/results/results.jsonl`.
- Demo notebook, first clean-node execution: the `attention_oom_safe` cell under a 4 GiB cap ran past the 1 h cell timeout. Cause: `effective_free_bytes` resolved the cap through `torch.cuda.get_per_process_memory_fraction(torch.device("cuda"))`, which raises for an index-less device, so the planner silently saw the uncapped 39 GiB, chose the monolithic path, and the fallback thrashed. Fixed (device index resolved explicitly); under a 0.6 GiB cap at 131K the planner now reports "monolithic needs 0.58 GiB > budget 0.45 GiB; itr=3 fits" and `attention_oom_safe` recovers in 3.5 s. Notebook sizes reduced (512K under a 2 GiB cap) and re-executed on a compute node.
- Found while re-running the notebook on a clean node: (1) `attention_oom_safe` keeps the fp32 accumulator on the device at every depth, so under a cap smaller than 8 bytes/element (accumulator + output) it cannot succeed at any depth -- at 512K under 2 GiB it walked itr=1..3 and failed; (2) building the task list is Python-side O(tasks x N): ~0.25 s per task at 512K, so itr=3 (343 tasks) spends ~60-90 s before the first kernel launch. Fix for (1): the fallback now tries each depth with the device accumulator and then with the host accumulator (`low_memory=True`) before going deeper. (2) is documented; the planner's preference for larger c at lower depth sidesteps it (c=73 at itr=1 is 73 tasks), and a vectorised task builder is a follow-up.
- Demo notebook executed end to end on a clean A100-80GB (job 13798786, 128 s, 0 error outputs): OOM recovery under a 2 GiB cap at 512K (20.8 s, peak 1.79 GiB, exact), the engine's knobs incl. c=13/31, stage trace, autograd with independent backward depth, planner tables, calibrate + `auto_attention` under a 2 GiB cap (planner: c=21 itr=1 acc=cpu host, ran at 1.46 GiB peak), autotune, devkit compare/quick_bench, Flex adapters (ALiBi exact / local-index bug caught / sliding window), kernel timings (FA-2 16.9 ms, v2 CQS-off 17.1, v2 CQS-on 24.4, v1 30.3), distributed docstring. Two more package fixes came out of it: `attention_oom_safe` now also tries the host accumulator per depth and casts its output on the host when the caller's tensors are host-resident; `devkit.Config` carries `(c, interest_set)`.

## Phase 4 (2026-09-12 evening): the CQS kernel in Triton (zero-build path)

`next/pkg/stream_cqsa/triton_kernel.py`: FlashAttention-2 forward in Triton with the CQS pair mask, same contract as the CUDA kernel (q/k/v [B,L,H,D], int64 group bits, block summaries, fp32 out + natural-log lse with -inf for empty rows). Structure follows the v2 CUDA design: row-side verdict once per program, per-tile verdict from two summary words (bitwise `tl.reduce`), fully masked tiles skipped by a scalar `if` (no K/V load, no GEMM), clear tiles run the plain path, mixed tiles apply the per-element AND; the causal diagonal is a separate short loop; online softmax in fp32/exp2 with the empty-row case guarded. Any hdim in {16,32,64,128}, fp16/bf16, ragged L.

Validation (interactive A100, contended so no timings): against the native kernel on real/zero/all-masked/random/half patterns, causal and not, L=2K-56K, hdim 64 and 128, bf16: rel diff 1.6-3.0e-4 (= fp16 P-rounding order), lse |d| <= 6e-6, identical -inf patterns, no NaN; against the fp32 dense reference the Triton and native kernels have the same error to two digits (e.g. 1.4e-4 vs 1.5e-4). `compare_kernels(triton_inner)`: "exact within rounding" at 131K itr=1 and itr=2/acc=cpu/host. Monolithic `triton_attention` vs flash-attn at 16K: 2.7e-4.

Engine wiring: when the CUDA extension is not importable, `stream_cqsa_forward` selects `triton_inner` automatically (verified by simulating a missing extension: 2.0e-4 vs the native engine); `shared_chunks` is forced off for any non-CUDA inner (the segmented chunk-pool view is CUDA-only) and `triton_inner` refuses a segmented input rather than reading it wrong. Test `test_triton_inner_exact` added. Clean-node benchmark (block-size sweep, kernel-level vs native/FA-2 at 16K-131K, engine-level at 256K/1M) submitted: job 13799436.
Not done: the backward in Triton (the engine's backward still needs the CUDA extension); the zero-build path is forward-only for now.

### Triton kernel benchmark (job 13799436, A100-SXM4-80GB, `logs/triton_bench.json`)

Block-size sweep at L=56K causal: best BLOCK_M=128, BLOCK_N=64, 4 warps, 3 stages (25.5 ms; 64x128 tiles 29 ms). Kernel-level, real bits, ms/call:

| N | L | causal | FA-2 (CUDA) | Triton plain | Triton CQS | native CQS (v11/v9) | Triton / native |
|---|---|---|---|---|---|---|---|
| 16K | 7K | yes | 0.47 | 0.50 | 0.84 | 1.40 | 0.60 |
| 32K | 14K | yes | 1.44 | 1.66 | 2.37 | 3.40 | 0.70 |
| 64K | 28K | yes | 5.08 | 5.78 | 7.62 | 8.13 | 0.94 |
| 131K | 56K | yes | 16.96 | 19.44 | 25.46 | 24.63 | 1.03 |
| 16K | 7K | no | 0.70 | 0.75 | 0.97 | 2.27 | 0.43 |
| 131K | 56K | no | 34.15 | 37.31 | 44.29 | 48.76 | 0.91 |

The Triton kernel's plain path is within 15% of FA-2, and with CQS on it is **faster than the native kernel below L=56K and at parity there** (the native kernel's mixed-tile path is what hurts it at small L; in Triton the per-element AND is cheap). Engine end to end (same job): 256K acc=GPU 0.69 s vs 0.53 s native (1.30x), 1M acc=GPU 10.9 vs 7.8 s (1.40x), 1M acc=CPU/host 13.3 vs 9.5 s (1.39x; the native arm also has shared_chunks). Outputs 2.2-2.5e-4 from the native engine (fp16 P rounding order), exact within rounding. The engine gap at 1M is larger than the kernel gap at 131K; a kernel-level run at L=449K/899K is queued (job below) to see whether the Triton kernel loses ground at long L.

### Triton kernel at long L (job 13799637, A100-SXM4-80GB): the per-tile `if` costs the software pipeline

| N | L | causal | FA-2 | Triton plain | Triton CQS | native CQS | Triton/native |
|---|---|---|---|---|---|---|---|
| 256K | 112K | yes | 67.4 | 75.3 | 97.9 | 83.6 | 1.17 |
| 1M | 449K | yes | 1086 | 1231 | 1569 | 1155 | 1.36 |
| 2M | 899K | yes | 4429 | 5025 | 6285 | 4499 | 1.40 |
| 1M | 449K | no | 2200 | 2398 | 2889 | 2683 | 1.08 |

The Triton plain path stays within 13% of FA-2 at every L, but the CQS path costs 27-31% over the plain path at every L, whereas the native v11 kernel's CQS path costs ~0-6% over its plain path at long L (22% masked tiles skipped ~ verdict overhead). Reason: wrapping the tile body in a scalar `if not masked` puts the K/V loads inside a conditional region, which Triton's software pipeliner cannot prefetch across, so the CQS loop runs unpipelined (`num_stages` is effectively 1). Fix in progress: iterate over *live runs* of key blocks (a per-row-block run table built on the host from the summaries, O(#distinct bit patterns x blocks)), so the inner loop has no verdict and pipelines exactly like the plain path -- the Triton counterpart of v11's live-tile loop.

Triton backward (`cqs_attention_backward`, two deterministic kernels dK/dV and dQ, global-lse form, `delta` accepted from the engine): vs the CUDA backward on real/zero/all-masked/random/half, causal and not, ragged, hdim 128, bf16: rel 2e-5..3e-4 (fp16 rounding); same error as the CUDA backward against an fp32 autograd reference to two digits (3.1e-4/3.1e-4/2.9e-4). Engine: `CQSA_BACKWARD=triton` or a missing extension routes `flash_attn_bwd_cqs_global_lse` to it; itr=1 and itr=2/host/acc=cpu agree with the CUDA backward to 3e-4; a fully extension-free autograd pass (Triton fwd + Triton bwd) agrees with the native one to 3e-4. Benchmark job 13799718 queued.

### Triton forward with live-run iteration (jobs 13799779 / 13799780, A100-SXM4-80GB)

The run table (`live_runs`: per row block, the runs of key blocks that are not fully masked, built from the summaries per distinct row word) removes the fully-masked tiles from the loop, so the tile body carries no `masked` conditional and Triton pipelines the K/V loads (`num_stages=3`) as in the plain kernel. Same validation set as before: exact within rounding, incl. multi-run patterns (itr=2 bits, stripes). ms/call, real bits:

| N | L | causal | FA-2 (CUDA) | Triton plain | Triton CQS | native CQS | Triton / native |
|---|---|---|---|---|---|---|---|
| 16K | 7K | yes | 0.42 | 0.46 | 1.48 | 1.27 | 1.17 |
| 32K | 14K | yes | 1.31 | 1.49 | 2.57 | 3.09 | 0.83 |
| 64K | 28K | yes | 4.58 | 5.26 | 6.89 | 8.05 | 0.86 |
| 131K | 56K | yes | 16.93 | 19.50 | 22.47 | 24.68 | **0.91** |
| 256K | 112K | yes | 72.5 | 75.1 | 84.9 | 83.4 | 1.02 |
| 1M | 449K | yes | 1090 | 1233 | 1274 | 1153 | 1.10 |
| 2M | 899K | yes | 4411 | 5000 | 5042 | 4456 | 1.13 |
| 131K | 56K | no | 34.14 | 37.34 | 36.42 | 48.81 | **0.75** |
| 1M | 449K | no | 2192 | 2390 | **2153** | 2686 | 0.80 |
| 2M | 899K | no | 9015 | 9862 | **8587** | 10698 | 0.80 |

The CQS overhead over the Triton plain path fell from +27-31% to +3-15% (causal) and became negative for non-causal (the skipped 22% of tiles now pay: the Triton CQS non-causal kernel is faster than FlashAttention-2 on the same L). Against the native CUDA kernel: faster below L~112K causal and at every L non-causal (0.75-0.80x), within 2-13% above. Engine end to end: 256K acc=GPU 0.59 s vs 0.53 native (1.11x), 1M acc=GPU 8.95 vs 7.81 (1.15x), 1M acc=CPU/host 11.2 vs 9.5 (1.18x, native has shared_chunks). The 16K row (1.17x) is the run-table build + launch overhead on a 1.4 ms kernel.

Triton backward (first version, per-tile `if`; job 13799718): 1.10-1.15x the CUDA backward causal, 1.41-1.44x non-causal at L=7K-56K; engine backward at 1M itr=1: 45.5 s vs 39.2 s CUDA (1.16x), max rel 1.2e-3 (fp16 dq atomics order + fp16 rounding). The live-run version of the backward (both kernels) is validated (2e-5..3e-4 vs CUDA on all patterns) and its benchmark is job 13799930.
