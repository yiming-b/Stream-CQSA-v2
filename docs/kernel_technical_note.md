# The Stream-CQSA native forward kernel, second generation (v11)

Technical note for future paper writing. All numbers are from
`next/LOG.md` (jobs cited there); the raw results are in `next/logs/*.json`.
Hardware: NVIDIA A100 (SXM4-80GB unless stated; the PCIe-40GB interactive
node was used for correctness only). fp16, head dim 64, 8 heads, batch 1.

## 1. What the kernel computes

A Stream-CQSA subproblem is a gathered subsequence of `L = N·(3/7)^itr`
tokens (three cyclic-quorum chunks at itr = 1). Every token carries an int64
*group-bits* word; the pair (row, col) is kept iff `bits[row] & bits[col] == 0`
(and `col ≤ row` when causal). The kernel is FlashAttention-2 (hdim 64, 128×128
tiles, 4 warps, sm80) with three additions: a per-tile *verdict* from
per-64-token block summaries (`blk_or`, `blk_and`) that classifies each
128×128 tile as *clear* (all pairs kept), *masked* (none kept) or *mixed*; a
per-element mask for mixed tiles; and an fp32 output path with the per-row
log-sum-exp, which the engine's exact merge needs. Everything the kernel
returns is what a monolithic FlashAttention call would return for the same
pair set, so the recomposed result is bit-for-bit what the engine computed
before (every variant below was checked bit-identical against the shipped
binary on real, all-clear, all-masked, random-mixed and ragged patterns,
causal and not).

## 2. Where the shipped kernel's time went

Two facts drove the redesign.

**Tile census.** Chunks are contiguous, so in a real subproblem 77.6 % of
causal tiles are clear, 22.2 % are fully masked (the non-owner chunks'
diagonal blocks) and only 0.2 % are mixed (N = 131072, itr 1; 0.5 % at itr 2).
The mixed-tile path is therefore nearly irrelevant at large L, and the masked
tiles are the only "free" work available: skipping them ideally makes the
kernel 0.78× a monolithic FA-2 call on the same L.

**The kernel is issue-bound, not memory-bound** (ncu, jobs 13765375/13765554):
DRAM < 1 %, L2 hit 98 %, one instruction issued every 3.3 cycles per
scheduler with two resident warps, dominant stall "fixed-latency dependency"
(integer ALU chains). Measured per 128×128 tile per warp:

| kernel / tile kind | executed instructions per warp per tile |
|---|---|
| upstream FA-2, live tile | 1 203 |
| shipped CQS kernel, live (clear) tile | 1 715 |
| shipped CQS kernel, fully masked ("skipped") tile | 524 |

So a *skipped* tile still cost 30 % of a live one and a live tile cost 43 %
more than FA-2's — all of it instructions executed before the tile's GEMM can
issue. The instruction surplus was the tile verdicts themselves: two
`cqs_tile_masked/clear` evaluations per tile, each with runtime-`blk_size`
integer divisions and modulos (≈ 20 instructions each, dependent) and small
loops over the summary words. A second, structural effect: with CQS *disabled*
the same binary executed the same instruction count as FA-2 (8.25 G vs
8.32 G) yet ran 1.22–1.33× slower, because it issued at 5.96 cycles per
instruction instead of 4.84: the runtime-predicated regions around the V load
/ QK GEMM / K prefetch, a runtime choice between two `softmax_rescale_o`
instantiations, and a late fp32 output store that kept the 32-register
accumulator live through the O smem round-trip break the straight-line loop
body that FA-2's schedule depends on.

## 3. The changes, in order of adoption

| step | change | effect (real causal workload, L = 56 K, SXM4) |
|---|---|---|
| v1 | the `Mask` object no longer carries the summary pointers; verdicts are computed from kernel params | −3…5 % |
| v4 | the row half of both verdicts (2 summary words per CTA row block) hoisted out of the KV loop | live tiles −9 % |
| v5 | compile-time block size (`kCqsBlk = 64`, checked on the host); column summaries prefetched one tile ahead so the verdict at the top of an iteration is a register AND | masked tiles 63 % → 45 % of a live tile |
| v7 | the two column-summary words of a full aligned tile read with one 16-byte vector load each, no loop, no clamps (general path only for the ragged last tile) | −15 % |
| v6/v10 | CQS on/off is a kernel template flag, dispatched at launch; one `Check_inf=true` softmax instantiation; fp32 store immediately after `normalize_softmax_lse` | CQS-off path now equals FA-2 (18.9 vs 18.7 ms) |
| v11 | steady loop iterates over *live* tiles only: the body is FA-2's straight-line block (wait/sync, V load, QK GEMM, wait/sync, K prefetch, mask, softmax, PV GEMM) with no predicated regions or `continue`; a search for the next live tile runs while the QK GEMM is in flight and steers the K prefetch to it | **−19 %** (30.2 → 24.4 ms); 4.03 G instructions vs FA-2's 3.71 G |

Rejected after measurement: v2 (staging a mixed tile's 128 column words in
shared memory: no effect — the per-element path is ALU-bound, 5.5× a live
tile, but runs on 0.2 % of tiles); v3 (skipping the K/V smem loads of masked
tiles: 5–9 % *slower*, because it added a second verdict per tile to an
issue-bound loop and the loads were never the bottleneck).

Correctness argument for v11's live-tile loop: live tiles are visited in the
same descending order as before, each exactly once; the online softmax state
therefore receives the same sequence of tiles, so the accumulation order and
hence the floating-point result are unchanged (confirmed bit-identical). The
K prefetch targets the next live tile; the first steady tile, if masked,
re-issues its K load after `cp.async.wait_all` (same thread, same smem
slots, so no barrier is needed).

## 4. Measured results

Clean-node A/B (job 13780947, A100-SXM4-80GB, one process per kernel, block
summaries precomputed as the engine does, ms per call at the itr = 1
subproblem shape L = 3N/7):

| N | causal | FA-2 (editable 2.8.3) | SDPA-flash (torch's stock FA-2) | shipped | v11 | v11 / shipped | v11 / FA-2 |
|---|---|---|---|---|---|---|---|
| 16 K | yes | 0.47 | 0.49 | 1.51 | 1.36 | 0.90 | 2.9 |
| 32 K | yes | 1.44 | 1.56 | 3.60 | 3.35 | 0.93 | 2.3 |
| 64 K | yes | 5.06 | 5.50 | 9.56 | 8.01 | 0.84 | 1.58 |
| 131 K | yes | 17.46 | 18.36 | 30.22 | 24.43 | **0.81** | 1.40 |
| 131 K | no | 34.20 | 36.18 | 53.38 | 48.32 (v9) | 0.91 | 1.41 |

With CQS disabled the same binary runs at 18.91 / 33.48 ms (FA-2: 17.46 /
34.20), i.e. the "CQS build" is no longer a slower FlashAttention when used as
one. Against a monolithic call the remaining 1.4× at 131 K decomposes as:
0.78× is the pair work after skipping masked tiles, the live-tile loop is
within 9 % of FA-2 in instruction count, and the rest is the `Check_inf`
softmax, the `!Is_even_MN` column-limit pass in `apply_mask` on every tile,
and instruction-cache pressure from the 20× larger CQS-on code (76 K vs 4–8 K
SASS instructions, almost all of it the unrolled mixed-tile path).

Accuracy: unchanged by construction (bit-identical to the shipped kernel);
against float64 on sampled rows the decomposed result has rel. error
1.9–2.1e-4 at N = 64 K–256 K versus 2.8–3.1e-4 for the monolithic fp16 call
(`quick_bench`, next/logs/quick_bench_*.json) — the merge accumulates in fp32
and each subproblem's softmax spans fewer keys, which slightly *reduces*
rounding error.

End to end (next/ engine + v11, one SXM4-80GB, Q/K/V on host, host
accumulator): 1.86× / 1.63× / 1.44× faster than the shipped engine at
N = 256 K / 1 M / 2 M, of which ~1.19× is the kernel on the compute stage.

## 5. Open issue: the non-causal CQS-on instantiation

For non-causal kernels the CQS-on instantiation of v10/v11 is compiled by
ptxas into a kernel that executes 23.6 G instructions where 4.0 G are
expected (L1 hit rate 35 % vs 62 %, 616–680 B stack): a register fragment is
demoted to local memory. The same source compiles correctly for the causal
instantiation, and the same non-causal code with a runtime flag (v9) is fine
(−10 % vs shipped). The interface therefore serves non-causal calls from v9
(`CQSA_CUDA_MODULE_NONCAUSAL`). A single-binary fix needs `-lineinfo` and a
source-level ncu pass. Variants were built with the fp16/hdim 64+128 kernel
set; a release build needs the `common` set (bf16 included).

## 6. Profiling method (for the paper's methodology section)

* Kernel timing: CUDA events around 10 launches after 3 warm-ups, one
  process per kernel binary, interleaved per shape, on an exclusively
  allocated node; block summaries precomputed once (the engine's behaviour —
  computing them per call inside the timed loop inflated early measurements
  by up to 3.5× at small N).
* Bit-exactness: fp32 outputs of every variant compared to the shipped binary
  with `max|Δ| == 0` on five mask patterns × causal/non-causal × three
  subsequence shapes including a ragged tail.
* Profiles: Nsight Compute on a compute node (performance counters are
  blocked on the shared interactive node), SM clock locked at 1.06 GHz by
  the profiler, sections SpeedOfLight / Occupancy / WarpStateStats /
  SchedulerStats / MemoryWorkloadAnalysis / InstructionStats.
* Static analysis: `cuobjdump --dump-resource-usage` (registers, stack) and
  `cuobjdump -sass` instruction counts per template instantiation.
