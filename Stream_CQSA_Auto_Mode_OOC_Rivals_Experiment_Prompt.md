# Additional experiments: automatic Stream-CQSA versus exact OOC attention

September 15, 2026. Implementation prompt for the existing Stream-CQSA experiment workspace. This does not concern the separate CQS-pairwise package.

## Objective and fixed method

Evaluate exact attention when a compute-capable GPU has limited remaining attention memory, including an A100 with about 10 GB available. Determine whether Stream-CQSA offers better feasibility, runtime or data movement than credible out-of-core (OOC) alternatives, not merely whether GPU-resident FA-2 runs out of memory.

**Use Stream-CQSA's existing automatic configuration mode in every primary comparison.** Inspect the actual API/configuration and retain its automatic depth, concurrency, capacity estimation and recovery decisions as implemented. Do not force a favorable manual decomposition or report an oracle-picked configuration as the automatic result. Log selected settings, changes, planning time and retries. Any optional manual experiment must be separate from the primary tables.

Reuse the current implementation and experiment harness. Inspect existing exact/OOC backends in [Stream-CQSA](https://github.com/yiming-b/Stream-CQSA.git) before adding rivals. Pin source revisions and record actual kernels; do not reimplement working normalization/backward machinery. Forward stable recomposition and backward using global normalization are already implemented in Stream-CQSA. No new model training recipe or broad workload sweep is requested here.

## Rivals to implement or integrate

### R1. GPU-resident FlashAttention-2

Run ordinary exact FA-2 forward and backward with matching shape, dtype, causal mask, scaling and head layout. Include required Q/K/V, outputs, saved statistics, gradients and workspace in memory accounting. Record OOM explicitly. This establishes the resident performance reference and feasibility boundary; it is not the only competitor.

If GQA expansion or another adapter is needed, identify and account for it. Do not silently compare different representations or kernels. Report unsupported shapes as unsupported rather than OOM.

### R2. Correct bounded-memory rectangular OOC attention

Keep canonical tensors in host RAM and use bounded GPU staging. For forward, retain a query tile and its output/normalization accumulator while streaming key/value tiles through it. Compute exact contributions with the original positions/causal mask and stable online softmax. Avoid a full score matrix, full GPU QKV copy or full GPU output accumulator when the budget cannot accommodate it.

Implement exact backward using the required global forward statistics and upstream derivatives. Accumulate dQ/dK/dV correctly across tiles using bounded GPU and/or host stores. Include all gradient transfers and repeated input reads. Ordinary independently normalized tile backward is not equivalent to full attention backward. Validate the kernel/API selected for partial contributions before performance testing.

Use a documented deterministic budget-based tile-sizing policy. A correct reference implementation establishes the baseline, but do not make an intentionally tiny tile or unoptimized Python loop the sole OOC rival. Reuse efficient local primitives where possible and disclose differences from Stream-CQSA's masked kernels.

### R3. Optimized adaptive rectangular OOC attention — primary rival

Build on R2 with memory-aware tile selection, bounded pinned buffers, asynchronous prefetch/double buffering where useful, and reuse of resident data. Tune query/key tile dimensions independently. Consider alternative traversal orders where they change reuse; do not insist on one poor schedule for both forward and backward.

Include any startup search/probing in cold timings. If tuning is cached, report warm reuse separately and specify its cache key. Give the rival a documented bounded tuning allowance; do not compare unlimited offline oracle tuning with Stream-CQSA automatic setup hidden or excluded. Neither method may exceed the actual memory budget through unaccounted buffers.

R2 and R3 can share an implementation with explicit scheduling options. The crucial contrast is automatic CQS versus a serious adaptive rectangular execution strategy. Both are OOC methods. Neither should compute masked causal work unnecessarily if the underlying kernel can avoid it; report unavoidable extra work.

### Optional full-model comparisons, only if the harness already supports them

For a full training-step study, relevant systems include [MEMO](https://arxiv.org/abs/2407.12117) and [Ulysses-Offload/FPDT](https://www.deepspeed.ai/tutorials/ulysses-offload/). Integrate a supported comparator if practical, but label its hardware/process layout and non-attention memory policies. These systems are not automatically interchangeable single-attention kernels. Do not let implementing a new full training stack block the operator-level experiments.

For inference, an existing offload runtime such as [FlexGen](https://arxiv.org/abs/2303.06865) is an optional application-level reference only with clearly matched precision and workload. Sparse attention, KV eviction, quantization and [StreamingLLM](https://arxiv.org/abs/2309.17453) belong in a separate quality–cost study; they are not exact-attention rivals. Do not require them for this experiment.

## Comparison conditions

Use the existing benchmark workload definitions. Match input data, seeds, shapes, precision, causal behavior and output residency. Report forward-only prefill, backward with its required forward state, and combined forward/backward separately. Do not describe operator timing as an end-to-end model result. Single-token decode is a different workload; if included, label it separately.

Use common canonical inputs and required final output locations for the main constrained-memory comparison. For example, all methods can start from host inputs and finish with host outputs/gradients; resident FA-2 must then pay its transfers. Additionally report a clearly labeled resident FA-2 kernel-only reference where it fits. Never divide transfer-inclusive time by kernel-only time and label the ratio an equivalent-work speedup.

Enforce and describe the budget mechanism. A 10 GB budget must account for attention-owned live tensors, staging buffers, saved backward state and kernel workspaces, while avoiding double-counting tensors shared with the model. Record baseline occupied memory, free device memory and peak total usage. A framework allocator limit may not constrain custom allocations; verify actual usage and log budget breaches. A full-sized GPU with reserved memory models constrained capacity, not reduced compute capability.

For proactive automatic execution, time planning through completion. For OOM recovery, include the failed attempt, cleanup, replanning and retry. Do not artificially induce an OOM in a method that would avoid it proactively. If methods take different routes, report those routes and the automatic end-to-end result.

## Metrics to report

| Category | Required fields |
|---|---|
| Configuration | GPU, host CPU/RAM, interconnect, shape, dtype, mask, head layout, memory budget, canonical input/output residency, source/software/kernel versions |
| Automatic choices | Stream-CQSA selected depth/concurrency and changes; rival tile sizes, buffer count, traversal; planning/probing time, cached or cold decision |
| Feasibility | Success, OOM, timeout, unsupported or numerical failure; achieved sequence length for the tested shape/budget; actual budget compliance |
| Correctness | Forward and dQ/dK/dV errors against a trusted reference; absolute/relative metrics, declared tolerances, NaN/Inf counts and near-zero handling |
| Runtime | Forward, backward, combined forward/backward; cold total and separately labeled warm total; resident kernel-only reference where applicable |
| Time breakdown | Planning, packing, H2D, kernels, D2H, merge, synchronization and recovery; distinguish overlapping durations from exposed critical-path time |
| Memory | Peak allocated/reserved GPU bytes where available, device-level usage, host RSS, pinned bytes, saved backward-state bytes and gradient storage |
| Data movement | Total H2D/D2H/peer bytes, repeated tensor reads, transfer counts, cache reuse and measured effective bandwidth |
| Compute efficiency | Useful interactions, extra evaluated interactions/recomputation, clearly defined useful FLOPs/s or tokens/s; actual kernel names |
| Reliability | Retry count, escalation history, failure reason and ability to continue subsequent cases after failure |

Use small/medium trusted-reference cases to establish numerical equivalence before performance runs. At large lengths where a dense reference is infeasible, label the validation method explicitly; completing a run is not sufficient evidence of correctness. Correctness checks must cover all-masked tiles, uneven lengths and the supported causal/head configurations.

Measure synchronized wall-clock time for operations involving CPU work and transfers; GPU events alone are insufficient. Profile representative runs separately so tracing does not contaminate headline timings. Do not sum concurrent stream durations and call the result elapsed time. Use repeated paired runs where affordable, report the number of repeats and median/spread, and retain raw results. Count JIT/build overhead separately from ordinary runtime planning.

## Required report and figures

Produce machine-readable per-run CSV/JSON records, configurations and trace/log paths, plus a concise Markdown report containing:

1. **Feasibility map:** sequence length versus available memory, marking each method's success/failure and successful runtime. No finite speedup ratio against an OOM or timeout.
2. **Runtime–memory curves:** automatic Stream-CQSA versus R1/R2/R3, with identical-work comparisons and forward/backward distinction.
3. **Transfer and phase breakdowns:** show whether gains come from less movement, better overlap, different kernel cost or recovery behavior.
4. **Automatic configuration table:** chosen settings and startup/recovery overhead; no hidden manual selection.
5. **Correctness summary and limitations:** especially kernel/precision differences, unsupported configurations and any full-model extrapolations.

Report the best credible completed rival for each matched case as well as individual baselines. Preserve losses and ties. The conclusion should identify the regimes in which CQS is useful, not assume that all OOC workloads favor it.

If real edge/workstation hardware is available, report it separately. A capped A100 is not an edge-device performance proxy. On integrated CPU/GPU systems, shared physical DRAM means host placement does not provide an independent larger memory pool; measure actual total memory and transfers. Do not claim edge suitability from capped-A100 results alone.
