# Additional experiments: automatic Stream-CQSA versus exact OOC attention

September 15, 2026. Implementation prompt for the existing Stream-CQSA experiment workspace. This does not concern the separate CQS-pairwise package.

> ## Status and plan (annotated 2026-09-15, after inspecting the workspace)
>
> **Already in place (reused, not rebuilt)**
> - *Automatic mode.* `stream_cqsa.attention()` plans from a hardware description (`hardware_from_dict({"cuda:0": "10GiB", ...})`), enumerates (c, itr, accumulator placement, host residency, concurrency), runs the fastest feasible point and escalates on OOM. Every run records the plan and its reason, `itr`, `itr_reached`, `oom_retries`, planning time (`plan.reason`, `info`). The paper harness (`experiments/paper/run_paper_experiment.py`, `--itr-list auto`, `--isolate`) already logs these per row. Nothing is forced by hand in the primary tables; a manual sweep exists separately (`results/profile_sweep/`).
> - *R1.* The harness's `flash` method: FlashAttention-2 forward and backward, isolated subprocess per measurement, OOM recorded as a row, peak allocated / reserved / nvidia-smi memory, host bytes and RSS. Runs at the 80 GB budget exist (`results/paper_rerun/` when the supplements finish); the constrained budgets below are new runs.
> - *Exact machinery.* Stable forward recomposition and the global-normalization backward are Stream-CQSA's; the rivals reuse FlashAttention-2's forward (`flash_attn_with_kvcache`, bottom-right causal alignment) and backward primitive (`_flash_attn_backward` with the **global** lse and Δ = rowsum(dO∘O)), which is exactly the per-tile backward with global normalization; no re-derivation.
> - *R2 forward, partially.* CQS-prefill's paged arm (`/scratch/gpfs/AKEY/yb2807/CQS-prefill/benchmarks/methods.py: paged_prefill`) is a bounded-memory rectangular forward: host-resident K/V, query chunks against key windows, fp32 log-sum-exp merge, H2D/D2H byte counts. It is ported into the package as the R2 forward.
> - *Budget mechanism.* `torch.cuda.set_per_process_memory_fraction` caps the allocator (the demo and devkit use it) **and** the planner gets the same number through the hardware dict; actual usage is verified with `max_memory_allocated` / `max_memory_reserved` / nvidia-smi and breaches are logged. A capped A100 models capacity only (noted in the report).
>
> **Not in place (implemented for this study)**
> - *R2 backward* and the host-side dQ/dK/dV accumulation with bounded device stores: new (`stream_cqsa/baselines/rect_ooc.py`).
> - *R3*: memory-aware independent q/kv tile sizing, pinned double-buffered staging with asynchronous prefetch, K/V-window reuse across query tiles (kv-outer traversal for the backward), a bounded tuning allowance (a fixed set of tile candidates probed on the first call, cold time reported; warm reuse keyed on (shape, dtype, budget)).
> - *Harness methods* `rect_ooc` (R2) and `rect_ooc_adaptive` (R3) with transfer-byte, phase and budget-compliance fields; the feasibility map / runtime-memory / breakdown figures and the report generator (`experiments/ooc_rivals/`).
> - The package's existing `backends/exact/rectangular_ooc_backend.py` is a GPU-only dense torch reference (no host residency, no bounded staging) and is used only as a small-N correctness cross-check, not as a rival.
>
> **Scope decisions**
> - In scope: one attention layer, B=1 H=8 D=64 fp16 causal (the paper's shape), budgets {10, 20, 40, 80 GiB} on A100 80GB PCIe nodes (one node type per comparison), N from 256K up to the first OOM/timeout per method, forward, backward (with its forward), and combined; all methods start from host inputs and end with host outputs/gradients; a resident kernel-only FA-2 reference where it fits, labelled as such. 3 repeats, median and spread, raw JSONL kept. Correctness against float64 on sampled rows at every N and full float64 at N <= 1M; all-masked tiles, uneven lengths and both traversal orders covered by unit tests.
> - Out of scope here: MEMO / FPDT / FlexGen (no full-model harness exists; not built for this study), single-token decode, sparse/eviction methods, edge hardware (none available; nothing is claimed about it), GQA (the shape has H_q = H_kv; other layouts reported as unsupported).
>
> **Execution order**: rival implementation + unit tests -> harness methods + budget enforcement -> 2 GiB-cap smoke on the interactive GPU -> gpu-short grid (one job per budget x direction) -> analysis and report. Progress and job ids in `experiments/ooc_rivals/LOG.md`.

## Objective and fixed method

Evaluate exact attention when a compute-capable GPU has limited remaining attention memory, including an A100 with about 10 GB available. Determine whether Stream-CQSA offers better feasibility, runtime or data movement than credible out-of-core (OOC) alternatives, not merely whether GPU-resident FA-2 runs out of memory.

**Use Stream-CQSA's existing automatic configuration mode in every primary comparison.** Inspect the actual API/configuration and retain its automatic depth, concurrency, capacity estimation and recovery decisions as implemented. Do not force a favorable manual decomposition or report an oracle-picked configuration as the automatic result. Log selected settings, changes, planning time and retries. Any optional manual experiment must be separate from the primary tables.

Reuse the current implementation and experiment harness. Inspect existing exact/OOC backends in [Stream-CQSA](https://github.com/yiming-b/Stream-CQSA.git) before adding rivals. Pin source revisions and record actual kernels; do not reimplement working normalization/backward machinery. Forward stable recomposition and backward using global normalization are already implemented in Stream-CQSA. No new model training recipe or broad workload sweep is requested here.

## Rivals to implement or integrate

### R1. GPU-resident FlashAttention-2

> *Status:* done in the harness (`flash`), reused; only the constrained budgets are new runs. GQA: unsupported for this shape, reported as such.

Run ordinary exact FA-2 forward and backward with matching shape, dtype, causal mask, scaling and head layout. Include required Q/K/V, outputs, saved statistics, gradients and workspace in memory accounting. Record OOM explicitly. This establishes the resident performance reference and feasibility boundary; it is not the only competitor.

If GQA expansion or another adapter is needed, identify and account for it. Do not silently compare different representations or kernels. Report unsupported shapes as unsupported rather than OOM.

### R2. Correct bounded-memory rectangular OOC attention

> *Status:* forward ported from CQS-prefill's paged prefill (FlashAttention-2 kernel per tile, fp32 merge); backward new, on FA-2's backward primitive with the global lse/Δ; deterministic tile sizing from the budget (largest power-of-two q tile such that q tile + kv window + accumulators + workspace fit at half the budget). Differences from Stream-CQSA's masked kernels disclosed in the report: the rival evaluates only live (q, kv) tiles, no CQS masking, no redundancy.

Keep canonical tensors in host RAM and use bounded GPU staging. For forward, retain a query tile and its output/normalization accumulator while streaming key/value tiles through it. Compute exact contributions with the original positions/causal mask and stable online softmax. Avoid a full score matrix, full GPU QKV copy or full GPU output accumulator when the budget cannot accommodate it.

Implement exact backward using the required global forward statistics and upstream derivatives. Accumulate dQ/dK/dV correctly across tiles using bounded GPU and/or host stores. Include all gradient transfers and repeated input reads. Ordinary independently normalized tile backward is not equivalent to full attention backward. Validate the kernel/API selected for partial contributions before performance testing.

Use a documented deterministic budget-based tile-sizing policy. A correct reference implementation establishes the baseline, but do not make an intentionally tiny tile or unoptimized Python loop the sole OOC rival. Reuse efficient local primitives where possible and disclose differences from Stream-CQSA's masked kernels.

### R3. Optimized adaptive rectangular OOC attention — primary rival

> *Status:* new; shares the implementation with R2 (`schedule="fixed"` vs `"adaptive"`). Adaptive: tile candidates probed on the first call (bounded: <= 6 candidates x 1 tile each, timed into the cold run), pinned double buffers with prefetch on a copy stream, kv-outer traversal for the backward so each K/V window is read once per pass, warm reuse keyed on (N, H, D, dtype, causal, budget).

Build on R2 with memory-aware tile selection, bounded pinned buffers, asynchronous prefetch/double buffering where useful, and reuse of resident data. Tune query/key tile dimensions independently. Consider alternative traversal orders where they change reuse; do not insist on one poor schedule for both forward and backward.

Include any startup search/probing in cold timings. If tuning is cached, report warm reuse separately and specify its cache key. Give the rival a documented bounded tuning allowance; do not compare unlimited offline oracle tuning with Stream-CQSA automatic setup hidden or excluded. Neither method may exceed the actual memory budget through unaccounted buffers.

R2 and R3 can share an implementation with explicit scheduling options. The crucial contrast is automatic CQS versus a serious adaptive rectangular execution strategy. Both are OOC methods. Neither should compute masked causal work unnecessarily if the underlying kernel can avoid it; report unavoidable extra work.

### Optional full-model comparisons, only if the harness already supports them

> *Status:* not supported by the harness; out of scope for this study (stated in the report's limitations).

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
