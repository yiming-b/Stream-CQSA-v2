# Changelog

## v2.2.1 (2026-09-15)

* Wave engine default budget: min(2M, max(512K, N)) packed tokens per wave instead of a flat 2M
  (`max_wave_tokens` / `CQSA_MAX_WAVE_TOKENS` / `CQSA_MIN_WAVE_TOKENS` still override). Peak device
  memory at 512K, itr=1: 4.8 GiB instead of 6.6 (the 2M budget cost the packed partial outputs of the
  whole call); a cap of ~N tokens measured <= 1.3% slower on an A100-SXM4-80GB (131K-512K).
* Routing: with `kernel="auto"` the wave engine is taken only for >= 20 subproblems (c >= 21 at
  itr=1, or itr=2; `CQSA_WAVE_MIN_SUBPROBLEMS`), where it measured faster than the classic engine;
  below that the classic engine is faster and uses less device memory (profile sweep, v2.2.0).

## v2.2.0 (2026-09-15)

* Wave engine on both kernels: `wave_forward(..., kernel="cuda"|"triton"|"auto")`. The Triton
  wave kernel is the CQS Triton forward generalised to the wave layout (W subproblems padded to a
  common length, per-subproblem tables and lengths, one launch per wave) with a deterministic
  torch merge, so the wave engine needs no compiled extension; `attention(kernel="wave-cuda" |
  "wave-triton")` selects it. The wave backward stays on the CUDA kernel.
* Wave engine host accumulator (`accumulate_on_gpu=False`, the classic engine's acc=CPU) on both
  kernels: each wave is merged on the device over the tokens it touches and folded into fp32 host
  accumulators; the device never holds the full output. `attention()` routes host-accumulator
  forwards to the wave engine.
* Quorum sets: c=3 {0,1} and c=133 {0,1,3,12,20,34,38,81,88,94,104,109} (both verified perfect
  difference sets); the planner enumerates every registered set, so the automatic configuration
  covers them.
* Profile sweep with a kernel axis (classic CUDA / classic Triton / wave CUDA / wave Triton) and
  c in {3, ..., 133}.
* Routing: `attention()` keeps the classic engine for host-accumulator (acc=CPU) forwards, where its
  overlapped per-subproblem transfers beat the wave engine's per-wave host merge by 5-7% at 1M-2M
  tokens on an A100-SXM4-80GB; `kernel="wave-cuda"` / `"wave-triton"` select the wave engine's
  host accumulator explicitly.
* Measurement note: Della's `gpu80` pool on gpu-short is A100 80GB PCIe; the SXM4 cards need
  `--constraint=sxm`. The paper re-run tables are PCIe, the profile sweep is SXM4-80GB.

## v2.1.2 (2026-09-15)

* Triton backward by default (`CQSA_BACKWARD=cuda` restores the CUDA backward): faster at every
  length measured (8.4M fwd+bwd 2946 s vs 3190 s) and needs no compiled extension.
* Native wave kernel on the release page for every cell: companion wheel `stream_cqsa_native`
  (python 3.10/3.11/3.12 x torch 2.5/2.6 x CUDA 12.4 x sm80/sm90) next to the classic wheel;
  workflow dispatch `native-all` and `attach_to=<tag>`; release job creates the release itself.
* Pure wheel and sdist published to PyPI on tag pushes (`pip install stream-cqsa`), when the
  repository variable `PYPI_PUBLISH` is set.
* One-parameter profiles (c in 7..133; N x {itr, acc}) with quadratic / linear fits:
  `benchmarks/profile_sweep.py`, `results/profile_sweep/`, README section. Quorum sets for
  c=91 and c=133 (Singer, q=9 and q=11) added to `QUORUM_SETS`.
* Distributed backward reduces the gradient chunks in place (halves the per-rank host footprint
  of the reduction).
* Slurm headers sized from the host tensors actually held; the paper harness records the host RSS
  peak (`mem_host_rss_peak_mib`).

## v2.1.1 (2026-09-14)

* Native wave kernel built for fp16 and bf16 at head dim 64 (head dim 128 forwards run on the
  classic engine); `attention()` routes by a capability probe of the installed build.
* `min_itr` (the paper's auto*) on the forward and backward; the depth planner returns cached
  allocator blocks before reading free memory and judges feasibility with one subproblem in flight.
* Paper harness: per-measurement process isolation (`--isolate`), imports the package under test.
* Release wheels: CUDA-extension wheels per python / torch / CUDA / GPU architecture from the
  workflow, each cell as two wheels (`stream_cqsa` with the classic extensions and the companion
  `stream_cqsa_native` with the wave kernel, `CQSA_ONLY_NATIVE=1` in `setup.py`); a torch 2.10 /
  CUDA 13 wheel with both built on della.

## v2.1.0 (2026-09-13)

* `stream_cqsa.attention(q, k, v, is_causal=...)`: one entry point with the signature of
  `torch.nn.functional.scaled_dot_product_attention`; monolithic when the call fits, exact
  decomposition when it does not, on the best kernel available; autograd; host or device inputs.
* `patch_sdpa()` / `patched_sdpa()`: route `F.scaled_dot_product_attention` through it.
* `estimate(N, ...)`: dry run (memory, configuration, predicted time, forward and backward).
* `doctor()` / `python -m stream_cqsa.doctor` / `stream-cqsa-doctor`: environment, kernels,
  device limits, exactness check.
* `verbose=True` / `CQSA_VERBOSE=1`: banner, progress bar with time remaining, closing line.
* Native multi-subproblem "wave" kernel (`native/`, `stream_cqsa.native_wave`): several
  subproblems per launch, in-place reads, deterministic batched merge, backward.
* Calibration cache (`calibrate()` persists per GPU model; `plan()` loads it).
* Clearer out-of-memory messages; warning for head-major host tensors.
* Packaging: pure-Python build (`CQSA_SKIP_EXT=1`), extras, console script, wheel workflow.

## v2.0.0

* Second-generation CUDA kernel (v11), pipelined engine, multi-device execution, automatic
  configuration, independent forward/backward depth, developer kit, Triton kernels, adapters.
