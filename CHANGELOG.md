# Changelog

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
