# Exact Attention Backends and Inner Kernels

This module provides correctness-first exact attention reproductions for the
Stream-CQSA OOM-recovery experiments.

Backends:

- `SDPABackend`: top-level PyTorch SDPA reference backend.
- `RectangularOutOfCoreBackend`: top-level exact rectangular blocked attention
  with online softmax. This reproduction implements GPU-only blocking and
  reports zero transfer bytes.

Inner kernels:

- `DenseExactInnerKernel`: applies `CQS_keep AND causal_keep` and exposes
  stable row-max shifted numerator/denominator state using explicit dense
  PyTorch math. `SDPAInnerKernel` remains as a backward-compatible alias.
- `RectangularOutOfCoreInnerKernel`: applies the same local keep mask and
  computes raw numerator/denominator terms by Q-block x KV-block online softmax.

Tensor layout is `[B, L, H, D]` for all exact backend and inner-kernel APIs.

Important limitations:

- `DenseExactInnerKernel` does not use PyTorch SDPA internally because SDPA does
  not expose the denominator or row-max state required by Stream-CQSA
  recomposition. It materializes local scores, so its scalability is limited to
  local CQS subproblems where that dense local score tensor fits.
- `RectangularOutOfCoreBackend` is exact and blocked, but this reproduction does
  not implement CPU-to-GPU KV paging.
- The pure-PyTorch reference Stream-CQSA runner is differentiable, but the
  production CUDA streaming autograd path is not wired to these custom exact
  inner kernels.
