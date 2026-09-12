# LongNet-Style Stream-CQSA Reproduction

This module is a simplified reproduction for testing whether Stream-CQSA can wrap
a global sparse/dilated approximate attention pattern. It is not an optimized
implementation of the LongNet paper.

The implemented mask rule is global-position aware:

```text
keep(q, k) if distance < segment_length * dilation_rate
           and distance % dilation_rate == 0
```

In causal mode `distance = q_position - k_position` and must be non-negative.
In non-causal mode `distance = abs(q_position - k_position)`.

The important Stream-CQSA algebra is preserved:

```text
Keep_r = CQS_keep_r AND LongNet_keep_global AND causal_keep
Num_r  = sum_j Keep_r(t,j) exp(q_t k_j^T / sqrt(d)) V_j
Den_r  = sum_j Keep_r(t,j) exp(q_t k_j^T / sqrt(d))
O      = sum_r Num_r / sum_r Den_r
```

Files:

- `longnet_mask.py`: schedule validation, global mask construction, CQS/local mask intersection.
- `longnet_backend.py`: full reference LongNet-style backend, materializing dense scores and masks.
- `longnet_inner_kernel.py`: Stream-CQSA-compatible inner kernel returning numerator and denominator.
- `configs/longnet_small.yaml`: small default config for correctness and smoke runs.

Limitations:

- This is a correctness-first Python/PyTorch reference path.
- Backward for `LongNetInnerKernel` is not wired into Stream-CQSA and is labeled unsupported.
- Large language-model and PG19 evaluation require a separate model/checkpoint integration step.

