"""LongNet-style sparse/dilated attention reproduction."""

from .longnet_backend import LongNetBackend
from .longnet_inner_kernel import LongNetInnerKernel, stream_cqsa_longnet_reference
from .longnet_mask import (
    LongNetSchedule,
    build_longnet_keep_mask,
    build_longnet_cqs_keep_mask,
    cqs_masked_pairs,
)

__all__ = [
    "LongNetBackend",
    "LongNetInnerKernel",
    "LongNetSchedule",
    "build_longnet_keep_mask",
    "build_longnet_cqs_keep_mask",
    "cqs_masked_pairs",
    "stream_cqsa_longnet_reference",
]

