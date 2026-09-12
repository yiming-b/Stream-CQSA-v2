from __future__ import annotations

from typing import Any, Sequence

import torch

from .online_softmax import rectangular_attention_num_den
from .sdpa_inner_kernel import DenseExactInnerKernel


class RectangularOutOfCoreInnerKernel(DenseExactInnerKernel):
    """Stream-CQSA exact inner kernel using rectangular blocked online softmax."""

    name: str = "rectangular_ooc_inner"
    supports_training: bool = True
    supports_inference: bool = True
    supports_stable_recomposition: bool = False

    def __init__(
        self,
        *,
        q_block_size: int = 128,
        kv_block_size: int = 128,
        allow_local_position_fallback: bool = False,
    ) -> None:
        super().__init__(allow_local_position_fallback=allow_local_position_fallback)
        self.q_block_size = int(q_block_size)
        self.kv_block_size = int(kv_block_size)
        self.last_stats: dict[str, Any] = {}

    def forward_num_den(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        local_positions: torch.Tensor | Sequence[int] | None = None,
        cqs_mask: dict[str, Any],
        causal: bool,
        softmax_scale: float,
        return_keep_mask: bool = False,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q, k, and v must be rank-4 tensors [B, L, H, D].")
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(f"q, k, and v must have identical shapes. Got {q.shape}, {k.shape}, {v.shape}.")
        _, L, H, _ = [int(x) for x in q.shape]
        if int(cqs_mask["local_size"]) != L:
            raise ValueError(f"cqs_mask local_size={cqs_mask['local_size']} does not match tensor L={L}")
        keep = self.keep_mask(
            local_positions=local_positions,
            cqs_mask=cqs_mask,
            causal=bool(causal),
            num_heads=H,
            device=q.device,
        )
        num, den, stats = rectangular_attention_num_den(
            q,
            k,
            v,
            keep_mask=keep,
            causal=False,
            softmax_scale=float(softmax_scale),
            q_block_size=int(self.q_block_size),
            kv_block_size=int(self.kv_block_size),
        )
        self.last_stats = dict(stats)
        if return_keep_mask:
            return num, den, keep
        return num, den

    def __call__(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cqs_mask: dict[str, Any],
        softmax_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_num_den(
            q,
            k,
            v,
            cqs_mask=cqs_mask,
            causal=True,
            softmax_scale=float(softmax_scale),
        )
