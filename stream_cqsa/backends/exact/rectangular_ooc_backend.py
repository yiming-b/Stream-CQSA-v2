from __future__ import annotations

from typing import Any, Sequence

import torch

from .online_softmax import normalize_num_den, rectangular_attention_num_den


class RectangularOutOfCoreBackend:
    """
    Exact rectangular blocked attention baseline with online softmax.

    This implementation is GPU-only blocking by default. It exposes a transfer
    byte counter, which is zero in GPU-only mode.
    """

    name: str = "rectangular_ooc"
    exact: bool = True
    supports_training: bool = True
    supports_inference: bool = True

    def __init__(
        self,
        *,
        q_block_size: int = 256,
        kv_block_size: int = 256,
        streaming_mode: str = "gpu",
    ) -> None:
        self.q_block_size = int(q_block_size)
        self.kv_block_size = int(kv_block_size)
        self.streaming_mode = str(streaming_mode).strip().lower()
        if self.streaming_mode != "gpu":
            raise ValueError("Only GPU-only rectangular blocking is implemented in this reproduction.")
        self.last_stats: dict[str, Any] = {}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        positions: torch.Tensor | Sequence[int] | None = None,
        causal: bool,
        softmax_scale: float | None = None,
        attention_mask: torch.Tensor | None = None,
        return_num_den: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        num, den, stats = rectangular_attention_num_den(
            q,
            k,
            v,
            positions=positions,
            causal=bool(causal),
            softmax_scale=softmax_scale,
            q_block_size=int(self.q_block_size),
            kv_block_size=int(self.kv_block_size),
            attention_mask=attention_mask if attention_mask is not None and attention_mask.dtype == torch.bool else None,
        )
        self.last_stats = dict(stats)
        out = normalize_num_den(num, den)
        if not return_num_den:
            return out.to(dtype=q.dtype)
        return out, {"num": num, "den": den, "stats": stats}

    __call__ = forward

