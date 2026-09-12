from __future__ import annotations

from typing import Any, Sequence

import torch

from .longnet_mask import build_longnet_keep_mask


class LongNetBackend:
    """
    Full simplified LongNet-style sparse/dilated attention backend.

    This is a reference implementation for correctness and OOM-boundary
    experiments. It materializes a dense score tensor and dense boolean keep mask,
    so it is intentionally not the optimized LongNet kernel from the paper.

    Tensor layout is [B, L, H, D].
    """

    name: str = "longnet"
    exact: bool = False
    supports_training: bool = True
    supports_inference: bool = True

    def __init__(
        self,
        *,
        segment_lengths: Sequence[int] | Sequence[Sequence[int]] = (32, 64, 128),
        dilation_rates: Sequence[int] | Sequence[Sequence[int]] = (1, 2, 4),
    ) -> None:
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        positions: torch.Tensor,
        causal: bool,
        softmax_scale: float | None = None,
        return_num_den: bool = False,
        return_mask: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q, k, and v must be rank-4 tensors with layout [B, L, H, D].")
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(f"q, k, and v must have identical shapes. Got {q.shape}, {k.shape}, {v.shape}.")
        B, L, H, D = [int(x) for x in q.shape]
        if softmax_scale is None:
            softmax_scale = float(D) ** -0.5

        keep = build_longnet_keep_mask(
            positions,
            segment_lengths=self.segment_lengths,
            dilation_rates=self.dilation_rates,
            causal=bool(causal),
            num_heads=H,
            device=q.device,
        )
        if keep.ndim == 2:
            keep = keep.unsqueeze(0).expand(H, -1, -1)
        if tuple(keep.shape) != (H, L, L):
            raise RuntimeError(f"Unexpected LongNet mask shape {tuple(keep.shape)}, expected {(H, L, L)}")

        q_bhld = q.transpose(1, 2).float()
        k_bhld = k.transpose(1, 2).float()
        v_bhld = v.transpose(1, 2).float()

        scores = torch.matmul(q_bhld, k_bhld.transpose(-2, -1)) * float(softmax_scale)
        scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))

        weights = torch.exp(scores)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        den = weights.sum(dim=-1)
        num = torch.matmul(weights, v_bhld).transpose(1, 2).contiguous()
        num = torch.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
        den = torch.nan_to_num(den, nan=0.0, posinf=0.0, neginf=0.0)

        out = torch.zeros((B, L, H, D), dtype=torch.float32, device=q.device)
        den_blhd = den.transpose(1, 2).unsqueeze(-1)
        valid = den_blhd.gt(0)
        out = torch.where(valid, num / den_blhd.clamp_min(torch.finfo(torch.float32).tiny), out)

        if not (return_num_den or return_mask):
            return out.to(dtype=q.dtype)

        aux: dict[str, torch.Tensor] = {}
        if return_num_den:
            aux["num"] = num
            aux["den"] = den
        if return_mask:
            aux["keep_mask"] = keep
        return out, aux

    __call__ = forward

