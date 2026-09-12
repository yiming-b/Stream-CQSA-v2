from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .online_softmax import causal_keep_from_positions, dense_attention_num_den, normalize_num_den


class SDPABackend:
    """Top-level exact attention backend backed by PyTorch SDPA."""

    name: str = "sdpa"
    exact: bool = True
    supports_training: bool = True
    supports_inference: bool = True

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
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q, k, and v must be rank-4 tensors [B, L, H, D].")
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(f"q, k, and v must have identical shapes. Got {q.shape}, {k.shape}, {v.shape}.")
        _, L, H, D = [int(x) for x in q.shape]
        if softmax_scale is None:
            softmax_scale = float(D) ** -0.5

        attn_mask = attention_mask
        use_is_causal = bool(causal) and positions is None and attention_mask is None
        if bool(causal) and not use_is_causal:
            causal_keep = causal_keep_from_positions(
                positions,
                positions,
                query_length=L,
                key_length=L,
                causal=True,
                device=q.device,
            )
            if attn_mask is None:
                attn_mask = causal_keep
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.to(device=q.device) & causal_keep
            else:
                additive = torch.zeros((L, L), dtype=torch.float32, device=q.device)
                additive = additive.masked_fill(~causal_keep, float("-inf"))
                attn_mask = attn_mask.to(device=q.device) + additive

        q_bhld = q.transpose(1, 2)
        k_bhld = k.transpose(1, 2)
        v_bhld = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q_bhld,
            k_bhld,
            v_bhld,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=bool(use_is_causal),
            scale=float(softmax_scale),
        ).transpose(1, 2).contiguous()

        if not return_num_den:
            return out

        num, den = dense_attention_num_den(
            q,
            k,
            v,
            positions=positions,
            causal=bool(causal),
            softmax_scale=float(softmax_scale),
            attention_mask=attention_mask if attention_mask is not None and attention_mask.dtype == torch.bool else None,
        )
        return out, {"num": num, "den": den, "dense_reference_out": normalize_num_den(num, den)}

    __call__ = forward

