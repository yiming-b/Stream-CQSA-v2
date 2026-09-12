from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch

from ..interface import flash_attn_func_cqs_group_bits


def _group_runs_to_group_bits(local_size: int, group_runs: Sequence[Sequence[Tuple[int, int]]]) -> np.ndarray:
    bits = np.zeros((local_size,), dtype=np.int64)
    for bit_id, runs in enumerate(group_runs):
        if bit_id >= 63:
            raise ValueError("Too many unique group runs for int64 bit encoding.")
        bit = np.int64(1) << np.int64(bit_id)
        for s, e in runs:
            si, ei = int(s), int(e)
            if ei > si:
                bits[si:ei] |= bit
    return bits


def default_subsequence_attention(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cqs_mask: Dict[str, Any],
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FlashAttention-based default subsequence kernel.

    Returns:
    - Num_i: [B, L, H, D] float32 numerator
    - Den_i: [B, H, L] float32 denominator
    """
    B_q, L_q, H_q, D_q = [int(x) for x in q.shape]
    local_size = int(cqs_mask["local_size"])
    group_bits_cpu = cqs_mask.get("group_bits_cpu", None)
    if isinstance(group_bits_cpu, torch.Tensor):
        group_bits = group_bits_cpu.to(q.device, dtype=torch.int64, non_blocking=True)
    else:
        group_bits_np = _group_runs_to_group_bits(local_size, cqs_mask["group_runs"])
        group_bits = torch.from_numpy(group_bits_np).to(q.device, dtype=torch.int64, non_blocking=True)

    attn_out_i, log_den_i, _ = flash_attn_func_cqs_group_bits(
        q,
        k,
        v,
        group_bits,
        dropout_p=0.0,
        softmax_scale=float(softmax_scale),
        causal=False,
        return_attn_probs=True,
    )

    # Normalize denominator layout to [B, H, L]
    den_i = torch.nan_to_num(torch.exp(log_den_i.float()), nan=0.0, posinf=0.0, neginf=0.0)
    if den_i.ndim != 3:
        raise RuntimeError(f"Unexpected denominator rank: {tuple(den_i.shape)}")
    if tuple(den_i.shape) == (B_q, H_q, L_q):
        pass
    elif tuple(den_i.shape) == (B_q, L_q, H_q):
        den_i = den_i.transpose(1, 2).contiguous()
    else:
        raise RuntimeError(
            "Unexpected denominator layout from FA kernel: "
            f"{tuple(den_i.shape)} (expected {(B_q, H_q, L_q)} or {(B_q, L_q, H_q)})"
        )

    # Normalize attention output layout to [B, L, H, D]
    num_i = attn_out_i.float()
    if num_i.ndim != 4:
        raise RuntimeError(f"Unexpected attention output rank: {tuple(num_i.shape)}")
    if tuple(num_i.shape) == (B_q, L_q, H_q, D_q):
        pass
    elif tuple(num_i.shape) == (B_q, H_q, L_q, D_q):
        num_i = num_i.transpose(1, 2).contiguous()
    else:
        raise RuntimeError(
            "Unexpected attention output layout from FA kernel: "
            f"{tuple(num_i.shape)} (expected {(B_q, L_q, H_q, D_q)} "
            f"or {(B_q, H_q, L_q, D_q)})"
        )

    den_i_t = den_i.transpose(1, 2).unsqueeze(-1)  # [B, L, H, 1]
    num_i.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    num_i.mul_(den_i_t)
    num_i.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    return num_i, den_i
