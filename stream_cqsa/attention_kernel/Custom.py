from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch


def _group_bits_to_dense_mask(
    *,
    group_bits: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Build dense local mask [L, L] from per-token int64 group bits.
    True means masked.
    """
    bits = group_bits.to(device=device, dtype=torch.int64).contiguous()
    bits_row = bits.view(bits.numel(), 1)
    bits_col = bits.view(1, bits.numel())
    return torch.bitwise_and(bits_row, bits_col).ne(0)


def _group_runs_to_dense_mask(
    *,
    local_size: int,
    group_runs: Sequence[Sequence[Tuple[int, int]]],
    device: torch.device,
) -> torch.Tensor:
    """
    Build dense local mask [L, L] where True means masked.
    """
    mask = torch.zeros((int(local_size), int(local_size)), dtype=torch.bool, device=device)
    for runs in group_runs:
        idx_parts: List[torch.Tensor] = []
        for s, e in runs:
            si, ei = int(s), int(e)
            if ei > si:
                idx_parts.append(torch.arange(si, ei, device=device, dtype=torch.long))
        if len(idx_parts) == 0:
            continue
        idx = idx_parts[0] if len(idx_parts) == 1 else torch.cat(idx_parts, dim=0)
        mask[idx.unsqueeze(1), idx.unsqueeze(0)] = True
    return mask


def custom_attn(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cqs_mask: Dict[str, Any],
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Naive reference subsequence attention plugin.

    Steps:
    1) R_i = (Q @ K^T) * scale
    2) apply CQS mask M_i on R_i (masked positions -> -inf)
    3) P_i = exp(R_i)   (no softmax normalization)
    4) Den_i = row_sum(P_i)
    5) Num_i = P_i @ V

    Returns:
    - Num_i: [B, L, H, D] float32
    - Den_i: [B, H, L] float32
    """
    local_size = int(cqs_mask["local_size"])
    group_bits_cpu = cqs_mask.get("group_bits_cpu", None)
    if isinstance(group_bits_cpu, torch.Tensor):
        M_i = _group_bits_to_dense_mask(
            group_bits=group_bits_cpu,
            device=q.device,
        )
    else:
        M_i = _group_runs_to_dense_mask(
            local_size=local_size,
            group_runs=cqs_mask["group_runs"],
            device=q.device,
        )  # [L, L], True means masked

    q_bhld = q.transpose(1, 2).float()  # [B, H, L, D]
    k_bhld = k.transpose(1, 2).float()  # [B, H, L, D]
    v_bhld = v.transpose(1, 2).float()  # [B, H, L, D]

    R_i = torch.matmul(q_bhld, k_bhld.transpose(-2, -1)) * float(softmax_scale)  # [B, H, L, L]
    if bool(M_i.any().item()):
        R_i = R_i.masked_fill(M_i.unsqueeze(0).unsqueeze(0), float("-inf"))

    P_i = torch.exp(R_i)  # [B, H, L, L]
    P_i = torch.nan_to_num(P_i, nan=0.0, posinf=0.0, neginf=0.0)

    Den_i = P_i.sum(dim=-1)  # [B, H, L]
    Num_i = torch.matmul(P_i, v_bhld).transpose(1, 2).contiguous()  # [B, L, H, D]
    Num_i = torch.nan_to_num(Num_i, nan=0.0, posinf=0.0, neginf=0.0)
    Den_i = torch.nan_to_num(Den_i, nan=0.0, posinf=0.0, neginf=0.0)
    return Num_i, Den_i
