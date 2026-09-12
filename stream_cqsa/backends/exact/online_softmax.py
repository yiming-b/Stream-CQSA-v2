from __future__ import annotations

from typing import Any, Sequence

import torch


def _positions_1d(
    positions: torch.Tensor | Sequence[int] | None,
    *,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    if positions is None:
        return torch.arange(int(length), dtype=torch.long, device=device)
    pos = torch.as_tensor(positions, dtype=torch.long, device=device)
    if pos.ndim == 2:
        first = pos[0]
        if bool((pos != first.unsqueeze(0)).any().item()):
            raise ValueError("Batched positions are supported only when each batch row has the same positions.")
        pos = first
    if pos.ndim != 1 or int(pos.numel()) != int(length):
        raise ValueError(f"Expected rank-1 positions with length {length}, got {tuple(pos.shape)}")
    return pos.contiguous()


def causal_keep_from_positions(
    query_positions: torch.Tensor | Sequence[int] | None,
    key_positions: torch.Tensor | Sequence[int] | None = None,
    *,
    query_length: int,
    key_length: int | None = None,
    causal: bool,
    device: torch.device,
) -> torch.Tensor:
    key_length_i = int(query_length if key_length is None else key_length)
    if not bool(causal):
        return torch.ones((int(query_length), key_length_i), dtype=torch.bool, device=device)
    q_pos = _positions_1d(query_positions, length=int(query_length), device=device)
    k_pos = _positions_1d(key_positions if key_positions is not None else query_positions, length=key_length_i, device=device)
    return k_pos.view(1, key_length_i).le(q_pos.view(int(query_length), 1))


def cqs_keep_mask(cqs_mask: dict[str, Any], *, device: torch.device | str | None = None) -> torch.Tensor:
    """Return a dense local CQS keep mask [L, L], where True means the pair contributes."""
    local_size = int(cqs_mask["local_size"])
    group_bits = cqs_mask.get("group_bits_cpu", cqs_mask.get("group_bits", None))
    if device is None:
        device_t = group_bits.device if isinstance(group_bits, torch.Tensor) else torch.device("cpu")
    else:
        device_t = torch.device(device)

    if isinstance(group_bits, torch.Tensor):
        bits = group_bits.to(device=device_t, dtype=torch.int64).contiguous()
        bits_row = bits.view(local_size, 1)
        bits_col = bits.view(1, local_size)
        masked = torch.bitwise_and(bits_row, bits_col).ne(0)
        return ~masked

    masked = torch.zeros((local_size, local_size), dtype=torch.bool, device=device_t)
    for runs in cqs_mask.get("group_runs", []):
        idx_parts: list[torch.Tensor] = []
        for s, e in runs:
            si, ei = int(s), int(e)
            if ei > si:
                idx_parts.append(torch.arange(si, ei, dtype=torch.long, device=device_t))
        if not idx_parts:
            continue
        idx = idx_parts[0] if len(idx_parts) == 1 else torch.cat(idx_parts, dim=0)
        masked[idx.unsqueeze(1), idx.unsqueeze(0)] = True
    return ~masked


def normalize_num_den(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """Normalize Num:[B,L,H,D] by Den:[B,H,L], returning [B,L,H,D]."""
    den_blhd = den.float().transpose(1, 2).unsqueeze(-1)
    out = torch.zeros_like(num.float())
    return torch.where(
        den_blhd.gt(0),
        num.float() / den_blhd.clamp_min(torch.finfo(torch.float32).tiny),
        out,
    )


def _combine_keep_mask(
    *,
    keep_mask: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    H: int,
    Lq: int,
    Lk: int,
    device: torch.device,
) -> torch.Tensor | None:
    keep = None
    if keep_mask is not None:
        keep = keep_mask.to(device=device, dtype=torch.bool)
        if keep.ndim == 2:
            keep = keep.unsqueeze(0).expand(int(H), -1, -1)
        if tuple(keep.shape) != (int(H), int(Lq), int(Lk)):
            raise ValueError(f"keep_mask shape {tuple(keep.shape)} does not match {(H, Lq, Lk)}")
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attn_keep = attention_mask.to(device=device)
            if attn_keep.ndim == 2:
                attn_keep = attn_keep.unsqueeze(0).expand(int(H), -1, -1)
            if tuple(attn_keep.shape) != (int(H), int(Lq), int(Lk)):
                raise ValueError(f"attention_mask shape {tuple(attn_keep.shape)} does not match {(H, Lq, Lk)}")
            keep = attn_keep if keep is None else (keep & attn_keep)
        else:
            raise ValueError("Numeric additive attention_mask is supported by SDPABackend only, not num/den helpers.")
    return keep


def dense_attention_num_den(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    keep_mask: torch.Tensor | None = None,
    positions: torch.Tensor | Sequence[int] | None = None,
    causal: bool = False,
    softmax_scale: float | None = None,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Explicit dense exact attention numerator/denominator.

    Tensor layout is [B, L, H, D]. Returns Num:[B,L,H,D], Den:[B,H,L].
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 tensors [B, L, H, D].")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, and v must have identical shapes. Got {q.shape}, {k.shape}, {v.shape}.")
    B, L, H, D = [int(x) for x in q.shape]
    if softmax_scale is None:
        softmax_scale = float(D) ** -0.5
    keep = _combine_keep_mask(
        keep_mask=keep_mask,
        attention_mask=attention_mask,
        H=H,
        Lq=L,
        Lk=L,
        device=q.device,
    )
    if bool(causal):
        causal_keep = causal_keep_from_positions(
            positions,
            positions,
            query_length=L,
            key_length=L,
            causal=True,
            device=q.device,
        )
        keep = causal_keep.unsqueeze(0).expand(H, -1, -1) if keep is None else (keep & causal_keep.unsqueeze(0))

    q_bhld = q.transpose(1, 2).float()
    k_bhld = k.transpose(1, 2).float()
    v_bhld = v.transpose(1, 2).float()
    scores = torch.matmul(q_bhld, k_bhld.transpose(-2, -1)) * float(softmax_scale)
    if keep is not None:
        scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
    weights = torch.exp(scores)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    den = weights.sum(dim=-1)
    num = torch.matmul(weights, v_bhld).transpose(1, 2).contiguous()
    return torch.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0), torch.nan_to_num(
        den, nan=0.0, posinf=0.0, neginf=0.0
    )


def rectangular_attention_num_den(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    keep_mask: torch.Tensor | None = None,
    positions: torch.Tensor | Sequence[int] | None = None,
    causal: bool = False,
    softmax_scale: float | None = None,
    q_block_size: int = 256,
    kv_block_size: int = 256,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """
    Rectangular blocked exact attention using online softmax.

    Returns raw Num/Den terms for compatibility with Stream-CQSA's current merge.
    The online state is rescaled by exp(row_max) at the end of each Q block.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 tensors [B, L, H, D].")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, and v must have identical shapes. Got {q.shape}, {k.shape}, {v.shape}.")
    B, L, H, D = [int(x) for x in q.shape]
    qb = int(q_block_size)
    kb = int(kv_block_size)
    if qb <= 0 or kb <= 0:
        raise ValueError(f"q_block_size and kv_block_size must be positive, got {qb}, {kb}")
    if softmax_scale is None:
        softmax_scale = float(D) ** -0.5

    keep = _combine_keep_mask(
        keep_mask=keep_mask,
        attention_mask=attention_mask,
        H=H,
        Lq=L,
        Lk=L,
        device=q.device,
    )
    if bool(causal):
        causal_keep = causal_keep_from_positions(
            positions,
            positions,
            query_length=L,
            key_length=L,
            causal=True,
            device=q.device,
        )
        keep = causal_keep.unsqueeze(0).expand(H, -1, -1) if keep is None else (keep & causal_keep.unsqueeze(0))

    q_bhld = q.transpose(1, 2).float()
    k_bhld = k.transpose(1, 2).float()
    v_bhld = v.transpose(1, 2).float()
    num_out = torch.zeros((B, L, H, D), dtype=torch.float32, device=q.device)
    den_out = torch.zeros((B, H, L), dtype=torch.float32, device=q.device)
    blocks = 0

    for q0 in range(0, L, qb):
        q1 = min(L, q0 + qb)
        q_blk = q_bhld[:, :, q0:q1, :]
        q_len = int(q1 - q0)
        m = torch.full((B, H, q_len), float("-inf"), dtype=torch.float32, device=q.device)
        l = torch.zeros((B, H, q_len), dtype=torch.float32, device=q.device)
        num = torch.zeros((B, H, q_len, D), dtype=torch.float32, device=q.device)

        for k0 in range(0, L, kb):
            k1 = min(L, k0 + kb)
            scores = torch.matmul(q_blk, k_bhld[:, :, k0:k1, :].transpose(-2, -1)) * float(softmax_scale)
            if keep is not None:
                scores = scores.masked_fill(~keep[:, q0:q1, k0:k1].unsqueeze(0), float("-inf"))
            m_block = scores.max(dim=-1).values
            m_new = torch.maximum(m, m_block)
            finite_new = torch.isfinite(m_new)
            old_scale = torch.where(torch.isfinite(m), torch.exp(m - m_new), torch.zeros_like(m_new))
            shifted = scores - m_new.unsqueeze(-1)
            shifted = torch.where(
                finite_new.unsqueeze(-1),
                shifted,
                torch.full_like(shifted, float("-inf")),
            )
            p = torch.exp(shifted)
            p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            l = old_scale * l + p.sum(dim=-1)
            num = old_scale.unsqueeze(-1) * num + torch.matmul(p, v_bhld[:, :, k0:k1, :])
            m = m_new
            blocks += 1

        raw_scale = torch.where(torch.isfinite(m), torch.exp(m), torch.zeros_like(m))
        den_raw = l * raw_scale
        num_raw = num * raw_scale.unsqueeze(-1)
        den_out[:, :, q0:q1] = torch.nan_to_num(den_raw, nan=0.0, posinf=0.0, neginf=0.0)
        num_out[:, q0:q1, :, :] = torch.nan_to_num(
            num_raw.transpose(1, 2).contiguous(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    stats = {
        "q_block_size": int(qb),
        "kv_block_size": int(kb),
        "num_q_blocks": int((L + qb - 1) // qb),
        "num_kv_blocks": int((L + kb - 1) // kb),
        "num_score_blocks": int(blocks),
        "streaming_mode": "gpu",
        "transfer_bytes": 0,
    }
    return num_out, den_out, stats

