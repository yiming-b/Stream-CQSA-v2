from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class LongNetSchedule:
    """Simplified LongNet-style dilated mask schedule."""

    segment_lengths: tuple[int, ...] = (32, 64, 128)
    dilation_rates: tuple[int, ...] = (1, 2, 4)

    def __post_init__(self) -> None:
        if len(self.segment_lengths) == 0:
            raise ValueError("segment_lengths must be non-empty")
        if len(self.segment_lengths) != len(self.dilation_rates):
            raise ValueError(
                "segment_lengths and dilation_rates must have the same length. "
                f"Got {len(self.segment_lengths)} and {len(self.dilation_rates)}."
            )
        for name, values in (
            ("segment_lengths", self.segment_lengths),
            ("dilation_rates", self.dilation_rates),
        ):
            bad = [int(x) for x in values if int(x) <= 0]
            if bad:
                raise ValueError(f"{name} must contain positive integers, got {bad}")

    @classmethod
    def from_values(
        cls,
        segment_lengths: Sequence[int],
        dilation_rates: Sequence[int],
    ) -> "LongNetSchedule":
        return cls(
            segment_lengths=tuple(int(x) for x in segment_lengths),
            dilation_rates=tuple(int(x) for x in dilation_rates),
        )


def _is_nested_schedule(values: Sequence[Any]) -> bool:
    if len(values) == 0:
        return False
    first = values[0]
    return isinstance(first, (list, tuple))


def _normalize_head_schedules(
    *,
    segment_lengths: Sequence[int] | Sequence[Sequence[int]],
    dilation_rates: Sequence[int] | Sequence[Sequence[int]],
    num_heads: int | None,
) -> list[LongNetSchedule]:
    seg_any = list(segment_lengths)
    dil_any = list(dilation_rates)
    seg_nested = _is_nested_schedule(seg_any)
    dil_nested = _is_nested_schedule(dil_any)
    if seg_nested != dil_nested:
        raise ValueError("segment_lengths and dilation_rates must both be flat or both be per-head nested.")

    if seg_nested:
        if num_heads is not None and len(seg_any) != int(num_heads):
            raise ValueError(f"Expected {num_heads} per-head schedules, got {len(seg_any)}")
        if len(seg_any) != len(dil_any):
            raise ValueError("Per-head segment_lengths and dilation_rates must have the same outer length.")
        return [
            LongNetSchedule.from_values(seg_values, dil_values)
            for seg_values, dil_values in zip(seg_any, dil_any)
        ]

    schedule = LongNetSchedule.from_values(seg_any, dil_any)
    n_heads = 1 if num_heads is None else int(num_heads)
    if n_heads <= 0:
        raise ValueError(f"num_heads must be positive when provided, got {num_heads}")
    return [schedule for _ in range(n_heads)]


def _positions_1d(
    positions: torch.Tensor | Sequence[int] | None,
    *,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    if positions is None:
        return torch.arange(int(length), device=device, dtype=torch.long)
    if not isinstance(positions, torch.Tensor):
        positions_t = torch.as_tensor(positions, dtype=torch.long, device=device)
    else:
        positions_t = positions.to(device=device, dtype=torch.long)
    if positions_t.ndim == 2:
        first = positions_t[0]
        if bool((positions_t != first.unsqueeze(0)).any().item()):
            raise ValueError("Batched positions are supported only when every batch row has the same positions.")
        positions_t = first
    if positions_t.ndim != 1:
        raise ValueError(f"positions must be rank 1 or rank 2, got shape {tuple(positions_t.shape)}")
    if int(positions_t.numel()) != int(length):
        raise ValueError(f"Expected {length} positions, got {int(positions_t.numel())}")
    return positions_t.contiguous()


def build_longnet_keep_mask(
    positions: torch.Tensor | Sequence[int] | None,
    *,
    key_positions: torch.Tensor | Sequence[int] | None = None,
    segment_lengths: Sequence[int] | Sequence[Sequence[int]] = (32, 64, 128),
    dilation_rates: Sequence[int] | Sequence[Sequence[int]] = (1, 2, 4),
    causal: bool = True,
    num_heads: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Build a global-position-aware simplified LongNet keep mask.

    Returns:
      - [Lq, Lk] when num_heads is None and the schedule is flat
      - [H, Lq, Lk] when num_heads is provided or schedules are per-head

    Simplified rule used by this reproduction:
      keep(q, k) if any schedule entry satisfies
      global_distance < segment_length * dilation_rate and
      global_distance % dilation_rate == 0.

    In causal mode, global_distance = q_position - k_position and must be >= 0.
    In non-causal mode, global_distance = abs(q_position - k_position).
    """
    if device is None:
        if isinstance(positions, torch.Tensor):
            device_t = positions.device
        elif isinstance(key_positions, torch.Tensor):
            device_t = key_positions.device
        else:
            device_t = torch.device("cpu")
    else:
        device_t = torch.device(device)

    q_len = int(len(positions)) if positions is not None and not isinstance(positions, torch.Tensor) else None
    if isinstance(positions, torch.Tensor):
        q_len = int(positions.shape[-1])
    if q_len is None:
        raise ValueError("positions must be provided when length cannot be inferred")

    if key_positions is None:
        k_len = q_len
    elif isinstance(key_positions, torch.Tensor):
        k_len = int(key_positions.shape[-1])
    else:
        k_len = int(len(key_positions))

    q_pos = _positions_1d(positions, length=q_len, device=device_t)
    k_pos = _positions_1d(key_positions if key_positions is not None else positions, length=k_len, device=device_t)
    schedules = _normalize_head_schedules(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        num_heads=num_heads,
    )

    delta = q_pos.view(q_len, 1) - k_pos.view(1, k_len)
    if causal:
        distance = delta
        direction_keep = delta.ge(0)
    else:
        distance = delta.abs()
        direction_keep = torch.ones((q_len, k_len), dtype=torch.bool, device=device_t)

    head_masks: list[torch.Tensor] = []
    for schedule in schedules:
        keep = torch.zeros((q_len, k_len), dtype=torch.bool, device=device_t)
        for segment_length, dilation_rate in zip(schedule.segment_lengths, schedule.dilation_rates):
            span = int(segment_length) * int(dilation_rate)
            keep_entry = direction_keep & distance.lt(int(span)) & torch.remainder(distance, int(dilation_rate)).eq(0)
            keep |= keep_entry
        head_masks.append(keep)

    if num_heads is None and not _is_nested_schedule(list(segment_lengths)):
        return head_masks[0]
    return torch.stack(head_masks, dim=0)


def cqs_masked_pairs(cqs_mask: dict[str, Any], *, device: torch.device | str | None = None) -> torch.Tensor:
    """Return a dense local CQS mask [L, L], where True means the pair is excluded by CQS."""
    local_size = int(cqs_mask["local_size"])
    if device is None:
        group_bits = cqs_mask.get("group_bits_cpu", None)
        device_t = group_bits.device if isinstance(group_bits, torch.Tensor) else torch.device("cpu")
    else:
        device_t = torch.device(device)

    group_bits = cqs_mask.get("group_bits_cpu", cqs_mask.get("group_bits", None))
    if isinstance(group_bits, torch.Tensor):
        bits = group_bits.to(device=device_t, dtype=torch.int64).contiguous()
        bits_row = bits.view(local_size, 1)
        bits_col = bits.view(1, local_size)
        return torch.bitwise_and(bits_row, bits_col).ne(0)

    mask = torch.zeros((local_size, local_size), dtype=torch.bool, device=device_t)
    for runs in cqs_mask.get("group_runs", []):
        idx_parts: list[torch.Tensor] = []
        for s, e in runs:
            si, ei = int(s), int(e)
            if ei > si:
                idx_parts.append(torch.arange(si, ei, dtype=torch.long, device=device_t))
        if not idx_parts:
            continue
        idx = idx_parts[0] if len(idx_parts) == 1 else torch.cat(idx_parts, dim=0)
        mask[idx.unsqueeze(1), idx.unsqueeze(0)] = True
    return mask


def build_longnet_cqs_keep_mask(
    *,
    local_positions: torch.Tensor | Sequence[int],
    cqs_mask: dict[str, Any],
    segment_lengths: Sequence[int] | Sequence[Sequence[int]] = (32, 64, 128),
    dilation_rates: Sequence[int] | Sequence[Sequence[int]] = (1, 2, 4),
    causal: bool = True,
    num_heads: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Build Keep_r = CQS_keep AND LongNet_keep AND causal_keep for a CQS subproblem.

    Returns [H, L, L], where True means the local query-key pair contributes to
    the unnormalized numerator and denominator.
    """
    device_t = torch.device(device) if device is not None else (
        local_positions.device if isinstance(local_positions, torch.Tensor) else torch.device("cpu")
    )
    longnet_keep = build_longnet_keep_mask(
        local_positions,
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        causal=causal,
        num_heads=int(num_heads),
        device=device_t,
    )
    if longnet_keep.ndim == 2:
        longnet_keep = longnet_keep.unsqueeze(0).expand(int(num_heads), -1, -1)
    cqs_keep = ~cqs_masked_pairs(cqs_mask, device=device_t)
    return longnet_keep & cqs_keep.unsqueeze(0)

