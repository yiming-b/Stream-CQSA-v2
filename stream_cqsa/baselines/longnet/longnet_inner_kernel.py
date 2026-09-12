from __future__ import annotations

from itertools import product
from typing import Any, Sequence

import torch

from ...cqs_mask import CQS_mask
from ...offload import TransferTracker
from .longnet_mask import build_longnet_cqs_keep_mask


class LongNetInnerKernel:
    """
    Stream-CQSA-compatible inner kernel for simplified LongNet attention.

    Tensor layout is [B, L, H, D]. The kernel returns raw local softmax
    numerator/denominator terms, not normalized local outputs:

      Num_i: [B, L, H, D]
      Den_i: [B, H, L]
    """

    name: str = "longnet_inner"
    supports_training: bool = False
    supports_inference: bool = True

    def __init__(
        self,
        *,
        segment_lengths: Sequence[int] | Sequence[Sequence[int]] = (32, 64, 128),
        dilation_rates: Sequence[int] | Sequence[Sequence[int]] = (1, 2, 4),
        allow_local_position_fallback: bool = False,
    ) -> None:
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.allow_local_position_fallback = bool(allow_local_position_fallback)

    def _resolve_local_positions(
        self,
        *,
        local_positions: torch.Tensor | Sequence[int] | None,
        cqs_mask: dict[str, Any],
        local_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if local_positions is not None:
            return torch.as_tensor(local_positions, dtype=torch.long, device=device)
        for key in ("local_positions", "token_ids_cpu", "token_ids"):
            value = cqs_mask.get(key, None)
            if value is not None:
                return torch.as_tensor(value, dtype=torch.long, device=device)
        if self.allow_local_position_fallback:
            return torch.arange(int(local_size), dtype=torch.long, device=device)
        raise ValueError(
            "LongNetInnerKernel requires original global token positions. "
            "Pass local_positions or include token_ids/token_ids_cpu in cqs_mask."
        )

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
            raise ValueError("q, k, and v must be rank-4 tensors with layout [B, L, H, D].")
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(f"q, k, and v must have identical shapes. Got {q.shape}, {k.shape}, {v.shape}.")
        B, L, H, D = [int(x) for x in q.shape]
        if int(cqs_mask["local_size"]) != L:
            raise ValueError(f"cqs_mask local_size={cqs_mask['local_size']} does not match tensor L={L}")

        positions = self._resolve_local_positions(
            local_positions=local_positions,
            cqs_mask=cqs_mask,
            local_size=L,
            device=q.device,
        )
        keep = build_longnet_cqs_keep_mask(
            local_positions=positions,
            cqs_mask=cqs_mask,
            segment_lengths=self.segment_lengths,
            dilation_rates=self.dilation_rates,
            causal=bool(causal),
            num_heads=H,
            device=q.device,
        )
        if tuple(keep.shape) != (H, L, L):
            raise RuntimeError(f"Unexpected local keep mask shape {tuple(keep.shape)}, expected {(H, L, L)}")

        q_bhld = q.transpose(1, 2).float()
        k_bhld = k.transpose(1, 2).float()
        v_bhld = v.transpose(1, 2).float()
        scores = torch.matmul(q_bhld, k_bhld.transpose(-2, -1)) * float(softmax_scale)
        scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))

        weights = torch.exp(scores)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        den_i = weights.sum(dim=-1)
        num_i = torch.matmul(weights, v_bhld).transpose(1, 2).contiguous()
        num_i = torch.nan_to_num(num_i, nan=0.0, posinf=0.0, neginf=0.0)
        den_i = torch.nan_to_num(den_i, nan=0.0, posinf=0.0, neginf=0.0)

        if return_keep_mask:
            return num_i, den_i, keep
        return num_i, den_i

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


def stream_cqsa_longnet_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    positions: torch.Tensor | Sequence[int] | None = None,
    c: int = 7,
    interest_set: Sequence[int] = (0, 1, 3),
    itr: int = 1,
    causal: bool = True,
    softmax_scale: float | None = None,
    inner_kernel: LongNetInnerKernel | None = None,
    offload_intermediates: bool = True,
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """
    Pure Python Stream-CQSA(LongNet) reference runner.

    This mirrors Stream-CQSA's numerator/denominator merge but keeps everything in
    PyTorch tensors for small correctness experiments.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 tensors with layout [B, L, H, D].")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, and v must have identical shapes. Got {q.shape}, {k.shape}, {v.shape}.")
    B, N, H, D = [int(x) for x in q.shape]
    if softmax_scale is None:
        softmax_scale = float(D) ** -0.5
    tracker = TransferTracker(q.device)
    positions_t = tracker.time_stage(
        "t_prepare_ms",
        lambda: torch.arange(N, dtype=torch.long, device=q.device)
        if positions is None
        else torch.as_tensor(positions, dtype=torch.long, device=q.device),
    )
    if positions_t.ndim != 1 or int(positions_t.numel()) != N:
        raise ValueError(f"positions must be rank-1 with length {N}, got {tuple(positions_t.shape)}")

    kernel = inner_kernel if inner_kernel is not None else LongNetInnerKernel()
    mask_engine = CQS_mask(interest_set=tuple(int(x) for x in interest_set), c=int(c))
    offload = bool(offload_intermediates)
    merge_device = torch.device("cpu") if offload else q.device
    num_global = tracker.time_stage(
        "t_prepare_ms",
        lambda: torch.zeros((B, N, H, D), dtype=torch.float32, device=merge_device),
    )
    den_global = tracker.time_stage(
        "t_prepare_ms",
        lambda: torch.zeros((B, H, N), dtype=torch.float32, device=merge_device),
    )
    paths = tracker.time_stage(
        "t_prepare_ms",
        lambda: [tuple()] if int(itr) <= 0 else list(product(range(int(c)), repeat=int(itr))),
    )
    den_stats: list[dict[str, float | int | tuple[int, ...]]] = []

    for path in paths:
        def prepare_subproblem():
            mask_one_p = mask_engine.gen_mask(
                N=N,
                num_itr=int(itr),
                quorum_idx=list(path),
                interest_set=tuple(int(x) for x in interest_set),
                c=int(c),
                include_trace=False,
            )
            token_ids_cpu_p = torch.as_tensor(mask_one_p["token_ids"], dtype=torch.long, device=torch.device("cpu"))
            token_ids_p = token_ids_cpu_p.to(device=q.device)
            if int(token_ids_p.numel()) == 0:
                return mask_one_p, token_ids_cpu_p, token_ids_p, None, None, None, None
            q_i_p = q.index_select(1, token_ids_p)
            k_i_p = k.index_select(1, token_ids_p)
            v_i_p = v.index_select(1, token_ids_p)
            local_positions_p = positions_t.index_select(0, token_ids_p)
            return mask_one_p, token_ids_cpu_p, token_ids_p, local_positions_p, q_i_p, k_i_p, v_i_p

        mask_one, token_ids_cpu, token_ids, local_positions, q_i, k_i, v_i = tracker.time_stage(
            "t_prepare_ms", prepare_subproblem
        )
        if int(token_ids.numel()) == 0:
            del mask_one, token_ids_cpu, token_ids
            tracker.flush_cuda_cache()
            continue
        num_i, den_i = tracker.time_stage(
            "t_compute_ms",
            lambda: kernel.forward_num_den(
                q_i,
                k_i,
                v_i,
                local_positions=local_positions,
                cqs_mask=mask_one,
                causal=bool(causal),
                softmax_scale=float(softmax_scale),
            ),
        )
        if offload:
            num_i_merge, den_i_merge = tracker.to_cpu(num_i, den_i)
            token_ids_merge = token_ids_cpu
        else:
            num_i_merge, den_i_merge = num_i, den_i
            token_ids_merge = token_ids

        def merge_subproblem():
            num_global.index_add_(1, token_ids_merge, num_i_merge)
            den_global.index_add_(2, token_ids_merge, den_i_merge)
            den_i_stats = den_i_merge.detach()
            return {
                "path": tuple(int(x) for x in path),
                "local_size": int(token_ids.numel()),
                "den_min": float(den_i_stats.min().item()) if den_i_stats.numel() else 0.0,
                "den_max": float(den_i_stats.max().item()) if den_i_stats.numel() else 0.0,
                "zero_rows": int(den_i_stats.eq(0).sum().item()) if den_i_stats.numel() else 0,
            }

        den_stats.append(
            tracker.time_stage(
                "t_cpu_merge_ms",
                merge_subproblem,
            )
        )
        del (
            mask_one,
            token_ids_cpu,
            token_ids,
            token_ids_merge,
            q_i,
            k_i,
            v_i,
            local_positions,
            num_i,
            den_i,
            num_i_merge,
            den_i_merge,
        )
        tracker.flush_cuda_cache()

    def finalize():
        out_merge_f = torch.zeros_like(num_global)
        den_blhd = den_global.transpose(1, 2).unsqueeze(-1)
        valid = den_blhd.gt(0)
        return torch.where(valid, num_global / den_blhd.clamp_min(torch.finfo(torch.float32).tiny), out_merge_f)

    out_merge = tracker.time_stage("t_finalize_ms", finalize)
    out = tracker.to_device(out_merge) if offload else out_merge
    if not return_aux:
        return out
    transfer_stats = tracker.as_dict()
    return out, {
        "num": num_global,
        "den": den_global,
        "paths": paths,
        "den_stats": den_stats,
        "num_subseq": int(len(paths)),
        "cpu_offload": offload,
        "communication_ms": transfer_stats["communication_ms"],
        "transfer_stats": transfer_stats,
        "itr": int(itr),
        "c": int(c),
        "interest_set": tuple(int(x) for x in interest_set),
    }
