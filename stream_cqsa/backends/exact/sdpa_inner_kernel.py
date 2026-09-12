from __future__ import annotations

from itertools import product
from typing import Any, Sequence

import torch

from ...cqs_mask import CQS_mask
from ...offload import TransferTracker
from .online_softmax import causal_keep_from_positions, cqs_keep_mask, normalize_num_den


def _safe_raw_scale(row_max: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(row_max), torch.exp(row_max), torch.zeros_like(row_max))


def normalize_scaled_num_den(num_scaled: torch.Tensor, den_scaled: torch.Tensor) -> torch.Tensor:
    """Normalize scaled online-softmax state without reconstructing raw exp(score)."""
    return normalize_num_den(num_scaled, den_scaled)


class DenseExactInnerKernel:
    """
    Stream-CQSA-compatible exact dense local attention kernel.

    This dense exact inner path exposes local numerator/denominator state because
    PyTorch SDPA returns only normalized outputs. For Stream-CQSA recomposition,
    the stable path uses row-max shifted online-softmax state rather than raw
    exp(score) accumulators.
    """

    name: str = "dense_exact_inner"
    supports_training: bool = True
    supports_inference: bool = True
    supports_stable_recomposition: bool = True
    scalability_note: str = "Dense exact local scores; stable row-max shifted Num/Den state for CQS recomposition."

    def __init__(self, *, allow_local_position_fallback: bool = False) -> None:
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
            "DenseExactInnerKernel requires original global token positions. "
            "Pass local_positions or include token_ids/token_ids_cpu in cqs_mask."
        )

    def keep_mask(
        self,
        *,
        local_positions: torch.Tensor | Sequence[int] | None,
        cqs_mask: dict[str, Any],
        causal: bool,
        num_heads: int,
        device: torch.device,
    ) -> torch.Tensor:
        local_size = int(cqs_mask["local_size"])
        positions = self._resolve_local_positions(
            local_positions=local_positions,
            cqs_mask=cqs_mask,
            local_size=local_size,
            device=device,
        )
        keep = cqs_keep_mask(cqs_mask, device=device)
        if bool(causal):
            causal_keep = causal_keep_from_positions(
                positions,
                positions,
                query_length=local_size,
                key_length=local_size,
                causal=True,
                device=device,
            )
            keep = keep & causal_keep
        return keep.unsqueeze(0).expand(int(num_heads), -1, -1)

    def _scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        keep: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        q_bhld = q.transpose(1, 2).float()
        k_bhld = k.transpose(1, 2).float()
        scores = torch.matmul(q_bhld, k_bhld.transpose(-2, -1)) * float(softmax_scale)
        return scores.masked_fill(~keep.unsqueeze(0), float("-inf"))

    def _validate_local_inputs(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cqs_mask: dict[str, Any]) -> tuple[int, int]:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q, k, and v must be rank-4 tensors [B, L, H, D].")
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(f"q, k, and v must have identical shapes. Got {q.shape}, {k.shape}, {v.shape}.")
        _, L, H, _ = [int(x) for x in q.shape]
        if int(cqs_mask["local_size"]) != L:
            raise ValueError(f"cqs_mask local_size={cqs_mask['local_size']} does not match tensor L={L}")
        return L, H

    def forward_row_max(
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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        L, H = self._validate_local_inputs(q, k, v, cqs_mask)
        keep = self.keep_mask(
            local_positions=local_positions,
            cqs_mask=cqs_mask,
            causal=bool(causal),
            num_heads=H,
            device=q.device,
        )
        row_max = self._scores(q, k, keep=keep, softmax_scale=float(softmax_scale)).max(dim=-1).values
        if return_keep_mask:
            return row_max, keep
        return row_max

    def forward_scaled_num_den(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        row_max: torch.Tensor,
        local_positions: torch.Tensor | Sequence[int] | None = None,
        cqs_mask: dict[str, Any],
        causal: bool,
        softmax_scale: float,
        return_keep_mask: bool = False,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        L, H = self._validate_local_inputs(q, k, v, cqs_mask)
        if tuple(row_max.shape) != (int(q.shape[0]), H, L):
            raise ValueError(f"row_max shape {tuple(row_max.shape)} does not match {(int(q.shape[0]), H, L)}")
        keep = self.keep_mask(
            local_positions=local_positions,
            cqs_mask=cqs_mask,
            causal=bool(causal),
            num_heads=H,
            device=q.device,
        )
        scores = self._scores(q, k, keep=keep, softmax_scale=float(softmax_scale))
        shifted = scores - row_max.to(device=q.device, dtype=torch.float32).unsqueeze(-1)
        shifted = torch.where(
            torch.isfinite(row_max).to(device=q.device).unsqueeze(-1),
            shifted,
            torch.full_like(shifted, float("-inf")),
        )
        weights = torch.exp(shifted)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        den_scaled = weights.sum(dim=-1)
        num_scaled = torch.matmul(weights, v.transpose(1, 2).float()).transpose(1, 2).contiguous()
        num_scaled = torch.nan_to_num(num_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        den_scaled = torch.nan_to_num(den_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        if return_keep_mask:
            return num_scaled, den_scaled, keep
        return num_scaled, den_scaled

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
        row_max, keep = self.forward_row_max(
            q,
            k,
            v,
            local_positions=local_positions,
            cqs_mask=cqs_mask,
            causal=bool(causal),
            softmax_scale=float(softmax_scale),
            return_keep_mask=True,
        )
        num_scaled, den_scaled = self.forward_scaled_num_den(
            q,
            k,
            v,
            row_max=row_max,
            local_positions=local_positions,
            cqs_mask=cqs_mask,
            causal=bool(causal),
            softmax_scale=float(softmax_scale),
        )
        raw_scale = _safe_raw_scale(row_max)
        den = den_scaled * raw_scale
        num = num_scaled * raw_scale.transpose(1, 2).unsqueeze(-1)
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


class SDPAInnerKernel(DenseExactInnerKernel):
    """Backward-compatible alias for the dense exact Stream-CQSA inner kernel."""


def stream_cqsa_exact_reference(
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
    inner_kernel: Any | None = None,
    offload_intermediates: bool = True,
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Small pure-PyTorch reference Stream-CQSA runner for exact inner kernels."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 tensors [B, L, H, D].")
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

    kernel = inner_kernel if inner_kernel is not None else DenseExactInnerKernel()
    mask_engine = CQS_mask(interest_set=tuple(int(x) for x in interest_set), c=int(c))
    paths = tracker.time_stage(
        "t_prepare_ms",
        lambda: [tuple()] if int(itr) <= 0 else list(product(range(int(c)), repeat=int(itr))),
    )
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
    den_stats: list[dict[str, Any]] = []

    if bool(getattr(kernel, "supports_stable_recomposition", False)):
        row_max_global = tracker.time_stage(
            "t_prepare_ms",
            lambda: torch.full((B, H, N), float("-inf"), dtype=torch.float32, device=merge_device),
        )

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
                local_positions_p = positions_t.index_select(0, token_ids_p)
                q_i_p = q.index_select(1, token_ids_p)
                k_i_p = k.index_select(1, token_ids_p)
                v_i_p = v.index_select(1, token_ids_p)
                return mask_one_p, token_ids_cpu_p, token_ids_p, local_positions_p, q_i_p, k_i_p, v_i_p

            mask_one, token_ids_cpu, token_ids, local_positions, q_i, k_i, v_i = tracker.time_stage(
                "t_prepare_ms", prepare_subproblem
            )
            if int(token_ids.numel()) == 0:
                del mask_one, token_ids_cpu, token_ids
                tracker.flush_cuda_cache()
                continue

            local_max = tracker.time_stage(
                "t_compute_ms",
                lambda: kernel.forward_row_max(
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
                local_max_merge = tracker.to_cpu(local_max)[0]
                token_ids_merge = token_ids_cpu
            else:
                local_max_merge = local_max
                token_ids_merge = token_ids

            def merge_row_max():
                current = row_max_global.index_select(2, token_ids_merge)
                row_max_global.index_copy_(2, token_ids_merge, torch.maximum(current, local_max_merge))

            tracker.time_stage("t_cpu_merge_ms", merge_row_max)
            del (
                mask_one,
                token_ids_cpu,
                token_ids,
                token_ids_merge,
                q_i,
                k_i,
                v_i,
                local_max,
                local_max_merge,
                local_positions,
            )
            tracker.flush_cuda_cache()

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
                    return mask_one_p, token_ids_cpu_p, token_ids_p, None, None, None, None, None
                local_positions_p = positions_t.index_select(0, token_ids_p)
                q_i_p = q.index_select(1, token_ids_p)
                k_i_p = k.index_select(1, token_ids_p)
                v_i_p = v.index_select(1, token_ids_p)
                row_max_i_p = row_max_global.index_select(2, token_ids_cpu_p if offload else token_ids_p)
                if offload:
                    row_max_i_p = tracker.to_device(row_max_i_p)
                return mask_one_p, token_ids_cpu_p, token_ids_p, local_positions_p, q_i_p, k_i_p, v_i_p, row_max_i_p

            mask_one, token_ids_cpu, token_ids, local_positions, q_i, k_i, v_i, row_max_i = tracker.time_stage(
                "t_prepare_ms", prepare_subproblem
            )
            if int(token_ids.numel()) == 0:
                del mask_one, token_ids_cpu, token_ids
                tracker.flush_cuda_cache()
                continue
            num_i, den_i = tracker.time_stage(
                "t_compute_ms",
                lambda: kernel.forward_scaled_num_den(
                    q_i,
                    k_i,
                    v_i,
                    row_max=row_max_i,
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

            def merge_scaled_subproblem():
                num_global.index_add_(1, token_ids_merge, num_i_merge)
                den_global.index_add_(2, token_ids_merge, den_i_merge)
                den_i_stats = den_i_merge.detach()
                return {
                    "path": tuple(int(x) for x in path),
                    "local_size": int(token_ids.numel()),
                    "den_scaled_min": float(den_i_stats.min().item()) if den_i_stats.numel() else 0.0,
                    "den_scaled_max": float(den_i_stats.max().item()) if den_i_stats.numel() else 0.0,
                    "zero_rows": int(den_i_stats.eq(0).sum().item()) if den_i_stats.numel() else 0,
                }

            den_stats.append(tracker.time_stage("t_cpu_merge_ms", merge_scaled_subproblem))
            del (
                mask_one,
                token_ids_cpu,
                token_ids,
                token_ids_merge,
                q_i,
                k_i,
                v_i,
                num_i,
                den_i,
                num_i_merge,
                den_i_merge,
                row_max_i,
                local_positions,
            )
            tracker.flush_cuda_cache()

        out_merge = tracker.time_stage("t_finalize_ms", lambda: normalize_scaled_num_den(num_global, den_global))
        out = tracker.to_device(out_merge) if offload else out_merge
        if not return_aux:
            return out
        raw_scale = tracker.time_stage("t_finalize_ms", lambda: _safe_raw_scale(row_max_global))
        den_raw = tracker.time_stage("t_finalize_ms", lambda: den_global * raw_scale)
        num_raw = tracker.time_stage(
            "t_finalize_ms",
            lambda: num_global * raw_scale.transpose(1, 2).unsqueeze(-1),
        )
        transfer_stats = tracker.as_dict()
        return out, {
            "num": num_raw,
            "den": den_raw,
            "num_scaled": num_global,
            "den_scaled": den_global,
            "row_max": row_max_global,
            "stable_softmax": True,
            "paths": paths,
            "num_subseq": int(len(paths)),
            "den_stats": den_stats,
            "cpu_offload": offload,
            "communication_ms": transfer_stats["communication_ms"],
            "transfer_stats": transfer_stats,
            "c": int(c),
            "interest_set": tuple(int(x) for x in interest_set),
            "itr": int(itr),
        }

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
            local_positions_p = positions_t.index_select(0, token_ids_p)
            q_i_p = q.index_select(1, token_ids_p)
            k_i_p = k.index_select(1, token_ids_p)
            v_i_p = v.index_select(1, token_ids_p)
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
            num_i,
            den_i,
            num_i_merge,
            den_i_merge,
            local_positions,
        )
        tracker.flush_cuda_cache()

    out_merge = tracker.time_stage("t_finalize_ms", lambda: normalize_num_den(num_global, den_global))
    out = tracker.to_device(out_merge) if offload else out_merge
    if not return_aux:
        return out
    transfer_stats = tracker.as_dict()
    return out, {
        "num": num_global,
        "den": den_global,
        "paths": paths,
        "num_subseq": int(len(paths)),
        "den_stats": den_stats,
        "cpu_offload": offload,
        "communication_ms": transfer_stats["communication_ms"],
        "transfer_stats": transfer_stats,
        "c": int(c),
        "interest_set": tuple(int(x) for x in interest_set),
        "itr": int(itr),
    }
