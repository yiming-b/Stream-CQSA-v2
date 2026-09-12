from __future__ import annotations

import os
from itertools import product
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
import time

from .cqs_mask import CQS_mask
from .interface import flash_attn_bwd_cqs_group_bits
from .attention_kernel.FA import default_subsequence_attention
from .attention_kernel.Custom import custom_attn
from .streamed_fwd_bwd import (
    compute_dnum_dden_gpu_staged,
    streamed_backward_phase,
    streamed_forward_phase,
)

_LAST_STREAMED_TRACE: dict[str, Any] | None = None


def reset_last_streamed_trace() -> None:
    global _LAST_STREAMED_TRACE
    _LAST_STREAMED_TRACE = None


def get_last_streamed_trace(*, reset: bool = False) -> dict[str, Any] | None:
    global _LAST_STREAMED_TRACE
    trace = _LAST_STREAMED_TRACE
    if trace is None:
        return None
    out = dict(trace)
    if reset:
        _LAST_STREAMED_TRACE = None
    return out


def _set_last_streamed_trace(trace: dict[str, Any]) -> None:
    global _LAST_STREAMED_TRACE
    _LAST_STREAMED_TRACE = dict(trace)


def _force_naive_default_backward() -> bool:
    """
    Correctness fallback for default FA backward.
    Set STREAM_CQSA_FORCE_NAIVE_DEFAULT_BWD=0 to disable.
    """
    raw = str(os.environ.get("STREAM_CQSA_FORCE_NAIVE_DEFAULT_BWD", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _is_cuda_oom_exception(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    cur: BaseException | None = exc
    while cur is not None:
        if oom_type is not None and isinstance(cur, oom_type):
            return True
        cur = cur.__cause__
    msg = str(exc).lower()
    return (
        ("out of memory" in msg)
        or ("cuda out of memory" in msg)
        or ("cuda_oom" in msg)
        or ("cuda oom" in msg)
        or (("oom" in msg) and ("cuda" in msg))
    )


def _max_itr_for_sequence_len(*, n_tokens: int, c: int) -> int:
    n = int(n_tokens)
    c_i = int(c)
    if n <= 0:
        return 0
    if c_i <= 1:
        return 0
    itr_max = 0
    while int(c_i ** int(itr_max + 1)) <= int(n):
        itr_max += 1
    return int(itr_max)


def _dense_mask_from_group_runs(
    *,
    local_size: int,
    group_runs: Sequence[Sequence[Tuple[int, int]]],
    device: torch.device,
) -> torch.Tensor:
    """
    Build dense local mask [L, L], where True means masked.
    """
    M_i = torch.zeros((int(local_size), int(local_size)), dtype=torch.bool, device=device)
    for runs in group_runs:
        idx_parts = []
        for s, e in runs:
            si, ei = int(s), int(e)
            if ei > si:
                idx_parts.append(torch.arange(si, ei, device=device, dtype=torch.long))
        if len(idx_parts) == 0:
            continue
        idx = idx_parts[0] if len(idx_parts) == 1 else torch.cat(idx_parts, dim=0)
        M_i[idx.unsqueeze(1), idx.unsqueeze(0)] = True
    return M_i


def _local_num_den(
    *,
    q_blhd: torch.Tensor,
    k_blhd: torch.Tensor,
    v_blhd: torch.Tensor,
    M_i: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unnormalized local attention:
      P = exp(scores_masked), Num = P @ V, Den = row_sum(P)
    Shapes:
      q/k/v: [B, L, H, D], M_i: [L, L]
      returns Num:[B, L, H, D], Den:[B, H, L]
    """
    q_bhld = q_blhd.transpose(1, 2).float()
    k_bhld = k_blhd.transpose(1, 2).float()
    v_bhld = v_blhd.transpose(1, 2).float()

    R_i = torch.matmul(q_bhld, k_bhld.transpose(-2, -1)) * float(softmax_scale)  # [B,H,L,L]
    if bool(M_i.any().item()):
        R_i = R_i.masked_fill(M_i.unsqueeze(0).unsqueeze(0), float("-inf"))

    p = torch.exp(R_i)
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

    den = p.sum(dim=-1)  # [B,H,L]
    num = torch.matmul(p, v_bhld).transpose(1, 2).contiguous()  # [B,L,H,D]
    num = torch.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
    den = torch.nan_to_num(den, nan=0.0, posinf=0.0, neginf=0.0)
    return num, den


def _group_runs_to_group_bits(
    *,
    local_size: int,
    group_runs: Sequence[Sequence[Tuple[int, int]]],
    device: torch.device,
) -> torch.Tensor:
    bits_np = np.zeros((int(local_size),), dtype=np.int64)
    for bit_id, runs in enumerate(group_runs):
        if bit_id >= 63:
            raise ValueError("Too many unique group-runs for int64 bit encoding.")
        bit = np.int64(1) << np.int64(bit_id)
        for s, e in runs:
            si, ei = int(s), int(e)
            if ei > si:
                bits_np[si:ei] |= bit
    return torch.from_numpy(bits_np).to(device=device, dtype=torch.int64, non_blocking=True)


def _local_backward_naive_from_group_bits(
    *,
    q_blhd: torch.Tensor,
    k_blhd: torch.Tensor,
    v_blhd: torch.Tensor,
    d_num_blhd: torch.Tensor,
    d_den_bhl: torch.Tensor,
    group_bits: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Naive Python local backward for one subsequence.
    Inputs:
      q/k/v, d_num: [B, L, H, D]
      d_den: [B, H, L]
      group_bits: [L] int64
    Returns:
      d_q_i, d_k_i, d_v_i: [B, L, H, D] (float32)
    """
    q_bhld = q_blhd.transpose(1, 2).float()
    k_bhld = k_blhd.transpose(1, 2).float()
    v_bhld = v_blhd.transpose(1, 2).float()
    d_num_bhld = d_num_blhd.transpose(1, 2).float()
    d_den_bhl = d_den_bhl.float()

    bits = group_bits.to(device=q_blhd.device, dtype=torch.int64).contiguous()
    bits_row = bits.view(1, 1, bits.numel(), 1)
    bits_col = bits.view(1, 1, 1, bits.numel())
    M_i = torch.bitwise_and(bits_row, bits_col).ne(0)

    R_i = torch.matmul(q_bhld, k_bhld.transpose(-2, -1)) * float(softmax_scale)
    R_i = R_i.masked_fill(M_i, float("-inf"))

    P_i = torch.exp(R_i)
    P_i = torch.nan_to_num(P_i, nan=0.0, posinf=0.0, neginf=0.0)

    d_v_bhld = torch.matmul(P_i.transpose(-2, -1), d_num_bhld)
    dP_i = torch.matmul(d_num_bhld, v_bhld.transpose(-2, -1))
    dP_i = dP_i + d_den_bhl.unsqueeze(-1)
    dR_i = dP_i * P_i

    d_q_bhld = torch.matmul(dR_i, k_bhld)
    d_q_bhld.mul_(float(softmax_scale))
    d_k_bhld = torch.matmul(dR_i.transpose(-2, -1), q_bhld)
    d_k_bhld.mul_(float(softmax_scale))

    d_q_i = torch.nan_to_num(d_q_bhld.transpose(1, 2).contiguous(), nan=0.0, posinf=0.0, neginf=0.0)
    d_k_i = torch.nan_to_num(d_k_bhld.transpose(1, 2).contiguous(), nan=0.0, posinf=0.0, neginf=0.0)
    d_v_i = torch.nan_to_num(d_v_bhld.transpose(1, 2).contiguous(), nan=0.0, posinf=0.0, neginf=0.0)
    return d_q_i, d_k_i, d_v_i


class _StreamCQSAFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_itr: int,
        c: int,
        interest_set: tuple[int, ...],
        softmax_scale: float,
        use_naive_backward: bool,
        use_streamed_backward: bool,
        max_parallel_subseq_fwd: int,
        max_parallel_subseq_bwd: int,
        schedule_mode: str,
    ) -> torch.Tensor:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("Q/K/V must be rank-4 [B,H,N,D].")
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(f"Q/K/V shape mismatch: {q.shape}, {k.shape}, {v.shape}")
        if q.device != k.device or q.device != v.device:
            raise ValueError("Q/K/V must be on the same device.")
        if q.device.type != "cuda" and (not bool(use_streamed_backward)):
            raise ValueError(
                "Autograd stream_cqsa path with CPU tensors requires use_streamed_backward=True."
            )

        B, H, N, D = [int(x) for x in q.shape]
        num_itr_i = int(num_itr)
        c_i = int(c)
        interest = tuple(int(x) for x in interest_set)

        if bool(use_streamed_backward):
            max_k_fwd = max(1, int(max_parallel_subseq_fwd))
            max_k_bwd = max(1, int(max_parallel_subseq_bwd))
            schedule_mode_n = str(schedule_mode).strip().lower()
            if schedule_mode_n not in ("event", "round"):
                raise ValueError("schedule_mode must be one of: event, round")

            reset_last_streamed_trace()
            fwd_t0 = time.perf_counter()
            fwd = streamed_forward_phase(
                N=int(N),
                D=int(D),
                B=int(B),
                H=int(H),
                c=int(c_i),
                interest_set=interest,
                itr_max=int(num_itr_i),
                max_parallel_subseq=int(max_k_fwd),
                schedule_mode=schedule_mode_n,
                dtype=q.dtype,
                subseq_attention_fn=(custom_attn if bool(use_naive_backward) else default_subsequence_attention),
                source_q=q.detach(),
                source_k=k.detach(),
                source_v=v.detach(),
                qkv_generation_mode="random",
                seed=0,
                input_std=0.1,
                max_rounds=None,
                t0_wall=float(time.perf_counter()),
                timeline_rows=[],
                timing_ms={},
            )
            fwd_wall_ms = float((time.perf_counter() - fwd_t0) * 1000.0)

            out_cpu = fwd["output_cpu"]
            # streamed_forward_phase returns token-major [B, N, H, D].
            # Autograd path must return head-major [B, H, N, D] to match Q/K/V layout.
            out = out_cpu.permute(0, 2, 1, 3).contiguous().to(device=q.device, dtype=torch.float32)
            ctx.save_for_backward(q, k, v)
            ctx.streamed = True
            ctx.use_naive_backward = bool(use_naive_backward)
            ctx.force_naive_default_bwd = bool(
                (not bool(use_naive_backward)) and _force_naive_default_backward()
            )
            ctx.softmax_scale = float(softmax_scale)
            ctx.num_global_cpu = fwd["num_global_cpu"]
            ctx.den_global_cpu = fwd["den_global_cpu"]
            ctx.processed_paths = fwd["processed_paths"]
            ctx.num_itr = int(num_itr_i)
            ctx.c = int(c_i)
            ctx.interest_set = interest
            ctx.max_parallel_subseq_bwd = int(max_k_bwd)
            ctx.schedule_mode = str(schedule_mode_n)
            ctx.fwd_timing_ms = dict(fwd.get("timing_ms", {}))
            ctx.fwd_timeline_rows = list(fwd.get("timeline_rows", []))
            ctx.fwd_wall_ms = float(fwd_wall_ms)
            ctx.fwd_num_paths = int(len(fwd.get("processed_paths", [])))
            return out

        mask_engine = CQS_mask()
        kernel_fn = custom_attn if bool(use_naive_backward) else default_subsequence_attention
        all_paths = [tuple()] if num_itr_i <= 0 else list(product(range(c_i), repeat=num_itr_i))

        global_num = torch.zeros((B, H, N, D), dtype=torch.float32, device=q.device)
        global_den = torch.zeros((B, H, N), dtype=torch.float32, device=q.device)

        path_infos: list[Dict[str, Any]] = []
        for path in all_paths:
            mask_one = mask_engine.gen_mask(
                N=N,
                num_itr=num_itr_i,
                quorum_idx=list(path),
                interest_set=interest,
                c=c_i,
                include_trace=False,
            )
            token_ids_np = np.asarray(mask_one["token_ids"], dtype=np.int64)
            local_size = int(mask_one["local_size"])
            if local_size <= 0 or token_ids_np.size == 0:
                continue

            token_ids = torch.from_numpy(token_ids_np).to(device=q.device, dtype=torch.long)
            q_i = q.index_select(2, token_ids).permute(0, 2, 1, 3).contiguous()
            k_i = k.index_select(2, token_ids).permute(0, 2, 1, 3).contiguous()
            v_i = v.index_select(2, token_ids).permute(0, 2, 1, 3).contiguous()

            group_bits = _group_runs_to_group_bits(
                local_size=local_size,
                group_runs=mask_one["group_runs"],
                device=q.device,
            )
            num_i, den_i = kernel_fn(
                q=q_i,
                k=k_i,
                v=v_i,
                cqs_mask={
                    "local_size": local_size,
                    "group_bits_cpu": group_bits.detach().cpu(),
                },
                softmax_scale=float(softmax_scale),
            )  # num_i:[B,L,H,D], den_i:[B,H,L]

            global_num.index_add_(2, token_ids, num_i.transpose(1, 2))
            global_den.index_add_(2, token_ids, den_i)

            path_infos.append(
                {
                    "token_ids": token_ids,
                    "group_bits": group_bits,
                }
            )

        den_safe = global_den.clamp_min(1e-12)
        out = global_num / den_safe.unsqueeze(-1)
        out = torch.where(global_den.unsqueeze(-1) > 0, out, torch.zeros_like(out))

        ctx.save_for_backward(q, k, v, global_num, global_den)
        ctx.path_infos = path_infos
        ctx.softmax_scale = float(softmax_scale)
        ctx.use_naive_backward = bool(use_naive_backward)
        ctx.force_naive_default_bwd = bool(
            (not bool(use_naive_backward)) and _force_naive_default_backward()
        )
        ctx.streamed = False
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        if bool(getattr(ctx, "streamed", False)):
            q, k, v = ctx.saved_tensors
            # grad_out arrives as [B, H, N, D] because forward returned [B, H, N, D].
            # compute_dnum_dden expects token-major [B, N, H, D].
            grad_out_tok = grad_out.float().detach().cpu().permute(0, 2, 1, 3).contiguous()
            d_num, d_den = compute_dnum_dden_gpu_staged(
                num_global_cpu=ctx.num_global_cpu,
                den_global_cpu=ctx.den_global_cpu,
                grad_out_cpu=grad_out_tok,
            )
            kernel_name = (
                "custom_attn"
                if (bool(ctx.use_naive_backward) or bool(getattr(ctx, "force_naive_default_bwd", False)))
                else "default_subsequence_attention"
            )
            c_i = int(ctx.c)
            n_i = int(q.shape[2])
            itr_try = int(ctx.num_itr)
            max_itr_allowed = _max_itr_for_sequence_len(n_tokens=int(n_i), c=int(c_i))
            while True:
                paths_try = (
                    ctx.processed_paths
                    if int(itr_try) == int(ctx.num_itr)
                    else ([tuple()] if int(itr_try) <= 0 else list(product(range(int(c_i)), repeat=int(itr_try))))
                )
                try:
                    bwd_t0 = time.perf_counter()
                    bwd = streamed_backward_phase(
                        N=int(q.shape[2]),
                        D=int(q.shape[3]),
                        B=int(q.shape[0]),
                        H=int(q.shape[1]),
                        c=int(c_i),
                        interest_set=tuple(int(x) for x in ctx.interest_set),
                        itr_max=int(itr_try),
                        max_parallel_subseq_bwd=int(ctx.max_parallel_subseq_bwd),
                        schedule_mode=str(ctx.schedule_mode),
                        dtype=q.dtype,
                        attention_kernel_name=kernel_name,
                        source_q=q.detach(),
                        source_k=k.detach(),
                        source_v=v.detach(),
                        qkv_generation_mode="random",
                        seed=0,
                        input_std=0.1,
                        processed_paths=paths_try,
                        max_rounds=None,
                        d_num_global_cpu=d_num,
                        d_den_global_cpu=d_den,
                        t0_wall=float(time.perf_counter()),
                        timeline_rows=[],
                        timing_ms={},
                    )
                    bwd_wall_ms = float((time.perf_counter() - bwd_t0) * 1000.0)
                    break
                except RuntimeError as exc:
                    if not _is_cuda_oom_exception(exc):
                        raise
                    if int(c_i) <= 1:
                        raise RuntimeError(
                            "CUDA OOM in streamed backward and c<=1, cannot reduce subsequence size via itr."
                        ) from exc
                    nxt_itr = int(itr_try + 1)
                    if int(nxt_itr) > int(max_itr_allowed):
                        raise RuntimeError(
                            f"CUDA OOM persists in streamed backward up to max itr={max_itr_allowed} "
                            f"(N={int(n_i)}, c={int(c_i)})."
                        ) from exc
                    print(
                        f"[cqsa_stream] CUDA OOM at itr={int(itr_try)}; retry with itr={int(nxt_itr)}.",
                        flush=True,
                    )
                    itr_try = int(nxt_itr)
                    torch.cuda.empty_cache()
            fwd_timing = dict(getattr(ctx, "fwd_timing_ms", {}) or {})
            bwd_timing = dict(bwd.get("timing_ms", {}) or {})
            fwd_rows = list(getattr(ctx, "fwd_timeline_rows", []) or [])
            bwd_rows_raw = list(bwd.get("timeline_rows", []) or [])
            if len(fwd_rows) > 0:
                fwd_t_max = max(float(r.get("t_rel_s", 0.0)) for r in fwd_rows)
            else:
                fwd_t_max = 0.0
            bwd_rows = []
            for r in bwd_rows_raw:
                row = dict(r)
                row["t_rel_s"] = float(row.get("t_rel_s", 0.0)) + float(fwd_t_max)
                bwd_rows.append(row)
            _set_last_streamed_trace(
                {
                    "itr_fwd": int(getattr(ctx, "num_itr", 0)),
                    "itr_bwd": int(itr_try),
                    "num_paths_processed": int(getattr(ctx, "fwd_num_paths", 0)),
                    "fwd_timing_ms": fwd_timing,
                    "bwd_timing_ms": bwd_timing,
                    "fwd_wall_ms": float(getattr(ctx, "fwd_wall_ms", 0.0)),
                    "bwd_wall_ms": float(bwd_wall_ms),
                    "timeline_rows": [*fwd_rows, *bwd_rows],
                }
            )
            d_q = bwd["d_q_global_cpu"].to(device=q.device, dtype=q.dtype)
            d_k = bwd["d_k_global_cpu"].to(device=q.device, dtype=q.dtype)
            d_v = bwd["d_v_global_cpu"].to(device=q.device, dtype=q.dtype)
            return d_q, d_k, d_v, None, None, None, None, None, None, None, None, None

        q, k, v, global_num, global_den = ctx.saved_tensors
        softmax_scale = float(ctx.softmax_scale)
        path_infos = ctx.path_infos

        grad_out_f = grad_out.float()
        den_safe = global_den.clamp_min(1e-12)
        d_num = grad_out_f / den_safe.unsqueeze(-1)
        d_den = -(grad_out_f * global_num).sum(dim=-1) / (den_safe * den_safe)

        valid = global_den > 0
        d_num = torch.where(valid.unsqueeze(-1), d_num, torch.zeros_like(d_num))
        d_den = torch.where(valid, d_den, torch.zeros_like(d_den))

        d_q = torch.zeros_like(q, dtype=torch.float32)
        d_k = torch.zeros_like(k, dtype=torch.float32)
        d_v = torch.zeros_like(v, dtype=torch.float32)

        for info in path_infos:
            token_ids = info["token_ids"]
            group_bits = info["group_bits"]

            q_i = q.index_select(2, token_ids).permute(0, 2, 1, 3).contiguous()
            k_i = k.index_select(2, token_ids).permute(0, 2, 1, 3).contiguous()
            v_i = v.index_select(2, token_ids).permute(0, 2, 1, 3).contiguous()

            d_o_i = d_num.index_select(2, token_ids).permute(0, 2, 1, 3).contiguous()  # [B,L,H,D]
            d_s_i = d_den.index_select(2, token_ids).contiguous()  # [B,H,L]
            if bool(ctx.use_naive_backward) or bool(getattr(ctx, "force_naive_default_bwd", False)):
                d_q_i, d_k_i, d_v_i = _local_backward_naive_from_group_bits(
                    q_blhd=q_i,
                    k_blhd=k_i,
                    v_blhd=v_i,
                    d_num_blhd=d_o_i,
                    d_den_bhl=d_s_i,
                    group_bits=group_bits,
                    softmax_scale=float(softmax_scale),
                )  # [B,L,H,D]
            else:
                d_o_i_kernel = d_o_i.to(dtype=q_i.dtype)
                d_s_i_kernel = d_s_i.to(dtype=q_i.dtype)
                d_q_i, d_k_i, d_v_i = flash_attn_bwd_cqs_group_bits(
                    dout_num=d_o_i_kernel,
                    dden=d_s_i_kernel,
                    q=q_i,
                    k=k_i,
                    v=v_i,
                    cqs_group_bits=group_bits,
                    softmax_scale=float(softmax_scale),
                )  # [B,L,H,D]

            d_q.index_add_(2, token_ids, d_q_i.transpose(1, 2).float())
            d_k.index_add_(2, token_ids, d_k_i.transpose(1, 2).float())
            d_v.index_add_(2, token_ids, d_v_i.transpose(1, 2).float())

        return d_q.to(dtype=q.dtype), d_k.to(dtype=k.dtype), d_v.to(dtype=v.dtype), None, None, None, None, None, None, None, None, None


def stream_cqsa_autograd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_itr: int,
    c: int = 7,
    interest_set: Sequence[int] = (0, 1, 3),
    softmax_scale: float | None = None,
    use_naive_backward: bool = False,
    use_streamed_backward: bool = False,
    max_parallel_subseq_fwd: int = 1,
    max_parallel_subseq_bwd: int | None = None,
    schedule_mode: str = "event",
) -> torch.Tensor:
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    max_k_fwd = max(1, int(max_parallel_subseq_fwd))
    if max_parallel_subseq_bwd is None:
        max_k_bwd = int(max_k_fwd)
    else:
        max_k_bwd = max(1, int(max_parallel_subseq_bwd))
    n_i = int(q.shape[2])
    c_i = int(c)
    itr_try = int(num_itr)
    max_itr_allowed = _max_itr_for_sequence_len(n_tokens=int(n_i), c=int(c_i))
    if int(itr_try) > int(max_itr_allowed):
        raise ValueError(
            f"num_itr={itr_try} exceeds max valid itr={max_itr_allowed} for N={int(n_i)} and c={int(c_i)}."
        )
    while True:
        try:
            return _StreamCQSAFunction.apply(
                q,
                k,
                v,
                int(itr_try),
                int(c_i),
                tuple(int(x) for x in interest_set),
                float(softmax_scale),
                bool(use_naive_backward),
                bool(use_streamed_backward),
                int(max_k_fwd),
                int(max_k_bwd),
                str(schedule_mode),
            )
        except RuntimeError as exc:
            if not _is_cuda_oom_exception(exc):
                raise
            if int(c_i) <= 1:
                raise RuntimeError(
                    "CUDA OOM in stream_cqsa_autograd and c<=1, cannot reduce subsequence size via itr."
                ) from exc
            nxt_itr = int(itr_try + 1)
            if int(nxt_itr) > int(max_itr_allowed):
                raise RuntimeError(
                    f"CUDA OOM persists in stream_cqsa_autograd up to max itr={max_itr_allowed} "
                    f"(N={int(n_i)}, c={int(c_i)})."
                ) from exc
            print(
                f"[cqsa_stream] CUDA OOM at itr={int(itr_try)}; retry with itr={int(nxt_itr)}.",
                flush=True,
            )
            itr_try = int(nxt_itr)
            torch.cuda.empty_cache()
