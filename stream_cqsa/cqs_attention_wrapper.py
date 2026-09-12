from __future__ import annotations

import math
import time
from typing import Any, Dict, Sequence

import torch
from .interface import flash_attn_func_cqs, flash_attn_func_cqsa

_CQSA_FAST_C = 7
_CQSA_FAST_INTEREST = (0, 1, 3)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _profile_block(
    enabled: bool,
    device: torch.device,
    timings: dict[str, float],
    key: str,
):
    if not enabled:
        return None
    _sync_device(device)
    return (key, time.perf_counter())


def _profile_block_end(
    enabled: bool,
    device: torch.device,
    timings: dict[str, float],
    token: tuple[str, float] | None,
) -> None:
    if not enabled or token is None:
        return
    key, t0 = token
    _sync_device(device)
    timings[key] = timings.get(key, 0.0) + (time.perf_counter() - t0) * 1000.0


def compute_chunk_layout(N: int, c: int) -> tuple[list[int], list[int], list[int]]:
    if N < 0:
        raise ValueError(f"N must be >= 0, got {N}")
    if c <= 0:
        raise ValueError(f"c must be > 0, got {c}")

    q, r = divmod(N, c)
    bound = c - r
    sizes = [q for _ in range(c)]
    if r > 0:
        for i in range(bound, c):
            sizes[i] += 1

    starts = [0 for _ in range(c)]
    for i in range(1, c):
        starts[i] = starts[i - 1] + sizes[i - 1]
    ends = [starts[i] + sizes[i] for i in range(c)]
    return sizes, starts, ends


def cqs_quorum_for_subseq(
    subseq_i: int,
    c: int,
    interest_set: Sequence[int],
) -> list[int]:
    if c <= 0:
        raise ValueError(f"c must be > 0, got {c}")
    return [((ele + subseq_i) % c) for ele in interest_set]


def build_quorum_indices(
    sizes: Sequence[int],
    starts: Sequence[int],
    ends: Sequence[int],
    quorum_chunks: Sequence[int],
) -> tuple[torch.Tensor, Dict[int, int], Dict[int, int]]:
    quorum_tokens: list[torch.Tensor] = []
    local_start: dict[int, int] = {}
    local_end: dict[int, int] = {}
    offset = 0
    for c in quorum_chunks:
        local_start[c] = offset
        offset += int(sizes[c])
        local_end[c] = offset
        quorum_tokens.append(torch.arange(int(starts[c]), int(ends[c]), dtype=torch.long))
    if quorum_tokens:
        quorum_of_token = torch.cat(quorum_tokens)
    else:
        quorum_of_token = torch.empty(0, dtype=torch.long)
    return quorum_of_token, local_start, local_end


def _build_local_default_cqs_mask(
    subseq_i: int,
    quorum_chunks: Sequence[int],
    local_start: Dict[int, int],
    local_end: Dict[int, int],
    *,
    device: torch.device,
) -> torch.Tensor:
    offset = 0
    for c in quorum_chunks:
        offset = max(offset, int(local_end[c]))
    mask = torch.ones((offset, offset), dtype=torch.bool, device=device)
    for c in quorum_chunks:
        if c == subseq_i:
            continue
        s, e = int(local_start[c]), int(local_end[c])
        if e > s:
            mask[s:e, s:e] = False
    return mask


def _subseq_dense_attention(
    q_sub: torch.Tensor,
    k_sub: torch.Tensor,
    v_sub: torch.Tensor,
    keep_mask: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q/k/v: [B, L, H, D], keep_mask: [L, L] bool.
    B, L, H, D = q_sub.shape
    _ = (B, D)
    q_bhld = q_sub.transpose(1, 2)  # [B, H, L, D]
    k_bhld = k_sub.transpose(1, 2)  # [B, H, L, D]
    v_bhld = v_sub.transpose(1, 2)  # [B, H, L, D]

    mask = keep_mask
    if causal:
        causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool, device=q_sub.device))
        mask = mask & causal_mask
    mask_bhll = mask.view(1, 1, L, L)

    logits = torch.matmul(q_bhld, k_bhld.transpose(-2, -1)) * softmax_scale
    logits = logits.masked_fill(~mask_bhll, -torch.inf)

    lse = torch.logsumexp(logits, dim=-1)  # [B, H, L], -inf if row is fully masked.
    denom = torch.exp(lse)  # [B, H, L]
    probs = torch.exp(logits - lse.unsqueeze(-1))
    probs = torch.where(mask_bhll, probs, torch.zeros_like(probs))

    # Handle fully-masked rows robustly.
    all_masked = ~mask.view(1, 1, L, L).any(dim=-1)
    probs = torch.where(all_masked.unsqueeze(-1), torch.zeros_like(probs), probs)

    out = torch.matmul(probs, v_bhld)  # [B, H, L, D]
    return out.transpose(1, 2), denom  # [B, L, H, D], [B, H, L]


def _subseq_flash_noncausal(
    q_sub: torch.Tensor,
    k_sub: torch.Tensor,
    v_sub: torch.Tensor,
    *,
    subseq_i: int,
    quorum_chunks: Sequence[int],
    local_start: Dict[int, int],
    local_end: Dict[int, int],
    softmax_scale: float,
    parallel_query_chunks: bool,
    max_parallel_streams: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    owner_local_idx = -1
    for idx, c in enumerate(quorum_chunks):
        if int(c) == int(subseq_i):
            owner_local_idx = idx
            break
    if owner_local_idx < 0:
        raise RuntimeError("owner chunk is missing from quorum; invalid CQSA layout")

    _ = (parallel_query_chunks, max_parallel_streams, local_start)
    cqs_chunk_ends = torch.tensor(
        [int(local_end[c]) for c in quorum_chunks],
        dtype=torch.int32,
        device=q_sub.device,
    )
    out_local, lse_local, _ = flash_attn_func_cqs(
        q_sub,
        k_sub,
        v_sub,
        cqs_chunk_ends=cqs_chunk_ends,
        cqs_owner_chunk=owner_local_idx,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
        return_attn_probs=True,
    )
    denom_local = lse_local.to(torch.float32).exp()
    return out_local, denom_local


def _gather_quorum_cat(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    starts: Sequence[int],
    ends: Sequence[int],
    quorum_chunks: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, int], dict[int, int]]:
    q_parts: list[torch.Tensor] = []
    k_parts: list[torch.Tensor] = []
    v_parts: list[torch.Tensor] = []
    local_start: dict[int, int] = {}
    local_end: dict[int, int] = {}
    offset = 0
    for c in quorum_chunks:
        s, e = int(starts[c]), int(ends[c])
        local_start[c] = offset
        if e > s:
            q_parts.append(q[:, s:e, :, :])
            k_parts.append(k[:, s:e, :, :])
            v_parts.append(v[:, s:e, :, :])
            offset += (e - s)
        local_end[c] = offset
    if offset == 0:
        empty = q[:, :0, :, :]
        return empty, empty, empty, local_start, local_end
    q_sub = q_parts[0] if len(q_parts) == 1 else torch.cat(q_parts, dim=1)
    k_sub = k_parts[0] if len(k_parts) == 1 else torch.cat(k_parts, dim=1)
    v_sub = v_parts[0] if len(v_parts) == 1 else torch.cat(v_parts, dim=1)
    return q_sub, k_sub, v_sub, local_start, local_end


def _accumulate_quorum_by_slices(
    global_num: torch.Tensor,
    global_den: torch.Tensor,
    num_i_acc: torch.Tensor,
    den_i_acc: torch.Tensor,
    quorum_chunks: Sequence[int],
    starts: Sequence[int],
    ends: Sequence[int],
    local_start: Dict[int, int],
    local_end: Dict[int, int],
) -> None:
    for c in quorum_chunks:
        gs, ge = int(starts[c]), int(ends[c])
        ls, le = int(local_start[c]), int(local_end[c])
        if ge <= gs or le <= ls:
            continue
        global_num[:, gs:ge, :, :] += num_i_acc[:, ls:le, :, :]
        global_den[:, :, gs:ge] += den_i_acc[:, :, ls:le]


def cqs_attention_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    c: int = 7,
    interest_set: Sequence[int] = (0, 1, 3),
    num_itr: int = 1,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    dropout_p: float = 0.0,
    eps: float = 1e-12,
    return_denominator: bool = False,
    return_debug: bool = False,
    profile: bool = False,
    parallel_query_chunks: bool = False,
    max_parallel_streams: int | None = None,
    use_fused_cqsa: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """
    CQS attention implemented inside flash-attention repo.

    Inputs:
      q, k, v: [B, N, H, D] (same as flash_attn_func).

    Behavior:
      1) Split sequence into CQS subsequences from (c, interest_set).
      2) Apply deterministic default CQS mask per subsequence:
         mask non-owner intra-chunk diagonal blocks.
      3) Compute subsequence outputs and softmax denominators.
      4) Merge with exact weighted rule:
         global_O += out_i * denom_i
         global_S += denom_i
         out = global_O / global_S

    Notes:
      - Fast path uses FlashAttention v2 only for non-causal, dropout=0.
      - For causal / dropout>0 / grad-enabled tensors, this falls back to dense masked attention
        for correctness.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be rank-4 [B, N, H, D] tensors")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, v must share shape; got {q.shape}, {k.shape}, {v.shape}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"q, k, v must share dtype; got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.device != k.device or q.device != v.device:
        raise ValueError(f"q, k, v must be on same device; got {q.device}, {k.device}, {v.device}")
    if dropout_p != 0.0 and causal is False:
        # Non-causal fast path currently only supports dropout_p == 0.
        pass

    B, N, H, D = q.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    profile_enabled = bool(profile)
    timings_ms: dict[str, float] = {}
    total_tok = _profile_block(profile_enabled, q.device, timings_ms, "total_ms")

    # Single-call fused path in CUDA/C++ for the default CQSA configuration.
    if (
        use_fused_cqsa
        and
        q.is_cuda
        and c == _CQSA_FAST_C
        and tuple(int(x) for x in interest_set) == _CQSA_FAST_INTEREST
        and not (q.requires_grad or k.requires_grad or v.requires_grad)
    ):
        fused_tok = _profile_block(profile_enabled, q.device, timings_ms, "fused_cqsa_ms")
        out, den = flash_attn_func_cqsa(
            q,
            k,
            v,
            num_itr=int(num_itr),
            dropout_p=dropout_p,
            softmax_scale=float(softmax_scale),
            causal=bool(causal),
            return_denominator=True,
        )
        _profile_block_end(profile_enabled, q.device, timings_ms, fused_tok)
        _profile_block_end(profile_enabled, q.device, timings_ms, total_tok)
        if not return_denominator and not return_debug:
            return out
        if not return_debug:
            return out, den
        debug = {
            "c": c,
            "interest_set": tuple(int(x) for x in interest_set),
            "num_itr": int(num_itr),
            "causal": bool(causal),
            "softmax_scale": float(softmax_scale),
            "dropout_p": float(dropout_p),
            "used_fused_cqsa": True,
        }
        if profile_enabled:
            debug["timings_ms"] = timings_ms
        return out, den, debug

    if int(num_itr) != 1:
        raise RuntimeError(
            "num_itr != 1 is only supported in the fused CQSA C++/CUDA path "
            "(use_fused_cqsa=True with default c/interest_set)."
        )

    layout_tok = _profile_block(profile_enabled, q.device, timings_ms, "layout_ms")
    sizes, starts, ends = compute_chunk_layout(int(N), c)
    _profile_block_end(profile_enabled, q.device, timings_ms, layout_tok)
    accum_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    alloc_tok = _profile_block(profile_enabled, q.device, timings_ms, "alloc_ms")
    global_num = torch.zeros((B, N, H, D), dtype=accum_dtype, device=q.device)
    global_den = torch.zeros((B, H, N), dtype=accum_dtype, device=q.device)
    _profile_block_end(profile_enabled, q.device, timings_ms, alloc_tok)
    per_subseq_denominators: list[dict[str, Any]] = [] if return_debug else []
    per_subseq_timings: list[dict[str, float]] = [] if profile_enabled else []

    use_flash_fast_path = (
        (not causal)
        and float(dropout_p) == 0.0
        and (not (q.requires_grad or k.requires_grad or v.requires_grad))
        and q.is_cuda
    )

    for subseq_i in range(c):
        quorum_chunks = cqs_quorum_for_subseq(subseq_i, c, interest_set)
        gather_tok = _profile_block(profile_enabled, q.device, timings_ms, "gather_ms")
        q_sub, k_sub, v_sub, local_start, local_end = _gather_quorum_cat(
            q, k, v, starts, ends, quorum_chunks
        )
        _profile_block_end(profile_enabled, q.device, timings_ms, gather_tok)
        if q_sub.shape[1] == 0:
            continue

        attn_tok = _profile_block(profile_enabled, q.device, timings_ms, "attn_ms")
        if use_flash_fast_path:
            out_i, den_i = _subseq_flash_noncausal(
                q_sub,
                k_sub,
                v_sub,
                subseq_i=subseq_i,
                quorum_chunks=quorum_chunks,
                local_start=local_start,
                local_end=local_end,
                softmax_scale=float(softmax_scale),
                parallel_query_chunks=parallel_query_chunks,
                max_parallel_streams=max_parallel_streams,
            )
        else:
            keep_mask = _build_local_default_cqs_mask(
                subseq_i,
                quorum_chunks,
                local_start,
                local_end,
                device=q.device,
            )
            out_i, den_i = _subseq_dense_attention(
                q_sub,
                k_sub,
                v_sub,
                keep_mask=keep_mask,
                softmax_scale=float(softmax_scale),
                causal=bool(causal),
            )
        _profile_block_end(profile_enabled, q.device, timings_ms, attn_tok)

        merge_tok = _profile_block(profile_enabled, q.device, timings_ms, "merge_ms")
        den_i_acc = den_i.to(accum_dtype)  # [B, H, L]
        num_i_acc = out_i.to(accum_dtype) * den_i_acc.transpose(1, 2).unsqueeze(-1)  # [B, L, H, D]
        _accumulate_quorum_by_slices(
            global_num,
            global_den,
            num_i_acc,
            den_i_acc,
            quorum_chunks,
            starts,
            ends,
            local_start,
            local_end,
        )
        _profile_block_end(profile_enabled, q.device, timings_ms, merge_tok)
        if profile_enabled:
            per_subseq_timings.append(
                {
                    "subseq_i": int(subseq_i),
                    "tokens": int(q_sub.shape[1]),
                }
            )
        if return_debug:
            quorum_idx_cpu, _, _ = build_quorum_indices(sizes, starts, ends, quorum_chunks)
            per_subseq_denominators.append(
                {
                    "subseq_i": int(subseq_i),
                    "quorum_chunks": tuple(int(c) for c in quorum_chunks),
                    "quorum_indices": quorum_idx_cpu,
                    "denominator": den_i_acc.detach().cpu(),
                }
            )

    finalize_tok = _profile_block(profile_enabled, q.device, timings_ms, "finalize_ms")
    out = torch.zeros_like(global_num)
    covered = global_den > 0
    if covered.any():
        out = global_num / global_den.transpose(1, 2).unsqueeze(-1).clamp_min(eps)
        out = torch.where(covered.transpose(1, 2).unsqueeze(-1), out, torch.zeros_like(out))
    out = out.to(q.dtype)
    _profile_block_end(profile_enabled, q.device, timings_ms, finalize_tok)
    _profile_block_end(profile_enabled, q.device, timings_ms, total_tok)

    if not return_denominator and not return_debug:
        return out

    den_out = global_den
    if not return_debug:
        return out, den_out

    debug = {
        "sizes": sizes,
        "starts": starts,
        "ends": ends,
        "c": c,
        "interest_set": tuple(int(x) for x in interest_set),
        "causal": bool(causal),
        "softmax_scale": float(softmax_scale),
        "dropout_p": float(dropout_p),
        "used_flash_fast_path": bool(use_flash_fast_path),
        "parallel_query_chunks": bool(parallel_query_chunks),
        "max_parallel_streams": max_parallel_streams,
        "covered": covered,
        "num_uncovered": int((~covered).sum().item()),
        "per_subseq_denominators": per_subseq_denominators,
    }
    if profile_enabled:
        timings_ms["num_subseq"] = float(len(per_subseq_timings))
        debug["timings_ms"] = timings_ms
        debug["per_subseq_timing_meta"] = per_subseq_timings
    return out, den_out, debug


def cqs_attention_func_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    c: int = 7,
    interest_set: Sequence[int] = (0, 1, 3),
    num_itr: int = 1,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    dropout_p: float = 0.0,
    eps: float = 1e-12,
    return_denominator: bool = False,
    return_debug: bool = False,
    profile: bool = False,
    max_parallel_subseq_streams: int | None = None,
    parallel_query_chunks: bool = False,
    max_parallel_query_streams: int | None = None,
    use_fused_cqsa: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """
    CQS attention with 2-phase subsequence fork-join.

    Phase 1 (fork): each subsequence computes local numerator/denominator independently.
    Phase 2 (join): local buffers are reduced into global_num/global_den on the default stream.

    This avoids write races on overlapping token indices while allowing CUDA stream parallelism
    across subsequences.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be rank-4 [B, N, H, D] tensors")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, v must share shape; got {q.shape}, {k.shape}, {v.shape}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"q, k, v must share dtype; got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.device != k.device or q.device != v.device:
        raise ValueError(f"q, k, v must be on same device; got {q.device}, {k.device}, {v.device}")

    B, N, H, D = q.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    profile_enabled = bool(profile)
    timings_ms: dict[str, float] = {}
    total_tok = _profile_block(profile_enabled, q.device, timings_ms, "total_ms")

    # For the hardcoded CQSA config, fused C++/CUDA path is faster than Python
    # fork-join orchestration.
    if (
        use_fused_cqsa
        and
        q.is_cuda
        and c == _CQSA_FAST_C
        and tuple(int(x) for x in interest_set) == _CQSA_FAST_INTEREST
        and not (q.requires_grad or k.requires_grad or v.requires_grad)
    ):
        return cqs_attention_func(
            q,
            k,
            v,
            c=c,
            interest_set=interest_set,
            num_itr=num_itr,
            causal=causal,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            eps=eps,
            return_denominator=return_denominator,
            return_debug=return_debug,
            profile=profile,
            parallel_query_chunks=parallel_query_chunks,
            max_parallel_streams=max_parallel_query_streams,
            use_fused_cqsa=use_fused_cqsa,
        )

    layout_tok = _profile_block(profile_enabled, q.device, timings_ms, "layout_ms")
    sizes, starts, ends = compute_chunk_layout(int(N), c)
    _profile_block_end(profile_enabled, q.device, timings_ms, layout_tok)
    accum_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    alloc_tok = _profile_block(profile_enabled, q.device, timings_ms, "alloc_ms")
    global_num = torch.zeros((B, N, H, D), dtype=accum_dtype, device=q.device)
    global_den = torch.zeros((B, H, N), dtype=accum_dtype, device=q.device)
    _profile_block_end(profile_enabled, q.device, timings_ms, alloc_tok)
    per_subseq_denominators: list[dict[str, Any]] = [] if return_debug else []

    use_flash_fast_path = (
        (not causal)
        and float(dropout_p) == 0.0
        and (not (q.requires_grad or k.requires_grad or v.requires_grad))
        and q.is_cuda
    )

    work_items: list[tuple[int, list[int]]] = []
    work_tok = _profile_block(profile_enabled, q.device, timings_ms, "work_items_ms")
    for subseq_i in range(c):
        quorum_chunks = cqs_quorum_for_subseq(subseq_i, c, interest_set)
        has_tokens = any(int(ends[c]) > int(starts[c]) for c in quorum_chunks)
        if not has_tokens:
            continue
        work_items.append((subseq_i, quorum_chunks))
    _profile_block_end(profile_enabled, q.device, timings_ms, work_tok)

    if not q.is_cuda:
        raise RuntimeError("cqs_attention_func_parallel is CUDA-only in performance mode")
    if len(work_items) <= 1:
        return cqs_attention_func(
            q,
            k,
            v,
            c=c,
            interest_set=interest_set,
            num_itr=num_itr,
            causal=causal,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            eps=eps,
            return_denominator=return_denominator,
            return_debug=return_debug,
            profile=profile,
            parallel_query_chunks=parallel_query_chunks,
            max_parallel_streams=max_parallel_query_streams,
            use_fused_cqsa=use_fused_cqsa,
        )

    if int(num_itr) != 1:
        raise RuntimeError(
            "num_itr != 1 is only supported in the fused CQSA C++/CUDA path "
            "(use_fused_cqsa=True with default c/interest_set)."
        )

    num_streams = len(work_items) if max_parallel_subseq_streams is None else min(
        len(work_items), max(1, int(max_parallel_subseq_streams))
    )
    streams = [torch.cuda.Stream(device=q.device) for _ in range(num_streams)]
    default_stream = torch.cuda.current_stream(device=q.device)

    # Fork: compute local outputs on independent streams.
    fork_tok = _profile_block(profile_enabled, q.device, timings_ms, "fork_stage_ms")
    staged_results: list[
        tuple[
            torch.cuda.Stream,
            int,
            tuple[int, ...],
            dict[int, int],
            dict[int, int],
            torch.Tensor,
            torch.Tensor,
        ]
    ] = []
    for i, (subseq_i, quorum_chunks) in enumerate(work_items):
        stream = streams[i % num_streams]
        with torch.cuda.stream(stream):
            q_sub, k_sub, v_sub, local_start, local_end = _gather_quorum_cat(
                q, k, v, starts, ends, quorum_chunks
            )

            if use_flash_fast_path:
                out_i, den_i = _subseq_flash_noncausal(
                    q_sub,
                    k_sub,
                    v_sub,
                    subseq_i=subseq_i,
                    quorum_chunks=quorum_chunks,
                    local_start=local_start,
                    local_end=local_end,
                    softmax_scale=float(softmax_scale),
                    parallel_query_chunks=parallel_query_chunks,
                    max_parallel_streams=max_parallel_query_streams,
                )
            else:
                keep_mask = _build_local_default_cqs_mask(
                    subseq_i,
                    quorum_chunks,
                    local_start,
                    local_end,
                    device=q.device,
                )
                out_i, den_i = _subseq_dense_attention(
                    q_sub,
                    k_sub,
                    v_sub,
                    keep_mask=keep_mask,
                    softmax_scale=float(softmax_scale),
                    causal=bool(causal),
                )

            den_i_acc = den_i.to(accum_dtype)  # [B, H, L]
            num_i_acc = out_i.to(accum_dtype) * den_i_acc.transpose(1, 2).unsqueeze(-1)  # [B, L, H, D]
            staged_results.append(
                (
                    stream,
                    int(subseq_i),
                    tuple(int(c) for c in quorum_chunks),
                    local_start,
                    local_end,
                    num_i_acc,
                    den_i_acc,
                )
            )
    _profile_block_end(profile_enabled, q.device, timings_ms, fork_tok)

    # Join: reduce on default stream (race-free accumulation).
    join_tok = _profile_block(profile_enabled, q.device, timings_ms, "join_stage_ms")
    for stream, subseq_i, quorum_chunks_t, local_start, local_end, num_i_acc, den_i_acc in staged_results:
        default_stream.wait_stream(stream)
        _accumulate_quorum_by_slices(
            global_num,
            global_den,
            num_i_acc,
            den_i_acc,
            quorum_chunks_t,
            starts,
            ends,
            local_start,
            local_end,
        )
        if return_debug:
            quorum_idx_cpu, _, _ = build_quorum_indices(sizes, starts, ends, quorum_chunks_t)
            per_subseq_denominators.append(
                {
                    "subseq_i": subseq_i,
                    "quorum_chunks": quorum_chunks_t,
                    "quorum_indices": quorum_idx_cpu,
                    "denominator": den_i_acc.detach().cpu(),
                }
            )
    _profile_block_end(profile_enabled, q.device, timings_ms, join_tok)

    finalize_tok = _profile_block(profile_enabled, q.device, timings_ms, "finalize_ms")
    out = torch.zeros_like(global_num)
    covered = global_den > 0
    if covered.any():
        out = global_num / global_den.transpose(1, 2).unsqueeze(-1).clamp_min(eps)
        out = torch.where(covered.transpose(1, 2).unsqueeze(-1), out, torch.zeros_like(out))
    out = out.to(q.dtype)
    _profile_block_end(profile_enabled, q.device, timings_ms, finalize_tok)
    _profile_block_end(profile_enabled, q.device, timings_ms, total_tok)

    if not return_denominator and not return_debug:
        return out

    den_out = global_den
    if not return_debug:
        return out, den_out

    debug = {
        "sizes": sizes,
        "starts": starts,
        "ends": ends,
        "c": c,
        "interest_set": tuple(int(x) for x in interest_set),
        "causal": bool(causal),
        "softmax_scale": float(softmax_scale),
        "dropout_p": float(dropout_p),
        "used_flash_fast_path": bool(use_flash_fast_path),
        "parallel_subseq": True,
        "max_parallel_subseq_streams": max_parallel_subseq_streams,
        "parallel_query_chunks": bool(parallel_query_chunks),
        "max_parallel_query_streams": max_parallel_query_streams,
        "covered": covered,
        "num_uncovered": int((~covered).sum().item()),
        "per_subseq_denominators": per_subseq_denominators,
    }
    if profile_enabled:
        timings_ms["num_work_items"] = float(len(work_items))
        debug["timings_ms"] = timings_ms
    return out, den_out, debug


__all__ = [
    "compute_chunk_layout",
    "cqs_quorum_for_subseq",
    "build_quorum_indices",
    "cqs_attention_func",
    "cqs_attention_func_parallel",
]
