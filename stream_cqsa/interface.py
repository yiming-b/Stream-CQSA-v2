from __future__ import annotations

from typing import Optional

import torch

import importlib as _importlib
import os as _os

# next/: which compiled extension to use. Kernel variants build as
# cqsa_cuda_next_<name> (see next/kernel/mk_variant.sh) and coexist on
# PYTHONPATH, so one process can be pointed at any of them for A/B runs.
CQSA_CUDA_MODULE = _os.environ.get("CQSA_CUDA_MODULE", "cqsa_cuda")
try:
    cqsa_cuda = _importlib.import_module(CQSA_CUDA_MODULE)
except Exception as exc:  # pragma: no cover - depends on local CUDA build ABI
    cqsa_cuda = None
    _CQSA_CUDA_IMPORT_ERROR = exc
else:
    _CQSA_CUDA_IMPORT_ERROR = None


# next/: optional second extension for NON-causal forward calls. The best
# causal kernel (v11) carries a non-causal CQS-on instantiation that ptxas
# compiles into a 3x slower kernel (see next/LOG.md), while v9's non-causal
# path is fine; both are bit-exact, so the interface can pick per call.
# Default: the same module for both.
# v2 default: the second extension built by setup.py (cqsa_cuda_nc); falls back
# to the causal module when it is absent (correct, just slower non-causal).
CQSA_CUDA_MODULE_NONCAUSAL = _os.environ.get("CQSA_CUDA_MODULE_NONCAUSAL",
                                             "cqsa_cuda_nc" if CQSA_CUDA_MODULE == "cqsa_cuda" else CQSA_CUDA_MODULE)
if CQSA_CUDA_MODULE_NONCAUSAL == CQSA_CUDA_MODULE:
    cqsa_cuda_noncausal = cqsa_cuda
else:
    try:
        cqsa_cuda_noncausal = _importlib.import_module(CQSA_CUDA_MODULE_NONCAUSAL)
    except Exception as exc:  # pragma: no cover
        cqsa_cuda_noncausal = cqsa_cuda
        if "CQSA_CUDA_MODULE_NONCAUSAL" in _os.environ:
            _CQSA_CUDA_IMPORT_ERROR = _CQSA_CUDA_IMPORT_ERROR or exc


def _require_cqsa_cuda(causal: bool = True):
    mod = cqsa_cuda if causal else cqsa_cuda_noncausal
    if mod is None:
        raise RuntimeError(
            "cqsa_cuda is not importable in this Python environment. "
            "Rebuild the CUDA extension with build_cqsa.sh or use pure Python backends. "
            f"Original import error: {_CQSA_CUDA_IMPORT_ERROR!r}"
        )
    return mod


def maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_attn_probs: bool = False,
    return_lse: bool = False,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    cuda_ext = _require_cqsa_cuda(causal=bool(causal))
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, s_dmask, _ = cuda_ext.fwd(
        q,
        k,
        v,
        None,
        alibi_slopes,
        float(dropout_p),
        float(softmax_scale),
        bool(causal),
        int(window_size[0]),
        int(window_size[1]),
        float(softcap),
        bool(return_attn_probs and dropout_p > 0),
        None,
    )
    # The kernel always produces softmax_lse; it was simply being dropped. The
    # backward needs it, so `return_lse` hands it back without a second pass.
    if return_lse:
        return out, softmax_lse
    return out if not return_attn_probs else (out, softmax_lse, s_dmask)


def flash_attn_func_cqs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cqs_chunk_ends: torch.Tensor,
    cqs_owner_chunk: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_attn_probs: bool = False,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    cuda_ext = _require_cqsa_cuda()
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if cqs_chunk_ends.device != q.device:
        cqs_chunk_ends = cqs_chunk_ends.to(device=q.device)
    if cqs_chunk_ends.dtype != torch.int32:
        cqs_chunk_ends = cqs_chunk_ends.to(dtype=torch.int32)
    cqs_chunk_ends = cqs_chunk_ends.contiguous()

    out, softmax_lse, s_dmask, _ = cuda_ext.fwd_cqs(
        q,
        k,
        v,
        None,
        alibi_slopes,
        float(dropout_p),
        float(softmax_scale),
        bool(causal),
        int(window_size[0]),
        int(window_size[1]),
        float(softcap),
        bool(return_attn_probs and dropout_p > 0),
        None,
        cqs_chunk_ends,
        int(cqs_owner_chunk),
    )
    return out if not return_attn_probs else (out, softmax_lse, s_dmask)


def flash_attn_func_cqsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_itr: int = 1,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    return_denominator: bool = False,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    cuda_ext = _require_cqsa_cuda()
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, den = cuda_ext.fwd_cqsa(
        q,
        k,
        v,
        float(dropout_p),
        float(softmax_scale),
        bool(causal),
        int(num_itr),
    )
    return (out, den) if return_denominator else out


CQS_BLK_SIZE = 64


def cqs_block_summaries(
    group_bits: torch.Tensor, blk_size: int = CQS_BLK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-block OR / AND summaries of the group bits.

    They let the kernel decide in O(1) whether a whole (row-block, col-block)
    tile contains any masked pair. Without them, merely enabling CQS forces the
    generic per-element masking path on every tile -- measured ~4.3x slower than
    plain FlashAttention even with an all-zero mask.

    Cheap and shape-only, so callers should build this once per subproblem
    alongside the mask, not per attention call.
    """
    import numpy as np

    # Single ufunc.reduce in numpy. A per-column Python loop over torch tensors
    # cost 198 ms per subproblem at N=131072: torch defaults to one intra-op
    # thread per core (80 here) and OpenMP fork/join dominates tiny reductions.
    if isinstance(group_bits, torch.Tensor):
        bits_np = group_bits.reshape(-1).to(torch.int64).cpu().numpy()
    else:
        bits_np = np.asarray(group_bits, dtype=np.int64).reshape(-1)

    L = int(bits_np.shape[0])
    nblk = (L + blk_size - 1) // blk_size
    pad = nblk * blk_size - L
    if pad:
        # OR ignores 0; AND must ignore -1 (all ones).
        or_src = np.concatenate([bits_np, np.zeros(pad, dtype=np.int64)])
        and_src = np.concatenate([bits_np, np.full(pad, -1, dtype=np.int64)])
    else:
        or_src, and_src = bits_np, bits_np

    blk_or = np.bitwise_or.reduce(or_src.reshape(nblk, blk_size), axis=1)
    blk_and = np.bitwise_and.reduce(and_src.reshape(nblk, blk_size), axis=1)
    return (
        torch.from_numpy(np.ascontiguousarray(blk_or)),
        torch.from_numpy(np.ascontiguousarray(blk_and)),
    )


def flash_attn_func_cqs_group_bits(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cqs_group_bits: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_attn_probs: bool = False,
    cqs_blk_or: Optional[torch.Tensor] = None,
    cqs_blk_and: Optional[torch.Tensor] = None,
    cqs_blk_size: int = CQS_BLK_SIZE,
    fp32_out: bool = False,
    cqs_block_base: Optional[torch.Tensor] = None,
    cqs_seg_align: int = 0,
    out: Optional[torch.Tensor] = None,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    cuda_ext = _require_cqsa_cuda(causal=bool(causal))
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if cqs_group_bits.device != q.device:
        cqs_group_bits = cqs_group_bits.to(device=q.device)
    if cqs_group_bits.dtype != torch.int64:
        cqs_group_bits = cqs_group_bits.to(dtype=torch.int64)
    cqs_group_bits = cqs_group_bits.contiguous()

    if cqs_blk_or is None or cqs_blk_and is None:
        cqs_blk_or, cqs_blk_and = cqs_block_summaries(cqs_group_bits, int(cqs_blk_size))
    cqs_blk_or = cqs_blk_or.to(device=q.device, dtype=torch.int64).contiguous()
    cqs_blk_and = cqs_blk_and.to(device=q.device, dtype=torch.int64).contiguous()

    # fp32 copy of the normalised output. The kernel holds acc_o in fp32
    # registers anyway; the stock epilogue rounds it to the input dtype, and
    # that rounding is ~83% of a subproblem's error.
    # With segmented input q is a strided view over a LONGER tensor, so
    # empty_like(q) would allocate by max linear index rather than by numel.
    # Give the kernel an explicit contiguous output in that case.
    if out is None and cqs_block_base is not None:
        out = torch.empty(tuple(q.shape), device=q.device, dtype=q.dtype)

    acc_out = None
    if fp32_out:
        # Match `out`'s strides: the kernel reuses the o_* element strides for
        # this buffer. When out is allocated by the kernel it is empty_like(q),
        # and q here is typically a transposed *view*, hence not contiguous.
        if out is not None:
            acc_out = torch.empty_strided(
                out.shape, out.stride(), device=q.device, dtype=torch.float32
            )
        else:
            acc_out = torch.empty_strided(
                q.shape, q.stride(), device=q.device, dtype=torch.float32
            )

    out, softmax_lse, s_dmask, _ = cuda_ext.fwd_cqs_group_bits(
        q,
        k,
        v,
        out,
        alibi_slopes,
        float(dropout_p),
        float(softmax_scale),
        bool(causal),
        int(window_size[0]),
        int(window_size[1]),
        float(softcap),
        bool(return_attn_probs and dropout_p > 0),
        None,
        cqs_group_bits,
        cqs_blk_or,
        cqs_blk_and,
        int(cqs_blk_size),
        acc_out,
        cqs_block_base,
        int(cqs_seg_align) if cqs_block_base is not None else None,
    )
    if fp32_out:
        out = acc_out
    return out if not return_attn_probs else (out, softmax_lse, s_dmask)


def flash_attn_bwd_cqs_group_bits(
    *,
    dout_num: torch.Tensor,
    dden: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cqs_group_bits: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward for local CQS attention with per-token group-bit mask.

    Inputs:
      dout_num: [B, L, H, D] gradient wrt local numerator
      dden: [B, H, L] gradient wrt local denominator
      q/k/v: [B, L, H, D]
      cqs_group_bits: [L] int64 bitmask
    Returns:
      dQ, dK, dV each in [B, L, H, D]
    """
    cuda_ext = _require_cqsa_cuda()
    if not hasattr(cuda_ext, "bwd_cqs_group_bits"):
        raise RuntimeError(
            "cqsa_cuda does not provide bwd_cqs_group_bits. Rebuild the extension (build_cqsa.sh)."
        )
    dout_num, dden, q, k, v = [maybe_contiguous(x) for x in (dout_num, dden, q, k, v)]
    if cqs_group_bits.device != q.device:
        cqs_group_bits = cqs_group_bits.to(device=q.device)
    if cqs_group_bits.dtype != torch.int64:
        cqs_group_bits = cqs_group_bits.to(dtype=torch.int64)
    cqs_group_bits = cqs_group_bits.contiguous()
    dQ, dK, dV = cuda_ext.bwd_cqs_group_bits(
        dout_num,
        dden,
        q,
        k,
        v,
        cqs_group_bits,
        float(softmax_scale),
    )
    return dQ, dK, dV


def flash_attn_bwd_cqs_global_lse(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cqs_group_bits: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool = False,
    cqs_blk_or: Optional[torch.Tensor] = None,
    cqs_blk_and: Optional[torch.Tensor] = None,
    cqs_blk_size: int = 64,
    dsoftmax_sum: Optional[torch.Tensor] = None,
):
    """
    Backward for one CQS subproblem, using the **global** log-sum-exp.

    All tensors in FlashAttention layout: q/k/v/dout/out ``[B, L, H, D]``,
    ``softmax_lse`` ``[B, H, L]``.

    ``softmax_lse`` must be the global lse gathered to this subsequence's
    tokens, not a locally-recomputed one. That is what makes this exact: the
    kernel then forms ``p = exp(s - lse) <= 1``, which is the true global
    attention weight for the retained pairs, so the standard softmax backward
    applies and the per-subproblem gradients simply add.

    This replaces ``flash_attn_bwd_cqs_group_bits`` (the ``cqsa_numden_mode``
    path), which consumes dNum/dDen and therefore needs ``Den = exp(lse)`` --
    infinite above ``lse ~= 88.7`` in fp32, silently zeroing the gradients.
    """
    if cqsa_cuda is None or _os.environ.get("CQSA_BACKWARD", "").lower() == "triton":
        # Zero-build path: the Triton backward (same global-lse contract).
        from .triton_kernel import cqs_attention_backward
        return cqs_attention_backward(dout, q, k, v, softmax_lse, cqs_group_bits, causal=causal,
                                      scale=float(softmax_scale), delta=dsoftmax_sum, out=out,
                                      blk_or=cqs_blk_or, blk_and=cqs_blk_and, blk_size=int(cqs_blk_size))
    cuda_ext = _require_cqsa_cuda()
    if dsoftmax_sum is not None:
        # With dsoftmax_sum supplied, the preprocess pass is skipped and O is
        # never read -- but the shape checks still want a tensor of O's shape,
        # so dout stands in. Callers pass None for `out` to make that explicit.
        if out is None:
            out = dout
        dsoftmax_sum = dsoftmax_sum.to(device=q.device, dtype=torch.float32).contiguous()
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    if cqs_group_bits.device != q.device:
        cqs_group_bits = cqs_group_bits.to(device=q.device)
    cqs_group_bits = cqs_group_bits.to(dtype=torch.int64).contiguous()
    softmax_lse = softmax_lse.to(dtype=torch.float32).contiguous()

    # Block summaries drive the mask's O(1) per-tile early-out. Omitting them
    # does not merely forgo an optimisation: every tile then runs the generic
    # per-element masking loop, which reads cqs_group_bits from global memory
    # twice per element inside the kernel's hottest loop.
    if cqs_blk_or is None or cqs_blk_and is None:
        cqs_blk_or, cqs_blk_and = cqs_block_summaries(cqs_group_bits, int(cqs_blk_size))
    cqs_blk_or = cqs_blk_or.to(device=q.device, dtype=torch.int64).contiguous()
    cqs_blk_and = cqs_blk_and.to(device=q.device, dtype=torch.int64).contiguous()

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    cuda_ext.bwd(
        dout, q, k, v, out, softmax_lse,
        dq, dk, dv,
        None,                      # alibi_slopes
        0.0,                       # p_dropout
        float(softmax_scale),
        bool(causal),
        -1, -1,                    # window
        0.0,                       # softcap
        False,                     # deterministic
        None,                      # gen
        None,                      # rng_state
        cqs_group_bits,
        cqs_blk_or,
        cqs_blk_and,
        int(cqs_blk_size),
        dsoftmax_sum,
    )
    return dq, dk, dv
