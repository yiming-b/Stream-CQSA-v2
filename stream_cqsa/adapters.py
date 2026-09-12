"""
Adapters that turn a monolithic attention implementation into a Stream-CQSA
inner kernel without writing CUDA.

What an inner kernel must provide (see devkit.compare_kernels):

    inner(q_i, k_i, v_i, group_bits, *, causal, scale, **_) -> (out_i, lse_i)

i.e. attention restricted to the pair set {(r, c): bits[r] & bits[c] == 0,
(c <= r if causal)}, returning the normalised output ``out_i [B, L, H, D]``
AND the per-row log-sum-exp of the kept scores ``lse_i [B, L, H]`` (both
token-major, float32). Any implementation that (a) accepts an
arbitrary pairwise mask and (b) returns the log-sum-exp can therefore be
plugged in mechanically. Two such families exist in PyTorch today:

* ``flex_inner(score_mod=None)`` -- torch's FlexAttention. The CQS pair set is
  expressed as a ``mask_mod`` (a block mask is built per subproblem, so the
  masked tiles are skipped by the compiled kernel too), and ``return_lse=True``
  gives the lse. Any attention that FlexAttention can express -- ALiBi,
  soft-capping, sliding windows, document masks, relative biases -- becomes a
  Stream-CQSA inner kernel by passing its ``score_mod``/extra mask, with no
  other change. That is the "automatic conversion" for the score-modification
  family.

* ``dense_inner(mono_masked)`` -- for small subproblems, any callable that
  takes a dense boolean mask and returns (out, lse) (or just out, in which
  case the lse is recomputed densely). This is the differential-testing route.

Kernels that return NO log-sum-exp cannot be adapted losslessly: the merge of
subproblems needs the row statistics, and recovering them from the output
alone is impossible (the output is normalised). For those the only exact
option is a second pass that computes the lse (``lse_from_scores``), which
costs another QK^T; the devkit reports that overhead honestly.
"""
from __future__ import annotations

from typing import Callable

import torch

_flex_cache: dict = {}


def _bits_pair_mask(bits: torch.Tensor, causal: bool):
    def mask_mod(b, h, q_idx, kv_idx):
        keep = (bits[q_idx] & bits[kv_idx]) == 0
        if causal:
            keep = keep & (kv_idx <= q_idx)
        return keep
    return mask_mod


def flex_inner(score_mod: Callable | None = None, *, extra_mask_mod: Callable | None = None,
               block_size: int = 128, compile: bool = True, global_positions: bool = True) -> Callable:
    """
    Build a Stream-CQSA inner kernel from torch FlexAttention.

    `score_mod(score, b, h, q_idx, kv_idx)` and `extra_mask_mod(b, h, q_idx, kv_idx)`
    are written against GLOBAL token positions (as in the monolithic call): a
    subproblem sees a gathered subsequence, so the engine hands the inner kernel
    the gather index (`token_ids`) and the adapter remaps local indices through
    it before calling the user's functions. That is the one thing an automatic
    conversion must get right for position-dependent attention (ALiBi, sliding
    windows, document masks); with `global_positions=False` the raw local
    indices are passed instead (only correct for position-free score_mods).
    """
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    fa = torch.compile(flex_attention, dynamic=False) if compile else flex_attention

    def inner(q_i, k_i, v_i, group_bits, *, causal: bool, scale: float, token_ids=None, **_):
        B, L, H, D = q_i.shape
        dev = q_i.device
        bits = group_bits.to(dev, torch.int64)
        mm = _bits_pair_mask(bits, causal)
        ids = token_ids.to(dev, torch.int64) if (token_ids is not None and global_positions) else None
        if extra_mask_mod is not None:
            base = mm
            if ids is not None:
                mm = lambda b, h, qi, ki: base(b, h, qi, ki) & extra_mask_mod(b, h, ids[qi], ids[ki])
            else:
                mm = lambda b, h, qi, ki: base(b, h, qi, ki) & extra_mask_mod(b, h, qi, ki)
        sm = score_mod
        if score_mod is not None and ids is not None:
            sm = lambda score, b, h, qi, ki: score_mod(score, b, h, ids[qi], ids[ki])
        bm = create_block_mask(mm, B=None, H=None, Q_LEN=L, KV_LEN=L, device=dev, BLOCK_SIZE=block_size)
        qb, kb, vb = (t.transpose(1, 2) for t in (q_i, k_i, v_i))     # [B,H,L,D]
        out, lse = fa(qb, kb, vb, score_mod=sm, block_mask=bm, scale=float(scale), return_lse=True)
        # rows with no kept pair: flex returns lse=-inf and out=0, which is the contract.
        # Engine contract: out_i [B,L,H,D], lse_i [B,L,H] (token-major, like the CUDA kernel's return).
        return out.transpose(1, 2).float(), lse.float().transpose(1, 2)

    return inner


def lse_from_scores(q_i, k_i, v_i, keep: torch.Tensor, scale: float) -> torch.Tensor:
    """Dense fp32 per-row lse over kept pairs (second pass for kernels without lse). [B,H,L]"""
    s = torch.matmul(q_i.transpose(1, 2).float(), k_i.transpose(1, 2).float().transpose(-1, -2)) * float(scale)
    s = s.masked_fill(~keep[None, None], float("-inf"))
    return torch.logsumexp(s, dim=-1)


def dense_inner(mono_masked: Callable, *, returns_lse: bool = True) -> Callable:
    """
    Wrap ``mono_masked(q, k, v, mask, scale) -> (out [B,H,L,D], lse [B,H,L])`` (or
    just out if returns_lse=False) where mask is a dense bool [L, L] (True = keep).
    Only for small L (the mask is O(L^2)).
    """
    def inner(q_i, k_i, v_i, group_bits, *, causal: bool, scale: float, **_):
        L = q_i.shape[1]
        bits = group_bits.to(q_i.device, torch.int64)
        keep = (bits[:, None] & bits[None, :]) == 0
        if causal:
            keep = keep & torch.ones(L, L, dtype=torch.bool, device=q_i.device).tril()
        qb, kb, vb = (t.transpose(1, 2) for t in (q_i, k_i, v_i))
        r = mono_masked(qb, kb, vb, keep, float(scale))
        if returns_lse:
            out, lse = r
        else:
            out, lse = r, lse_from_scores(q_i, k_i, v_i, keep, scale)
        return out.transpose(1, 2).float(), lse.float().transpose(1, 2)   # [B,L,H,D], [B,L,H]
    return inner


def sdpa_masked(qb, kb, vb, keep, scale):
    """Reference monolithic-with-mask kernel (torch SDPA, no lse): use with dense_inner(..., returns_lse=False)."""
    import torch.nn.functional as F
    return F.scaled_dot_product_attention(qb, kb, vb, attn_mask=keep, scale=scale)
