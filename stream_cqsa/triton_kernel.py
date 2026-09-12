"""
Stream-CQSA inner kernel in Triton: FlashAttention-2 forward with the CQS
pair mask, no CUDA build required.

    from stream_cqsa.triton_kernel import triton_inner, triton_attention

    out_i, lse_i = triton_inner(q_i, k_i, v_i, group_bits, causal=True, scale=s,
                                blk_or=..., blk_and=...)          # engine contract
    out = triton_attention(q, k, v, causal=True)                  # monolithic, [B,H,N,D]

Contract (identical to the CUDA kernel's, so `stream_cqsa_forward(..., inner=triton_inner)`
is a drop-in): q/k/v are [B, L, H, D] token-major (any strides), `group_bits`
is int64 [L] and pair (row, col) is kept iff ``bits[row] & bits[col] == 0``
(and col <= row when causal). Returns the normalised output in float32
[B, L, H, D] and the per-row log-sum-exp float32 [B, L, H] (natural log,
-inf for rows with no kept pair, exactly as the CUDA kernel reports it).

Design, mirroring what made the v2 CUDA kernel fast (docs/kernel_technical_note.md):

* Per-64-token block summaries (``blk_or``, ``blk_and``) give an O(1) verdict
  per 128x64 tile: *masked* (skip the tile entirely -- no K/V load, no GEMM),
  *clear* (plain FlashAttention tile), or *mixed* (apply the per-element
  bit test). In a real subproblem ~22% of causal tiles are masked and <1% mixed.
  The row half of the verdict is computed once per program.
* The causal diagonal is handled in a separate short loop so the main loop
  has no causal test (the Triton tutorial's structure).
* Online softmax in float32 with exp2; the -inf/-inf case of empty rows is
  guarded so a row that never sees a kept pair ends with out=0, lse=-inf.

Performance: Triton's FlashAttention-2 forward on A100 is roughly 0.7-0.85x of
the CUDA one at hdim 64; this kernel inherits that ratio (measured numbers in
LOG.md). It is the zero-build path; the CUDA kernel remains the fast path.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _cqs_attn_fwd_kernel(
    Q, K, V, Out, Lse,
    BITS, BLK_OR, BLK_AND, RUN_S, RUN_E, NRUNS, MAXR,
    sm_scale_log2,            # softmax scale * log2(e)
    L, NUM_BLK,
    stride_qb, stride_ql, stride_qh, stride_qd,
    stride_kb, stride_kl, stride_kh, stride_kd,
    stride_vb, stride_vl, stride_vh, stride_vd,
    stride_ob, stride_ol, stride_oh, stride_od,
    stride_lb, stride_lh, stride_ll,
    H,
    CAUSAL: tl.constexpr, CQS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    CQS_BLK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n_base = tl.arange(0, BLOCK_N)
    row_ok = offs_m < L

    # Q tile [BLOCK_M, BLOCK_D]
    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_ql + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=row_ok[:, None], other=0.0)

    # Row-side CQS summaries, once per program.
    if CQS:
        row_bits = tl.load(BITS + offs_m, mask=row_ok, other=0)                     # int64 [BLOCK_M]
        rb0 = (pid_m * BLOCK_M) // CQS_BLK
        rblk = rb0 + tl.arange(0, BLOCK_M // CQS_BLK)
        rblk_ok = rblk < NUM_BLK
        # bitwise OR / AND over the row block's summary words (tl.reduce with a custom combine)
        ro = tl.load(BLK_OR + rblk, mask=rblk_ok, other=0)
        ra = tl.load(BLK_AND + rblk, mask=rblk_ok, other=-1)
        row_or = tl.reduce(ro, 0, _bor)
        row_and = tl.reduce(ra, 0, _band)
        # cqs_tile_masked's row preconditions: whole row block inside the sequence
        row_full = (pid_m * BLOCK_M + BLOCK_M) <= L
        row_may_mask = row_full & (row_and != 0)
        row_clear = row_or == 0
    else:
        row_bits = tl.zeros([BLOCK_M], dtype=tl.int64)
        row_or = tl.zeros([], dtype=tl.int64)
        row_and = tl.zeros([], dtype=tl.int64)
        row_may_mask = False
        row_clear = True

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Key range: causal -> up to the diagonal; the last tile(s) that intersect
    # the diagonal are handled in the second loop with the causal test.
    if CAUSAL:
        n_end = tl.minimum((pid_m + 1) * BLOCK_M, L)          # exclusive
        n_diag_start = (pid_m * BLOCK_M) // BLOCK_N * BLOCK_N  # first key block touching the diagonal
    else:
        n_end = L
        n_diag_start = L

    # ---- main loop: live runs of tiles strictly below the diagonal ----
    # The run table (built on the host from the summaries) removes the
    # fully-masked tiles from the iteration space, so the tile body carries no
    # `masked` conditional and Triton pipelines its K/V loads across iterations
    # exactly as in the plain kernel. (A scalar `if` around the body would
    # block the software pipeliner: measured +27-31% at every L.)
    if CQS:
        nr = tl.load(NRUNS + pid_m)
        for i in range(0, nr):
            rs = tl.load(RUN_S + pid_m * MAXR + i)
            re = tl.minimum(tl.load(RUN_E + pid_m * MAXR + i), n_diag_start)
            for start_n in range(rs, re, BLOCK_N):
                m_i, l_i, acc = _process_tile(
                    q, K, V, BITS, BLK_OR, BLK_AND, b, h, start_n, L, NUM_BLK,
                    stride_kb, stride_kl, stride_kh, stride_kd, stride_vb, stride_vl, stride_vh, stride_vd,
                    offs_m, offs_n_base, offs_d, row_bits, row_or, row_and, row_may_mask, row_clear,
                    m_i, l_i, acc, sm_scale_log2,
                    False, CQS, False, BLOCK_N, CQS_BLK)
    else:
        for start_n in range(0, n_diag_start, BLOCK_N):
            m_i, l_i, acc = _process_tile(
                q, K, V, BITS, BLK_OR, BLK_AND, b, h, start_n, L, NUM_BLK,
                stride_kb, stride_kl, stride_kh, stride_kd, stride_vb, stride_vl, stride_vh, stride_vd,
                offs_m, offs_n_base, offs_d, row_bits, row_or, row_and, row_may_mask, row_clear,
                m_i, l_i, acc, sm_scale_log2,
                False, CQS, False, BLOCK_N, CQS_BLK)

    # ---- diagonal / tail loop: causal test on ----
    for start_n in range(n_diag_start, n_end, BLOCK_N):
        m_i, l_i, acc = _process_tile(
            q, K, V, BITS, BLK_OR, BLK_AND, b, h, start_n, L, NUM_BLK,
            stride_kb, stride_kl, stride_kh, stride_kd, stride_vb, stride_vl, stride_vh, stride_vd,
            offs_m, offs_n_base, offs_d, row_bits, row_or, row_and, row_may_mask, row_clear,
            m_i, l_i, acc, sm_scale_log2,
            CAUSAL, CQS, True, BLOCK_N, CQS_BLK)

    # ---- epilogue ----
    has = l_i > 0.0
    l_safe = tl.where(has, l_i, 1.0)
    out = acc / l_safe[:, None]
    out = tl.where(has[:, None], out, 0.0)
    lse = tl.where(has, (m_i + tl.log2(l_safe)) / LOG2E_C(), float("-inf"))   # natural log
    o_ptrs = Out + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out, mask=row_ok[:, None])
    l_ptrs = Lse + b * stride_lb + h * stride_lh + offs_m * stride_ll
    tl.store(l_ptrs, lse, mask=row_ok)


@triton.jit
def LOG2E_C():
    return 1.4426950408889634


@triton.jit
def _bor(a, b):
    return a | b


@triton.jit
def _band(a, b):
    return a & b


@triton.jit
def _process_tile(
    q, K, V, BITS, BLK_OR, BLK_AND, b, h, start_n, L, NUM_BLK,
    stride_kb, stride_kl, stride_kh, stride_kd, stride_vb, stride_vl, stride_vh, stride_vd,
    offs_m, offs_n_base, offs_d, row_bits, row_or, row_and, row_may_mask, row_clear,
    m_i, l_i, acc, sm_scale_log2,
    CAUSAL_TILE: tl.constexpr, CQS: tl.constexpr, CHECK_MASKED: tl.constexpr, BLOCK_N: tl.constexpr, CQS_BLK: tl.constexpr,
):
    offs_n = start_n + offs_n_base
    col_ok = offs_n < L
    # ---- tile verdict from the column summaries ----
    if CQS:
        cb0 = start_n // CQS_BLK
        cblk = cb0 + tl.arange(0, BLOCK_N // CQS_BLK)
        cblk_ok = cblk < NUM_BLK
        co = tl.load(BLK_OR + cblk, mask=cblk_ok, other=0)
        col_or = tl.reduce(co, 0, _bor)
        clear = row_clear | ((row_or & col_or) == 0)
        if CHECK_MASKED:
            ca = tl.load(BLK_AND + cblk, mask=cblk_ok, other=-1)
            col_and = tl.reduce(ca, 0, _band)
            col_full = (start_n + BLOCK_N) <= L
            masked = row_may_mask & col_full & ((row_and & col_and) != 0)
        else:
            masked = False          # the run table already excluded masked tiles
    else:
        masked = False
        clear = True

    if not masked:
        k_ptrs = K + b * stride_kb + h * stride_kh + offs_n[None, :] * stride_kl + offs_d[:, None] * stride_kd   # [D, N]
        k = tl.load(k_ptrs, mask=col_ok[None, :], other=0.0)
        qk = tl.dot(q, k) * sm_scale_log2                                        # [M, N], log2 units
        # padding / causal / CQS masks
        qk = tl.where(col_ok[None, :], qk, float("-inf"))
        if CAUSAL_TILE:
            qk = tl.where(offs_n[None, :] <= offs_m[:, None], qk, float("-inf"))
        if CQS:
            if not clear:
                col_bits = tl.load(BITS + offs_n, mask=col_ok, other=0)
                drop = (row_bits[:, None] & col_bits[None, :]) != 0
                qk = tl.where(drop, float("-inf"), qk)
        # online softmax (base 2)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)                     # m_i=-inf -> 0
        p = tl.exp2(qk - m_safe[:, None])                 # qk=-inf -> 0
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v_ptrs = V + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vl + offs_d[None, :] * stride_vd   # [N, D]
        v = tl.load(v_ptrs, mask=col_ok[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new
    return m_i, l_i, acc


def _block_reduce(words: torch.Tensor, blk_tokens: int, cqs_blk: int, L: int, op: str):
    """Reduce per-64-token summary words to per-(blk_tokens)-block words. [nblk_out]"""
    per = blk_tokens // cqs_blk
    n_out = (L + blk_tokens - 1) // blk_tokens
    fill = 0 if op == "or" else -1
    w = words
    pad = n_out * per - w.numel()
    if pad > 0:
        w = torch.cat([w, w.new_full((pad,), fill)])
    w = w.view(n_out, per)
    out = w[:, 0].clone()
    for j in range(1, per):
        out = (out | w[:, j]) if op == "or" else (out & w[:, j])
    return out


def live_runs(blk_and: torch.Tensor, L: int, block_r: int, block_c: int, cqs_blk: int = 64):
    """
    For each row block (block_r tokens) the runs of column blocks (block_c
    tokens) that are NOT fully masked, as token offsets: (run_s, run_e) int32
    [R, MAXR] and n_runs int32 [R]. A tile is fully masked iff both blocks are
    whole and (row_and & col_and) != 0 -- the same verdict as the kernels'.
    Built per distinct row word (a handful in real subproblems), O(patterns x C).
    """
    dev = blk_and.device
    R = (L + block_r - 1) // block_r
    C = (L + block_c - 1) // block_c
    row_and = _block_reduce(blk_and, block_r, cqs_blk, L, "and")
    col_and = _block_reduce(blk_and, block_c, cqs_blk, L, "and")
    row_full = (torch.arange(R, device=dev) + 1) * block_r <= L
    col_full = (torch.arange(C, device=dev) + 1) * block_c <= L
    row_key = torch.where(row_full, row_and, torch.zeros_like(row_and))       # a partial row block is never masked
    uniq, inv = torch.unique(row_key, return_inverse=True)
    starts_list, ends_list = [], []
    maxr = 1
    for u in uniq.tolist():
        if u == 0:
            starts_list.append(torch.zeros(1, dtype=torch.int32, device=dev)); ends_list.append(torch.full((1,), L, dtype=torch.int32, device=dev)); continue
        live = ~(col_full & ((col_and & u) != 0))                              # [C] bool
        pad = torch.zeros(1, dtype=torch.bool, device=dev)
        d = torch.diff(torch.cat([pad, live, pad]).to(torch.int8))
        st = torch.nonzero(d == 1).flatten(); en = torch.nonzero(d == -1).flatten()
        starts_list.append((st * block_c).to(torch.int32)); ends_list.append(torch.clamp(en * block_c, max=L).to(torch.int32))
        maxr = max(maxr, int(st.numel()))
    run_s = torch.zeros(len(uniq), maxr, dtype=torch.int32, device=dev)
    run_e = torch.zeros(len(uniq), maxr, dtype=torch.int32, device=dev)
    n_runs = torch.zeros(len(uniq), dtype=torch.int32, device=dev)
    for i, (st, en) in enumerate(zip(starts_list, ends_list)):
        run_s[i, :st.numel()] = st; run_e[i, :en.numel()] = en; n_runs[i] = st.numel()
    return run_s[inv].contiguous(), run_e[inv].contiguous(), n_runs[inv].contiguous(), maxr


def _summaries(bits: torch.Tensor, blk: int = 64):
    L = bits.numel()
    nblk = (L + blk - 1) // blk
    pad = nblk * blk - L
    b_or = torch.cat([bits, bits.new_zeros(pad)]) if pad else bits
    b_and = torch.cat([bits, bits.new_full((pad,), -1)]) if pad else bits
    b_or = b_or.view(nblk, blk); b_and = b_and.view(nblk, blk)
    # bitwise reductions on GPU without a Python loop over blocks: fold columns
    o = b_or[:, 0].clone(); a = b_and[:, 0].clone()
    for j in range(1, blk):
        o |= b_or[:, j]; a &= b_and[:, j]
    return o.contiguous(), a.contiguous()


def cqs_attention_forward(q, k, v, group_bits=None, *, causal: bool, scale: float | None = None,
                          blk_or=None, blk_and=None, blk_size: int = 64,
                          block_m: int = 128, block_n: int = 64, num_warps: int = 4, num_stages: int = 3):
    """
    q/k/v: [B, L, H, D] (token-major; strides are honoured). Returns
    (out float32 [B, L, H, D], lse float32 [B, H, L]).
    """
    B, L, H, D = q.shape
    assert k.shape == q.shape == v.shape
    assert D in (16, 32, 64, 128), "head dim must be a power of two <= 128"
    if scale is None:
        scale = D ** -0.5
    dev = q.device
    cqs = group_bits is not None
    if cqs:
        bits = group_bits.to(dev, torch.int64).contiguous()
        assert bits.numel() == L
        if blk_or is None or blk_and is None:
            blk_or, blk_and = _summaries(bits, blk_size)
        blk_or = blk_or.to(dev, torch.int64).contiguous(); blk_and = blk_and.to(dev, torch.int64).contiguous()
        num_blk = blk_or.numel()
        assert blk_size == 64 and block_m % 64 == 0 and block_n % 64 == 0
        run_s, run_e, n_runs, maxr = live_runs(blk_and, L, block_m, block_n, blk_size)
    else:
        bits = torch.zeros(1, dtype=torch.int64, device=dev); blk_or = bits; blk_and = bits; num_blk = 0
        run_s = run_e = torch.zeros(1, dtype=torch.int32, device=dev); n_runs = run_s; maxr = 1
    out = torch.empty(B, L, H, D, dtype=torch.float32, device=dev)
    lse = torch.empty(B, H, L, dtype=torch.float32, device=dev)
    grid = (triton.cdiv(L, block_m), B * H)
    _cqs_attn_fwd_kernel[grid](
        q, k, v, out, lse, bits, blk_or, blk_and, run_s, run_e, n_runs, maxr,
        float(scale) * LOG2E, L, num_blk,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        H,
        CAUSAL=bool(causal), CQS=cqs,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=D, CQS_BLK=64,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out, lse


def triton_inner(q_i, k_i, v_i, group_bits, *, causal: bool, scale: float,
                 blk_or=None, blk_and=None, blk_size: int = 64, block_base=None, **_):
    """Stream-CQSA inner kernel (engine contract): (out [B,L,H,D] fp32, lse [B,L,H] fp32)."""
    if block_base is not None:
        # The engine's shared_chunks mode hands the kernel a segmented (chunk-pool)
        # view that only the CUDA kernel can address. Refuse loudly rather than
        # read the wrong memory; the engine disables shared_chunks for this inner.
        raise ValueError("triton_inner does not support segmented (shared_chunks) inputs")
    out, lse = cqs_attention_forward(q_i, k_i, v_i, group_bits, causal=causal, scale=scale,
                                     blk_or=blk_or, blk_and=blk_and, blk_size=blk_size)
    return out, lse.transpose(1, 2)


def triton_attention(q, k, v, *, causal: bool = False, scale: float | None = None, return_lse: bool = False):
    """Monolithic attention, q/k/v [B, H, N, D] -> out [B, H, N, D] in the input dtype (Triton, no build)."""
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))          # [B, N, H, D] views
    out, lse = cqs_attention_forward(qt, kt, vt, None, causal=causal, scale=scale)
    out = out.transpose(1, 2).to(q.dtype)
    return (out, lse) if return_lse else out


# ===========================================================================
# Backward (global-lse form): two deterministic kernels, dK/dV and dQ
# ===========================================================================

@triton.jit
def _tile_verdict(BLK_OR, BLK_AND, start_row, start_col, L, NUM_BLK,
                  BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, CQS_BLK: tl.constexpr):
    """(masked, clear) for the tile rows [start_row, +BLOCK_R) x cols [start_col, +BLOCK_C)."""
    rblk = start_row // CQS_BLK + tl.arange(0, BLOCK_R // CQS_BLK)
    cblk = start_col // CQS_BLK + tl.arange(0, BLOCK_C // CQS_BLK)
    ro = tl.load(BLK_OR + rblk, mask=rblk < NUM_BLK, other=0)
    ra = tl.load(BLK_AND + rblk, mask=rblk < NUM_BLK, other=-1)
    co = tl.load(BLK_OR + cblk, mask=cblk < NUM_BLK, other=0)
    ca = tl.load(BLK_AND + cblk, mask=cblk < NUM_BLK, other=-1)
    row_or = tl.reduce(ro, 0, _bor); row_and = tl.reduce(ra, 0, _band)
    col_or = tl.reduce(co, 0, _bor); col_and = tl.reduce(ca, 0, _band)
    full = ((start_row + BLOCK_R) <= L) & ((start_col + BLOCK_C) <= L)
    masked = full & ((row_and & col_and) != 0)
    clear = (row_or & col_or) == 0
    return masked, clear


@triton.jit
def _tile_clear(BLK_OR, start_row, start_col, NUM_BLK, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, CQS_BLK: tl.constexpr):
    """True when no pair of the tile can be masked (OR summaries disjoint)."""
    rblk = start_row // CQS_BLK + tl.arange(0, BLOCK_R // CQS_BLK)
    cblk = start_col // CQS_BLK + tl.arange(0, BLOCK_C // CQS_BLK)
    ro = tl.load(BLK_OR + rblk, mask=rblk < NUM_BLK, other=0)
    co = tl.load(BLK_OR + cblk, mask=cblk < NUM_BLK, other=0)
    return (tl.reduce(ro, 0, _bor) & tl.reduce(co, 0, _bor)) == 0


@triton.jit
def _cqs_attn_bwd_dkdv_kernel(
    Q, K, V, DO, LSE, DELTA, DK, DV, BITS, BLK_OR, BLK_AND, RUN_S, RUN_E, NRUNS, MAXR,
    sm_scale, L, NUM_BLK, H,
    stride_qb, stride_ql, stride_qh, stride_qd,
    stride_kb, stride_kl, stride_kh, stride_kd,
    stride_vb, stride_vl, stride_vh, stride_vd,
    stride_db, stride_dl, stride_dh, stride_dd,       # dout
    stride_lb, stride_lh, stride_ll,                  # lse and delta share this layout [B,H,L]
    stride_ob, stride_ol, stride_oh, stride_od,       # dk/dv output [B,L,H,D]
    CAUSAL: tl.constexpr, CQS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, CQS_BLK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    col_ok = offs_n < L
    k = tl.load(K + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_kl + offs_d[None, :] * stride_kd, mask=col_ok[:, None], other=0.0)
    v = tl.load(V + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vl + offs_d[None, :] * stride_vd, mask=col_ok[:, None], other=0.0)
    if CQS:
        col_bits = tl.load(BITS + offs_n, mask=col_ok, other=0)
    else:
        col_bits = tl.zeros([BLOCK_N], dtype=tl.int64)
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    qk_scale = sm_scale * LOG2E_C()
    if CAUSAL:
        m_start = (start_n // BLOCK_M) * BLOCK_M       # first query block that can see this key block
    else:
        m_start = 0
    # Live runs of query blocks for this key block (run table built on the host);
    # the tile body has no `masked` conditional, so the loads pipeline.
    if CQS:
        nr = tl.load(NRUNS + pid_n)
    else:
        nr = 1
    for i in range(0, nr):
        if CQS:
            rs = tl.maximum(tl.load(RUN_S + pid_n * MAXR + i), m_start)
            re = tl.load(RUN_E + pid_n * MAXR + i)
        else:
            rs = m_start
            re = L
        for start_m in range(rs, re, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            row_ok = offs_m < L
            if CQS:
                clear = _tile_clear(BLK_OR, start_m, start_n, NUM_BLK, BLOCK_M, BLOCK_N, CQS_BLK)
            else:
                clear = True
            q = tl.load(Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_ql + offs_d[None, :] * stride_qd, mask=row_ok[:, None], other=0.0)
            do = tl.load(DO + b * stride_db + h * stride_dh + offs_m[:, None] * stride_dl + offs_d[None, :] * stride_dd, mask=row_ok[:, None], other=0.0)
            lse = tl.load(LSE + b * stride_lb + h * stride_lh + offs_m * stride_ll, mask=row_ok, other=float("inf"))
            delta = tl.load(DELTA + b * stride_lb + h * stride_lh + offs_m * stride_ll, mask=row_ok, other=0.0)
            qk = tl.dot(q, tl.trans(k)) * qk_scale                                   # [M, N] log2 units
            keep = row_ok[:, None] & col_ok[None, :]
            if CAUSAL:
                keep = keep & (offs_n[None, :] <= offs_m[:, None])
            if CQS:
                if not clear:
                    row_bits = tl.load(BITS + offs_m, mask=row_ok, other=0)
                    keep = keep & ((row_bits[:, None] & col_bits[None, :]) == 0)
            lse_fin = lse != float("inf")
            lse_fin = lse_fin & (lse != float("-inf"))
            lse2 = tl.where(lse_fin, lse * LOG2E_C(), 0.0)
            p = tl.exp2(qk - lse2[:, None])
            p = tl.where(keep & lse_fin[:, None], p, 0.0)
            dv += tl.dot(tl.trans(p).to(do.dtype), do)                             # [N, D]
            dp = tl.dot(do, tl.trans(v))                                           # [M, N]
            ds = p * (dp - delta[:, None])
            dk += tl.dot(tl.trans(ds).to(q.dtype), q)                              # [N, D]
    dk = dk * sm_scale
    tl.store(DK + b * stride_ob + h * stride_oh + offs_n[:, None] * stride_ol + offs_d[None, :] * stride_od, dk.to(DK.dtype.element_ty), mask=col_ok[:, None])
    tl.store(DV + b * stride_ob + h * stride_oh + offs_n[:, None] * stride_ol + offs_d[None, :] * stride_od, dv.to(DV.dtype.element_ty), mask=col_ok[:, None])


@triton.jit
def _cqs_attn_bwd_dq_kernel(
    Q, K, V, DO, LSE, DELTA, DQ, BITS, BLK_OR, BLK_AND, RUN_S, RUN_E, NRUNS, MAXR,
    sm_scale, L, NUM_BLK, H,
    stride_qb, stride_ql, stride_qh, stride_qd,
    stride_kb, stride_kl, stride_kh, stride_kd,
    stride_vb, stride_vl, stride_vh, stride_vd,
    stride_db, stride_dl, stride_dh, stride_dd,
    stride_lb, stride_lh, stride_ll,
    stride_ob, stride_ol, stride_oh, stride_od,
    CAUSAL: tl.constexpr, CQS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, CQS_BLK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    row_ok = offs_m < L
    q = tl.load(Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_ql + offs_d[None, :] * stride_qd, mask=row_ok[:, None], other=0.0)
    do = tl.load(DO + b * stride_db + h * stride_dh + offs_m[:, None] * stride_dl + offs_d[None, :] * stride_dd, mask=row_ok[:, None], other=0.0)
    lse = tl.load(LSE + b * stride_lb + h * stride_lh + offs_m * stride_ll, mask=row_ok, other=float("inf"))
    delta = tl.load(DELTA + b * stride_lb + h * stride_lh + offs_m * stride_ll, mask=row_ok, other=0.0)
    lse_fin = (lse != float("inf")) & (lse != float("-inf"))
    lse2 = tl.where(lse_fin, lse * LOG2E_C(), 0.0)
    if CQS:
        row_bits = tl.load(BITS + offs_m, mask=row_ok, other=0)
    else:
        row_bits = tl.zeros([BLOCK_M], dtype=tl.int64)
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    qk_scale = sm_scale * LOG2E_C()
    if CAUSAL:
        n_end = tl.minimum(start_m + BLOCK_M, L)
    else:
        n_end = L
    if CQS:
        nr = tl.load(NRUNS + pid_m)
    else:
        nr = 1
    for i in range(0, nr):
        if CQS:
            rs = tl.load(RUN_S + pid_m * MAXR + i)
            re = tl.minimum(tl.load(RUN_E + pid_m * MAXR + i), n_end)
        else:
            rs = 0
            re = n_end
        for start_n in range(rs, re, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            col_ok = offs_n < L
            if CQS:
                clear = _tile_clear(BLK_OR, start_m, start_n, NUM_BLK, BLOCK_M, BLOCK_N, CQS_BLK)
            else:
                clear = True
            k = tl.load(K + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_kl + offs_d[None, :] * stride_kd, mask=col_ok[:, None], other=0.0)
            v = tl.load(V + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vl + offs_d[None, :] * stride_vd, mask=col_ok[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * qk_scale
            keep = row_ok[:, None] & col_ok[None, :]
            if CAUSAL:
                keep = keep & (offs_n[None, :] <= offs_m[:, None])
            if CQS:
                if not clear:
                    col_bits = tl.load(BITS + offs_n, mask=col_ok, other=0)
                    keep = keep & ((row_bits[:, None] & col_bits[None, :]) == 0)
            p = tl.exp2(qk - lse2[:, None])
            p = tl.where(keep & lse_fin[:, None], p, 0.0)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k)
    dq = dq * sm_scale
    tl.store(DQ + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od, dq.to(DQ.dtype.element_ty), mask=row_ok[:, None])


def cqs_attention_backward(dout, q, k, v, lse, group_bits=None, *, causal: bool, scale: float | None = None,
                           delta=None, out=None, blk_or=None, blk_and=None, blk_size: int = 64,
                           block_m: int = 64, block_n: int = 64, num_warps: int = 4, num_stages: int = 2):
    """
    Backward of one CQS subproblem in the global-lse form (the engine's contract):
    dout/q/k/v/out [B, L, H, D], lse [B, H, L] (natural log, GLOBAL), returns
    (dq, dk, dv) in the input dtype, [B, L, H, D]. `delta` = rowsum(dout*out)
    [B, H, L] fp32 may be passed (the engine computes it once globally);
    otherwise it is computed here from `out`.
    """
    B, L, H, D = q.shape
    if scale is None:
        scale = D ** -0.5
    dev = q.device
    if delta is None:
        assert out is not None, "need `out` or `delta`"
        delta = (dout.float() * out.float()).sum(-1).transpose(1, 2).contiguous()     # [B, H, L]
    delta = delta.to(dev, torch.float32)
    lse = lse.to(dev, torch.float32)
    if lse.stride() != delta.stride():
        delta = delta.contiguous(); lse = lse.contiguous()
    cqs = group_bits is not None
    if cqs:
        bits = group_bits.to(dev, torch.int64).contiguous()
        if blk_or is None or blk_and is None:
            blk_or, blk_and = _summaries(bits, blk_size)
        blk_or = blk_or.to(dev, torch.int64).contiguous(); blk_and = blk_and.to(dev, torch.int64).contiguous()
        num_blk = blk_or.numel()
        assert blk_size == 64 and block_m % 64 == 0 and block_n % 64 == 0
    else:
        bits = torch.zeros(1, dtype=torch.int64, device=dev); blk_or = bits; blk_and = bits; num_blk = 0
    dq = torch.empty_like(q); dk = torch.empty_like(k); dv = torch.empty_like(v)
    if cqs:
        q_rs, q_re, q_nr, q_maxr = live_runs(blk_and, L, block_m, block_n, blk_size)   # per query block: key runs
        k_rs, k_re, k_nr, k_maxr = live_runs(blk_and, L, block_n, block_m, blk_size)   # per key block: query runs
    else:
        z = torch.zeros(1, dtype=torch.int32, device=dev)
        q_rs = q_re = q_nr = k_rs = k_re = k_nr = z; q_maxr = k_maxr = 1
    common = dict(CAUSAL=bool(causal), CQS=cqs, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=D, CQS_BLK=64,
                  num_warps=num_warps, num_stages=num_stages)
    st = lambda t: (t.stride(0), t.stride(1), t.stride(2), t.stride(3))
    _cqs_attn_bwd_dkdv_kernel[(triton.cdiv(L, block_n), B * H)](
        q, k, v, dout, lse, delta, dk, dv, bits, blk_or, blk_and, k_rs, k_re, k_nr, k_maxr, float(scale), L, num_blk, H,
        *st(q), *st(k), *st(v), *st(dout), lse.stride(0), lse.stride(1), lse.stride(2), *st(dk), **common)
    _cqs_attn_bwd_dq_kernel[(triton.cdiv(L, block_m), B * H)](
        q, k, v, dout, lse, delta, dq, bits, blk_or, blk_and, q_rs, q_re, q_nr, q_maxr, float(scale), L, num_blk, H,
        *st(q), *st(k), *st(v), *st(dout), lse.stride(0), lse.stride(1), lse.stride(2), *st(dq), **common)
    return dq, dk, dv
