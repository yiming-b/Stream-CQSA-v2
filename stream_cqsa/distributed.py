"""
Multi-device Stream-CQSA forward.

Design. The c**itr subproblems are independent and recompose through a
max-shifted merge of per-token (m, l, acc) statistics, and that merge is
associative and commutative. So:

  1. Every rank builds the identical task list and takes a round-robin shard
     of it (`task_subset`), then runs the unmodified single-device engine on
     its shard. No communication during compute.
  2. Each rank ends with a normalized partial output out_r and a partial
     log-sum-exp lse_r (-inf on tokens its shard never touched). These are
     exactly a "subproblem result" in the engine's own merge_lse() sense, with
     local l = 1. The cross-rank merge is therefore the same formula:

         m'   = max_r lse_r                      (all_reduce MAX)
         w_r  = exp(lse_r - m')                  (0 where lse_r = -inf)
         acc  = sum_r  w_r * out_r               (all_reduce SUM)
         l    = sum_r  w_r                       (all_reduce SUM)
         out  = acc / l          lse = m' + log l

     Both exponents are <= 0, so nothing overflows -- the same reason the
     single-device merge is safe.

  3. The reduction is chunked over the token axis so the fp32 [B,H,N,D]
     accumulator never has to sit on one device whole. With the recommended
     host-accumulator configuration out_r lives in host memory; each chunk is
     staged to the device, reduced, and returned.

Only standard NCCL collectives. The result is bit-for-bit the same function as
the single-device path up to fp32 summation order.

Requires the `task_subset` kwarg added to stream_cqsa_forward in next/pkg.
"""
from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.distributed as dist

from .stable_stream import stream_cqsa_forward, stream_cqsa_backward, build_tasks_cached


def _shard(n_tasks: int, rank: int, world: int) -> list[int]:
    return list(range(rank, n_tasks, world))


def dist_stream_cqsa_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
    itr: int, causal: bool = False, group=None,
    chunk_tokens: int = 1 << 20, output: str = "replicated",
    **engine_kw: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    q/k/v: [B, H, N, D], identical on every rank (host or device resident).
    Returns (out [B,H,N,D] fp32, info). With output="replicated" every rank
    holds the full output; with "sharded" rank r holds tokens
    [r*N/world, (r+1)*N/world) and the rest are zero.
    """
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("torch.distributed must be initialised (torchrun)")
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    dev = torch.device("cuda", torch.cuda.current_device())
    B, H, N, D = q.shape
    itr = int(itr)
    c = int(engine_kw.get("c", 7))
    interest_set = tuple(engine_kw.get("interest_set", (0, 1, 3)))

    # Identical task list on every rank; shard it round-robin so the largest
    # subproblems (there is some variation from uneven chunks) spread out.
    tasks = build_tasks_cached(
        N, itr, B=B, H=H, D=D, itemsize=q.element_size(),
        sorted_gather=bool(engine_kw.get("sorted_gather", True)),
        pin=False, c=c, interest_set=interest_set, seg_align=0)
    mine = _shard(len(tasks), rank, world)

    t0 = time.perf_counter()
    out_r, info = stream_cqsa_forward(
        q, k, v, itr=itr, causal=causal, task_subset=mine, **engine_kw)
    lse_r = info["lse"]                                   # [B, H, N], -inf where untouched
    torch.cuda.synchronize(dev)
    t_local = time.perf_counter() - t0

    # ---- cross-rank merge, chunked over tokens ---------------------------
    t1 = time.perf_counter()
    out = torch.zeros_like(out_r) if output == "replicated" else torch.zeros_like(out_r)
    lse = torch.empty_like(lse_r)
    lo_mine = (N * rank) // world
    hi_mine = (N * (rank + 1)) // world
    for s in range(0, N, chunk_tokens):
        e = min(N, s + chunk_tokens)
        lse_c = lse_r[:, :, s:e].to(dev, non_blocking=True).contiguous()   # [B,H,n]
        m = lse_c.clone()
        dist.all_reduce(m, op=dist.ReduceOp.MAX, group=group)
        w = torch.where(torch.isfinite(lse_c), torch.exp(lse_c - m), torch.zeros_like(lse_c))
        # NCCL needs contiguous tensors; a sliced [B,H,n,D] view is not, and
        # elementwise ops preserve the view's strides.
        acc = (out_r[:, :, s:e, :].to(dev, non_blocking=True) * w.unsqueeze(-1)).contiguous()
        acc = torch.nan_to_num(acc, nan=0.0)               # untouched rows are 0/0 locally
        l = w.contiguous().clone()
        dist.all_reduce(acc, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(l, op=dist.ReduceOp.SUM, group=group)
        res = acc / l.clamp_min(1e-30).unsqueeze(-1)
        lse_g = torch.where(torch.isfinite(m), m + torch.log(l.clamp_min(1e-30)),
                            torch.full_like(m, float("-inf")))
        if output == "replicated":
            out[:, :, s:e, :].copy_(res.to(out.device))
        else:
            a, b = max(s, lo_mine), min(e, hi_mine)
            if b > a:
                out[:, :, a:b, :].copy_(res[:, :, a - s:b - s, :].to(out.device))
        lse[:, :, s:e].copy_(lse_g.to(lse.device))
        del lse_c, m, w, acc, l, res, lse_g
    torch.cuda.synchronize(dev)
    t_merge = time.perf_counter() - t1

    info = dict(info)
    info.update(rank=rank, world=world, tasks_mine=mine, n_tasks=len(tasks),
                t_local_s=t_local, t_merge_s=t_merge, lse=lse, output=output)
    return out, info


def dist_stream_cqsa_backward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    dout: torch.Tensor, out: torch.Tensor, lse: torch.Tensor, *,
    itr: int, causal: bool = False, group=None,
    chunk_tokens: int = 1 << 20, **engine_kw: Any,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, Any]]:
    """
    Multi-device backward. q/k/v/dout/out are [B, H, N, D] and lse is [B, H, N]
    (the GLOBAL lse from dist_stream_cqsa_forward), identical on every rank.

    Each rank runs the unmodified single-device backward on its round-robin
    shard of the c**itr subproblems (`task_subset`) and obtains fp32 partial
    dq/dk/dv restricted to that shard's pair set. The gradient of each token
    is a plain sum over the subproblems that touch it, so the cross-rank step
    is one all_reduce(SUM) per gradient, chunked over tokens. Every rank ends
    with the full fp32 gradients (replicated).
    """
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("torch.distributed must be initialised (torchrun)")
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    dev = torch.device("cuda", torch.cuda.current_device())
    B, H, N, D = q.shape
    itr = int(itr)
    c = int(engine_kw.get("c", 7))
    interest_set = tuple(engine_kw.get("interest_set", (0, 1, 3)))
    tasks = build_tasks_cached(
        N, itr, B=B, H=H, D=D, itemsize=q.element_size(),
        sorted_gather=bool(engine_kw.get("sorted_gather", True)),
        pin=False, c=c, interest_set=interest_set, seg_align=0)
    mine = _shard(len(tasks), rank, world)

    t0 = time.perf_counter()
    grads = stream_cqsa_backward(
        q, k, v, dout, out, lse, itr=itr, causal=causal, task_subset=mine, **engine_kw)
    torch.cuda.synchronize(dev)
    t_local = time.perf_counter() - t0

    t1 = time.perf_counter()
    outs = []
    for g in grads:                                       # [B, H, N, D] fp32 views
        full = torch.empty_like(g)
        for s in range(0, N, chunk_tokens):
            e = min(N, s + chunk_tokens)
            buf = g[:, :, s:e, :].to(dev, non_blocking=True).contiguous()
            dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=group)
            full[:, :, s:e, :].copy_(buf.to(full.device))
            del buf
        outs.append(full)
    torch.cuda.synchronize(dev)
    t_merge = time.perf_counter() - t1
    info = dict(rank=rank, world=world, tasks_mine=mine, n_tasks=len(tasks),
                t_local_s=t_local, t_merge_s=t_merge)
    return (outs[0], outs[1], outs[2]), info
