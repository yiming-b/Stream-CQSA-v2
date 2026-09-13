"""
Numerically stable Stream-CQSA forward path, with an event-driven scheduler.

Why this module exists
----------------------
The shipped path (``attention_kernel/FA.py`` + the ``index_add_`` merge in
``streamed_fwd_bwd.py``) recomposes subproblems in *unshifted* coordinates::

    den_i = exp(lse_i)                      # FA.py
    num_i = out_i * den_i
    num_global.index_add_(...); den_global.index_add_(...)

``exp`` overflows fp32 above ``lse ~= 88.7``. The overflow is then swallowed by
``nan_to_num(..., posinf=0.0)``, so an overflowing subproblem contributes
*exactly zero* instead of raising. Measured on A100 with D=64, L=878
(``experiments/kernel/audit_cuda_local.py``):

    input_std   lse_max   den == 0
        1.0         7.1      0/878
        3.0        48.9      0/878
        6.0       195.6    740/878
        9.0       440.0    878/878

That is silent wrong output, not a crash. Scores of that size are ordinary for
un-normalised activations, so this is a real correctness bug rather than a
theoretical one.

This module keeps every contribution in *max-shifted* coordinates and never
materialises ``exp(lse)``. It uses the compatibility contract from the design
note: a FlashAttention result ``(out_i, lse_i)`` is merged as

    local_m = lse_i,  local_l = 1,  local_acc = out_i

which is exact, because ``exp(lse_i) * out_i`` is the unshifted numerator and
``exp(lse_i) * 1`` the unshifted denominator.

Causal attention
----------------
Causality must be decided on *global* token ids. Rather than teach the kernel
about token ids, this module gathers each subsequence in ascending chunk order
(``sorted_gather=True``), which makes local order identical to global order at
every decomposition level. A plain local lower-triangular mask is then exactly
the global causal mask, so the stock FlashAttention causal path is reused
unchanged. See ``tests/test_reference.py::test_sorted_gather_makes_local_triangle_exact``.
"""

from __future__ import annotations

import os as _os

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

import torch

from .reference import all_paths, group_bits_for_path, segment_block_base
from .progress import Progress, verbose_enabled, describe_call, expected_seconds
import sys

__all__ = [
    "StableAccumulator",
    "SubproblemTask",
    "TraceRecorder",
    "build_tasks",
    "build_tasks_cached",
    "clear_task_cache",
    "estimate_task_bytes",
    "choose_parallelism",
    "plan_decomposition",
    "estimate_peak_bytes",
    "estimate_monolithic_bytes",
    "local_stats_flash",
    "local_stats_torch",
    "stream_cqsa_forward",
    "stream_cqsa_backward",
    "DeviceChunkPool",
]

_GIB = float(1 << 30)

# Torch defaults to one intra-op thread per core (80 on a Della node). For the
# merge's medium-sized elementwise/gather work that is catastrophic: measured at
# N=131072, L=56175 a single CPU merge costs 3233 ms at 80 threads and 140 ms at
# 8 -- a 23x difference, and throughput actually *falls* past 16 threads because
# OpenMP fork/join and cache contention dominate. This is the third place in the
# project where the 80-thread default cost more than an order of magnitude (the
# test suite and cqs_block_summaries were the others).
_CPU_MERGE_THREADS_DEFAULT = 8

# Run the host scatter on a worker thread so it overlaps device work?
# Measured: NO. The stage total collapses (1685 -> 15 ms at N=262144) because
# the work moves off the issue loop, but wall time gets *worse* -- 16.2% at
# N=65536, 4.0% at N=262144 (medians of 5, after warmup). The host side is
# memory-bandwidth-bound, so overlapping the scatter with the gather makes two
# bandwidth-bound streams contend rather than hiding either, and the thread
# handoff adds latency on top. Default off; the toggle stays so the claim can be
# re-checked on hardware with more host bandwidth, where it may flip.
# next/: default ON. The docstring of flush() below measured the scatter at
# 1685 ms against 5115 ms of compute at N=262144, "entirely hideable", yet it
# shipped off. "0" restores the shipped behaviour for A/B.
_SCATTER_ASYNC = os.environ.get("CQSA_SCATTER_ASYNC", "0") != "0"
# next/: forward host-accumulator merge on a worker thread, with a
# non-blocking D2H into pinned slots. Default ON here; "0" restores the
# shipped blocking path for A/B measurement.
_FWD_MERGE_ASYNC = os.environ.get("CQSA_FWD_MERGE_ASYNC", "1") != "0"

# Drop O from the per-subproblem transfers by precomputing rowsum(dO*O) once
# and gathering it like the lse (see _global_dpsum). Implemented, exact, and
# measured -- and it LOSES, so it is off by default.
#
#   N=262144, itr=2, streamed:  O-path 3985 ms  vs  dpsum 9630 ms
#   N= 65536, itr=2, streamed:  O-path 1404 ms  vs  dpsum 2488 ms
#   device-resident: within 1% either way.
#
# The stage totals predicted a ~4.6% win (O is 1/5 of the gather and of the
# H2D). They were wrong because they attribute *overlapped* time: O rides the
# same pinned staging pipeline as q/k/v/dO and hides behind compute, so its
# marginal wall cost is far below its stage total. Removing it saves little,
# while computing the global dP_sum has to read dO and O once from pageable
# host memory -- 124 ms of unoverlapped pageable H2D, before the loop starts.
# A stage total is an upper bound on what removing that stage can save, never
# an estimate of it.
#
# Kept behind the flag rather than deleted: it is validated exact (gradients
# match float64 identically on both paths) and would plausibly win where the
# host tensors are already pinned, or on a host with more memory bandwidth.
_BWD_USE_DPSUM = os.environ.get("CQSA_BWD_DPSUM", "0") != "0"


class _cpu_threads:
    """Scoped torch intra-op thread count; restores the previous value."""

    def __init__(self, n: int | None):
        self.n = n
        self.prev = None

    def __enter__(self):
        if self.n:
            self.prev = torch.get_num_threads()
            torch.set_num_threads(int(self.n))
        return self

    def __exit__(self, *exc):
        if self.prev is not None:
            torch.set_num_threads(self.prev)
        return False


# ---------------------------------------------------------------------------
# Global accumulator
# ---------------------------------------------------------------------------


class StableAccumulator:
    """
    Max-shifted global accumulator.

    Invariant: the true unshifted numerator for token ``n`` is
    ``exp(m[n]) * acc[n]`` and the denominator ``exp(m[n]) * l[n]``. Only the
    ratio is ever needed, so ``exp(m)`` is never formed.

    Storage is ``[B, N, H, D]`` -- token-major, matching the FlashAttention I/O
    layout. The scheduler therefore never transposes a per-subproblem tensor:
    gathers and merges are plain ``index_*`` calls on dim 1. Keeping the
    accumulator in ``[B, H, N, D]`` instead forced a strided
    ``transpose(1,2).contiguous()`` copy of Q/K/V and of the output for every
    subproblem, which the scheduler trace showed dominating actual attention
    compute.
    """

    def __init__(self, B: int, H: int, N: int, D: int, *, device, dtype=torch.float32):
        self.acc = torch.zeros((B, N, H, D), device=device, dtype=dtype)
        self.l = torch.zeros((B, N, H), device=device, dtype=dtype)
        self.m = torch.full((B, N, H), float("-inf"), device=device, dtype=dtype)
        # Set only if a merge failed after it had begun writing. The OOM
        # recovery retries a failed subproblem, which is sound exactly while no
        # merge is half applied, so the recovery consults this before retrying.
        self.dirty = False

    @property
    def device(self):
        return self.acc.device

    def merge_lse(self, idx: torch.Tensor, out_i: torch.Tensor, lse_i: torch.Tensor) -> None:
        """
        Merge a FlashAttention-style subproblem result.

        idx   : [L] global token ids (distinct within a subproblem)
        out_i : [B, L, H, D] normalised local output (FA layout)
        lse_i : [B, L, H] local log-sum-exp
        """
        self._merge(idx, out_i, torch.ones_like(lse_i), lse_i)

    def merge_stats(
        self,
        idx: torch.Tensor,
        local_acc: torch.Tensor,
        local_l: torch.Tensor,
        local_m: torch.Tensor,
    ) -> None:
        """Merge native shifted statistics, all in ``[B, L, H, *]`` layout."""
        self._merge(idx, local_acc, local_l, local_m)

    def _merge(self, idx, local_acc, local_l, local_m) -> None:
        idx = idx.to(self.acc.device, dtype=torch.long, non_blocking=True)
        local_acc = local_acc.to(self.acc.dtype)
        local_l = local_l.to(self.l.dtype)
        local_m = local_m.to(self.m.dtype)

        # A local row with no retained keys carries no information, but the two
        # inner kernels signal it differently: the torch fallback returns -inf,
        # while FlashAttention returns **+inf**. Taking max(old_m, +inf) would
        # make old_scale = exp(old_m - inf) = 0 and silently ERASE the token's
        # real contribution from earlier subproblems -- observed as 586 of 2048
        # tokens (chunks 0 and 1) going empty under causal masking. Normalise
        # any non-finite lse to -inf first, so "no keys" always means "no
        # contribution" regardless of which kernel produced it.
        local_valid = torch.isfinite(local_m)
        neg_inf = torch.full_like(local_m, float("-inf"))
        local_m = torch.where(local_valid, local_m, neg_inf)

        old_m = self.m.index_select(1, idx)
        new_m = torch.maximum(old_m, local_m)

        # A token not yet touched has old_m = -inf. Both cases must scale to 0,
        # and -inf - -inf would be NaN.
        old_scale = torch.where(
            torch.isfinite(old_m), torch.exp(old_m - new_m), torch.zeros_like(old_m)
        )
        new_scale = torch.where(
            local_valid, torch.exp(local_m - new_m), torch.zeros_like(local_m)
        )

        # Mask the invalid rows' payload rather than relying on multiplication:
        # local_acc may hold inf/NaN there, and 0 * inf is NaN.
        zero_acc = torch.zeros_like(local_acc)
        contrib_acc = torch.where(
            local_valid.unsqueeze(-1), local_acc * new_scale.unsqueeze(-1), zero_acc
        )
        contrib_l = torch.where(local_valid, local_l * new_scale, torch.zeros_like(local_l))

        acc = self.acc.index_select(1, idx) * old_scale.unsqueeze(-1) + contrib_acc
        lsum = self.l.index_select(1, idx) * old_scale + contrib_l

        # Commit. Every allocation this method needs has already happened
        # above, and index_copy_ allocates nothing, so an out-of-memory failure
        # lands before the first write and leaves the accumulator untouched --
        # which is what makes retrying a failed subproblem sound. Should a write
        # nonetheless fail partway, the accumulator holds a half-merged
        # subproblem that no retry can undo, so record it and let the caller
        # fail loudly rather than recover into a wrong answer.
        self.acc.index_copy_(1, idx, acc)
        try:
            self.l.index_copy_(1, idx, lsum)
            self.m.index_copy_(1, idx, new_m)
        except BaseException:
            self.dirty = True
            raise

    def lse(self) -> torch.Tensor:
        """
        Global log-sum-exp per token, ``[B, H, N]``.

        ``m + log(l)`` -- finite for any score magnitude, unlike ``exp(lse)``.
        This is what the backward needs from every subproblem, and computing it
        here means the backward never has to recompute or invert anything.
        """
        lse = self.m + torch.log(self.l.clamp_min(1e-30))
        return torch.where(torch.isfinite(self.m), lse,
                           torch.full_like(lse, float("-inf"))).transpose(1, 2)

    def output(self) -> torch.Tensor:
        """Final attention output in ``[B, H, N, D]``."""
        out = self.acc / self.l.clamp_min(1e-30).unsqueeze(-1)
        return out.transpose(1, 2)

    def untouched_tokens(self) -> int:
        """Tokens that received no contribution -- a scheduling/coverage bug."""
        return int((~torch.isfinite(self.m)).all(dim=2).sum().item())


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@dataclass
class SubproblemTask:
    path_idx: int
    path: tuple[int, ...]
    itr: int
    token_ids: torch.Tensor          # CPU, pinned when possible
    group_bits: torch.Tensor         # CPU int64
    local_size: int
    estimated_mem_gib: float
    status: str = "queued"
    stream_id: int = -1
    extra: dict[str, Any] = field(default_factory=dict)


def estimate_task_bytes(local_size: int, B: int, H: int, D: int, itemsize: int) -> int:
    """
    Working-set estimate for one subproblem: Q/K/V plus the fp32 output and lse,
    plus FlashAttention's own scratch. Deliberately generous -- the scheduler
    uses it to stay under a memory budget, so overestimating is the safe error.
    """
    qkv = 3 * B * local_size * H * D * itemsize
    out = B * local_size * H * D * itemsize
    out_f32 = B * H * local_size * D * 4
    lse = B * H * local_size * 4
    scratch = int(0.25 * (qkv + out))
    return int(qkv + out + out_f32 + lse + scratch)


_TASK_CACHE: "dict[tuple, list[SubproblemTask]]" = {}
_TASK_CACHE_MAX = 8


def build_tasks_cached(*args, **kwargs) -> list[SubproblemTask]:
    """
    Memoised ``build_tasks``.

    The task list is a pure function of the *shape* (N, itr, c, sorted_gather)
    and the memory estimate inputs -- never of Q/K/V. Rebuilding it per call
    cost 4.8 s at N=131072 (35 s with pinning), which dwarfed the 0.8 s of
    actual attention. Real workloads call attention repeatedly at one shape, so
    this is the common case, not a benchmark artefact.

    Tasks carry mutable status/stream_id fields, so they are reset on hand-out.
    """
    key = (args, tuple(sorted(kwargs.items())))
    tasks = _TASK_CACHE.get(key)
    if tasks is None:
        tasks = build_tasks(*args, **kwargs)
        if len(_TASK_CACHE) >= _TASK_CACHE_MAX:
            _TASK_CACHE.pop(next(iter(_TASK_CACHE)))
        _TASK_CACHE[key] = tasks
    for t in tasks:
        t.status = "queued"
        t.stream_id = -1
    return tasks


def clear_task_cache() -> None:
    _TASK_CACHE.clear()


def is_oom(exc: BaseException) -> bool:
    """
    Whether an exception is an out-of-memory failure.

    Matching on the message alone misses an OutOfMemoryError whose text does not
    contain the usual phrase, and matching on the type alone misses the plain
    RuntimeError that some allocation paths and inner kernels raise. The
    recovery has to see both, since anything it fails to recognise is re-raised
    to the caller instead of being recovered from.
    """
    return isinstance(exc, torch.cuda.OutOfMemoryError) or \
        "out of memory" in str(exc).lower()


def max_depth_for(N: int, c: int = 7) -> int:
    """
    Deepest decomposition the sequence structurally admits.

    A subsequence at depth k holds roughly N/c**k tokens, so the recursion stops
    when a chunk can no longer hold a token. This is the only principled bound on
    depth: everything shallower is a policy choice, and nothing deeper is
    meaningful.
    """
    n, ci = int(N), int(c)
    if n <= 0 or ci <= 1:
        return 0
    k = 0
    while ci ** (k + 1) <= n:
        k += 1
    return k


def refine_task(
    task: "SubproblemTask",
    N: int,
    *,
    B: int,
    H: int,
    D: int,
    itemsize: int,
    sorted_gather: bool = True,
    pin: bool = True,
    c: int = 7,
    interest_set: Sequence[int] = (0, 1, 3),
    seg_align: int = 0,
) -> list["SubproblemTask"]:
    """
    Decompose one subproblem a further level, returning its ``c`` children.

    A child's path is the parent's path with one index appended, and
    ``group_bits_for_path`` derives its mask from the whole path. The children
    therefore inherit the parent's masking as well as applying their own, which
    is what makes their retained pairs partition exactly the parent's retained
    pairs. Replacing a task by its children mid-run is consequently sound: the
    merge is additive and order-independent, so provided the parent contributed
    nothing before it failed, the result is unchanged.

    Refining a single task rather than the whole problem is also what produces
    the hybrid schedules of the paper: branches that fit stay coarse, and only
    those that failed are split.
    """
    kids: list[SubproblemTask] = []
    align = int(seg_align) if seg_align else 1
    for j in range(int(c)):
        path = tuple(task.path) + (int(j),)
        ids_np, bits_np = group_bits_for_path(
            N, path, c, interest_set, sorted_gather=sorted_gather, align=align)
        L = int(ids_np.shape[0])
        if L == 0:
            continue
        ids = torch.from_numpy(ids_np.astype(np.int64, copy=False))
        bits = torch.from_numpy(bits_np.astype(np.int64, copy=False))
        from .interface import CQS_BLK_SIZE, cqs_block_summaries
        blk_or, blk_and = cqs_block_summaries(bits, CQS_BLK_SIZE)
        seg_base = None
        if seg_align:
            seg_base = torch.from_numpy(
                segment_block_base(N, path, c=c, interest_set=interest_set,
                                   align=int(seg_align)))
        if pin and torch.cuda.is_available():
            ids = ids.pin_memory()
            bits = bits.pin_memory()
            blk_or = blk_or.pin_memory()
            blk_and = blk_and.pin_memory()
            if seg_base is not None:
                seg_base = seg_base.pin_memory()
        kids.append(
            SubproblemTask(
                path_idx=task.path_idx * int(c) + j,
                path=path,
                itr=int(task.itr) + 1,
                token_ids=ids,
                group_bits=bits,
                local_size=L,
                estimated_mem_gib=estimate_task_bytes(L, B, H, D, itemsize) / _GIB,
                extra={"blk_or": blk_or, "blk_and": blk_and, "seg_base": seg_base},
            )
        )
    return kids


def build_tasks(
    N: int,
    itr: int,
    *,
    B: int,
    H: int,
    D: int,
    itemsize: int,
    sorted_gather: bool = True,
    pin: bool = True,
    c: int = 7,
    interest_set: Sequence[int] = (0, 1, 3),
    seg_align: int = 0,
) -> list[SubproblemTask]:
    """
    Materialise every leaf subproblem's token ids and CQS group bits.

    ``(c, interest_set)`` must be a perfect (Singer) difference set: every
    non-zero residue mod ``c`` occurs exactly once as a difference of two
    members, which is what makes the off-diagonal chunk pairs covered exactly
    once. ``c = l^2 - l + 1`` for ``l = len(interest_set)``.
    """
    tasks: list[SubproblemTask] = []
    for path_idx, path in enumerate(all_paths(itr, c)):
        align = int(seg_align) if seg_align else 1
        ids_np, bits_np = group_bits_for_path(
            N, path, c, interest_set, sorted_gather=sorted_gather, align=align)
        L = int(ids_np.shape[0])
        if L == 0:
            continue
        ids = torch.from_numpy(ids_np.astype(np.int64, copy=False))
        bits = torch.from_numpy(bits_np.astype(np.int64, copy=False))
        # Tile summaries are shape-only, so build them once here rather than on
        # every attention call.
        from .interface import CQS_BLK_SIZE, cqs_block_summaries
        blk_or, blk_and = cqs_block_summaries(bits, CQS_BLK_SIZE)
        seg_base = None
        if seg_align:
            seg_base = torch.from_numpy(
                segment_block_base(N, path, c=c, interest_set=interest_set,
                                   align=int(seg_align)))
        if pin and torch.cuda.is_available():
            ids = ids.pin_memory()
            bits = bits.pin_memory()
            blk_or = blk_or.pin_memory()
            blk_and = blk_and.pin_memory()
            if seg_base is not None:
                seg_base = seg_base.pin_memory()
        tasks.append(
            SubproblemTask(
                path_idx=path_idx,
                path=tuple(int(x) for x in path),
                itr=int(itr),
                token_ids=ids,
                group_bits=bits,
                local_size=L,
                estimated_mem_gib=estimate_task_bytes(L, B, H, D, itemsize) / _GIB,
                extra={"blk_or": blk_or, "blk_and": blk_and, "seg_base": seg_base},
            )
        )
    return tasks


def choose_parallelism(
    tasks: Sequence[SubproblemTask],
    *,
    device,
    safety: float = 0.75,
    reserve_gib: float = 0.0,
    max_streams: int = 8,
    B: int = 1,
    H: int = 1,
    D: int = 64,
    block_m: int = 128,
    host_streaming: bool = False,
    itemsize: int = 2,
    pinned_budget_gib: float = 2.0,
) -> int:
    """
    Pick how many subproblems may be in flight, bounded by *both* memory and
    SM occupancy.

    Memory alone is the wrong bound. A subsequence of length L issues about
    ``ceil(L/block_m) * B * H`` thread blocks; once that exceeds the SM count the
    GPU is already saturated and extra streams cannot overlap any *compute* --
    they only overlap gather/H2D with compute. Measured on A100 (108 SMs),
    N=32768, H=8: one subproblem issues 880 blocks (8.1x the SMs), and going
    from 2 to 8 streams changed wall time by under 1%:

        n_par        1        2        4        8
        N=32768  160.0ms  140.4ms  139.0ms  138.9ms
        N=65536  585.4ms  544.8ms  543.7ms  543.2ms

    So when a single task already saturates the device, cap at 2 -- enough to
    keep one task's transfer/gather under another's compute, without holding
    extra working sets or paying per-stream bookkeeping.
    """
    if not tasks:
        return 1

    free_b = effective_free_bytes(device)
    budget = max(0.0, free_b / _GIB - reserve_gib) * float(safety)
    worst = max(t.estimated_mem_gib for t in tasks)
    mem_cap = max_streams if worst <= 0 else max(1, math.floor(budget / worst))

    largest_L = max(t.local_size for t in tasks)
    blocks = max(1, -(-largest_L // int(block_m))) * max(1, int(B)) * max(1, int(H))
    try:
        n_sm = torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        n_sm = 108
    # >= 2x the SMs means a single task fills the machine, so extra streams
    # cannot overlap COMPUTE. That is the whole argument for the cap -- and it
    # only applies when there is nothing else to overlap.
    occupancy_cap = 2 if blocks >= 2 * n_sm else max_streams

    if host_streaming:
        # With host-resident inputs each task also carries an H2D transfer, and
        # transfers overlap compute regardless of SM saturation, so the SM cap
        # does not apply. Device peak is flat in n_par (831 -> 835 MiB at
        # N=131072; +16 MiB at N=1048576) because per-task input buffers are
        # freed at kernel launch and recycled.
        #
        # The binding cost is instead PINNED HOST memory: one staging set per
        # slot, sized for the largest task, and pinned pages are unswappable.
        # At N=524288 that is 0.64 GiB per slot -- 4.5 GiB at n_par=7 -- while
        # the speed gain shrinks with N (32% at N=131072, 11% at N=524288). So
        # bound the total rather than the count.
        per_slot_pinned = 3 * largest_L * B * H * (D * itemsize)
        pinned_cap = max(1, int(pinned_budget_gib * _GIB // max(1, per_slot_pinned)))
        occupancy_cap = min(max_streams, pinned_cap)

    return int(max(1, min(max_streams, len(tasks), mem_cap, occupancy_cap)))


def per_el_floor(itemsize: int, stream_from_host: bool,
                 accumulate_on_gpu: bool) -> int:
    """Device-resident bytes per element that no decomposition depth reduces."""
    per_el = 4
    if not stream_from_host:
        per_el += 3 * itemsize
    if accumulate_on_gpu:
        per_el += 4
    return per_el


def estimate_peak_bytes(N: int, itr: int, *, B: int, H: int, D: int, itemsize: int,
                       n_par: int = 2, c: int = 7, l: int = 3,
                       stream_from_host: bool = False,
                       accumulate_on_gpu: bool = True) -> int:
    """
    Predicted peak device memory for a Stream-CQSA forward.

    Two parts, and only the second shrinks with ``itr``:

      floor     = (3*itemsize + 4 + 4) * N*H*D
                  Q/K/V, the fp32 accumulator, and the fp32 output. All
                  O(N*H*D) and completely independent of ``itr`` -- but only
                  those terms that are actually DEVICE-resident belong in a
                  device-memory estimate. ``stream_from_host`` moves Q/K/V to
                  the host and ``accumulate_on_gpu=False`` moves the
                  accumulator, and each must then be dropped from the floor.

                  Counting them regardless is what made the planner refuse to
                  decompose at N=16M: it reported an O(N*H*D) floor of 112 GiB
                  against a 59 GiB budget, concluded nothing fit, and fell back
                  to its deepest depth. The configuration it was estimating went
                  on to run in 32 GiB, and at a shallower depth would have run
                  1.6x faster at the same peak.
      transient = n_par * (3*itemsize + 4) * L*H*D,  L = N*(l/c)^itr
                  the gathered q_i/k_i/v_i and out_i in flight.

    Validated against measurement: predicted/measured peak is 1.36x at itr=1 and
    1.02x at itr=2, flat across N = 32768 .. 131072. The ratio does *not* improve
    with N because both terms are linear in N.
    """
    floor = per_el_floor(itemsize, stream_from_host, accumulate_on_gpu) * N * H * D
    L = N
    for _ in range(int(itr)):
        L = int(L * l / c)
    transient = int(n_par) * (3 * itemsize + 4) * L * H * D
    return int(floor + transient)


def effective_free_bytes(device) -> int:
    """
    Free device memory as the allocator will actually grant it: the driver's
    free bytes, capped by torch.cuda.set_per_process_memory_fraction when one
    is set (a memory cap used to simulate a smaller device).
    """
    free_b, total_b = torch.cuda.mem_get_info(device)
    # The fraction is kept per device INDEX; an index-less torch.device("cuda")
    # must be resolved to the current device or the lookup silently returns 1.0.
    idx = device.index if isinstance(device, torch.device) else device
    if idx is None or isinstance(idx, str):
        idx = torch.cuda.current_device()
    try:
        frac = torch.cuda.get_per_process_memory_fraction(int(idx))
    except Exception:
        frac = 1.0
    if frac < 1.0:
        allowed = int(frac * total_b) - int(torch.cuda.memory_reserved(device))
        free_b = max(0, min(free_b, allowed))
    return int(free_b)


def estimate_peak_bytes_bwd(N: int, itr: int, *, B: int, H: int, D: int, itemsize: int,
                           n_par: int = 1, c: int = 7, l: int = 3,
                           stream_from_host: bool = False,
                           accumulate_on_gpu: bool = True) -> int:
    """
    Predicted peak device memory for a Stream-CQSA BACKWARD (dq/dk/dv of a
    forward already computed). Same shape as `estimate_peak_bytes` but with the
    backward's operands:

      floor     device-resident O(N.H.D) terms: q/k/v/dout/out (5*itemsize) unless
                streamed from the host; the three fp32 gradient accumulators
                (12 bytes) unless accumulated on the host; the fp32 lse.
      transient per in-flight subproblem: q_i/k_i/v_i/dout_i/out_i (5*itemsize),
                the kernel's dq/dk/dv (3*4), its dq accumulation scratch (4)
                and lse/delta (8 per row).
    """
    per_el = 0
    if not stream_from_host:
        per_el += 5 * itemsize
    if accumulate_on_gpu:
        per_el += 3 * 4
    floor = per_el * N * B * H * D + N * B * H * 4
    L = N
    for _ in range(int(itr)):
        L = int(L * l / c)
    transient = int(n_par) * ((5 * itemsize + 3 * 4 + 4) * L * B * H * D + 8 * L * B * H)
    return int(floor + transient)


def plan_decomposition_bwd(N: int, *, B: int, H: int, D: int, itemsize: int, device,
                           safety: float = 0.75, reserve_gib: float = 0.0, c: int = 7,
                           interest_set: Sequence[int] = (0, 1, 3), max_itr: int = 3,
                           stream_from_host: bool = False, accumulate_on_gpu: bool = True) -> tuple[int, str]:
    """
    Depth for the backward, chosen from free memory with the backward's own
    estimate -- independent of the depth the forward used. Returns (itr, reason);
    itr >= 1 always (the native backward has no monolithic path).
    """
    if torch.device(device).type != "cuda":
        return 1, "cpu device"
    free_b = effective_free_bytes(device)
    budget = int(max(0.0, free_b - reserve_gib * _GIB) * float(safety))
    l = len(interest_set)
    for itr in range(1, int(max_itr) + 1):
        need = estimate_peak_bytes_bwd(N, itr, B=B, H=H, D=D, itemsize=itemsize, c=c, l=l,
                                       stream_from_host=stream_from_host, accumulate_on_gpu=accumulate_on_gpu)
        if need <= budget:
            return itr, f"backward itr={itr} fits ({need / _GIB:.2f} GiB <= {budget / _GIB:.2f} GiB budget)"
    return max_itr, f"no backward depth fits the {budget / _GIB:.2f} GiB budget; using itr={max_itr}"


def estimate_monolithic_bytes(N: int, *, B: int, H: int, D: int, itemsize: int) -> int:
    """
    Predicted peak for a single monolithic FlashAttention call: Q/K/V, the
    output, the fp32 lse, plus a modest allowance for kernel workspace.
    """
    qkv = 3 * N * H * D * itemsize
    out = N * H * D * itemsize
    lse = N * H * 4
    return int((qkv + out + lse) * 1.15)


def plan_decomposition(
    N: int,
    *,
    B: int,
    H: int,
    D: int,
    itemsize: int,
    device,
    safety: float = 0.75,
    reserve_gib: float = 0.0,
    c: int = 7,
    interest_set: Sequence[int] = (0, 1, 3),
    max_itr: int = 3,
    stream_from_host: bool = False,
    accumulate_on_gpu: bool = True,
) -> tuple[int, str]:
    """
    Choose the decomposition depth from measured free memory.

    Returns ``(itr, reason)``; **itr = 0 means do not decompose** -- call
    FlashAttention directly.

    The policy exists because below the OOM boundary Stream-CQSA is worse on
    *both* axes: ~1.6x slower than a monolithic call, and it adds decomposition
    error a monolithic call does not have. Above the boundary the monolithic
    call is not an option at any price. So the only question worth automating is
    which side of that boundary we are on -- unlike the precision knob, whose
    cost curve is flat and whose risk is not inferable from the data.

    Deeper ``itr`` is a *feasibility* knob only: cost grows 2.2x (itr=2) and 8.2x
    (itr=3) while peak memory saturates at the floor, so this returns the
    smallest depth that fits.
    """
    if torch.device(device).type != "cuda":
        return 1, "cpu device: decomposition depth not memory-limited"

    free_b = effective_free_bytes(device)
    budget = int(max(0.0, free_b - reserve_gib * _GIB) * float(safety))

    mono = estimate_monolithic_bytes(N, B=B, H=H, D=D, itemsize=itemsize)
    if mono <= budget:
        return 0, (f"monolithic fits ({mono / _GIB:.2f} GiB <= {budget / _GIB:.2f} GiB budget): "
                   "not decomposing is both faster and more accurate")

    l = len(interest_set)
    for itr in range(1, int(max_itr) + 1):
        need = estimate_peak_bytes(N, itr, B=B, H=H, D=D, itemsize=itemsize, c=c, l=l,
                                   stream_from_host=stream_from_host,
                                   accumulate_on_gpu=accumulate_on_gpu)
        if need <= budget:
            return itr, (f"monolithic needs {mono / _GIB:.2f} GiB > budget "
                         f"{budget / _GIB:.2f} GiB; itr={itr} fits at {need / _GIB:.2f} GiB")

    floor = per_el_floor(itemsize, stream_from_host, accumulate_on_gpu) * N * H * D
    return max_itr, (
        f"nothing fits the {budget / _GIB:.2f} GiB budget; using itr={max_itr}. "
        f"The O(N*H*D) floor alone is {floor / _GIB:.2f} GiB and deeper "
        "decomposition cannot reduce it -- offload or a smaller N/H/D is required"
    )


class DeviceChunkPool:
    """
    A device-resident pool of token chunks, shared across subsequences.

    Every subsequence is a union of ``l`` whole chunks, and any two subsequences
    share exactly one chunk (the difference set's lambda=1 property). So when
    several are loaded at once, the naive scheme stores some chunks twice. This
    pool stores each chunk once and hands out slots.

    Reference counting, not a plain residency list: a chunk is evicted only when
    no in-flight subsequence still holds it. That is what removes the race where
    S_0={0,1,3} releasing chunk 0 could evict it while S_6={6,0,2} is still
    using it, forcing a re-transfer.

    Slots are chunk-aligned, so the segmented-input block map can point straight
    at pool rows -- no gather and no contiguous per-subsequence buffer.
    """

    def __init__(self, n_slots, chunk_tokens, *, B, H, D, dtype, device, align=128):
        self.n_slots = int(n_slots)
        self.chunk_tokens = int(chunk_tokens)
        self.align = int(align)
        # One [B, n_slots*chunk_tokens, H, D] buffer per tensor, token-major so
        # the block map indexes rows directly.
        self.buf = [
            torch.empty((B, self.n_slots * self.chunk_tokens, H, D),
                        dtype=dtype, device=device)
            for _ in range(3)
        ]
        self.slot_of: dict[int, int] = {}
        self.refcount: dict[int, int] = {}
        # Host-side refcounts are not sufficient on their own: the count drops
        # when a subsequence's work is ENQUEUED, but the kernel may still be
        # reading the chunk. Evicting then overwrites live data -- measured as
        # relative error 1.28e-01 at 5 slots, while 6-7 slots (no eviction ever)
        # were correct. So each chunk also records the last GPU event that
        # touched it, and eviction waits on it.
        self.last_event: dict[int, torch.cuda.Event] = {}
        self.free: list[int] = list(range(self.n_slots))
        self.transfers = 0          # chunks actually copied H2D
        self.hits = 0               # chunks already resident
        self.evictions = 0

    def _evict_one(self) -> int:
        for cid, slot in list(self.slot_of.items()):
            if self.refcount.get(cid, 0) == 0:
                ev = self.last_event.pop(cid, None)
                if ev is not None:
                    ev.synchronize()      # the GPU must be done reading it
                del self.slot_of[cid]
                self.refcount.pop(cid, None)
                self.evictions += 1
                return slot
        raise RuntimeError(
            "chunk pool exhausted: every slot is referenced by an in-flight "
            "subsequence. Increase n_slots or reduce n_par."
        )

    def acquire(self, chunk_ids, sources, chunk_ranges) -> dict[int, int]:
        """
        Make ``chunk_ids`` resident and pin them. ``sources`` are the three
        host token-major tensors; ``chunk_ranges[cid]`` is ``(start, length)``
        in global token coordinates. Returns ``{chunk_id: slot}``.
        """
        assigned = {}
        for cid in chunk_ids:
            slot = self.slot_of.get(cid)
            if slot is None:
                slot = self.free.pop() if self.free else self._evict_one()
                gs, ln = chunk_ranges[cid]
                dst0 = slot * self.chunk_tokens
                for b, src in zip(self.buf, sources):
                    b[:, dst0:dst0 + ln].copy_(src[:, gs:gs + ln], non_blocking=True)
                self.slot_of[cid] = slot
                self.transfers += 1
            else:
                self.hits += 1
            self.refcount[cid] = self.refcount.get(cid, 0) + 1
            assigned[cid] = slot
        return assigned

    def release(self, chunk_ids, event: "torch.cuda.Event | None" = None) -> None:
        """
        Drop the host-side reference and record the GPU event that must complete
        before the chunk's slot may be reused.
        """
        for cid in chunk_ids:
            if cid in self.refcount:
                self.refcount[cid] = max(0, self.refcount[cid] - 1)
            if event is not None:
                self.last_event[cid] = event

    def block_base(self, chunk_ids, assigned, chunk_ranges) -> torch.Tensor:
        """
        Map each aligned block of the LOCAL subsequence to its row in the pool.

        Chunks are concatenated in ascending id (sorted gather), and both chunk
        lengths and slot bases are multiples of ``align``, so a block never
        straddles a slot boundary.
        """
        base = []
        for cid in sorted(chunk_ids):
            _, ln = chunk_ranges[cid]
            row0 = assigned[cid] * self.chunk_tokens
            base.extend(range(row0, row0 + ln, self.align))
        return torch.tensor(base, dtype=torch.int32, device=self.buf[0].device)

    def stats(self) -> dict[str, int]:
        total = self.transfers + self.hits
        return {
            "chunk_loads_requested": total,
            "chunk_transfers": self.transfers,
            "chunk_hits": self.hits,
            "evictions": self.evictions,
            "transfer_saving_pct": (100.0 * self.hits / total) if total else 0.0,
        }


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


class TraceRecorder:
    """
    One row per task stage, matching the schema in the codegen prompt.

    Rows are buffered in memory and written as JSONL, so a run can be replayed
    by the visualisation script without re-running the GPU work.
    """

    # duration_ms is wall-clock around the launch; cuda_ms is device time from
    # CUDA events (NaN for CPU stages and for the CPU fallback path).
    FIELDS = (
        "run_id", "phase", "path_idx", "path", "itr", "stage", "stream_id",
        "start_time_s", "end_time_s", "duration_ms", "cuda_ms", "local_size",
        "active_subseq", "gpu_used_gib", "gpu_free_gib", "cuda_allocated_gib",
        "cuda_reserved_gib", "status",
    )

    def __init__(self, run_id: str | None = None, *, enabled: bool = True, device=None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.enabled = bool(enabled)
        self.device = device
        self.rows: list[dict[str, Any]] = []
        self.t0 = time.perf_counter()

    def _mem(self) -> tuple[float, float, float, float]:
        # A CUDA-capable box can still be running the CPU fallback path.
        if not torch.cuda.is_available():
            return (0.0, 0.0, 0.0, 0.0)
        if self.device is not None and torch.device(self.device).type != "cuda":
            return (0.0, 0.0, 0.0, 0.0)
        free_b, total_b = torch.cuda.mem_get_info(self.device)
        return (
            (total_b - free_b) / _GIB,
            free_b / _GIB,
            torch.cuda.memory_allocated(self.device) / _GIB,
            torch.cuda.memory_reserved(self.device) / _GIB,
        )

    def record(
        self,
        *,
        stage: str,
        task: SubproblemTask | None,
        start: float,
        end: float,
        phase: str = "fwd",
        active_subseq: int = 0,
        status: str = "ok",
        stream_id: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        used, free, alloc, reserved = self._mem()
        self.rows.append(
            {
                "run_id": self.run_id,
                "phase": phase,
                "path_idx": -1 if task is None else task.path_idx,
                "path": "" if task is None else "-".join(str(x) for x in task.path),
                "itr": -1 if task is None else task.itr,
                "stage": stage,
                "stream_id": (task.stream_id if task is not None else -1)
                if stream_id is None else int(stream_id),
                "start_time_s": start - self.t0,
                "end_time_s": end - self.t0,
                "duration_ms": (end - start) * 1000.0,
                "cuda_ms": float("nan"),
                "local_size": 0 if task is None else task.local_size,
                "active_subseq": int(active_subseq),
                "gpu_used_gib": used,
                "gpu_free_gib": free,
                "cuda_allocated_gib": alloc,
                "cuda_reserved_gib": reserved,
                "status": status,
            }
        )

    # Bookkeeping rows: they carry the true end-to-end span but are not a stage
    # of work, so they are excluded from stage totals and from the Gantt lanes.
    SUMMARY_STAGES = frozenset({"run_total"})

    def stage_totals_ms(self, *, prefer_cuda: bool = True) -> dict[str, float]:
        """
        Summed time per stage. Uses device time where CUDA events resolved,
        falling back to wall-clock for CPU-side stages -- summing launch time
        would credit an async stage with almost nothing.

        Caveat: with more than one stream in flight, an event pair also spans
        the time its stream sat queued behind others, so these totals
        over-count. Attribute stage cost from a ``max_parallel=1`` run; there
        the totals match end-to-end wall time.
        """
        out: dict[str, float] = {}
        for r in self.rows:
            if r["stage"] in self.SUMMARY_STAGES:
                continue
            cuda_ms = r.get("cuda_ms", float("nan"))
            use = cuda_ms if (prefer_cuda and cuda_ms == cuda_ms) else r["duration_ms"]
            out[r["stage"]] = out.get(r["stage"], 0.0) + float(use)
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def write_jsonl(self, path) -> None:
        with open(path, "w") as fh:
            for r in self.rows:
                fh.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Inner kernels
# ---------------------------------------------------------------------------


def local_stats_flash(q, k, v, group_bits, *, causal: bool, scale: float,
                      blk_or=None, blk_and=None, blk_size: int = 64,
                      fp32_out: bool = True, block_base=None, seg_align: int = 0,
                      token_ids=None, **_unused):
    # token_ids / extra kwargs: accepted (and ignored) so that the scheduler's
    # full call, which also carries the segmented-input arguments, never falls
    # back to the bare call. Without this the shared_chunks path lost
    # block_base and read the wrong memory (caught by devkit.quick_bench).
    """
    CUDA inner kernel. q/k/v in ``[B, L, H, D]``.

    Returns ``(out_i [B,L,H,D] fp32, lse_i [B,L,H] fp32)``. Note this returns
    ``lse``, never ``exp(lse)`` -- that is the whole point.

    Everything stays token-major, so no transpose-copy of the (large) output is
    needed; only ``lse`` is transposed, and it is ``D`` times smaller.
    """
    from .interface import flash_attn_func_cqs_group_bits

    out, lse, _ = flash_attn_func_cqs_group_bits(
        q, k, v, group_bits,
        dropout_p=0.0, softmax_scale=float(scale), causal=bool(causal),
        return_attn_probs=True,
        cqs_blk_or=blk_or, cqs_blk_and=blk_and, cqs_blk_size=int(blk_size),
        fp32_out=bool(fp32_out),
        cqs_block_base=block_base, cqs_seg_align=int(seg_align),
    )
    # With fp32_out the kernel already wrote fp32, so .float() is a no-op view
    # rather than a full [B,L,H,D] cast+copy.
    return out.float(), lse.float().transpose(1, 2)


def local_stats_torch(q, k, v, group_bits, *, causal: bool, scale: float):
    """
    Pure-torch fallback with the same contract, for machines without the CUDA
    extension and for differential testing. q/k/v in ``[B, L, H, D]``.

    Materialises the dense local score matrix, so it is only for small L.
    """
    qb = q.transpose(1, 2).float()
    kb = k.transpose(1, 2).float()
    vb = v.transpose(1, 2).float()
    scores = torch.matmul(qb, kb.transpose(-2, -1)) * float(scale)

    bits = group_bits.to(scores.device, dtype=torch.long)
    keep = torch.bitwise_and(bits[:, None], bits[None, :]).eq(0)
    if causal:
        L = scores.shape[-1]
        keep = keep & torch.ones(L, L, dtype=torch.bool, device=scores.device).tril()

    scores = scores.masked_fill(~keep[None, None], float("-inf"))
    m = scores.max(dim=-1).values
    finite = torch.isfinite(m)
    safe_m = torch.where(finite, m, torch.zeros_like(m))
    p = torch.exp(scores - safe_m.unsqueeze(-1))
    p = torch.where(keep[None, None], p, torch.zeros_like(p))
    l = p.sum(dim=-1)
    acc = torch.matmul(p, vb)
    out = acc / l.clamp_min(1e-30).unsqueeze(-1)
    lse = torch.where(finite, m + torch.log(l.clamp_min(1e-30)),
                      torch.full_like(m, float("-inf")))
    # Return token-major to match the CUDA kernel's contract.
    return out.transpose(1, 2), lse.transpose(1, 2)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def stream_cqsa_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    # Automatic by default: the caller should not have to know what `itr` means
    # to get a correct result. An explicit int is the opt-out, for reproducing a
    # specific decomposition depth.
    itr: int | str = "auto",
    causal: bool = False,
    scale: float | None = None,
    inner: Callable | None = None,
    sorted_gather: bool = True,
    max_parallel: int | None = None,
    safety: float = 0.75,
    accumulate_on_gpu: bool = True,
    allow_escalation: bool = True,
    low_memory: bool = False,
    trace: TraceRecorder | None = None,
    c: int = 7,
    interest_set: Sequence[int] = (0, 1, 3),
    segmented: bool = False,
    seg_align: int = 128,
    stream_from_host: bool = False,
    cpu_threads: int | None = _CPU_MERGE_THREADS_DEFAULT,
    shared_chunks: bool = False,
    chunk_pool_slots: int | None = None,
    # next/: run only these task path_idx values. Used by the multi-device
    # driver, where each rank takes a shard of the identical task list.
    task_subset: Sequence[int] | None = None,
    # Announce the call, show a progress bar over the subproblems with the time
    # remaining (stderr). None defers to the CQSA_VERBOSE environment variable.
    verbose: bool | None = None,
    # Where the fp32 result is returned: None = the inputs' device (as before);
    # "acc" = wherever the accumulator lives (host with low_memory=True), which
    # spares a device copy the caller may not have room for.
    out_device=None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Stream-CQSA exact attention forward.

    q/k/v: ``[B, H, N, D]``. Returns ``(O [B,H,N,D] fp32, info)``.

    Subproblems run on independent CUDA streams; each is gathered, computed and
    merged as soon as its own compute finishes, so there is no round barrier.
    On OOM the scheduler halves the in-flight count, clears the cache and
    retries, rather than failing the run.
    """
    B, H, N, D = q.shape
    device = q.device
    if scale is None:
        scale = float(D) ** -0.5

    # Host-resident Q/K/V: the inputs live on the CPU and only the in-flight
    # subsequences are ever on the device. This is the framework's actual OOM
    # mechanism -- the device footprint loses its O(N.H.D) input term, so deeper
    # `itr` genuinely reduces it.
    #
    # Explicit opt-in on purpose. Inferring it from `q.device == cpu` would
    # silently hijack every CPU call, including the pure-torch reference path
    # that has no CUDA kernel to stream to.
    # `low_memory=True` is the discoverable spelling of accumulate_on_gpu=False:
    # it moves the fp32 [B,N,H,D] accumulator to the host. Deliberately NOT named
    # "safe" -- in this domain "safe" means numerically safe (safe softmax), and
    # this flag does not affect accuracy: the two paths differ by 1.5e-08, five
    # orders of magnitude below the fp16 error floor of ~3e-04. It trades time
    # for device memory and nothing else, so it is named for the axis it trades.
    #
    # It also only pays at large N. The accumulator is O(N), so at N=65536 it is
    # a rounding error in the peak (193 -> 192 MiB) while costing 2.4x the time;
    # at N=1M it is 2.0 GiB and often the difference between running and not.
    if low_memory:
        accumulate_on_gpu = False

    host_resident = bool(stream_from_host) and device.type == "cpu"
    if host_resident and not torch.cuda.is_available():
        raise ValueError("stream_from_host=True requires CUDA")
    if host_resident:
        if inner is local_stats_torch:
            raise ValueError(
                "host-resident streaming needs the CUDA inner kernel; pass "
                "CUDA tensors to use the pure-torch fallback"
            )
        device = torch.device("cuda")
        if inner is None:
            from . import interface as _iface
            if _iface.cqsa_cuda is None or _os.environ.get("CQSA_FORWARD", "").lower() == "triton":
                from .triton_kernel import triton_inner
                inner = triton_inner
        inner = inner or local_stats_flash

    if inner is None:
        inner = local_stats_flash if device.type == "cuda" else local_stats_torch
        if device.type == "cuda":
            from . import interface as _iface
            if _iface.cqsa_cuda is None or _os.environ.get("CQSA_FORWARD", "").lower() == "triton":
                # No compiled extension: the Triton kernel is the zero-build path
                # (same contract, ~0.7x the CUDA kernel's speed).
                from .triton_kernel import triton_inner
                inner = triton_inner
    if trace is None:
        trace = TraceRecorder(enabled=False, device=device)
    # shared_chunks hands the inner kernel a segmented chunk-pool view; only the
    # CUDA kernel addresses it, so any other inner kernel gets plain gathers.
    if shared_chunks and inner is not local_stats_flash:
        shared_chunks = False

    plan_reason = None
    if isinstance(itr, str):
        if itr != "auto":
            raise ValueError(f"itr must be an int or 'auto', got {itr!r}")
        itr, plan_reason = plan_decomposition(
            N, B=B, H=H, D=D, itemsize=q.element_size(), device=device,
            safety=safety, c=c, interest_set=interest_set,
            # Tell the planner where this call's O(N) terms will actually live.
            # Without these it estimates every configuration as if Q/K/V and the
            # accumulator were device-resident, which is exactly the
            # configuration the caller has asked it not to use.
            stream_from_host=bool(host_resident),
            accumulate_on_gpu=bool(accumulate_on_gpu),
        )

    if int(itr) == 0:
        # Below the OOM boundary a monolithic call wins on both speed and
        # accuracy, so decomposing would be strictly worse. Only reachable via
        # itr="auto"; an explicit itr=0 is treated the same way.
        if device.type != "cuda":
            raise ValueError("itr=0 (monolithic) requires CUDA")
        from .interface import flash_attn_func

        if host_resident:
            # itr="auto" can pick 0 even with stream_from_host=True: the planner
            # only asks whether a monolithic call fits, and below the OOM
            # boundary it does. The inputs are on the host though, so the
            # monolithic kernel cannot read them in place. Staging them is safe
            # precisely because the planner just certified they fit.
            q, k, v = (t.to(device, non_blocking=True) for t in (q, k, v))

        t0 = time.perf_counter()
        # Ask for the lse. Without it this path silently omits info["lse"], and
        # since itr="auto" can select itr=0, a caller following the documented
        # forward-then-backward flow would hit a KeyError purely because the
        # planner happened to decline to decompose.
        out_bhd, lse0 = flash_attn_func(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            softmax_scale=float(scale), causal=bool(causal), return_lse=True,
        )
        out = out_bhd.transpose(1, 2).float()
        if verbose_enabled(verbose):
            print(f"Stream-CQSA: forward of N={N} tokens fits in device memory -- one monolithic FlashAttention call, no decomposition",
                  file=sys.stderr, flush=True)
        return out, {
            "n_subproblems": 0,
            "itr": 0,
            "monolithic": True,
            "n_parallel": 1,
            "oom_retries": 0,
              "depth_escalations": 0,
              "itr_max_reached": 0,
              "depth_capped": False,
            "causal": bool(causal),
            "untouched_tokens": 0,
            "wall_s": time.perf_counter() - t0,
            "stage_totals_ms": {},
            "plan_reason": plan_reason,
            # [B, H, N], same contract as the decomposed path: the *global* lse.
            "lse": lse0,
        }

    # Segmented input needs sorted gather (runs must be ascending) and the
    # CUDA inner kernel; the torch fallback has no notion of a block map.
    use_seg = bool(segmented) and device.type == "cuda" and sorted_gather and not host_resident
    # The alignment the TASKS are built with, which is not the same as the
    # seg_align argument: outside segment/shared-chunk mode the token ids must
    # not be padded at all. Escalation rebuilds tasks mid-run and has to use
    # this same value -- padding a refined child to a 128-token boundary made it
    # LARGER than the parent it replaced (63 real tokens became 256), so the
    # recovery grew the subproblem it was called to shrink and could only ever
    # climb to the depth cap and fail.
    task_seg_align = int(seg_align) if (use_seg or shared_chunks) else 0
    tasks = build_tasks_cached(
        N, itr, B=B, H=H, D=D, itemsize=q.element_size(),
        sorted_gather=sorted_gather, pin=(device.type == "cuda"), c=c,
        interest_set=tuple(interest_set),
        # shared_chunks also needs block-aligned chunks: the pool's block map
        # assumes a tile never straddles a chunk (and therefore a slot) boundary.
        seg_align=task_seg_align,
    )
    if task_subset is not None:
        _keep = {int(i) for i in task_subset}
        tasks = [t for t in tasks if int(t.path_idx) in _keep]
    # Token-major views of the ORIGINAL tensors. stride(-1) == 1, so the kernel
    # accepts them without a copy, and every subproblem reads these same
    # buffers -- which is where the memory saving comes from.
    qt = q.transpose(1, 2) if use_seg else None
    kt = k.transpose(1, 2) if use_seg else None
    vt = v.transpose(1, 2) if use_seg else None


    on_cuda = device.type == "cuda"
    # Where the accumulator lives is an independent choice from where the inputs
    # live, and it is the dominant one for speed. Measured at N=131072, L=56173:
    # one merge costs 33.5 ms on the GPU and 4661.8 ms on the CPU (139x), so a
    # host accumulator adds ~32 s over seven subproblems. Keeping it on the
    # device costs an O(N.H.D) term (~516 MiB here) but is the difference
    # between usable and not. `accumulate_on_gpu` selects it explicitly.
    acc_device = device if (accumulate_on_gpu and on_cuda) else torch.device("cpu")
    accum = StableAccumulator(B, H, N, D, device=acc_device)

    if max_parallel is None:
        n_par = (choose_parallelism(tasks, device=device, safety=safety, B=B, H=H,
                                    host_streaming=host_resident, D=D,
                                    itemsize=q.element_size())
                 if on_cuda else 1)
    else:
        n_par = int(max_parallel)

    info: dict[str, Any] = {
        "n_subproblems": len(tasks),
        "itr": int(itr),
        "n_parallel": int(n_par),
        "oom_retries": 0,
        "depth_escalations": 0,
        "itr_max_reached": int(itr),
        # True only if a subproblem failed at the deepest depth N structurally
        # admits, which is the one case the recovery cannot act on.
        "depth_capped": False,
        "monolithic": False,
        "plan_reason": plan_reason,
        "sorted_gather": bool(sorted_gather),
        "segmented": bool(use_seg),
        "shared_chunks": bool(shared_chunks),
        "causal": bool(causal),
        "c": int(c),
        "interest_set": tuple(int(x) for x in interest_set),
        "max_local_size": max((t.local_size for t in tasks), default=0),
        "host_resident": bool(host_resident),
        "est_task_gib": max((t.estimated_mem_gib for t in tasks), default=0.0),
        "blocks_per_task": (max(1, -(-max((t.local_size for t in tasks), default=1) // 128)) * B * H),
    }

    progress = Progress(
        len(tasks), enabled=verbose_enabled(verbose), unit="subproblem",
        banner=describe_call(what="forward", N=N, B=B, H=H, D=D, c=int(c), itr=int(itr), n_tasks=len(tasks),
                             causal=bool(causal), device=str(device),
                             extra=f"{n_par} in flight, accumulator on {acc_device.type}"
                                   + (", Q/K/V streamed from host memory" if host_resident else "")),
        expected_s=expected_seconds(N=N, B=B, H=H, D=D, itr=int(itr), c=int(c), causal=bool(causal),
                                    acc=acc_device.type, stream_from_host=host_resident, n_par=int(n_par)))

    streams = [torch.cuda.Stream(device=device) for _ in range(n_par)] if on_cuda else [None]
    if on_cuda:
        # A freshly created stream carries no dependency on the caller's work.
        # Without this the gathers and kernels below start while whatever the
        # caller last issued is still running, and the allocator can hand these
        # streams blocks that work is still using. That showed up as entire
        # gradient tensors returning NaN whenever a call followed unsynchronised
        # GPU work, and disappearing whenever a synchronising read happened to
        # sit in between. The exit side is already covered by the synchronize
        # after the scheduling loop; this is the entry side.
        _caller = torch.cuda.current_stream(device)
        for _s in streams:
            _s.wait_stream(_caller)

    # Pinned staging, one buffer set per stream slot, sized for the largest
    # subproblem. Pageable H2D runs at roughly a third of pinned bandwidth, and
    # pinning per task would dominate, so these are allocated once and reused.
    # Shared chunk pool: store each token chunk on the device once and let
    # concurrent subsequences share it, instead of every subsequence carrying
    # its own copy of the l chunks it needs.
    chunk_pool = None
    chunk_ranges: dict[int, tuple[int, int]] = {}
    if shared_chunks:
        if not host_resident:
            raise ValueError(
                "shared_chunks requires stream_from_host=True; with device-resident "
                "inputs every chunk is already on the device exactly once"
            )
        if int(itr) != 1:
            raise ValueError(
                "shared_chunks currently supports itr=1, where a subsequence is a "
                f"union of whole top-level chunks (got itr={itr})"
            )
        from .reference import chunk_layout
        sizes, starts, _ = chunk_layout(N, c, seg_align)
        chunk_ranges = {i: (starts[i], sizes[i]) for i in range(c)}
        max_chunk = max(sizes)
        l = len(interest_set)
        # Enough slots for every in-flight subsequence's chunks, plus room to
        # keep shared ones resident rather than thrashing them.
        slots = chunk_pool_slots or min(c, n_par * l)

    pinned_slots = []
    q_tm = k_tm = v_tm = None
    if host_resident:
        # One-time transpose to token-major (~98 ms at N=32768); every
        # subproblem's gather is then a contiguous index_select on dim 1.
        if not q.transpose(1, 2).is_contiguous():
            _warn_head_major(N, H, D, q.element_size())
        q_tm = q.transpose(1, 2).contiguous()
        k_tm = k.transpose(1, 2).contiguous()
        v_tm = v.transpose(1, 2).contiguous()
        if shared_chunks:
            chunk_pool = DeviceChunkPool(
                slots, max_chunk, B=B, H=H, D=D, dtype=q.dtype,
                device=device, align=seg_align,
            )
        max_el = B * max((t.local_size for t in tasks), default=1) * H * D
        for _ in range(max(1, len(streams))):
            pinned_slots.append(tuple(
                torch.empty(max_el, dtype=q.dtype, pin_memory=True) for _ in range(3)
            ))

    # The merge is a read-modify-write of shared global state (acc/l/m). Running
    # it on the worker streams lets concurrent subproblems interleave their
    # read-modify-write on the same token rows, silently corrupting the result
    # (observed: relative error 1.0 at N=16384, itr=2, 8 streams). All merges
    # are therefore issued on ONE dedicated stream, ordered by issue order,
    # while gather+compute stay concurrent.
    merge_stream = torch.cuda.Stream(device=device) if on_cuda else None
    if merge_stream is not None:
        merge_stream.wait_stream(torch.cuda.current_stream(device))

    # next/ (track 2): with the accumulator on the host, the shipped path
    # copied each subproblem's output with a BLOCKING d2h and merged inline,
    # so the host could not launch subproblem i+1 until i had fully drained
    # -- measured flat in n_par (3005/2924/2921/2930 ms at N=64K for
    # n_par=1/2/4/8). This is the backward's flush()/scatter_pool design
    # brought to the forward: one pinned fp32 output slot per stream, a
    # non-blocking d2h with an event, and the merge on ONE worker thread.
    # One worker, not a pool: the accumulator is shared mutable state.
    out_slots: list[tuple[torch.Tensor, torch.Tensor]] = []
    merge_pool = None
    slot_merge: list = []
    merge_log: list = []   # (task, t_wait0, t_wait1, t_merge_end), flushed to trace after join
    if on_cuda and acc_device.type == "cpu" and _FWD_MERGE_ASYNC:
        _maxL = max((int(t.local_size) for t in tasks), default=1)
        for _ in range(max(1, len(streams))):
            out_slots.append((
                torch.empty(B * _maxL * H * D, dtype=torch.float32, pin_memory=True),
                torch.empty(B * _maxL * H, dtype=torch.float32, pin_memory=True),
            ))
        merge_pool = ThreadPoolExecutor(max_workers=1)
        slot_merge = [None] * max(1, len(streams))

    # Gather on dim 2 (contiguous [B,H,L,D]) then view as token-major. The
    # kernel only requires stride(-1) == 1 (flash_api.cpp:616), which a
    # transposed view satisfies, so no copy is needed here. Materialising a
    # token-major copy of Q/K/V up front would cost three extra full-size
    # buffers -- unacceptable in an OOM-recovery path.

    # perf_counter around an async CUDA call measures launch time, not GPU time.
    # Pair every GPU stage with events and resolve them after the final sync.
    pending_events: list[tuple[int, torch.cuda.Event, torch.cuda.Event]] = []

    # Backpressure: one done-event per slot. Before reusing a slot we wait on
    # its previous task, which bounds in-flight work to len(streams) working
    # sets. Without this the loop would queue every subproblem immediately and
    # memory would only be bounded incidentally, by the caching allocator's
    # per-stream reuse -- not by the budget we just computed.
    slot_done: list[torch.cuda.Event | None] = [None] * max(1, len(streams))
    in_flight = 0

    def run_one(task: SubproblemTask, slot: int) -> None:
        nonlocal in_flight
        task.stream_id = slot
        stream = streams[slot] if on_cuda else None
        ctx = torch.cuda.stream(stream) if stream is not None else _null_ctx()

        if on_cuda and slot < len(slot_done) and slot_done[slot] is not None:
            t_wait0 = time.perf_counter()
            slot_done[slot].synchronize()
            in_flight = max(0, in_flight - 1)
            trace.record(stage="wait", task=task, start=t_wait0,
                         end=time.perf_counter(), active_subseq=in_flight,
                         stream_id=slot)

        def mk_ev():
            return (torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True)) if on_cuda else (None, None)

        def stamp(stage, ev, t_start, t_end):
            row_idx = len(trace.rows)
            trace.record(stage=stage, task=task, start=t_start, end=t_end,
                         active_subseq=in_flight)
            if on_cuda and trace.enabled and ev[0] is not None:
                pending_events.append((row_idx, ev[0], ev[1]))

        with ctx:
            ev = mk_ev()
            t0 = time.perf_counter()
            if on_cuda:
                ev[0].record()
            idx = task.token_ids.to(device, non_blocking=True)
            bits = task.group_bits.to(device, non_blocking=True)
            blk_or = task.extra.get("blk_or")
            blk_and = task.extra.get("blk_and")
            if blk_or is not None:
                blk_or = blk_or.to(device, non_blocking=True)
                blk_and = blk_and.to(device, non_blocking=True)
            seg_base = task.extra.get("seg_base") if use_seg else None
            if chunk_pool is not None:
                # Only the chunks not already resident are transferred; the
                # block map then points the kernel at pool rows, so there is no
                # per-subsequence gather at all.
                cids = sorted({(task.path[0] + off) % c for off in interest_set})
                assigned = chunk_pool.acquire(cids, (q_tm, k_tm, v_tm), chunk_ranges)
                seg_base = chunk_pool.block_base(cids, assigned, chunk_ranges)
                task.extra["held_chunks"] = cids
                Lloc = int(task.local_size)
                pool_rows = chunk_pool.n_slots * chunk_pool.chunk_tokens
                q_i, k_i, v_i = (
                    b.as_strided((B, Lloc, H, D), b.stride()) for b in chunk_pool.buf
                )
            elif host_resident:
                # Host-resident Q/K/V: gather on the CPU and stream only this
                # subsequence to the device. The device never holds the full
                # N-token tensors, so its footprint is O(L.H.D), not O(N.H.D).
                #
                # The gather runs on the pre-transposed token-major copies and
                # writes straight into the pinned staging buffer. Gathering from
                # the [B,H,N,D] originals instead needs a
                # transpose(1,2).contiguous() per subproblem, which is
                # cache-hostile on the CPU: 166.6 ms versus 3.0 ms measured at
                # N=32768 (56x), and it dominated everything else.
                Lloc = int(task.local_size)
                buf = pinned_slots[slot]
                n_el = B * Lloc * H * D
                views = [b[:n_el].view(B, Lloc, H, D) for b in buf]
                torch.index_select(q_tm, 1, task.token_ids, out=views[0])
                torch.index_select(k_tm, 1, task.token_ids, out=views[1])
                torch.index_select(v_tm, 1, task.token_ids, out=views[2])
                q_i = views[0].to(device, non_blocking=True)
                k_i = views[1].to(device, non_blocking=True)
                v_i = views[2].to(device, non_blocking=True)
            elif seg_base is not None:
                seg_base = seg_base.to(device, non_blocking=True)
                # No gather at all: hand the kernel a length-L strided view over
                # the original token-major tensors. Row r of this view resolves
                # to global row base[r // align] + r % align inside the kernel.
                Lloc = int(task.local_size)
                q_i = qt.as_strided((B, Lloc, H, D), qt.stride())
                k_i = kt.as_strided((B, Lloc, H, D), kt.stride())
                v_i = vt.as_strided((B, Lloc, H, D), vt.stride())
            else:
                q_i = q.index_select(2, idx).transpose(1, 2)
                k_i = k.index_select(2, idx).transpose(1, 2)
                v_i = v.index_select(2, idx).transpose(1, 2)
            if on_cuda:
                ev[1].record()
            t1 = time.perf_counter()
            stamp("gather", ev, t0, t1)

            ev = mk_ev()
            if on_cuda:
                ev[0].record()
            try:
                out_i, lse_i = inner(q_i, k_i, v_i, bits, causal=causal, scale=float(scale),
                                     blk_or=blk_or, blk_and=blk_and,
                                     block_base=seg_base,
                                     seg_align=int(seg_align) if seg_base is not None else 0,
                                     # next/: global positions of the gathered tokens, for inner
                                     # kernels whose score depends on absolute position (ALiBi,
                                     # RoPE-in-kernel, document ids). idx is the gather index.
                                     token_ids=idx)
            except TypeError:
                # Inner kernels without tile summaries (the torch fallback).
                out_i, lse_i = inner(q_i, k_i, v_i, bits, causal=causal, scale=float(scale))
            if on_cuda:
                ev[1].record()
            del q_i, k_i, v_i
            t2 = time.perf_counter()
            stamp("compute", ev, t1, t2)

            # Event-time this the way the backward does. A host timer here
            # measures the blocking copy AND the wait for the asynchronous
            # kernel that precedes it, so the recorded interval was kernel plus
            # copy: at N=1M each subsequence reported 1732 ms against 1566 ms of
            # kernel, and the stage totals came to 1.95x the measured wall
            # clock. An event pair times the copy on the device, which is what
            # the stage claims to be.
            ev_d2h = mk_ev()
            if on_cuda and ev_d2h[0] is not None:
                ev_d2h[0].record()
            async_merge = merge_pool is not None
            d2h_done = None
            if acc_device.type == "cpu":
                idx = task.token_ids
                if async_merge:
                    # The slot's pinned buffers are free only once the merge
                    # that last read them has finished. Waiting HERE, not at
                    # slot acquisition, is what lets gather+h2d+compute of
                    # this task overlap that merge.
                    if slot < len(slot_merge) and slot_merge[slot] is not None:
                        slot_merge[slot].result()
                        slot_merge[slot] = None
                    _Lo = int(task.local_size)
                    _ob, _lb = out_slots[slot]
                    _ov = _ob[: B * _Lo * H * D].view(B, _Lo, H, D)
                    _lv = _lb[: B * _Lo * H].view(B, _Lo, H)
                    _ov.copy_(out_i, non_blocking=True)
                    _lv.copy_(lse_i, non_blocking=True)
                    out_i, lse_i = _ov, _lv
                    d2h_done = torch.cuda.Event()
                    d2h_done.record()
                else:
                    out_i = out_i.to("cpu", non_blocking=False)
                    lse_i = lse_i.to("cpu", non_blocking=False)
            if on_cuda and ev_d2h[1] is not None:
                ev_d2h[1].record()
            t3 = time.perf_counter()
            stamp("d2h", ev_d2h, t2, t3)

            if on_cuda:
                compute_done = torch.cuda.Event()
                compute_done.record()

        # Merge on the dedicated stream, after this task's compute has landed.
        merge_ctx = torch.cuda.stream(merge_stream) if merge_stream is not None else _null_ctx()
        with merge_ctx:
            ev = mk_ev()
            if on_cuda:
                merge_stream.wait_event(compute_done)
                # These were allocated on the worker stream; tell the allocator
                # they are still live on the merge stream so it cannot hand the
                # blocks back to the worker while the merge is still reading.
                # Only meaningful for device tensors -- with a host accumulator
                # they have already been copied to the CPU, where record_stream
                # does not exist.
                if out_i.is_cuda:
                    out_i.record_stream(merge_stream)
                    lse_i.record_stream(merge_stream)
                ev[0].record()
            if async_merge:
                def _merge_job(idx_=idx, o_=out_i, l_=lse_i, ev_=d2h_done, task_=task):
                    tw0 = time.perf_counter()
                    ev_.synchronize()               # this slot's d2h has landed
                    tw1 = time.perf_counter()
                    accum.merge_lse(idx_, o_, l_)
                    merge_log.append((task_, tw0, tw1, time.perf_counter()))
                slot_merge[slot] = merge_pool.submit(_merge_job)
                if on_cuda:
                    ev[1].record()
                del out_i, lse_i
                t4 = time.perf_counter()
            else:
                accum.merge_lse(idx, out_i, lse_i)
                if on_cuda:
                    ev[1].record()
                del out_i, lse_i
                t4 = time.perf_counter()
                stamp("merge", ev, t3, t4)

            if chunk_pool is not None:
                # Record the event this subsequence's work completes on, and
                # hand it to the pool: the host reference drops now, but the
                # slot cannot be reused until the GPU has actually finished.
                rel_ev = torch.cuda.Event()
                rel_ev.record()
                chunk_pool.release(task.extra.pop("held_chunks", []), rel_ev)

            if on_cuda and slot < len(slot_done):
                # Slot reuse must wait for the merge too, not just the compute.
                done = torch.cuda.Event()
                done.record()
                slot_done[slot] = done
                in_flight += 1
        task.status = "done"

    t_start = time.perf_counter()
    pending = list(tasks)
    slot = 0
    # Termination is already guaranteed without this: every recovery either
    # halves the stream count, which can happen only log2(n_parallel) times, or
    # refines a task, and refinement stops at the structural depth bound. The
    # budget is a backstop against an unforeseen loop, so it has to sit well
    # above what a legitimate run can reach -- a fixed 64 aborted deep
    # escalations, where one failure per task already exhausts it.
    oom_retry_budget = 64 + 16 * max(1, len(pending)) * max(1, max_depth_for(N, c))
    # Only worth scoping when CPU tensors are actually on the critical path.
    thread_ctx = _cpu_threads(cpu_threads if acc_device.type == "cpu" else None)
    with thread_ctx:
      while pending:
          task = pending.pop(0)
          try:
              run_one(task, slot % max(1, len(streams)))
              slot += 1
              progress.update(1)
          except RuntimeError as exc:
              if not is_oom(exc):
                  raise
              if getattr(accum, "dirty", False):
                  raise RuntimeError(
                      "Stream-CQSA: ran out of memory while committing a merge. "
                      "The accumulator holds a partially merged subproblem, so "
                      "retrying it would double count and recovery is unsafe. "
                      "Re-run with a lower max_parallel or a higher itr."
                  ) from exc
              info["oom_retries"] += 1
              task.status = "oom"
              trace.record(
                  stage="oom", task=task, start=time.perf_counter(),
                  end=time.perf_counter(), status="oom",
              )
              if on_cuda:
                  # The pure-torch fallback runs this same recovery on the CPU,
                  # where these would raise "No CUDA GPUs are available" and
                  # bury the failure they were called to clean up after.
                  torch.cuda.synchronize(device)
                  torch.cuda.empty_cache()
              if merge_pool is not None:
                  for _i, _f in enumerate(slot_merge):
                      if _f is not None:
                          _f.result(); slot_merge[_i] = None
              in_flight = 0
              if len(streams) > 1:
                  # First response is always to lower concurrency: it is free and
                  # costs no extra arithmetic.
                  streams = streams[: max(1, len(streams) // 2)]
                  info["n_parallel"] = len(streams)
                  slot_done = [None] * max(1, len(streams))
                  task.status = "queued"
                  pending.insert(0, task)
                  progress.set_total(progress.done + len(pending))
              else:
                  # Concurrency is already one, so the subproblem itself does not
                  # fit. Decompose it a further level and retry its children.
                  # Nothing was merged for this task -- the merge follows the
                  # kernel -- so replacing it double counts nothing, and the
                  # children's retained pairs partition exactly the parent's.
                  depth_cap = max_depth_for(N, c)
                  if not allow_escalation:
                      # A fixed-depth measurement wants the out-of-memory result
                      # itself. Refining here would answer a different question --
                      # what a hybrid schedule costs -- and file it under the
                      # depth that was asked for.
                      info["depth_capped"] = True
                      raise torch.cuda.OutOfMemoryError(
                          f"Stream-CQSA: a subproblem at itr={task.itr} does not "
                          f"fit and escalation is disabled (allow_escalation="
                          f"False), so this depth genuinely does not fit."
                      ) from exc
                  if int(task.itr) >= depth_cap:
                      info["depth_capped"] = True
                      raise torch.cuda.OutOfMemoryError(
                          f"Stream-CQSA: a subproblem at itr={task.itr} does not fit "
                          f"with one subsequence resident, and itr={depth_cap} is the "
                          f"deepest decomposition N={N} admits at c={c}. The O(N*H*D) "
                          f"resident terms dominate and no depth reduces them; offload "
                          f"them, or reduce N, H or D."
                      ) from exc
                  kids = refine_task(
                      task, N, B=B, H=H, D=D, itemsize=q.element_size(),
                      sorted_gather=sorted_gather, pin=True, c=c,
                      interest_set=interest_set, seg_align=task_seg_align,
                  )
                  info["depth_escalations"] += 1
                  info["itr_max_reached"] = max(
                      int(info.get("itr_max_reached", 0)),
                      max((int(t.itr) for t in kids), default=int(task.itr)))
                  slot_done = [None] * max(1, len(streams))
                  pending[0:0] = kids
                  progress.set_total(progress.done + len(pending))
              if info["oom_retries"] > oom_retry_budget:
                  raise
    if merge_pool is not None:
        for _f in slot_merge:
            if _f is not None:
                _f.result()
        merge_pool.shutdown(wait=True)
        # Recorded here rather than from the worker: stamp() reads
        # len(trace.rows) to index its cuda-event row, and a concurrent
        # append from another thread would shift that index.
        for _task, _w0, _w1, _m1 in merge_log:
            trace.record(stage="wait", task=_task, start=_w0, end=_w1)
            trace.record(stage="merge", task=_task, start=_w1, end=_m1)
    if on_cuda:
        torch.cuda.synchronize(device)

    # Device is idle now, so the event pairs can be resolved into real GPU
    # durations. duration_ms stays wall-clock (it shows launch/queueing on the
    # Gantt); cuda_ms is the actual device time for that stage.
    for row_idx, ev0, ev1 in pending_events:
        try:
            trace.rows[row_idx]["cuda_ms"] = float(ev0.elapsed_time(ev1))
        except RuntimeError:
            pass

    out = accum.output()
    # The backward needs the global lse; it is free here and finite for any
    # score magnitude, unlike exp(lse).
    info["lse"] = accum.lse()
    if chunk_pool is not None:
        info.update(chunk_pool.stats())
    info["untouched_tokens"] = accum.untouched_tokens()
    info["wall_s"] = time.perf_counter() - t_start
    # Launch timestamps understate the run: the host returns long before the
    # device drains. Record the true span so the visualisation reports wall
    # time rather than launch time.
    trace.record(stage="run_total", task=None, start=t_start,
                 end=t_start + info["wall_s"], status="done", stream_id=-1)
    info["stage_totals_ms"] = trace.stage_totals_ms() if trace.enabled else {}
    progress.close(f"{info['n_subproblems']} subproblems" + (f", {info['oom_retries']} OOM retries" if info['oom_retries'] else ""))
    if out_device == "acc":
        return out, info
    return out.to(device if out_device is None else out_device), info


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------



def _contiguous_runs(token_ids: torch.Tensor) -> list[tuple[int, int, int]]:
    """
    Split a sorted index vector into ``(dst_offset, src_start, length)`` runs.

    With ``sorted_gather`` a subsequence is the ascending union of a few whole
    chunks, and chunks are contiguous token ranges -- so the index vector is
    only 2-3 runs long in practice, not L scattered positions. Gathering and
    scattering by slice instead of by index is then the same bytes with none of
    the indirection: measured 22x faster on the host at L=12039 (5.5 ms ->
    0.2 ms), which matters because the host scatter had become the single
    largest stage of the streamed backward.

    Returns an empty list if the ids are not ascending, so callers fall back to
    index_select/index_add rather than computing something wrong.
    """
    if token_ids.numel() == 0:
        return []
    ids = token_ids
    d = ids[1:] - ids[:-1]
    if bool((d < 1).any()):          # not strictly ascending: no run structure
        return []
    brk = (d != 1).nonzero().flatten()
    runs, start = [], 0
    for b in brk.tolist():
        runs.append((start, int(ids[start]), b + 1 - start))
        start = b + 1
    runs.append((start, int(ids[start]), int(ids.numel()) - start))
    return runs


def _global_dpsum(dout: torch.Tensor, out: torch.Tensor,
                  chunk: int = 32768) -> torch.Tensor:
    """
    ``rowsum(dO * O)`` over the head dimension, as ``[B, H, N]`` fp32.

    FlashAttention's backward normally recomputes this in a preprocess pass, per
    call, from O. But it is a *global per-token* quantity -- exactly like the
    log-sum-exp -- so Stream-CQSA can compute it once for all N tokens and gather
    it per subproblem. O then never has to reach the device at all, which removes
    one of five streamed tensors and one full-size token-major host copy.

    Accumulated in fp32 to match the kernel, which converts to fp32 before
    multiplying. Doing the product in fp16 first would lose precision the
    preprocess pass does not lose. Chunked over the sequence so the fp32
    temporary stays bounded rather than scaling with N.
    """
    B, H, N, D = dout.shape
    dps = torch.empty((B, H, N), dtype=torch.float32, device=dout.device)

    # On the host this is bandwidth- and allocation-bound and disastrously slow
    # -- 2799 ms at N=262144, against the ~206 ms of gather+H2D that dropping O
    # saves. Doing the fp16 product without widening first is 7x faster but
    # costs 2.07e-04 relative error, the same order as the fp16 gradient itself,
    # so that trade is not available. Instead stream the chunks to the device,
    # reduce there, and bring back only the [B,H,N] result: ~50x cheaper than
    # the host loop and exact, because the widening still happens before the
    # multiply. Each token's O crosses the bus once here, versus once per
    # subproblem containing it (about 9x at itr=2) if the kernel recomputed it.
    if dout.device.type == "cpu" and torch.cuda.is_available():
        dev = torch.device("cuda", torch.cuda.current_device())
        # Slice per (b, h): dout/out are [B,H,N,D] contiguous, so dout[b, h] is
        # a contiguous [N,D] block and a chunk of it is a straight DMA. Slicing
        # [:, :, s0:e0] instead spans all heads and is non-contiguous, which
        # makes .to(device) materialise a host-side contiguous copy first --
        # measured 2195 ms at N=262144 versus the transfer alone.
        for bi in range(B):
            for hi in range(H):
                dsrc, osrc = dout[bi, hi], out[bi, hi]
                for s0 in range(0, N, chunk):
                    e0 = min(s0 + chunk, N)
                    a = dsrc[s0:e0].to(dev, non_blocking=True).float()
                    bb = osrc[s0:e0].to(dev, non_blocking=True).float()
                    dps[bi, hi, s0:e0] = (a * bb).sum(-1).cpu()
                    del a, bb
        return dps

    for s0 in range(0, N, chunk):
        e0 = min(s0 + chunk, N)
        dps[:, :, s0:e0] = (dout[:, :, s0:e0].float()
                            * out[:, :, s0:e0].float()).sum(-1)
    return dps


_WARNED_HEAD_MAJOR = [False]


def _warn_head_major(N, H, D, itemsize):
    """Once per process: host tensors stored head-major cost a full token-major copy."""
    if _WARNED_HEAD_MAJOR[0]:
        return
    _WARNED_HEAD_MAJOR[0] = True
    import warnings
    gb = 3 * N * H * D * itemsize / 2**30
    warnings.warn(
        f"Stream-CQSA: host-resident Q/K/V are stored head-major ([B,H,N,D] contiguous); streaming needs them "
        f"token-major, so a {gb:.1f} GiB copy is made per call. Keep host tensors token-major "
        f"(x.transpose(1,2).contiguous().transpose(1,2)) to avoid it.", RuntimeWarning, stacklevel=3)


def _stream_cqsa_backward_host(
    q, k, v, dout, out, lse, tasks, *,
    B, H, N, D, device, scale, causal, max_parallel, cpu_threads, bwd_fn,
    trace=None, accumulate_on_gpu=False, progress=None,
):
    """
    Host-resident backward. Q/K/V/dO/O/lse stay in CPU memory and each
    subsequence is gathered on the host, streamed to the device, and
    differentiated there.

    Where the dQ/dK/dV accumulators live is a separate choice, and it is the
    backward's analogue of the forward's ``accumulate_on_gpu``. They are three
    fp32 ``[B, N, H, D]`` buffers, so keeping them on the device costs
    ``12 * N * H * D`` bytes that no depth of decomposition reduces -- the same
    shape of O(N) device term the forward's accumulator has, and three times the
    size. On the host they cost a D2H copy and a scatter per subproblem instead,
    which the pipeline overlaps with the next subproblem's compute.

    Until this was exposed the two were welded together: streaming the inputs
    forced host accumulation, so there was no way to ask for the fast
    configuration, and a run labelled "acc=GPU" was in fact accumulating on the
    host. That made the two settings indistinguishable in every measurement.

    The accumulators are why this matters more for the backward than for the
    forward. The forward keeps one ``(acc, l, m)`` set; the backward keeps three
    full ``[B, N, H, D]`` fp32 tensors, which for fp16 inputs is 6x the size of
    Q alone. Device-resident, that term alone dominates and OOMs at exactly the
    N where the streamed forward still succeeds.
    """
    dtype = q.dtype
    # One-time transpose to token-major, so every per-subproblem gather is a
    # contiguous index_select on dim 1. Gathering from the [B,H,N,D] originals
    # instead needs a transpose().contiguous() per subproblem -- measured 56x
    # slower in the forward for exactly this reason.
    # Tried and rejected: allocating these pinned so each run could DMA straight
    # to the device, skipping the pinned staging buffer. It needs per-slot
    # device landing buffers sized to the largest subproblem, which doubled the
    # streamed path's device workspace (157 -> 337 MiB at N=65536, 626 -> 1346
    # at N=262144) for no measurable time win -- the pinned setup cost for five
    # full-size tensors offsets what the per-subproblem copy saves. Doubling
    # device memory is precisely the wrong trade for this path.
    q_tm = q.transpose(1, 2).contiguous()
    k_tm = k.transpose(1, 2).contiguous()
    v_tm = v.transpose(1, 2).contiguous()
    do_tm = dout.transpose(1, 2).contiguous().to(dtype)
    lse_c = lse.contiguous()                      # [B, H, N], gathered on dim 2
    # O is not staged or transferred at all: it exists only to form
    # rowsum(dO * O), which is global per token, so it is computed once here and
    # gathered per subproblem alongside the lse. Saves 1/5 of the H2D bytes,
    # 1/5 of the host gather, and one full-size token-major host copy.
    dps_c = (_global_dpsum(dout, out).contiguous()
             if _BWD_USE_DPSUM else None)          # [B, H, N]
    o_tm = None if _BWD_USE_DPSUM else out.transpose(1, 2).contiguous().to(dtype)

    acc_dev = device if accumulate_on_gpu else None
    dq = torch.zeros((B, N, H, D), dtype=torch.float32, device=acc_dev)
    dk = torch.zeros((B, N, H, D), dtype=torch.float32, device=acc_dev)
    dv = torch.zeros((B, N, H, D), dtype=torch.float32, device=acc_dev)

    if max_parallel is None:
        n_par = choose_parallelism(
            tasks, device=device, B=B, H=H, D=D,
            host_streaming=True, itemsize=q.element_size(),
        )
        # Each subproblem here carries 5 streamed inputs and 3 returned
        # gradients, versus 3 inputs and 1 output in the forward. Halve the
        # forward's choice so the device working set stays comparable.
        n_par = max(1, n_par // 2)
    else:
        n_par = max(1, int(max_parallel))
    n_par = min(n_par, len(tasks))

    streams = [torch.cuda.Stream(device=device) for _ in range(n_par)]
    # Same entry-side ordering as the forward: a new stream carries no
    # dependency on the caller's outstanding work, so it must be told to wait.
    _caller = torch.cuda.current_stream(device)
    for _s in streams:
        _s.wait_stream(_caller)
    max_L = max((int(t.local_size) for t in tasks), default=1)
    max_el = B * max_L * H * D
    # Pinned staging: 5 in (q,k,v,do,o) + 3 out (dq,dk,dv) + lse, per slot.
    stage_in = [
        tuple(torch.empty(max_el, dtype=dtype, pin_memory=True)
              for _ in range(4 if _BWD_USE_DPSUM else 5))
        for _ in range(n_par)
    ]
    stage_out = [
        tuple(torch.empty(max_el, dtype=dtype, pin_memory=True) for _ in range(3))
        for _ in range(n_par)
    ]
    stage_lse = [
        torch.empty(B * H * max_L, dtype=torch.float32, pin_memory=True)
        for _ in range(n_par)
    ]
    stage_dps = [
        torch.empty(B * H * max_L, dtype=torch.float32, pin_memory=True)
        for _ in range(n_par)
    ]
    # (event, token_ids, L) for a slot whose gradients are still in flight.
    slot_state: list[tuple | None] = [None] * n_par
    slot_task: list[Any] = [None] * n_par
    slot_scatter: list[Any] = [None] * n_par
    # torch's CPU kernels release the GIL, so a worker thread genuinely overlaps
    # the host scatter with device work rather than merely interleaving it.
    scatter_pool = (ThreadPoolExecutor(max_workers=1)
                    if (_SCATTER_ASYNC and len(tasks) > 1) else None)

    tr = trace if (trace is not None and trace.enabled) else None
    # GPU spans are async, so a host timer around them measures launch time.
    # Event-time the device work and host-time only the genuinely CPU stages
    # (the host gather and the index_add scatter).
    gpu_events: list[tuple[int, torch.cuda.Event, torch.cuda.Event]] = []

    def _scatter(slot: int, idx_cpu, Lloc: int, runs) -> tuple[float, float]:
        """Add one slot's gradients into the host accumulators."""
        if accumulate_on_gpu:
            return 0.0, 0.0          # already added on the device, in-line
        n_el = B * Lloc * H * D
        for acc, buf in zip((dq, dk, dv), stage_out[slot]):
            # The pair sets partition, so contributions simply add (algo:bwd
            # lines 17-19). Accumulate in fp32 regardless of input dtype.
            #
            # add_ takes the fp16/bf16 source directly: the in-place add
            # promotes to the fp32 accumulator's dtype, which is bit-identical
            # to .float() but skips materialising a full-size fp32 temporary
            # (measured 1.86x at L=12036, break-even once the copy is large
            # enough to be bandwidth-bound anyway).
            src = buf[:n_el].view(B, Lloc, H, D)
            if runs:
                for off, start, ln in runs:
                    acc[:, start:start + ln].add_(src[:, off:off + ln])
            else:
                acc.index_add_(1, idx_cpu, src)
        return 0.0, 0.0

    def flush(slot: int) -> None:
        """
        Retire one slot: wait for its D2H, then scatter into the accumulators.

        The scatter is handed to a single background worker so it overlaps the
        next subproblem's GPU work instead of stalling the issue loop. It was
        measured at 1685 ms against 5115 ms of compute at N=262144 -- entirely
        hideable, and the same structural fix the forward needed for its merge.
        One worker, not a pool: the accumulators are shared mutable state and
        concurrent adds to overlapping token ranges would race. Ordering among
        scatters is irrelevant (addition commutes); what matters is that only
        one runs at a time, and that a slot is not refilled until its own
        scatter has consumed the staging buffer.
        """
        st = slot_state[slot]
        if st is None:
            return
        ev, idx_cpu, Lloc = st[0], st[1], st[2]
        runs = st[3] if len(st) > 3 else None
        pending = st[4] if len(st) > 4 else None
        if pending is not None:
            pending.result()                      # previous scatter for this slot
        t_w0 = time.perf_counter()
        ev.synchronize()          # the D2H copy for this slot has landed
        t_w1 = time.perf_counter()
        if scatter_pool is not None:
            fut = scatter_pool.submit(_scatter, slot, idx_cpu, Lloc, runs)
        else:
            _scatter(slot, idx_cpu, Lloc, runs)
            fut = None
        if tr is not None:
            task = slot_task[slot]
            tr.record(stage="wait", task=task, start=t_w0, end=t_w1)
            tr.record(stage="scatter", task=task, start=t_w1,
                      end=time.perf_counter())
        slot_state[slot] = None
        slot_scatter[slot] = fut

    with _cpu_threads(cpu_threads):
        for i, task in enumerate(tasks):
            if progress is not None and i > 0:
                progress.update(1)
            slot = i % n_par
            # Backpressure: reusing a slot means its previous gradients must be
            # scattered first. This bounds in-flight work to n_par working sets
            # rather than leaving it to the caching allocator.
            flush(slot)
            # ...and the scatter must have finished reading stage_out[slot]
            # before this iteration overwrites it.
            if slot_scatter[slot] is not None:
                slot_scatter[slot].result()
                slot_scatter[slot] = None

            idx_cpu = task.token_ids
            Lloc = int(task.local_size)
            n_el = B * Lloc * H * D

            def mk():
                return (torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True))

            with torch.cuda.stream(streams[slot]):
                t_g0 = time.perf_counter()
                views = [
                    b[:n_el].view(B, Lloc, H, D) for b in stage_in[slot]
                ]   # q, k, v, dO -- O is not staged (see dps_c above)
                lv = stage_lse[slot][: B * H * Lloc].view(B, H, Lloc)
                dv_ = stage_dps[slot][: B * H * Lloc].view(B, H, Lloc)
                runs = task.extra.get("runs")
                if runs is None:
                    runs = _contiguous_runs(idx_cpu)
                    task.extra["runs"] = runs
                srcs = ((q_tm, k_tm, v_tm, do_tm) if _BWD_USE_DPSUM
                        else (q_tm, k_tm, v_tm, do_tm, o_tm))
                if runs:
                    for src, dst in zip(srcs, views):
                        for off, st, ln in runs:
                            dst[:, off:off + ln] = src[:, st:st + ln]
                    for off, st, ln in runs:
                        lv[:, :, off:off + ln] = lse_c[:, :, st:st + ln]
                        if _BWD_USE_DPSUM:
                            dv_[:, :, off:off + ln] = dps_c[:, :, st:st + ln]
                else:
                    for src, dst in zip(srcs, views):
                        torch.index_select(src, 1, idx_cpu, out=dst)
                    torch.index_select(lse_c, 2, idx_cpu, out=lv)
                    if _BWD_USE_DPSUM:
                        torch.index_select(dps_c, 2, idx_cpu, out=dv_)
                t_g1 = time.perf_counter()

                e_h2d = mk()
                if tr is not None:
                    e_h2d[0].record()
                moved = [x.to(device, non_blocking=True) for x in views]
                q_i, k_i, v_i, do_i = moved[:4]
                o_i = None if _BWD_USE_DPSUM else moved[4]
                lse_i = lv.to(device, non_blocking=True)
                dps_i = dv_.to(device, non_blocking=True) if _BWD_USE_DPSUM else None
                bits = task.group_bits.to(device, non_blocking=True)
                if tr is not None:
                    e_h2d[1].record()

                e_cmp = mk()
                if tr is not None:
                    e_cmp[0].record()
                dq_i, dk_i, dv_i = bwd_fn(
                    do_i, q_i, k_i, v_i, o_i, lse_i, bits,
                    softmax_scale=scale, causal=causal,
                    dsoftmax_sum=dps_i,
                )
                if tr is not None:
                    e_cmp[1].record()

                e_d2h = mk()
                if tr is not None:
                    e_d2h[0].record()
                if accumulate_on_gpu:
                    # The accumulators are already here, so the gradients never
                    # leave the device: no D2H, no pinned staging, no scatter
                    # worker. index_add_ promotes the fp16/bf16 source to the
                    # fp32 accumulator exactly as the host add_ does, so the two
                    # settings agree to the bit.
                    idx_dev = idx_cpu.to(device, non_blocking=True)
                    for acc, g in zip((dq, dk, dv), (dq_i, dk_i, dv_i)):
                        acc.index_add_(1, idx_dev, g.to(torch.float32))
                else:
                    for g, buf in zip((dq_i, dk_i, dv_i), stage_out[slot]):
                        buf[:n_el].view(B, Lloc, H, D).copy_(g, non_blocking=True)
                if tr is not None:
                    e_d2h[1].record()
                ev = torch.cuda.Event()
                ev.record()
            if tr is not None:
                tr.record(stage="gather", task=task, start=t_g0, end=t_g1)
                t_gpu = time.perf_counter()
                for stage, evp in (("h2d", e_h2d), ("compute", e_cmp),
                                   ("d2h", e_d2h)):
                    gpu_events.append((len(tr.rows), evp[0], evp[1]))
                    tr.record(stage=stage, task=task, start=t_g1, end=t_gpu)
            slot_state[slot] = (ev, idx_cpu, Lloc, task.extra.get("runs"))
            slot_task[slot] = task

        for slot in range(n_par):
            flush(slot)
        for slot in range(n_par):
            if slot_scatter[slot] is not None:
                slot_scatter[slot].result()
        if scatter_pool is not None:
            scatter_pool.shutdown(wait=True)

    if tr is not None:
        # Resolve device spans now that everything has drained. Host timers
        # around async launches measure launch cost, so stage_totals_ms()
        # prefers cuda_ms wherever it resolved.
        torch.cuda.synchronize(device)
        for row_idx, e0, e1 in gpu_events:
            tr.rows[row_idx]["cuda_ms"] = float(e0.elapsed_time(e1))

    return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2)


def stream_cqsa_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    itr: int | str = "auto",
    causal: bool = False,
    scale: float | None = None,
    sorted_gather: bool = True,
    c: int = 7,
    interest_set: Sequence[int] = (0, 1, 3),
    stream_from_host: bool = False,
    accumulate_on_gpu: bool = False,
    allow_escalation: bool = True,
    max_parallel: int | None = None,
    cpu_threads: int | None = _CPU_MERGE_THREADS_DEFAULT,
    trace: "TraceRecorder | None" = None,
    bwd_info: dict | None = None,
    task_subset: Sequence[int] | None = None,
    verbose: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stream-CQSA exact attention backward. All of q/k/v/dout/out are
    ``[B, H, N, D]``; ``lse`` is ``[B, H, N]`` and must be the **global**
    log-sum-exp from the forward (``StableAccumulator.lse()``).

    Overflow-free by construction: every subproblem receives the global lse, so
    the kernel forms ``p = exp(s - lse) <= 1`` and the standard softmax backward
    is exact on the retained pair set. No ``exp(lse)``, no ``Num``/``Den``.

    This is the global-lse form of Algorithm `algo:bwd`. It is algebraically
    the same backward, but it never materialises ``Num``/``Den``: with
    ``Den = exp(lse)`` the literal form overflows fp32 once ``lse > 88.7``, and
    ``P_i = exp(R_i)`` overflows whenever any retained score exceeds 88.7.
    Substituting ``Num = Den * O`` and folding ``Den`` into ``P`` turns every
    such quantity into ``exp(s - lse) <= 1``, so the identical gradient is
    computed entirely in the representable range.

    With ``stream_from_host=True`` and CPU inputs, Q/K/V/dO/O/lse stay in host
    memory and only one subsequence at a time is resident on the device: the
    loop over ``subseq_entries`` gathers ``idx``, computes, and scatters back,
    so no step needs the full N-token tensors. Device footprint is then
    ``O(n_par * L * H * D)`` rather than ``O(N * H * D)``, and the ``dQ/dK/dV``
    accumulators (3 x fp32, the largest term) live on the host.

    Returns ``(dq, dk, dv)`` in ``[B, H, N, D]``, fp32.
    """
    from .interface import flash_attn_bwd_cqs_global_lse

    B, H, N, D = q.shape
    host_resident = bool(stream_from_host) and q.device.type == "cpu"
    if isinstance(itr, str):
        # next/: the backward plans its own depth (independent of the forward's).
        if itr != "auto":
            raise ValueError(f"itr must be an int or 'auto', got {itr!r}")
        _dev = torch.device("cuda", torch.cuda.current_device()) if host_resident else q.device
        itr, _reason = plan_decomposition_bwd(
            N, B=B, H=H, D=D, itemsize=q.element_size(), device=_dev, c=c, interest_set=tuple(interest_set),
            stream_from_host=host_resident, accumulate_on_gpu=bool(accumulate_on_gpu))
        if bwd_info is not None:
            bwd_info["plan_reason"] = _reason
    if host_resident:
        if not torch.cuda.is_available():
            raise ValueError("stream_from_host=True requires CUDA")
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = q.device
        if device.type != "cuda":
            raise ValueError(
                "stream_cqsa_backward requires CUDA tensors; pass "
                "stream_from_host=True to stream CPU-resident inputs"
            )
    if scale is None:
        scale = float(D) ** -0.5

    tasks = build_tasks_cached(
        N, itr, B=B, H=H, D=D, itemsize=q.element_size(),
        sorted_gather=sorted_gather, pin=True, c=c,
        interest_set=tuple(interest_set),
    )
    # Multi-device: run only these subproblems (by path_idx). The returned
    # dq/dk/dv are then the fp32 partial sums over that subset, and the caller
    # adds the ranks' partials (all_reduce SUM); every task contributes to its
    # own tokens' gradients independently, so the sum over ranks equals the
    # single-device sum over all tasks up to fp32 summation order.
    if task_subset is not None:
        _keep = {int(i) for i in task_subset}
        tasks = [t for t in tasks if int(t.path_idx) in _keep]

    # The backward returns only gradients, so without this its escalations are
    # invisible: a caller reading info from the FORWARD sees the forward's
    # counters and concludes nothing happened. That is how an 8M backward could
    # refine itself to a depth nobody asked for and still be recorded as the
    # depth that was requested.
    if bwd_info is not None:
        bwd_info.update(itr_requested=int(itr), itr_reached=int(itr),
                        depth_escalations=0, oom_retries=0)

    if host_resident:
        # Every operand must be host-resident, or the per-subproblem gather
        # fails deep inside with an opaque cross-device error. Name the
        # offenders instead.
        stray = [
            n for n, t in (("k", k), ("v", v), ("dout", dout),
                           ("out", out), ("lse", lse))
            if t.device.type != "cpu"
        ]
        if stray:
            raise ValueError(
                "stream_from_host=True requires every operand on the CPU; "
                f"{', '.join(stray)} {'is' if len(stray) == 1 else 'are'} on "
                "the device. The forward returns out/lse wherever its "
                "accumulator lived -- call .cpu() on them before passing."
            )
        # The streamed path pre-sizes its pinned staging slots from the planned
        # task list, so a task cannot be refined in flight. Escalation is applied
        # at whole-call granularity instead: the accumulators are allocated
        # inside the call, so a failed attempt leaves nothing to unwind and the
        # retry simply rebuilds the task list one level deeper.
        depth_cap = max_depth_for(N, c)
        depth = int(itr)
        tasks_at_depth = tasks
        while True:
            try:
                progress = Progress(
                    len(tasks_at_depth), enabled=verbose_enabled(verbose), unit="subproblem",
                    banner=describe_call(what="backward", N=N, B=B, H=H, D=D, c=int(c), itr=int(depth),
                                         n_tasks=len(tasks_at_depth), causal=bool(causal), device=str(device),
                                         extra="Q/K/V/dO streamed from host memory"),
                    expected_s=expected_seconds(N=N, B=B, H=H, D=D, itr=int(depth), c=int(c), causal=bool(causal),
                                                direction="bwd", stream_from_host=True))
                res = _stream_cqsa_backward_host(
                    q, k, v, dout, out, lse, tasks_at_depth,
                    B=B, H=H, N=N, D=D, device=device, scale=float(scale),
                    causal=bool(causal), max_parallel=max_parallel,
                    cpu_threads=cpu_threads, bwd_fn=flash_attn_bwd_cqs_global_lse,
                    trace=trace, accumulate_on_gpu=bool(accumulate_on_gpu), progress=progress,
                )
                progress.update(1)
                progress.close(f"{len(tasks_at_depth)} subproblems")
                return res
            except RuntimeError as exc:
                if not is_oom(exc):
                    raise
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
                if not allow_escalation:
                    raise torch.cuda.OutOfMemoryError(
                        f"Stream-CQSA streamed backward: itr={depth} does not fit "
                        f"and escalation is disabled (allow_escalation=False), so "
                        f"this depth genuinely does not fit."
                    ) from exc
                if depth >= depth_cap:
                    raise torch.cuda.OutOfMemoryError(
                        f"Stream-CQSA streamed backward: itr={depth_cap} is the "
                        f"deepest decomposition N={N} admits at c={c}, and the "
                        f"call still does not fit. The O(N*H*D) resident terms "
                        f"dominate and no depth reduces them; reduce N, H or D."
                    ) from exc
                depth += 1
                if bwd_info is not None:
                    bwd_info["depth_escalations"] += 1
                    bwd_info["itr_reached"] = depth
                    bwd_info["oom_retries"] += 1
                tasks_at_depth = build_tasks_cached(
                    N, depth, B=B, H=H, D=D, itemsize=q.element_size(),
                    sorted_gather=sorted_gather, c=c, interest_set=interest_set,
                )
                if task_subset is not None:
                    # A whole-call escalation rebuilds the task list one level
                    # deeper; the children of this rank's shard are exactly the
                    # deeper tasks whose path starts with one of its paths.
                    _mine = {tuple(t.path) for t in tasks}
                    tasks_at_depth = [t for t in tasks_at_depth
                                      if tuple(t.path)[:len(next(iter(_mine)))] in _mine]

    dq = torch.zeros((B, N, H, D), device=device, dtype=torch.float32)
    dk = torch.zeros((B, N, H, D), device=device, dtype=torch.float32)
    dv = torch.zeros((B, N, H, D), device=device, dtype=torch.float32)

    # rowsum(dO * O) is global per token, so compute it once rather than letting
    # each subproblem's preprocess pass rederive it from a gathered O. O is then
    # never gathered at all -- one fewer full-size gather and device tensor per
    # subproblem. Same reasoning as the streamed path.
    dps = _global_dpsum(dout, out) if _BWD_USE_DPSUM else None

    # Set if a subproblem failed after it had begun accumulating. Retrying is
    # sound exactly while no partial contribution is in the buffers.
    nonlocal_dirty = [False]

    def _run_bwd_task(task):
        """Compute one subproblem and commit its contribution.

        Every allocation happens before the three index_add_ calls, so a failure
        leaves the accumulators untouched and the task may be retried or refined
        without double counting.
        """
        idx = task.token_ids.to(device, non_blocking=True)
        bits = task.group_bits.to(device, non_blocking=True)
        # Token-major gathers, matching the kernel's layout.
        q_i = q.index_select(2, idx).transpose(1, 2).contiguous()
        k_i = k.index_select(2, idx).transpose(1, 2).contiguous()
        v_i = v.index_select(2, idx).transpose(1, 2).contiguous()
        do_i = dout.index_select(2, idx).transpose(1, 2).contiguous()
        lse_i = lse.index_select(2, idx).contiguous()      # GLOBAL lse, gathered
        if _BWD_USE_DPSUM:
            o_i, dps_i = None, dps.index_select(2, idx).contiguous()
        else:
            o_i = out.index_select(2, idx).transpose(1, 2).contiguous().to(q.dtype)
            dps_i = None

        dq_i, dk_i, dv_i = flash_attn_bwd_cqs_global_lse(
            do_i, q_i, k_i, v_i, o_i, lse_i, bits,
            softmax_scale=float(scale), causal=bool(causal),
            dsoftmax_sum=dps_i,
        )
        # The pair sets partition, so contributions simply add. Cast all three
        # before accumulating any of them: index_add_ allocates nothing, so an
        # out-of-memory failure then lands before the first accumulation instead
        # of between them. That matters because the recovery re-runs the failed
        # subproblem and the accumulation is additive -- a commit left half
        # applied would be counted twice, silently, in the gradient.
        dq_f, dk_f, dv_f = dq_i.float(), dk_i.float(), dv_i.float()
        del dq_i, dk_i, dv_i
        dq.index_add_(1, idx, dq_f)
        try:
            dk.index_add_(1, idx, dk_f)
            dv.index_add_(1, idx, dv_f)
        except BaseException:
            nonlocal_dirty[0] = True
            raise

    # Same recovery policy as the forward. The backward has no concurrency to
    # surrender, so an out-of-memory failure goes straight to decomposing the
    # offending subproblem a further level and retrying its children.
    depth_cap = max_depth_for(N, c)
    pending = list(tasks)
    retries = 0
    # See the forward: the depth bound is what terminates, this only backstops.
    retry_budget = 64 + 16 * max(1, len(pending)) * max(1, depth_cap)
    progress = Progress(
        len(pending), enabled=verbose_enabled(verbose), unit="subproblem",
        banner=describe_call(what="backward", N=N, B=B, H=H, D=D, c=int(c), itr=int(itr), n_tasks=len(pending),
                             causal=bool(causal), device=str(device),
                             extra=f"gradients accumulated on {'device' if accumulate_on_gpu else 'host'}"),
        expected_s=expected_seconds(N=N, B=B, H=H, D=D, itr=int(itr), c=int(c), causal=bool(causal), direction="bwd"))
    while pending:
        task = pending.pop(0)
        try:
            _run_bwd_task(task)
            progress.update(1)
        except RuntimeError as exc:
            if not is_oom(exc):
                raise
            if nonlocal_dirty[0]:
                raise RuntimeError(
                    "Stream-CQSA backward: ran out of memory while accumulating a "
                    "subproblem's gradient. Part of its contribution is already in "
                    "the buffers, so retrying would double count and recovery is "
                    "unsafe. Re-run at a higher itr."
                ) from exc
            retries += 1
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            if not allow_escalation or int(task.itr) >= depth_cap \
                    or retries > retry_budget:
                raise torch.cuda.OutOfMemoryError(
                    f"Stream-CQSA backward: a subproblem at itr={task.itr} does "
                    f"not fit, and itr={depth_cap} is the deepest decomposition "
                    f"N={N} admits at c={c}. The O(N*H*D) resident terms dominate "
                    f"and no depth reduces them; offload them, or reduce N, H or D."
                ) from exc
            kids = refine_task(
                task, N, B=B, H=H, D=D, itemsize=q.element_size(),
                sorted_gather=sorted_gather, pin=True, c=c,
                interest_set=interest_set, seg_align=0,
            )
            if bwd_info is not None:
                bwd_info["depth_escalations"] += 1
                bwd_info["oom_retries"] = retries
                bwd_info["itr_reached"] = max(bwd_info["itr_reached"],
                                              max(int(t.itr) for t in kids))
            pending[0:0] = kids
            progress.set_total(progress.done + len(pending))

    progress.close(f"{progress.done} subproblems" + (f", {retries} OOM retries" if retries else ""))
    return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2)
