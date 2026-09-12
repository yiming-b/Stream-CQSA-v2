from __future__ import annotations

import math
import threading
import time
from collections import deque
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from itertools import product
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from .cqs_mask import CQS_mask
from .interface import flash_attn_bwd_cqs_group_bits

GIB = 1024**3


def _prep_worker_count(max_parallel: int) -> int:
    return max(1, min(8, int(max_parallel)))


def _prep_prefetch_limit(max_parallel: int) -> int:
    return max(2, int(max_parallel) + 2)


def _nvtx_available() -> bool:
    return hasattr(torch.cuda, "nvtx") and hasattr(torch.cuda.nvtx, "range_push")


@contextmanager
def _nvtx_range(message: str):
    if _nvtx_available():
        try:
            torch.cuda.nvtx.range_push(str(message))
        except Exception:
            yield
            return
        try:
            yield
        finally:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
    else:
        yield


def _snapshot_gpu_mem(stage: str, *, t0_wall: float) -> dict[str, Any]:
    free_b, total_b = torch.cuda.mem_get_info()
    used_b = int(total_b - free_b)
    allocated_b = int(torch.cuda.memory_allocated())
    reserved_b = int(torch.cuda.memory_reserved())
    peak_allocated_b = int(torch.cuda.max_memory_allocated())
    return {
        "t_rel_s": float(time.perf_counter() - t0_wall),
        "stage": str(stage),
        "gpu_used_gib": float(used_b) / GIB,
        "gpu_free_gib": float(free_b) / GIB,
        "gpu_total_gib": float(total_b) / GIB,
        "cuda_allocated_gib": float(allocated_b) / GIB,
        "cuda_reserved_gib": float(reserved_b) / GIB,
        "cuda_peak_allocated_gib": float(peak_allocated_b) / GIB,
    }


def _group_runs_to_group_bits(local_size: int, group_runs: Sequence[Sequence[Sequence[int] | tuple[int, int]]]) -> torch.Tensor:
    bits_np = np.zeros((int(local_size),), dtype=np.int64)
    for bit_id, runs in enumerate(group_runs):
        if bit_id >= 63:
            raise ValueError("Too many unique group-runs for int64 bit encoding.")
        bit = np.int64(1) << np.int64(bit_id)
        for se in runs:
            s, e = int(se[0]), int(se[1])
            if e > s:
                bits_np[s:e] |= bit
    return torch.from_numpy(bits_np)


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


def _local_backward_naive_staged_lowmem(
    *,
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    d_num_i_cpu: torch.Tensor,
    d_den_i_cpu: torch.Tensor,
    group_bits_cpu: torch.Tensor,
    on_cuda: bool,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Stage 1: Q/K (+ bits) -> R_i -> P_i
    if on_cuda:
        q_i = q_local
        k_i = k_local
        group_bits_gpu = group_bits_cpu.to(device=q_i.device, dtype=torch.int64, non_blocking=True)
    else:
        q_i = q_local.to("cuda", non_blocking=True)
        k_i = k_local.to("cuda", non_blocking=True)
        group_bits_gpu = group_bits_cpu.to("cuda", dtype=torch.int64, non_blocking=True)

    q_bhld = q_i.transpose(1, 2).float()
    k_bhld = k_i.transpose(1, 2).float()
    bits_row = group_bits_gpu.view(1, 1, group_bits_gpu.numel(), 1)
    bits_col = group_bits_gpu.view(1, 1, 1, group_bits_gpu.numel())
    M_i = torch.bitwise_and(bits_row, bits_col).ne(0)
    R_i = torch.matmul(q_bhld, k_bhld.transpose(-2, -1)) * float(softmax_scale)
    R_i = R_i.masked_fill(M_i, float("-inf"))
    P_i = torch.exp(R_i)
    P_i = torch.nan_to_num(P_i, nan=0.0, posinf=0.0, neginf=0.0)
    del q_i, k_i, q_bhld, k_bhld, group_bits_gpu, bits_row, bits_col, M_i, R_i

    # Stage 2: dNum_i -> dV_i, release temporary dNum copy.
    d_num_i = d_num_i_cpu.to("cuda", non_blocking=True)
    d_num_bhld = d_num_i.transpose(1, 2).float()
    d_v_bhld = torch.matmul(P_i.transpose(-2, -1), d_num_bhld)
    del d_num_i, d_num_bhld

    # Stage 3: dPi from (dNum_i, dDen_i, V_i), then dR_i = dPi * P_i.
    d_num_i = d_num_i_cpu.to("cuda", non_blocking=True)
    d_den_i = d_den_i_cpu.to("cuda", non_blocking=True)
    if on_cuda:
        v_i = v_local
    else:
        v_i = v_local.to("cuda", non_blocking=True)
    d_num_bhld = d_num_i.transpose(1, 2).float()
    d_den_bhl = d_den_i.float()
    v_bhld = v_i.transpose(1, 2).float()
    dP_i = torch.matmul(d_num_bhld, v_bhld.transpose(-2, -1))
    dP_i = dP_i + d_den_bhl.unsqueeze(-1)
    dR_i = dP_i * P_i
    del d_num_i, d_den_i, v_i, d_num_bhld, d_den_bhl, v_bhld, dP_i, P_i

    # Stage 4: dQ_i from dR_i and K_i (K_i can be reloaded from CPU).
    if on_cuda:
        k_i = k_local
    else:
        k_i = k_local.to("cuda", non_blocking=True)
    k_bhld = k_i.transpose(1, 2).float()
    d_q_bhld = torch.matmul(dR_i, k_bhld)
    d_q_bhld.mul_(float(softmax_scale))
    del k_i, k_bhld

    # Stage 5: dK_i from dR_i and Q_i (Q_i can be reloaded from CPU).
    if on_cuda:
        q_i = q_local
    else:
        q_i = q_local.to("cuda", non_blocking=True)
    q_bhld = q_i.transpose(1, 2).float()
    d_k_bhld = torch.matmul(dR_i.transpose(-2, -1), q_bhld)
    d_k_bhld.mul_(float(softmax_scale))
    del q_i, q_bhld, dR_i

    d_q_i = torch.nan_to_num(d_q_bhld.transpose(1, 2).contiguous(), nan=0.0, posinf=0.0, neginf=0.0)
    d_k_i = torch.nan_to_num(d_k_bhld.transpose(1, 2).contiguous(), nan=0.0, posinf=0.0, neginf=0.0)
    d_v_i = torch.nan_to_num(d_v_bhld.transpose(1, 2).contiguous(), nan=0.0, posinf=0.0, neginf=0.0)
    del d_q_bhld, d_k_bhld, d_v_bhld
    return d_q_i, d_k_i, d_v_i


def _path_seed(seed: int, path: tuple[int, ...]) -> int:
    s = int(seed) & ((1 << 64) - 1)
    for x in path:
        s = (s * 6364136223846793005 + (1 + int(x))) & ((1 << 64) - 1)
    return int(s & 0x7FFFFFFF)


def _make_qkv_for_path(
    *,
    source_q: torch.Tensor | None,
    source_k: torch.Tensor | None,
    source_v: torch.Tensor | None,
    token_ids_cpu: torch.Tensor,
    local_size: int,
    path: tuple[int, ...],
    B: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    qkv_generation_mode: str,
    seed: int,
    input_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    if source_q is not None:
        assert source_k is not None and source_v is not None
        if source_q.device.type == "cuda":
            idx_dev = token_ids_cpu.to(source_q.device, non_blocking=True)
            q_dev = source_q.index_select(2, idx_dev).permute(0, 2, 1, 3).contiguous()
            k_dev = source_k.index_select(2, idx_dev).permute(0, 2, 1, 3).contiguous()
            v_dev = source_v.index_select(2, idx_dev).permute(0, 2, 1, 3).contiguous()
            return q_dev, k_dev, v_dev, True

        q_cpu = source_q.index_select(2, token_ids_cpu).permute(0, 2, 1, 3).contiguous().pin_memory()
        k_cpu = source_k.index_select(2, token_ids_cpu).permute(0, 2, 1, 3).contiguous().pin_memory()
        v_cpu = source_v.index_select(2, token_ids_cpu).permute(0, 2, 1, 3).contiguous().pin_memory()
        return q_cpu, k_cpu, v_cpu, False

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(_path_seed(seed, path)))
    shape = (int(B), int(local_size), int(H), int(D))
    q_cpu = torch.empty(shape, dtype=dtype, pin_memory=True)
    k_cpu = torch.empty(shape, dtype=dtype, pin_memory=True)
    v_cpu = torch.empty(shape, dtype=dtype, pin_memory=True)

    mode = str(qkv_generation_mode).strip().lower()
    if mode == "constant":
        q_const = float(torch.rand((), generator=gen).item())
        k_const = float(torch.rand((), generator=gen).item())
        v_const = float(torch.rand((), generator=gen).item())
        q_cpu.fill_(q_const)
        k_cpu.fill_(k_const)
        v_cpu.fill_(v_const)
    elif mode == "random":
        q_cpu.normal_(mean=0.0, std=float(input_std), generator=gen)
        k_cpu.normal_(mean=0.0, std=float(input_std), generator=gen)
        v_cpu.normal_(mean=0.0, std=float(input_std), generator=gen)
    else:
        raise ValueError(f"qkv_generation_mode must be 'constant' or 'random', got {qkv_generation_mode}")
    return q_cpu, k_cpu, v_cpu, False


def _ensure_timing_dict(timing_ms: dict[str, float] | None) -> dict[str, float]:
    if timing_ms is None:
        timing_ms = {}
    defaults = {
        "fwd_mask_gen": 0.0,
        "fwd_qkv_cpu_gen": 0.0,
        "fwd_h2d": 0.0,
        "fwd_compute": 0.0,
        "fwd_sync": 0.0,
        "fwd_d2h": 0.0,
        "fwd_merge": 0.0,
        "cpu_dO_gen": 0.0,
        "cpu_dNum_dDen": 0.0,
        "bwd_mask_gen": 0.0,
        "bwd_qkv_cpu_gen": 0.0,
        "bwd_h2d": 0.0,
        "bwd_compute": 0.0,
        "bwd_sync": 0.0,
        "bwd_d2h": 0.0,
        "bwd_merge": 0.0,
    }
    for k, v in defaults.items():
        timing_ms.setdefault(k, v)
    return timing_ms


def _log_mem(
    timeline_rows: list[dict[str, Any]],
    *,
    t0_wall: float,
    stage: str,
    phase: str,
    paths_done: int,
    active_subseq: int,
    round_idx: int = -1,
    path_idx: int = -1,
) -> None:
    row = _snapshot_gpu_mem(stage=f"{phase}:{stage}", t0_wall=t0_wall)
    row["phase"] = str(phase)
    row["paths_done"] = int(paths_done)
    row["active_subseq"] = int(active_subseq)
    row["round"] = int(round_idx)
    row["path_idx"] = int(path_idx)
    timeline_rows.append(row)


def compute_paths(
    *,
    c: int,
    itr_max: int,
    max_parallel_subseq: int,
    max_rounds: int | None,
) -> tuple[list[tuple[int, ...]], int, int]:
    all_paths = [tuple()] if itr_max <= 0 else list(product(range(int(c)), repeat=int(itr_max)))
    num_paths_total = int(len(all_paths))
    if max_rounds is None:
        target_paths = int(num_paths_total)
    else:
        max_rounds_i = int(max_rounds)
        if max_rounds_i <= 0:
            raise ValueError("max_rounds must be > 0 when set.")
        target_paths = int(min(int(num_paths_total), int(max_rounds_i) * int(max_parallel_subseq)))
    return all_paths, num_paths_total, target_paths


def build_precomputed_path_cache(
    *,
    N: int,
    c: int,
    interest_set: Sequence[int],
    itr_max: int,
    paths: Sequence[tuple[int, ...]],
    bulk_limit: int = 4096,
) -> tuple[dict[tuple[int, ...], dict[str, Any]], dict[str, Any]]:
    cache: dict[tuple[int, ...], dict[str, Any]] = {}
    stats = {
        "mode": "none",
        "paths": 0,
        "bytes": 0,
        "wall_ms": 0.0,
    }
    norm_paths = [tuple(int(x) for x in path) for path in paths]
    if len(norm_paths) == 0:
        return cache, stats

    mask_engine = CQS_mask(interest_set=tuple(int(x) for x in interest_set), c=int(c))
    t0 = time.perf_counter()
    total_bytes = 0

    total_possible = int(1 if int(itr_max) <= 0 else int(c) ** int(itr_max))
    use_bulk = int(total_possible) <= int(bulk_limit) and len(norm_paths) == int(total_possible)

    if use_bulk:
        mask_all = mask_engine.gen_mask(
            N=int(N),
            num_itr=int(itr_max),
            quorum_idx=None,
            interest_set=tuple(int(x) for x in interest_set),
            c=int(c),
            include_trace=False,
        )
        masks = mask_all.get("masks", {})
        for path in norm_paths:
            mask_one = masks.get(tuple(path))
            if mask_one is None:
                continue
            token_ids_np = np.asarray(mask_one["token_ids"], dtype=np.int64)
            local_size = int(mask_one["local_size"])
            token_ids_cpu = torch.from_numpy(token_ids_np.astype(np.int64, copy=False)).to(torch.long)
            group_bits_cpu = _group_runs_to_group_bits(local_size, mask_one["group_runs"]).to(torch.long)
            lightweight_mask = {
                "local_size": int(local_size),
                "group_bits_cpu": group_bits_cpu,
                "token_ids_cpu": token_ids_cpu,
                "token_ids": token_ids_cpu.tolist(),
            }
            cache[tuple(path)] = {
                "path": tuple(path),
                "local_size": int(local_size),
                "token_ids_cpu": token_ids_cpu,
                "group_bits_cpu": group_bits_cpu,
                "mask_one": lightweight_mask,
            }
            total_bytes += int(token_ids_cpu.numel() * token_ids_cpu.element_size())
            total_bytes += int(group_bits_cpu.numel() * group_bits_cpu.element_size())
        stats["mode"] = "bulk_all"
    else:
        for path in norm_paths:
            mask_one = mask_engine.gen_mask(
                N=int(N),
                num_itr=int(itr_max),
                quorum_idx=list(path),
                interest_set=tuple(int(x) for x in interest_set),
                c=int(c),
                include_trace=False,
            )
            token_ids_np = np.asarray(mask_one["token_ids"], dtype=np.int64)
            local_size = int(mask_one["local_size"])
            token_ids_cpu = torch.from_numpy(token_ids_np.astype(np.int64, copy=False)).to(torch.long)
            group_bits_cpu = _group_runs_to_group_bits(local_size, mask_one["group_runs"]).to(torch.long)
            lightweight_mask = {
                "local_size": int(local_size),
                "group_bits_cpu": group_bits_cpu,
                "token_ids_cpu": token_ids_cpu,
                "token_ids": token_ids_cpu.tolist(),
            }
            cache[tuple(path)] = {
                "path": tuple(path),
                "local_size": int(local_size),
                "token_ids_cpu": token_ids_cpu,
                "group_bits_cpu": group_bits_cpu,
                "mask_one": lightweight_mask,
            }
            total_bytes += int(token_ids_cpu.numel() * token_ids_cpu.element_size())
            total_bytes += int(group_bits_cpu.numel() * group_bits_cpu.element_size())
        stats["mode"] = "per_path"

    stats["paths"] = int(len(cache))
    stats["bytes"] = int(total_bytes)
    stats["wall_ms"] = float((time.perf_counter() - t0) * 1000.0)
    return cache, stats


def streamed_forward_phase(
    *,
    N: int,
    D: int,
    B: int,
    H: int,
    c: int,
    interest_set: Sequence[int],
    itr_max: int,
    max_parallel_subseq: int,
    schedule_mode: str,
    dtype: torch.dtype,
    subseq_attention_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    source_q: torch.Tensor | None = None,
    source_k: torch.Tensor | None = None,
    source_v: torch.Tensor | None = None,
    qkv_generation_mode: str = "random",
    seed: int = 123,
    input_std: float = 0.1,
    max_rounds: int | None = None,
    t0_wall: float | None = None,
    timeline_rows: list[dict[str, Any]] | None = None,
    timing_ms: dict[str, float] | None = None,
    subseq_qkv_provider: Callable[[int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    precomputed_path_cache: Mapping[tuple[int, ...], dict[str, Any]] | None = None,
    preload_subseq: bool = False,
) -> dict[str, Any]:
    with _nvtx_range(
        f"cqsa:fwd:itr={int(itr_max)}:max_parallel={int(max_parallel_subseq)}:schedule={str(schedule_mode).strip().lower()}"
    ):
        if t0_wall is None:
            t0_wall = float(time.perf_counter())
        if timeline_rows is None:
            timeline_rows = []
        timing_ms = _ensure_timing_dict(timing_ms)

        schedule_mode_n = str(schedule_mode).strip().lower()
        if schedule_mode_n not in ("event", "round"):
            raise ValueError("schedule_mode must be one of: event, round")

        all_paths, num_paths_total, target_paths = compute_paths(
            c=int(c),
            itr_max=int(itr_max),
            max_parallel_subseq=int(max_parallel_subseq),
            max_rounds=max_rounds,
        )
        if target_paths <= 0:
            raise RuntimeError("No paths selected for streamed forward run.")

        mask_engine = CQS_mask(interest_set=tuple(int(x) for x in interest_set), c=int(c))
        softmax_scale = float(1.0 / math.sqrt(float(D)))

        num_global_cpu = torch.zeros((int(B), int(N), int(H), int(D)), dtype=torch.float32)
        den_global_cpu = torch.zeros((int(B), int(H), int(N)), dtype=torch.float32)
        processed_paths: list[tuple[int, ...]] = []

        streams = [torch.cuda.Stream(device="cuda") for _ in range(int(max_parallel_subseq))]
        inflight: list[dict[str, Any] | None] = [None for _ in range(int(max_parallel_subseq))]
        pending_merge: deque[dict[str, Any]] = deque()
        prep_workers = _prep_worker_count(int(max_parallel_subseq))
        eager_prepare_all = bool(
            preload_subseq
            or (
                precomputed_path_cache is not None
                and subseq_qkv_provider is not None
                and source_q is None
            )
        )
        prep_prefetch = int(target_paths) if eager_prepare_all else _prep_prefetch_limit(int(max_parallel_subseq))
        prep_executor = ThreadPoolExecutor(max_workers=prep_workers, thread_name_prefix="cqsa_fwd_prep")
        prep_pending: dict[Future[Any], int] = {}
        prep_ready: deque[dict[str, Any]] = deque()
        prep_submit_idx = 0
        paths_done = 0
        paths_skipped = 0
        progress_bar = None

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        _log_mem(timeline_rows, t0_wall=t0_wall, stage="fwd_start", phase="fwd", paths_done=0, active_subseq=0)

        def _ensure_progress_bar() -> None:
            nonlocal progress_bar
            if progress_bar is not None:
                return
            if target_paths <= 0:
                return
            if tqdm is not None:
                print(
                    f"[cqsa_stream] phase=fwd progress start: total_subseq={int(target_paths)}",
                    flush=True,
                )
                progress_bar = tqdm(
                    total=int(target_paths),
                    initial=int(paths_done),
                    desc="fwd",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                    leave=True,
                    dynamic_ncols=True,
                )
            else:
                print("[cqsa_stream] tqdm not installed; forward ETA progress bar disabled.", flush=True)

        def _prepare_path(path_idx: int, path: tuple[int, ...]) -> dict[str, Any]:
            with _nvtx_range(f"cqsa:fwd:path={int(path_idx)}:prepare"):
                cache_entry = None if precomputed_path_cache is None else precomputed_path_cache.get(tuple(path))
                if cache_entry is None:
                    with _nvtx_range(f"cqsa:fwd:path={int(path_idx)}:mask"):
                        t_mask0 = time.perf_counter()
                        mask_one_raw = mask_engine.gen_mask(
                            N=int(N),
                            num_itr=int(itr_max),
                            quorum_idx=list(path),
                            interest_set=tuple(int(x) for x in interest_set),
                            c=int(c),
                            include_trace=False,
                        )
                        mask_ms = (time.perf_counter() - t_mask0) * 1000.0
                    token_ids_np = np.asarray(mask_one_raw["token_ids"], dtype=np.int64)
                    local_size = int(mask_one_raw["local_size"])
                    if local_size <= 0 or token_ids_np.size == 0:
                        return {
                            "status": "skip",
                            "path": path,
                            "path_idx": int(path_idx),
                            "mask_ms": float(mask_ms),
                            "qkv_ms": 0.0,
                        }
                    token_ids_cpu = torch.from_numpy(token_ids_np.astype(np.int64, copy=False)).to(torch.long)
                    group_bits_cpu = _group_runs_to_group_bits(local_size, mask_one_raw["group_runs"]).to(torch.long)
                    mask_one = {
                        "local_size": int(local_size),
                        "group_bits_cpu": group_bits_cpu,
                        "token_ids_cpu": token_ids_cpu,
                        "token_ids": token_ids_cpu.tolist(),
                    }
                else:
                    mask_ms = 0.0
                    local_size = int(cache_entry["local_size"])
                    token_ids_cpu = cache_entry["token_ids_cpu"]
                    group_bits_cpu = cache_entry["group_bits_cpu"]
                    mask_one = cache_entry["mask_one"]
                    if "token_ids_cpu" not in mask_one:
                        mask_one = dict(mask_one)
                        mask_one["token_ids_cpu"] = token_ids_cpu
                        mask_one["token_ids"] = token_ids_cpu.tolist()
                if local_size <= 0 or int(token_ids_cpu.numel()) == 0:
                    return {
                        "status": "skip",
                        "path": path,
                        "path_idx": int(path_idx),
                        "mask_ms": float(mask_ms),
                        "qkv_ms": 0.0,
                    }

                needs_inline_qkv = bool(source_q is not None and source_q.device.type == "cuda")
                q_local: torch.Tensor | None = None
                k_local: torch.Tensor | None = None
                v_local: torch.Tensor | None = None
                on_cuda = False
                qkv_ms = 0.0

                if not needs_inline_qkv:
                    using_cached_provider = bool(source_q is None and subseq_qkv_provider is not None)
                    with _nvtx_range(f"cqsa:fwd:path={int(path_idx)}:qkv_prepare"):
                        t_qkv0 = time.perf_counter()
                        if using_cached_provider:
                            q_local, k_local, v_local = subseq_qkv_provider(int(local_size))
                            on_cuda = bool(isinstance(q_local, torch.Tensor) and q_local.device.type == "cuda")
                        else:
                            q_local, k_local, v_local, on_cuda = _make_qkv_for_path(
                                source_q=source_q,
                                source_k=source_k,
                                source_v=source_v,
                                token_ids_cpu=token_ids_cpu,
                                local_size=local_size,
                                path=path,
                                B=int(B),
                                H=int(H),
                                D=int(D),
                                dtype=dtype,
                                qkv_generation_mode=qkv_generation_mode,
                                seed=int(seed),
                                input_std=float(input_std),
                            )
                            qkv_ms = (time.perf_counter() - t_qkv0) * 1000.0

                return {
                    "status": "prepared",
                    "path": path,
                    "path_idx": int(path_idx),
                    "local_size": int(local_size),
                    "token_ids_cpu": token_ids_cpu,
                    "mask_one": mask_one,
                    "q_local": q_local,
                    "k_local": k_local,
                    "v_local": v_local,
                    "on_cuda": bool(on_cuda),
                    "needs_inline_qkv": bool(needs_inline_qkv),
                    "mask_ms": float(mask_ms),
                    "qkv_ms": float(qkv_ms),
                    "gpu_preloaded": False,
                }

        def _preload_prepared(prepared: dict[str, Any]) -> dict[str, Any]:
            if str(prepared.get("status", "")) != "prepared":
                return prepared
            if bool(prepared.get("gpu_preloaded", False)):
                return prepared
            path = tuple(int(x) for x in prepared["path"])
            path_idx = int(prepared["path_idx"])
            local_size = int(prepared["local_size"])
            token_ids_cpu = prepared["token_ids_cpu"]
            q_local = prepared.get("q_local", None)
            k_local = prepared.get("k_local", None)
            v_local = prepared.get("v_local", None)
            on_cuda = bool(prepared.get("on_cuda", False))
            mask_one = prepared["mask_one"]

            if bool(prepared.get("needs_inline_qkv", False)):
                with _nvtx_range(f"cqsa:fwd:path={path_idx}:preload_inline_qkv"):
                    t_qkv0 = time.perf_counter()
                    q_local, k_local, v_local, on_cuda = _make_qkv_for_path(
                        source_q=source_q,
                        source_k=source_k,
                        source_v=source_v,
                        token_ids_cpu=token_ids_cpu,
                        local_size=local_size,
                        path=path,
                        B=int(B),
                        H=int(H),
                        D=int(D),
                        dtype=dtype,
                        qkv_generation_mode=qkv_generation_mode,
                        seed=int(seed),
                        input_std=float(input_std),
                    )
                    timing_ms["fwd_qkv_cpu_gen"] += (time.perf_counter() - t_qkv0) * 1000.0

            with _nvtx_range(f"cqsa:fwd:path={path_idx}:preload_h2d"):
                t_h2d0 = time.perf_counter()
                if not on_cuda:
                    q_local = q_local.to("cuda", non_blocking=False)
                    k_local = k_local.to("cuda", non_blocking=False)
                    v_local = v_local.to("cuda", non_blocking=False)
                group_bits_cpu = mask_one.get("group_bits_cpu", None)
                if isinstance(group_bits_cpu, torch.Tensor):
                    if group_bits_cpu.device.type != "cuda" or group_bits_cpu.dtype != torch.int64:
                        group_bits_gpu = group_bits_cpu.to("cuda", dtype=torch.int64, non_blocking=False)
                    else:
                        group_bits_gpu = group_bits_cpu
                else:
                    group_bits_gpu = group_bits_cpu
                timing_ms["fwd_h2d"] += (time.perf_counter() - t_h2d0) * 1000.0

            prepared["q_local"] = q_local
            prepared["k_local"] = k_local
            prepared["v_local"] = v_local
            prepared["on_cuda"] = True
            prepared["needs_inline_qkv"] = False
            prepared["gpu_preloaded"] = True
            prepared["mask_one"] = {
                "local_size": int(local_size),
                "group_bits_cpu": group_bits_gpu,
                "token_ids_cpu": token_ids_cpu,
                "token_ids": token_ids_cpu.tolist(),
            }
            prepared["group_bits_cpu"] = group_bits_gpu
            _log_mem(
                timeline_rows,
                t0_wall=t0_wall,
                stage="subseq_preloaded",
                phase="fwd",
                paths_done=paths_done,
                active_subseq=sum(1 for t in inflight if t is not None),
                round_idx=max(0, int(paths_done // max(1, max_parallel_subseq))),
                path_idx=path_idx,
            )
            return prepared

        def _submit_more_prepared() -> None:
            nonlocal prep_submit_idx
            while prep_submit_idx < target_paths:
                queued = len(prep_pending) + len(prep_ready) + sum(1 for t in inflight if t is not None)
                if queued >= prep_prefetch:
                    break
                path_idx = int(prep_submit_idx)
                path = all_paths[path_idx]
                fut = prep_executor.submit(_prepare_path, path_idx, path)
                prep_pending[fut] = path_idx
                prep_submit_idx += 1

        def _harvest_prepared(timeout_s: float = 0.0) -> bool:
            if len(prep_pending) == 0:
                return False
            done, _ = wait(list(prep_pending.keys()), timeout=float(timeout_s), return_when=FIRST_COMPLETED)
            if len(done) == 0:
                return False
            for fut in done:
                prep_pending.pop(fut, None)
                prepared = fut.result()
                timing_ms["fwd_mask_gen"] += float(prepared.get("mask_ms", 0.0))
                timing_ms["fwd_qkv_cpu_gen"] += float(prepared.get("qkv_ms", 0.0))
                prep_ready.append(prepared)
                _log_mem(
                    timeline_rows,
                    t0_wall=t0_wall,
                    stage="subseq_prepared",
                    phase="fwd",
                    paths_done=paths_done,
                    active_subseq=sum(1 for t in inflight if t is not None),
                    round_idx=max(0, int(paths_done // max(1, max_parallel_subseq))),
                    path_idx=int(prepared.get("path_idx", -1)),
                )
            return True

        def _pop_ready_prepared() -> dict[str, Any] | None:
            while len(prep_ready) > 0:
                prepared = prep_ready.popleft()
                if int(prepared.get("path_idx", -1)) >= int(target_paths):
                    continue
                return prepared
            return None

        def _account_skip(path_idx: int) -> None:
            nonlocal paths_skipped
            paths_skipped += 1
            if progress_bar is not None:
                progress_bar.update(1)
            _log_mem(
                timeline_rows,
                t0_wall=t0_wall,
                stage="subseq_skip",
                phase="fwd",
                paths_done=paths_done,
                active_subseq=sum(1 for t in inflight if t is not None),
                round_idx=max(0, int(paths_done // max(1, max_parallel_subseq))),
                path_idx=int(path_idx),
            )

        def _launch_prepared(prepared: dict[str, Any], slot_idx: int) -> str:
            path = tuple(int(x) for x in prepared["path"])
            path_idx = int(prepared["path_idx"])
            token_ids_cpu = prepared["token_ids_cpu"]
            mask_one = prepared["mask_one"]
            q_local = prepared.get("q_local", None)
            k_local = prepared.get("k_local", None)
            v_local = prepared.get("v_local", None)
            on_cuda = bool(prepared.get("on_cuda", False))
            ev_h2d_start = torch.cuda.Event(enable_timing=True)
            ev_h2d_end = torch.cuda.Event(enable_timing=True)
            ev_compute_start = torch.cuda.Event(enable_timing=True)
            ev_compute_end = torch.cuda.Event(enable_timing=True)
            ev_d2h_start = torch.cuda.Event(enable_timing=True)
            ev_d2h_end = torch.cuda.Event(enable_timing=True)

            s = streams[slot_idx]
            try:
                with _nvtx_range(f"cqsa:fwd:path={path_idx}:slot={int(slot_idx)}:launch"):
                    with torch.cuda.stream(s):
                        if bool(prepared.get("needs_inline_qkv", False)):
                            # For CUDA source Q/K/V, gather on the launch stream to avoid
                            # default-stream gather + non-default compute races.
                            with _nvtx_range(f"cqsa:fwd:path={path_idx}:inline_qkv_prepare"):
                                t_qkv0 = time.perf_counter()
                                q_local, k_local, v_local, on_cuda = _make_qkv_for_path(
                                    source_q=source_q,
                                    source_k=source_k,
                                    source_v=source_v,
                                    token_ids_cpu=token_ids_cpu,
                                    local_size=int(prepared["local_size"]),
                                    path=path,
                                    B=int(B),
                                    H=int(H),
                                    D=int(D),
                                    dtype=dtype,
                                    qkv_generation_mode=qkv_generation_mode,
                                    seed=int(seed),
                                    input_std=float(input_std),
                                )
                                timing_ms["fwd_qkv_cpu_gen"] += (time.perf_counter() - t_qkv0) * 1000.0
                        with _nvtx_range(f"cqsa:fwd:path={path_idx}:slot={int(slot_idx)}:h2d"):
                            ev_h2d_start.record()
                            if on_cuda:
                                q_i = q_local
                                k_i = k_local
                                v_i = v_local
                            else:
                                q_i = q_local.to("cuda", non_blocking=True)
                                k_i = k_local.to("cuda", non_blocking=True)
                                v_i = v_local.to("cuda", non_blocking=True)
                            ev_h2d_end.record()

                        with _nvtx_range(f"cqsa:fwd:path={path_idx}:slot={int(slot_idx)}:compute"):
                            ev_compute_start.record()
                            num_i, den_i = subseq_attention_fn(
                                q=q_i,
                                k=k_i,
                                v=v_i,
                                cqs_mask=mask_one,
                                softmax_scale=float(softmax_scale),
                            )
                            if not isinstance(num_i, torch.Tensor) or not isinstance(den_i, torch.Tensor):
                                raise TypeError("subseq_attention_fn must return (Num_i, Den_i).")
                            num_i = torch.nan_to_num(num_i.float(), nan=0.0, posinf=0.0, neginf=0.0)
                            den_i = torch.nan_to_num(den_i.float(), nan=0.0, posinf=0.0, neginf=0.0)
                            ev_compute_end.record()
                        with _nvtx_range(f"cqsa:fwd:path={path_idx}:slot={int(slot_idx)}:d2h"):
                            ev_d2h_start.record()
                            num_i_cpu = torch.empty_like(num_i, device="cpu", pin_memory=True)
                            den_i_cpu = torch.empty_like(den_i, device="cpu", pin_memory=True)
                            num_i_cpu.copy_(num_i, non_blocking=True)
                            den_i_cpu.copy_(den_i, non_blocking=True)
                            ev_d2h_end.record()
                            del num_i, den_i
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    return "oom"
                raise

            inflight[slot_idx] = {
                "path": path,
                "path_idx": path_idx,
                "token_ids_cpu": token_ids_cpu,
                "num_i_cpu": num_i_cpu,
                "den_i_cpu": den_i_cpu,
                "ev_h2d_start": ev_h2d_start,
                "ev_h2d_end": ev_h2d_end,
                "ev_compute_start": ev_compute_start,
                "ev_compute_end": ev_compute_end,
                "ev_d2h_start": ev_d2h_start,
                "ev_d2h_end": ev_d2h_end,
                "inputs_keepalive": (
                    (q_local, k_local, v_local, mask_one.get("group_bits_cpu"))
                    if bool(prepared.get("gpu_preloaded", False))
                    else None
                ),
            }
            return "launched"

        def _collect_done(slot_idx: int) -> bool:
            task = inflight[slot_idx]
            if task is None:
                return False
            if not task["ev_d2h_end"].query():
                return False
            path_idx = int(task.get("path_idx", -1))
            timing_ms["fwd_h2d"] += float(task["ev_h2d_start"].elapsed_time(task["ev_h2d_end"]))
            timing_ms["fwd_compute"] += float(task["ev_compute_start"].elapsed_time(task["ev_compute_end"]))
            timing_ms["fwd_d2h"] += float(task["ev_d2h_start"].elapsed_time(task["ev_d2h_end"]))
            task.pop("inputs_keepalive", None)
            inflight[slot_idx] = None
            pending_merge.append(task)
            _log_mem(
                timeline_rows,
                t0_wall=t0_wall,
                stage="subseq_d2h_done",
                phase="fwd",
                paths_done=paths_done,
                active_subseq=sum(1 for x in inflight if x is not None),
                round_idx=max(0, int(paths_done // max(1, max_parallel_subseq))),
                path_idx=path_idx,
            )
            return True

        def _merge_one_pending() -> bool:
            nonlocal paths_done
            if not pending_merge:
                return False
            task = pending_merge.popleft()
            path_idx = int(task.get("path_idx", -1))
            with _nvtx_range(f"cqsa:fwd:path={path_idx}:merge"):
                t0 = time.perf_counter()
                token_ids_cpu = task["token_ids_cpu"]
                num_global_cpu.index_add_(1, token_ids_cpu, task["num_i_cpu"])
                den_global_cpu.index_add_(2, token_ids_cpu, task["den_i_cpu"])
                timing_ms["fwd_merge"] += (time.perf_counter() - t0) * 1000.0
            processed_paths.append(task["path"])
            paths_done += 1
            if progress_bar is not None:
                progress_bar.update(1)
            _log_mem(
                timeline_rows,
                t0_wall=t0_wall,
                stage="subseq_merge_done",
                phase="fwd",
                paths_done=paths_done,
                active_subseq=sum(1 for x in inflight if x is not None),
                round_idx=max(0, int(paths_done // max(1, max_parallel_subseq))),
                path_idx=path_idx,
            )
            _log_mem(
                timeline_rows,
                t0_wall=t0_wall,
                stage="subseq_done",
                phase="fwd",
                paths_done=paths_done,
                active_subseq=sum(1 for x in inflight if x is not None),
                round_idx=max(0, int(paths_done // max(1, max_parallel_subseq))),
                path_idx=path_idx,
            )
            task.clear()
            return True

        try:
            _ensure_progress_bar()
            _submit_more_prepared()
            if bool(preload_subseq):
                while prep_submit_idx < target_paths or len(prep_pending) > 0:
                    if not _harvest_prepared(timeout_s=0.0005):
                        if len(prep_pending) > 0:
                            t0 = time.perf_counter()
                            time.sleep(0.0005)
                            timing_ms["fwd_sync"] += (time.perf_counter() - t0) * 1000.0
                    _submit_more_prepared()
                preloaded_ready: deque[dict[str, Any]] = deque()
                while len(prep_ready) > 0:
                    prepared = prep_ready.popleft()
                    if str(prepared.get("status", "")) == "prepared":
                        prepared = _preload_prepared(prepared)
                    preloaded_ready.append(prepared)
                prep_ready = preloaded_ready
                _log_mem(
                    timeline_rows,
                    t0_wall=t0_wall,
                    stage="preload_done",
                    phase="fwd",
                    paths_done=paths_done,
                    active_subseq=0,
                    round_idx=0,
                )

            if schedule_mode_n == "event":
                _log_mem(
                    timeline_rows,
                    t0_wall=t0_wall,
                    stage="pipeline_start",
                    phase="fwd",
                    paths_done=0,
                    active_subseq=0,
                    round_idx=0,
                )
                with _nvtx_range("cqsa:fwd:event_scheduler"):
                    while (
                        (paths_done + paths_skipped) < target_paths
                        or any(t is not None for t in inflight)
                        or len(prep_pending) > 0
                        or len(prep_ready) > 0
                        or prep_submit_idx < target_paths
                    ):
                        made_progress = False

                        for i, t in enumerate(inflight):
                            if t is not None and t["ev_d2h_end"].query():
                                _collect_done(i)
                                made_progress = True

                        if _harvest_prepared(timeout_s=0.0):
                            made_progress = True
                        _submit_more_prepared()

                        while True:
                            free_slot = next((i for i, t in enumerate(inflight) if t is None), None)
                            if free_slot is None:
                                break
                            prepared = _pop_ready_prepared()
                            if prepared is None:
                                break
                            if str(prepared.get("status", "")) == "skip":
                                _account_skip(int(prepared.get("path_idx", -1)))
                                made_progress = True
                                _submit_more_prepared()
                                continue
                            status = _launch_prepared(prepared, int(free_slot))
                            if status == "oom":
                                raise RuntimeError(f"CUDA OOM in forward streaming at path={prepared['path']}.")
                            made_progress = True
                            _log_mem(
                                timeline_rows,
                                t0_wall=t0_wall,
                                stage="subseq_dispatch",
                                phase="fwd",
                                paths_done=paths_done,
                                active_subseq=sum(1 for x in inflight if x is not None),
                                round_idx=max(0, int(paths_done // max(1, max_parallel_subseq))),
                                path_idx=int(prepared.get("path_idx", -1)),
                            )
                            _submit_more_prepared()

                        if _merge_one_pending():
                            made_progress = True

                        if not made_progress:
                            if _harvest_prepared(timeout_s=0.0005):
                                continue
                            if (
                                not any(t is not None for t in inflight)
                                and len(pending_merge) == 0
                                and len(prep_pending) == 0
                                and len(prep_ready) == 0
                                and prep_submit_idx >= target_paths
                            ):
                                break
                            t0 = time.perf_counter()
                            time.sleep(0.0005)
                            timing_ms["fwd_sync"] += (time.perf_counter() - t0) * 1000.0
                            _log_mem(
                                timeline_rows,
                                t0_wall=t0_wall,
                                stage="pipeline_wait",
                                phase="fwd",
                                paths_done=paths_done,
                                active_subseq=sum(1 for x in inflight if x is not None),
                                round_idx=max(0, int(paths_done // max(1, max_parallel_subseq))),
                            )
            else:
                round_idx = 0
                with _nvtx_range("cqsa:fwd:round_scheduler"):
                    while (paths_done + paths_skipped) < target_paths:
                        launched: list[int] = []
                        _log_mem(
                            timeline_rows,
                            t0_wall=t0_wall,
                            stage="round_start",
                            phase="fwd",
                            paths_done=paths_done,
                            active_subseq=0,
                            round_idx=round_idx,
                        )
                        while len(launched) < int(max_parallel_subseq):
                            _submit_more_prepared()
                            if len(prep_ready) == 0 and not _harvest_prepared(timeout_s=0.0005):
                                if len(prep_pending) == 0 and prep_submit_idx >= target_paths:
                                    break
                                continue
                            prepared = _pop_ready_prepared()
                            if prepared is None:
                                if len(prep_pending) == 0 and prep_submit_idx >= target_paths:
                                    break
                                continue
                            if str(prepared.get("status", "")) == "skip":
                                _account_skip(int(prepared.get("path_idx", -1)))
                                continue
                            slot_idx = len(launched)
                            status = _launch_prepared(prepared, int(slot_idx))
                            if status == "oom":
                                raise RuntimeError(f"CUDA OOM in forward streaming at path={prepared['path']}.")
                            launched.append(int(slot_idx))

                        if len(launched) == 0:
                            break

                        _log_mem(
                            timeline_rows,
                            t0_wall=t0_wall,
                            stage="round_launch_done",
                            phase="fwd",
                            paths_done=paths_done,
                            active_subseq=len(launched),
                            round_idx=round_idx,
                        )
                        with _nvtx_range(f"cqsa:fwd:round={int(round_idx)}:sync"):
                            t0 = time.perf_counter()
                            for i in launched:
                                streams[i].synchronize()
                            timing_ms["fwd_sync"] += (time.perf_counter() - t0) * 1000.0
                        _log_mem(
                            timeline_rows,
                            t0_wall=t0_wall,
                            stage="round_synced",
                            phase="fwd",
                            paths_done=paths_done,
                            active_subseq=len(launched),
                            round_idx=round_idx,
                        )
                        for i in launched:
                            _collect_done(i)
                        while _merge_one_pending():
                            pass
                        _log_mem(
                            timeline_rows,
                            t0_wall=t0_wall,
                            stage="round_accum_done",
                            phase="fwd",
                            paths_done=paths_done,
                            active_subseq=len(launched),
                            round_idx=round_idx,
                        )
                        torch.cuda.empty_cache()
                        _log_mem(
                            timeline_rows,
                            t0_wall=t0_wall,
                            stage="round_cache_cleared",
                            phase="fwd",
                            paths_done=paths_done,
                            active_subseq=0,
                            round_idx=round_idx,
                        )
                        round_idx += 1
        finally:
            if progress_bar is not None:
                progress_bar.close()
            prep_executor.shutdown(wait=True, cancel_futures=False)

        _log_mem(timeline_rows, t0_wall=t0_wall, stage="fwd_done", phase="fwd", paths_done=paths_done, active_subseq=0)

        den_tok = den_global_cpu.transpose(1, 2).contiguous()
        den_safe = den_tok.clamp_min(1e-12)
        output_cpu = num_global_cpu / den_safe.unsqueeze(-1)
        output_cpu = torch.where(den_tok.unsqueeze(-1) > 0, output_cpu, torch.zeros_like(output_cpu))

        return {
            "num_global_cpu": num_global_cpu,
            "den_global_cpu": den_global_cpu,
            "output_cpu": output_cpu,
            "processed_paths": processed_paths,
            "paths_forward": int(paths_done),
            "target_paths": int(target_paths),
            "num_paths_total": int(num_paths_total),
            "timeline_rows": timeline_rows,
            "timing_ms": timing_ms,
            "t0_wall": float(t0_wall),
            "peak_cuda_allocated_gib": float(torch.cuda.max_memory_allocated()) / GIB,
        }


def compute_dnum_dden(
    *,
    num_global_cpu: torch.Tensor,
    den_global_cpu: torch.Tensor,
    grad_out_cpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # num_global_cpu: [B, N, H, D], den_global_cpu: [B, H, N], grad_out_cpu: [B, N, H, D]
    den_tok = den_global_cpu.transpose(1, 2).contiguous()  # [B, N, H]
    den_safe = den_tok.clamp_min(1e-12)
    d_num = grad_out_cpu.float() / den_safe.unsqueeze(-1)  # [B, N, H, D]
    d_den_tok = -(grad_out_cpu.float() * num_global_cpu).sum(dim=-1) / (den_safe * den_safe)  # [B,N,H]
    valid_tok = den_tok > 0
    d_num = torch.where(valid_tok.unsqueeze(-1), d_num, torch.zeros_like(d_num))
    d_den_tok = torch.where(valid_tok, d_den_tok, torch.zeros_like(d_den_tok))
    d_den = d_den_tok.transpose(1, 2).contiguous()  # [B, H, N]
    return d_num, d_den


def compute_dnum_dden_gpu_staged(
    *,
    num_global_cpu: torch.Tensor,
    den_global_cpu: torch.Tensor,
    grad_out_cpu: torch.Tensor,
    tokens_per_chunk: int = 131_072,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dNum/dDen on GPU in chunks, then materialize outputs on CPU.

    Layouts:
      - num_global_cpu: [B, N, H, D]
      - den_global_cpu: [B, H, N]
      - grad_out_cpu:   [B, N, H, D]
    Returns:
      - d_num_cpu: [B, N, H, D] (CPU float32)
      - d_den_cpu: [B, H, N]    (CPU float32)
    """
    if not torch.cuda.is_available():
        return compute_dnum_dden(
            num_global_cpu=num_global_cpu,
            den_global_cpu=den_global_cpu,
            grad_out_cpu=grad_out_cpu,
        )

    if int(tokens_per_chunk) <= 0:
        raise ValueError("tokens_per_chunk must be > 0")

    num_cpu = num_global_cpu.float().contiguous()
    den_cpu = den_global_cpu.float().contiguous()
    go_cpu = grad_out_cpu.float().contiguous()

    B_i, N_i, H_i, D_i = (int(x) for x in num_cpu.shape)
    if tuple(go_cpu.shape) != (B_i, N_i, H_i, D_i):
        raise ValueError("grad_out_cpu shape mismatch with num_global_cpu")
    if tuple(den_cpu.shape) != (B_i, H_i, N_i):
        raise ValueError("den_global_cpu shape mismatch with num_global_cpu")

    den_tok_cpu = den_cpu.transpose(1, 2).contiguous()  # [B, N, H]
    d_num_cpu = torch.empty_like(num_cpu, dtype=torch.float32, device="cpu")
    d_den_tok_cpu = torch.empty_like(den_tok_cpu, dtype=torch.float32, device="cpu")

    chunk = int(tokens_per_chunk)
    for s in range(0, N_i, chunk):
        e = min(N_i, s + chunk)
        num_gpu = num_cpu[:, s:e, :, :].to("cuda", non_blocking=False)
        den_gpu = den_tok_cpu[:, s:e, :].to("cuda", non_blocking=False)
        go_gpu = go_cpu[:, s:e, :, :].to("cuda", non_blocking=False)

        den_safe = den_gpu.clamp_min(1e-12)
        valid = den_gpu > 0
        d_num_gpu = go_gpu / den_safe.unsqueeze(-1)
        d_den_gpu = -(go_gpu * num_gpu).sum(dim=-1) / (den_safe * den_safe)
        d_num_gpu = torch.where(valid.unsqueeze(-1), d_num_gpu, torch.zeros_like(d_num_gpu))
        d_den_gpu = torch.where(valid, d_den_gpu, torch.zeros_like(d_den_gpu))

        d_num_cpu[:, s:e, :, :].copy_(d_num_gpu, non_blocking=False)
        d_den_tok_cpu[:, s:e, :].copy_(d_den_gpu, non_blocking=False)
        del num_gpu, den_gpu, go_gpu, den_safe, valid, d_num_gpu, d_den_gpu

    torch.cuda.synchronize()
    d_den_cpu = d_den_tok_cpu.transpose(1, 2).contiguous()  # [B, H, N]
    return d_num_cpu, d_den_cpu


def loss_grad_from_output(
    *,
    output_cpu: torch.Tensor,
    loss_type: str,
    loss_seed: int,
) -> torch.Tensor:
    key = str(loss_type).strip().lower()
    if key == "mse":
        numel = max(1, int(output_cpu.numel()))
        return (2.0 / float(numel)) * output_cpu.float()
    if key == "dot":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(loss_seed))
        return torch.randn_like(output_cpu, dtype=torch.float32, device="cpu", generator=gen)
    raise ValueError(f"Unknown bwd loss type '{loss_type}'. Use one of: mse, dot")


def streamed_backward_phase(
    *,
    N: int,
    D: int,
    B: int,
    H: int,
    c: int,
    interest_set: Sequence[int],
    itr_max: int,
    max_parallel_subseq_bwd: int,
    schedule_mode: str,
    dtype: torch.dtype,
    attention_kernel_name: str,
    source_q: torch.Tensor | None,
    source_k: torch.Tensor | None,
    source_v: torch.Tensor | None,
    qkv_generation_mode: str,
    seed: int,
    input_std: float,
    processed_paths: Sequence[tuple[int, ...]],
    max_rounds: int | None = None,
    d_num_global_cpu: torch.Tensor,
    d_den_global_cpu: torch.Tensor,
    t0_wall: float,
    timeline_rows: list[dict[str, Any]],
    timing_ms: dict[str, float],
    memory_cap_gib: float | None = None,
    auto_parallel_subseq_bwd: bool = False,
    reuse_qkv_across_paths: bool = False,
    prestage_qkv_on_gpu: bool = False,
    subseq_qkv_provider: Callable[[int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    cached_dnum_dden_by_path: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] | None = None,
    precomputed_path_cache: Mapping[tuple[int, ...], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    timing_ms = _ensure_timing_dict(timing_ms)
    schedule_mode_n = str(schedule_mode).strip().lower()
    if schedule_mode_n not in ("event", "round"):
        raise ValueError("schedule_mode must be one of: event, round")

    mask_engine = CQS_mask(interest_set=tuple(int(x) for x in interest_set), c=int(c))
    softmax_scale = float(1.0 / math.sqrt(float(D)))
    use_naive_bwd = str(attention_kernel_name).strip().lower() == "custom_attn"

    d_q_global_cpu = torch.zeros((int(B), int(H), int(N), int(D)), dtype=torch.float32)
    d_k_global_cpu = torch.zeros_like(d_q_global_cpu)
    d_v_global_cpu = torch.zeros_like(d_q_global_cpu)

    paths_done = 0
    num_paths_total = int(len(processed_paths))
    max_parallel_limit = max(1, int(num_paths_total))
    max_parallel_current = max(1, min(int(max_parallel_subseq_bwd), int(max_parallel_limit)))
    if max_rounds is None:
        max_rounds_i: int | None = None
        target_paths = int(num_paths_total)
    else:
        max_rounds_i = int(max_rounds)
        if max_rounds_i <= 0:
            raise ValueError("max_rounds must be > 0 when set.")
        target_paths = int(min(int(num_paths_total), int(max_rounds_i) * int(max_parallel_current)))

    streams = [torch.cuda.Stream(device="cuda") for _ in range(int(max_parallel_current))]
    inflight: list[dict[str, Any] | None] = [None for _ in range(int(max_parallel_current))]
    pending_merge: deque[dict[str, Any]] = deque()
    progress_bar = None
    bwd_oom_backoffs = 0
    first_round_peak_delta_gib: float | None = None
    first_round_per_subseq_gib: float | None = None
    paths_skipped = 0
    prep_workers = _prep_worker_count(int(max_parallel_current))
    eager_prepare_all = bool(
        precomputed_path_cache is not None
        and subseq_qkv_provider is not None
        and source_q is None
    )
    prep_executor = ThreadPoolExecutor(max_workers=prep_workers, thread_name_prefix="cqsa_bwd_prep")
    prep_pending: dict[Future[Any], int] = {}
    prep_ready: deque[dict[str, Any]] = deque()
    prep_submit_idx = 0
    reuse_qkv_lock = threading.Lock()
    cached_dnum_dden_lock = threading.Lock() if cached_dnum_dden_by_path is not None else None
    reuse_qkv = bool(reuse_qkv_across_paths) and (source_q is None)
    prestage_qkv = bool(prestage_qkv_on_gpu) and bool(reuse_qkv)
    reused_qkv_cpu_by_l: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    reused_qkv_gpu_by_slot_l: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    eager_prestaged_l: set[int] = set()

    _log_mem(timeline_rows, t0_wall=t0_wall, stage="bwd_start", phase="bwd", paths_done=0, active_subseq=0)

    def _ensure_progress_bar() -> None:
        nonlocal progress_bar
        if progress_bar is not None:
            return
        if target_paths <= 0:
            return
        if tqdm is not None:
            print(
                f"[cqsa_stream] phase=bwd progress start: total_subseq={int(target_paths)}",
                flush=True,
            )
            progress_bar = tqdm(
                total=int(target_paths),
                initial=int(paths_done),
                desc="bwd",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                leave=True,
                dynamic_ncols=True,
            )
        else:
            print("[cqsa_stream] tqdm not installed; backward ETA progress bar disabled.", flush=True)

    def _ensure_stream_capacity(required_k: int) -> None:
        nonlocal streams, inflight
        while len(streams) < int(required_k):
            streams.append(torch.cuda.Stream(device="cuda"))
            inflight.append(None)

    def _round_idx() -> int:
        return max(0, int(paths_done // max(1, int(max_parallel_current))))

    def _set_target_paths(new_target_paths: int, reason: str) -> None:
        nonlocal target_paths, progress_bar
        old_target = int(target_paths)
        target_paths = max(
            int(paths_done),
            int(prep_submit_idx),
            min(int(num_paths_total), int(new_target_paths)),
        )
        if progress_bar is not None:
            progress_bar.total = int(target_paths)
            progress_bar.refresh()
        if int(target_paths) != int(old_target):
            print(
                f"[cqsa_stream][bwd] target_paths update ({reason}): {old_target} -> {target_paths}",
                flush=True,
            )

    def _round_based_target_for_k(k: int) -> int:
        if max_rounds_i is None:
            return int(num_paths_total)
        return min(int(num_paths_total), int(max_rounds_i) * int(k))

    def _prepare_path(path_idx: int, path: tuple[int, ...]) -> dict[str, Any]:
        with _nvtx_range(f"cqsa:bwd:path={int(path_idx)}:prepare"):
            cache_entry = None if precomputed_path_cache is None else precomputed_path_cache.get(tuple(path))
            if cache_entry is None:
                with _nvtx_range(f"cqsa:bwd:path={int(path_idx)}:mask"):
                    t_mask0 = time.perf_counter()
                    mask_one_raw = mask_engine.gen_mask(
                        N=int(N),
                        num_itr=int(itr_max),
                        quorum_idx=list(path),
                        interest_set=tuple(int(x) for x in interest_set),
                        c=int(c),
                        include_trace=False,
                    )
                    mask_ms = (time.perf_counter() - t_mask0) * 1000.0
                token_ids_np = np.asarray(mask_one_raw["token_ids"], dtype=np.int64)
                local_size = int(mask_one_raw["local_size"])
                if local_size <= 0 or token_ids_np.size == 0:
                    return {
                        "status": "skip",
                        "path": path,
                        "path_idx": int(path_idx),
                        "mask_ms": float(mask_ms),
                        "qkv_ms": 0.0,
                        "split_ms": 0.0,
                    }
                token_ids_cpu = torch.from_numpy(token_ids_np.astype(np.int64, copy=False)).to(torch.long)
                group_bits_cpu = _group_runs_to_group_bits(local_size, mask_one_raw["group_runs"]).to(torch.long)
                mask_one = {
                    "local_size": int(local_size),
                    "group_bits_cpu": group_bits_cpu,
                    "token_ids_cpu": token_ids_cpu,
                    "token_ids": token_ids_cpu.tolist(),
                }
            else:
                mask_ms = 0.0
                local_size = int(cache_entry["local_size"])
                token_ids_cpu = cache_entry["token_ids_cpu"]
                group_bits_cpu = cache_entry["group_bits_cpu"]
                mask_one = cache_entry["mask_one"]
                if "token_ids_cpu" not in mask_one:
                    mask_one = dict(mask_one)
                    mask_one["token_ids_cpu"] = token_ids_cpu
                    mask_one["token_ids"] = token_ids_cpu.tolist()
            if local_size <= 0 or int(token_ids_cpu.numel()) == 0:
                return {
                    "status": "skip",
                    "path": path,
                    "path_idx": int(path_idx),
                    "mask_ms": float(mask_ms),
                    "qkv_ms": 0.0,
                    "split_ms": 0.0,
                }

            needs_inline_qkv = bool(source_q is not None and source_q.device.type == "cuda")
            q_local: torch.Tensor | None = None
            k_local: torch.Tensor | None = None
            v_local: torch.Tensor | None = None
            on_cuda = False
            qkv_ms = 0.0

            if not needs_inline_qkv:
                using_cached_provider = bool(source_q is None and subseq_qkv_provider is not None)
                with _nvtx_range(f"cqsa:bwd:path={int(path_idx)}:qkv_prepare"):
                    t_qkv0 = time.perf_counter()
                    if using_cached_provider:
                        q_local, k_local, v_local = subseq_qkv_provider(int(local_size))
                        on_cuda = bool(isinstance(q_local, torch.Tensor) and q_local.device.type == "cuda")
                    elif reuse_qkv:
                        with reuse_qkv_lock:
                            if int(local_size) not in reused_qkv_cpu_by_l:
                                q_reuse, k_reuse, v_reuse, _ = _make_qkv_for_path(
                                    source_q=None,
                                    source_k=None,
                                    source_v=None,
                                    token_ids_cpu=token_ids_cpu,
                                    local_size=local_size,
                                    path=tuple(),
                                    B=int(B),
                                    H=int(H),
                                    D=int(D),
                                    dtype=dtype,
                                    qkv_generation_mode=qkv_generation_mode,
                                    seed=int(seed),
                                    input_std=float(input_std),
                                )
                                reused_qkv_cpu_by_l[int(local_size)] = (q_reuse, k_reuse, v_reuse)
                            q_local, k_local, v_local = reused_qkv_cpu_by_l[int(local_size)]
                        on_cuda = False
                    else:
                        q_local, k_local, v_local, on_cuda = _make_qkv_for_path(
                            source_q=source_q,
                            source_k=source_k,
                            source_v=source_v,
                            token_ids_cpu=token_ids_cpu,
                            local_size=local_size,
                            path=path,
                            B=int(B),
                            H=int(H),
                            D=int(D),
                            dtype=dtype,
                            qkv_generation_mode=qkv_generation_mode,
                            seed=int(seed),
                            input_std=float(input_std),
                        )
                    if not using_cached_provider:
                        qkv_ms = (time.perf_counter() - t_qkv0) * 1000.0

        cache_key = tuple(int(x) for x in path)
        cached_pair = None
        if cached_dnum_dden_by_path is not None:
            if cached_dnum_dden_lock is None:
                cached_pair = cached_dnum_dden_by_path.get(cache_key)
            else:
                with cached_dnum_dden_lock:
                    cached_pair = cached_dnum_dden_by_path.get(cache_key)

        split_ms = 0.0
        if cached_pair is not None:
            d_num_i_cpu, d_den_i_cpu = cached_pair
        else:
            with _nvtx_range(f"cqsa:bwd:path={int(path_idx)}:split_dnum_dden"):
                t_split0 = time.perf_counter()
                d_num_i_cpu = d_num_global_cpu.index_select(1, token_ids_cpu).contiguous().pin_memory()
                d_den_i_cpu = d_den_global_cpu.index_select(2, token_ids_cpu).contiguous().pin_memory()
                split_ms = (time.perf_counter() - t_split0) * 1000.0
                if cached_dnum_dden_by_path is not None:
                    if cached_dnum_dden_lock is None:
                        cached_dnum_dden_by_path[cache_key] = (d_num_i_cpu, d_den_i_cpu)
                    else:
                        with cached_dnum_dden_lock:
                            existing = cached_dnum_dden_by_path.get(cache_key)
                            if existing is None:
                                cached_dnum_dden_by_path[cache_key] = (d_num_i_cpu, d_den_i_cpu)
                            else:
                                d_num_i_cpu, d_den_i_cpu = existing

        return {
            "status": "prepared",
            "path": path,
            "path_idx": int(path_idx),
            "local_size": int(local_size),
            "token_ids_cpu": token_ids_cpu,
            "group_bits_cpu": group_bits_cpu,
            "mask_one": mask_one,
            "q_local": q_local,
            "k_local": k_local,
            "v_local": v_local,
            "on_cuda": bool(on_cuda),
            "needs_inline_qkv": bool(needs_inline_qkv),
            "d_num_i_cpu": d_num_i_cpu,
            "d_den_i_cpu": d_den_i_cpu,
            "mask_ms": float(mask_ms),
            "qkv_ms": float(qkv_ms),
            "split_ms": float(split_ms),
        }

    def _submit_more_prepared() -> None:
        nonlocal prep_submit_idx
        while prep_submit_idx < target_paths:
            queued = (
                len(prep_pending)
                + len(prep_ready)
                + sum(1 for t in inflight[: int(max_parallel_current)] if t is not None)
            )
            if queued >= (int(target_paths) if eager_prepare_all else _prep_prefetch_limit(int(max_parallel_current))):
                break
            path_idx = int(prep_submit_idx)
            path = tuple(processed_paths[path_idx])
            fut = prep_executor.submit(_prepare_path, path_idx, path)
            prep_pending[fut] = path_idx
            prep_submit_idx += 1

    def _harvest_prepared(timeout_s: float = 0.0) -> bool:
        if len(prep_pending) == 0:
            return False
        done, _ = wait(list(prep_pending.keys()), timeout=float(timeout_s), return_when=FIRST_COMPLETED)
        if len(done) == 0:
            return False
        for fut in done:
            prep_pending.pop(fut, None)
            prepared = fut.result()
            if prepared is None:
                raise RuntimeError(
                    "Backward path preparation returned None. "
                    "Expected a dict with status='prepared' or status='skip'."
                )
            timing_ms["bwd_mask_gen"] += float(prepared.get("mask_ms", 0.0))
            timing_ms["bwd_qkv_cpu_gen"] += float(prepared.get("qkv_ms", 0.0))
            timing_ms["cpu_dNum_dDen"] += float(prepared.get("split_ms", 0.0))
            prep_ready.append(prepared)
            _log_mem(
                timeline_rows,
                t0_wall=t0_wall,
                stage="subseq_prepared",
                phase="bwd",
                paths_done=paths_done,
                active_subseq=sum(1 for t in inflight[: int(max_parallel_current)] if t is not None),
                round_idx=_round_idx(),
                path_idx=int(prepared.get("path_idx", -1)),
            )
        return True

    def _pop_ready_prepared() -> dict[str, Any] | None:
        while len(prep_ready) > 0:
            prepared = prep_ready.popleft()
            if int(prepared.get("path_idx", -1)) >= int(target_paths):
                continue
            return prepared
        return None

    def _account_skip(path_idx: int) -> None:
        nonlocal paths_skipped
        paths_skipped += 1
        if progress_bar is not None:
            progress_bar.update(1)
        _log_mem(
            timeline_rows,
            t0_wall=t0_wall,
            stage="subseq_skip",
            phase="bwd",
            paths_done=paths_done,
            active_subseq=sum(1 for t in inflight[: int(max_parallel_current)] if t is not None),
            round_idx=_round_idx(),
            path_idx=int(path_idx),
        )

    def _launch_prepared(prepared: dict[str, Any], slot_idx: int) -> str:
        nonlocal prestage_qkv
        path = tuple(int(x) for x in prepared["path"])
        path_idx = int(prepared["path_idx"])
        _log_mem(
            timeline_rows,
            t0_wall=t0_wall,
            stage="subseq_prepare_start",
            phase="bwd",
            paths_done=paths_done,
            active_subseq=sum(1 for t in inflight[: int(max_parallel_current)] if t is not None),
            round_idx=_round_idx(),
            path_idx=path_idx,
        )
        _log_mem(
            timeline_rows,
            t0_wall=t0_wall,
            stage="subseq_mask_ready",
            phase="bwd",
            paths_done=paths_done,
            active_subseq=sum(1 for t in inflight[: int(max_parallel_current)] if t is not None),
            round_idx=_round_idx(),
            path_idx=path_idx,
        )

        token_ids_cpu = prepared["token_ids_cpu"]
        local_size = int(prepared["local_size"])
        group_bits_cpu = prepared["group_bits_cpu"]
        q_local = prepared.get("q_local", None)
        k_local = prepared.get("k_local", None)
        v_local = prepared.get("v_local", None)
        on_cuda = bool(prepared.get("on_cuda", False))
        d_num_i_cpu = prepared["d_num_i_cpu"]
        d_den_i_cpu = prepared["d_den_i_cpu"]

        if (not bool(prepared.get("needs_inline_qkv", False))) and reuse_qkv and prestage_qkv:
            if int(local_size) not in eager_prestaged_l:
                try:
                    for slot_pre in range(int(max_parallel_current)):
                        key = (int(slot_pre), int(local_size))
                        if key in reused_qkv_gpu_by_slot_l:
                            continue
                        qg = q_local.to("cuda", non_blocking=True)
                        kg = k_local.to("cuda", non_blocking=True)
                        vg = v_local.to("cuda", non_blocking=True)
                        reused_qkv_gpu_by_slot_l[key] = (qg, kg, vg)
                    eager_prestaged_l.add(int(local_size))
                    print(
                        f"[cqsa_stream][bwd] pre-staged reused QKV on GPU for L={int(local_size)} "
                        f"across {int(max_parallel_current)} slots.",
                        flush=True,
                    )
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        reused_qkv_gpu_by_slot_l.clear()
                        eager_prestaged_l.clear()
                        torch.cuda.empty_cache()
                        print(
                            f"[cqsa_stream][bwd] QKV pre-stage OOM at L={int(local_size)}; "
                            "falling back to reused CPU QKV (no pre-stage).",
                            flush=True,
                        )
                        prestage_qkv = False
                    else:
                        raise
            if prestage_qkv:
                cached = reused_qkv_gpu_by_slot_l.get((int(slot_idx), int(local_size)))
                if cached is not None:
                    q_local, k_local, v_local = cached
                    on_cuda = True

        _log_mem(
            timeline_rows,
            t0_wall=t0_wall,
            stage="subseq_inputs_ready",
            phase="bwd",
            paths_done=paths_done,
            active_subseq=sum(1 for t in inflight[: int(max_parallel_current)] if t is not None),
            round_idx=_round_idx(),
            path_idx=path_idx,
        )

        ev_h2d_start = torch.cuda.Event(enable_timing=True)
        ev_h2d_end = torch.cuda.Event(enable_timing=True)
        ev_compute_start = torch.cuda.Event(enable_timing=True)
        ev_compute_end = torch.cuda.Event(enable_timing=True)
        ev_d2h_start = torch.cuda.Event(enable_timing=True)
        ev_d2h_end = torch.cuda.Event(enable_timing=True)

        s = streams[slot_idx]
        d_q_i_cpu: torch.Tensor | None = None
        d_k_i_cpu: torch.Tensor | None = None
        d_v_i_cpu: torch.Tensor | None = None
        try:
            with _nvtx_range(f"cqsa:bwd:path={path_idx}:slot={int(slot_idx)}:launch"):
                with torch.cuda.stream(s):
                    if bool(prepared.get("needs_inline_qkv", False)):
                        # For CUDA source Q/K/V, gather on the launch stream to avoid
                        # default-stream gather + non-default compute races.
                        with _nvtx_range(f"cqsa:bwd:path={path_idx}:inline_qkv_prepare"):
                            t_qkv0 = time.perf_counter()
                            q_local, k_local, v_local, on_cuda = _make_qkv_for_path(
                                source_q=source_q,
                                source_k=source_k,
                                source_v=source_v,
                                token_ids_cpu=token_ids_cpu,
                                local_size=local_size,
                                path=path,
                                B=int(B),
                                H=int(H),
                                D=int(D),
                                dtype=dtype,
                                qkv_generation_mode=qkv_generation_mode,
                                seed=int(seed),
                                input_std=float(input_std),
                            )
                            timing_ms["bwd_qkv_cpu_gen"] += (time.perf_counter() - t_qkv0) * 1000.0
                    with _nvtx_range(f"cqsa:bwd:path={path_idx}:slot={int(slot_idx)}:h2d_compute"):
                        ev_h2d_start.record()
                        if bool(use_naive_bwd):
                            ev_h2d_end.record()
                            ev_compute_start.record()
                            d_q_i, d_k_i, d_v_i = _local_backward_naive_staged_lowmem(
                                q_local=q_local,
                                k_local=k_local,
                                v_local=v_local,
                                d_num_i_cpu=d_num_i_cpu,
                                d_den_i_cpu=d_den_i_cpu,
                                group_bits_cpu=group_bits_cpu,
                                on_cuda=bool(on_cuda),
                                softmax_scale=float(softmax_scale),
                            )
                        else:
                            if on_cuda:
                                q_i = q_local
                                k_i = k_local
                                v_i = v_local
                            else:
                                q_i = q_local.to("cuda", non_blocking=True)
                                k_i = k_local.to("cuda", non_blocking=True)
                                v_i = v_local.to("cuda", non_blocking=True)
                            d_num_i = d_num_i_cpu.to("cuda", dtype=q_i.dtype, non_blocking=True)
                            d_den_i = d_den_i_cpu.to("cuda", dtype=q_i.dtype, non_blocking=True)
                            group_bits_gpu = group_bits_cpu.to("cuda", non_blocking=True)
                            ev_h2d_end.record()
                            ev_compute_start.record()

                            d_q_i, d_k_i, d_v_i = flash_attn_bwd_cqs_group_bits(
                                dout_num=d_num_i,
                                dden=d_den_i,
                                q=q_i,
                                k=k_i,
                                v=v_i,
                                cqs_group_bits=group_bits_gpu,
                                softmax_scale=float(softmax_scale),
                            )
                            del q_i, k_i, v_i, d_num_i, d_den_i, group_bits_gpu
                        d_q_i = torch.nan_to_num(d_q_i.float(), nan=0.0, posinf=0.0, neginf=0.0)
                        d_k_i = torch.nan_to_num(d_k_i.float(), nan=0.0, posinf=0.0, neginf=0.0)
                        d_v_i = torch.nan_to_num(d_v_i.float(), nan=0.0, posinf=0.0, neginf=0.0)
                        ev_compute_end.record()
                    with _nvtx_range(f"cqsa:bwd:path={path_idx}:slot={int(slot_idx)}:d2h"):
                        ev_d2h_start.record()
                        d_q_i_cpu = torch.empty_like(d_q_i, device="cpu", pin_memory=True)
                        d_k_i_cpu = torch.empty_like(d_k_i, device="cpu", pin_memory=True)
                        d_v_i_cpu = torch.empty_like(d_v_i, device="cpu", pin_memory=True)
                        d_q_i_cpu.copy_(d_q_i, non_blocking=True)
                        d_k_i_cpu.copy_(d_k_i, non_blocking=True)
                        d_v_i_cpu.copy_(d_v_i, non_blocking=True)
                        ev_d2h_end.record()
                        del d_q_i, d_k_i, d_v_i
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                return "oom"
            raise

        if d_q_i_cpu is None or d_k_i_cpu is None or d_v_i_cpu is None:
            raise RuntimeError("Backward staged D2H failed to produce CPU tensors.")

        inflight[slot_idx] = {
            "token_ids_cpu": token_ids_cpu,
            "d_q_i_cpu": d_q_i_cpu,
            "d_k_i_cpu": d_k_i_cpu,
            "d_v_i_cpu": d_v_i_cpu,
            "path_idx": int(path_idx),
            "ev_h2d_start": ev_h2d_start,
            "ev_h2d_end": ev_h2d_end,
            "ev_compute_start": ev_compute_start,
            "ev_compute_end": ev_compute_end,
            "ev_d2h_start": ev_d2h_start,
            "ev_d2h_end": ev_d2h_end,
        }
        return "launched"

    def _collect_done(slot_idx: int) -> bool:
        task = inflight[slot_idx]
        if task is None:
            return False
        if not task["ev_d2h_end"].query():
            return False

        path_idx = int(task.get("path_idx", -1))

        _log_mem(
            timeline_rows,
            t0_wall=t0_wall,
            stage="subseq_drain_start",
            phase="bwd",
            paths_done=paths_done,
            active_subseq=sum(1 for t in inflight if t is not None),
            round_idx=_round_idx(),
            path_idx=path_idx,
        )

        with _nvtx_range(f"cqsa:bwd:path={path_idx}:drain"):
            timing_ms["bwd_h2d"] += float(task["ev_h2d_start"].elapsed_time(task["ev_h2d_end"]))
            timing_ms["bwd_compute"] += float(task["ev_compute_start"].elapsed_time(task["ev_compute_end"]))
            timing_ms["bwd_d2h"] += float(task["ev_d2h_start"].elapsed_time(task["ev_d2h_end"]))
        _log_mem(
            timeline_rows,
            t0_wall=t0_wall,
            stage="subseq_d2h_done",
            phase="bwd",
            paths_done=paths_done,
            active_subseq=sum(1 for t in inflight if t is not None),
            round_idx=_round_idx(),
            path_idx=path_idx,
        )

        inflight[slot_idx] = None
        pending_merge.append(task)
        return True

    def _merge_one_pending() -> bool:
        nonlocal paths_done
        if not pending_merge:
            return False

        task = pending_merge.popleft()
        path_idx = int(task.get("path_idx", -1))
        d_q_i_cpu = task["d_q_i_cpu"]
        d_k_i_cpu = task["d_k_i_cpu"]
        d_v_i_cpu = task["d_v_i_cpu"]

        with _nvtx_range(f"cqsa:bwd:path={path_idx}:merge"):
            t0 = time.perf_counter()
            token_ids_cpu = task["token_ids_cpu"]
            d_q_global_cpu.index_add_(2, token_ids_cpu, d_q_i_cpu.transpose(1, 2))
            d_k_global_cpu.index_add_(2, token_ids_cpu, d_k_i_cpu.transpose(1, 2))
            d_v_global_cpu.index_add_(2, token_ids_cpu, d_v_i_cpu.transpose(1, 2))
            timing_ms["bwd_merge"] += (time.perf_counter() - t0) * 1000.0
        _log_mem(
            timeline_rows,
            t0_wall=t0_wall,
            stage="subseq_merge_done",
            phase="bwd",
            paths_done=paths_done,
            active_subseq=sum(1 for t in inflight if t is not None),
            round_idx=_round_idx(),
            path_idx=path_idx,
        )

        del d_q_i_cpu, d_k_i_cpu, d_v_i_cpu
        del task["d_q_i_cpu"], task["d_k_i_cpu"], task["d_v_i_cpu"]
        paths_done += 1
        if progress_bar is not None:
            progress_bar.update(1)
        _log_mem(
            timeline_rows,
            t0_wall=t0_wall,
            stage="subseq_done",
            phase="bwd",
            paths_done=paths_done,
            active_subseq=sum(1 for t in inflight if t is not None),
            round_idx=_round_idx(),
            path_idx=path_idx,
        )
        task.clear()
        return True

    def _downscale_parallel(old_k: int, new_k: int, reason: str, path: tuple[int, ...] | None, path_idx: int | None) -> None:
        nonlocal max_parallel_current, bwd_oom_backoffs
        max_parallel_current = int(max(1, min(int(new_k), int(max_parallel_limit))))
        bwd_oom_backoffs += 1
        _set_target_paths(_round_based_target_for_k(int(max_parallel_current)), reason=f"{reason}_downscale")
        print(
            f"[cqsa_stream][bwd] parallel downscale ({reason}): {int(old_k)} -> {int(max_parallel_current)}"
            + (f", requeue path={path}" if path is not None else ""),
            flush=True,
        )
        _log_mem(
            timeline_rows,
            t0_wall=t0_wall,
            stage=f"parallel_downscale_{reason}",
            phase="bwd",
            paths_done=paths_done,
            active_subseq=sum(1 for t in inflight if t is not None),
            round_idx=_round_idx(),
            path_idx=(-1 if path_idx is None else int(path_idx)),
        )

    try:
        _ensure_progress_bar()
        _submit_more_prepared()

        if schedule_mode_n == "event":
            _log_mem(
                timeline_rows,
                t0_wall=t0_wall,
                stage="pipeline_start",
                phase="bwd",
                paths_done=paths_done,
                active_subseq=0,
                round_idx=_round_idx(),
            )
            with _nvtx_range("cqsa:bwd:event_scheduler"):
                while (
                    (paths_done + paths_skipped) < target_paths
                    or any(t is not None for t in inflight[: int(max_parallel_current)])
                    or len(pending_merge) > 0
                    or len(prep_pending) > 0
                    or len(prep_ready) > 0
                    or prep_submit_idx < target_paths
                ):
                    made_progress = False

                    for i, t in enumerate(inflight[: int(max_parallel_current)]):
                        if t is not None and t["ev_d2h_end"].query():
                            _collect_done(i)
                            made_progress = True

                    if _harvest_prepared(timeout_s=0.0):
                        made_progress = True
                    _submit_more_prepared()

                    while True:
                        free_slot = next(
                            (i for i, t in enumerate(inflight[: int(max_parallel_current)]) if t is None),
                            None,
                        )
                        if free_slot is None:
                            break
                        prepared = _pop_ready_prepared()
                        if prepared is None:
                            break
                        if str(prepared.get("status", "")) == "skip":
                            _account_skip(int(prepared.get("path_idx", -1)))
                            made_progress = True
                            _submit_more_prepared()
                            continue

                        status = _launch_prepared(prepared, int(free_slot))
                        if status == "oom":
                            old_k = int(max_parallel_current)
                            if old_k <= 1:
                                raise RuntimeError(f"CUDA OOM in backward streaming at path={prepared['path']}.")
                            prep_ready.appendleft(prepared)
                            _downscale_parallel(
                                old_k=int(old_k),
                                new_k=int(old_k - 1),
                                reason="runtime_oom",
                                path=tuple(prepared["path"]),
                                path_idx=int(prepared["path_idx"]),
                            )
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                            made_progress = True
                            break

                        made_progress = True
                        _log_mem(
                            timeline_rows,
                            t0_wall=t0_wall,
                            stage="subseq_dispatch",
                            phase="bwd",
                            paths_done=paths_done,
                            active_subseq=sum(1 for x in inflight[: int(max_parallel_current)] if x is not None),
                            round_idx=_round_idx(),
                            path_idx=int(prepared.get("path_idx", -1)),
                        )
                        _submit_more_prepared()

                    if _merge_one_pending():
                        made_progress = True

                    if not made_progress:
                        if _harvest_prepared(timeout_s=0.0005):
                            continue
                        if (
                            not any(t is not None for t in inflight[: int(max_parallel_current)])
                            and len(pending_merge) == 0
                            and len(prep_pending) == 0
                            and len(prep_ready) == 0
                            and prep_submit_idx >= target_paths
                        ):
                            break
                        t0 = time.perf_counter()
                        time.sleep(0.0005)
                        timing_ms["bwd_sync"] += (time.perf_counter() - t0) * 1000.0
                        _log_mem(
                            timeline_rows,
                            t0_wall=t0_wall,
                            stage="pipeline_wait",
                            phase="bwd",
                            paths_done=paths_done,
                            active_subseq=sum(1 for t in inflight[: int(max_parallel_current)] if t is not None),
                            round_idx=_round_idx(),
                        )
        else:
            round_idx = 0
            with _nvtx_range("cqsa:bwd:round_scheduler"):
                while (paths_done + paths_skipped) < target_paths:
                    launched: list[int] = []
                    _log_mem(
                        timeline_rows,
                        t0_wall=t0_wall,
                        stage="round_start",
                        phase="bwd",
                        paths_done=paths_done,
                        active_subseq=0,
                        round_idx=round_idx,
                    )
                    oom_in_round = False
                    while len(launched) < int(max_parallel_current):
                        _ensure_stream_capacity(int(max_parallel_current))
                        _submit_more_prepared()
                        if len(prep_ready) == 0 and not _harvest_prepared(timeout_s=0.0005):
                            if len(prep_pending) == 0 and prep_submit_idx >= target_paths:
                                break
                            continue
                        prepared = _pop_ready_prepared()
                        if prepared is None:
                            if len(prep_pending) == 0 and prep_submit_idx >= target_paths:
                                break
                            continue
                        if str(prepared.get("status", "")) == "skip":
                            _account_skip(int(prepared.get("path_idx", -1)))
                            continue

                        slot_idx = len(launched)
                        status = _launch_prepared(prepared, int(slot_idx))
                        if status == "oom":
                            old_k = int(max_parallel_current)
                            if old_k <= 1:
                                raise RuntimeError(f"CUDA OOM in backward streaming at path={prepared['path']}.")
                            prep_ready.appendleft(prepared)
                            _downscale_parallel(
                                old_k=int(old_k),
                                new_k=int(old_k - 1),
                                reason="runtime_oom",
                                path=tuple(prepared["path"]),
                                path_idx=int(prepared["path_idx"]),
                            )
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                            oom_in_round = True
                            break
                        launched.append(int(slot_idx))
                    if len(launched) == 0:
                        if oom_in_round:
                            round_idx += 1
                            continue
                        break

                    _log_mem(
                        timeline_rows,
                        t0_wall=t0_wall,
                        stage="round_launch_done",
                        phase="bwd",
                        paths_done=paths_done,
                        active_subseq=len(launched),
                        round_idx=round_idx,
                    )
                    with _nvtx_range(f"cqsa:bwd:round={int(round_idx)}:sync"):
                        t0 = time.perf_counter()
                        for i in launched:
                            streams[i].synchronize()
                        timing_ms["bwd_sync"] += (time.perf_counter() - t0) * 1000.0
                    _log_mem(
                        timeline_rows,
                        t0_wall=t0_wall,
                        stage="round_synced",
                        phase="bwd",
                        paths_done=paths_done,
                        active_subseq=len(launched),
                        round_idx=round_idx,
                    )
                    for i in launched:
                        _collect_done(i)
                    while _merge_one_pending():
                        pass
                    _log_mem(
                        timeline_rows,
                        t0_wall=t0_wall,
                        stage="round_accum_done",
                        phase="bwd",
                        paths_done=paths_done,
                        active_subseq=len(launched),
                        round_idx=round_idx,
                    )
                    torch.cuda.empty_cache()
                    _log_mem(
                        timeline_rows,
                        t0_wall=t0_wall,
                        stage="round_cache_cleared",
                        phase="bwd",
                        paths_done=paths_done,
                        active_subseq=0,
                        round_idx=round_idx,
                    )
                    round_idx += 1
    finally:
        if progress_bar is not None:
            progress_bar.close()
        prep_executor.shutdown(wait=True, cancel_futures=False)

    _log_mem(timeline_rows, t0_wall=t0_wall, stage="bwd_done", phase="bwd", paths_done=paths_done, active_subseq=0)

    return {
        "d_q_global_cpu": d_q_global_cpu,
        "d_k_global_cpu": d_k_global_cpu,
        "d_v_global_cpu": d_v_global_cpu,
        "paths_backward": int(paths_done),
        "target_paths": int(target_paths),
        "num_paths_total": int(num_paths_total),
        "timeline_rows": timeline_rows,
        "timing_ms": timing_ms,
        "final_max_parallel_subseq_bwd": int(max_parallel_current),
        "first_round_peak_delta_gib": (
            None if first_round_peak_delta_gib is None else float(first_round_peak_delta_gib)
        ),
        "first_round_per_subseq_gib": (
            None if first_round_per_subseq_gib is None else float(first_round_per_subseq_gib)
        ),
        "bwd_oom_backoffs": int(bwd_oom_backoffs),
    }
