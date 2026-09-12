from __future__ import annotations

import argparse
import csv
import math
import queue
import subprocess
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from .cqs_mask import CQS_mask
from .cqsa_probe_capacity import CQSAProbe, ProbeAttempt
from .config import (
    DEFAULT_PROBE_MAX_NUM_ITR,
    DEFAULT_PROBE_START_NUM_ITR,
    MEM_BUDGET_COE,
    DEFAULT_PROBING_HISTORY_DIR,
    GLOBAL_STREAM_SEED,
    STREAM_CQSA_ROOT,
)
from .attention_kernel.FA import default_subsequence_attention
from .attention_kernel.Custom import custom_attn
from .memory_fitting import fitting_mem, pred_mem
from .autograd_op import stream_cqsa_autograd

DUMMY_QKV_PATH = Path(STREAM_CQSA_ROOT) / "dummy_matrix" / "N_100K_D_128.pt"


def _dtype_to_str(dtype: torch.dtype) -> str:
    s = str(dtype)
    if s.startswith("torch."):
        return s[len("torch.") :]
    return s


def _attention_kernel_name(kernel_fn: Callable[..., Any]) -> str:
    name = getattr(kernel_fn, "__name__", None)
    if isinstance(name, str) and len(name) > 0:
        return name
    return kernel_fn.__class__.__name__


def _device_history_filename() -> str:
    try:
        dev_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception:
        dev_name = "unknown_device"
    safe = dev_name.replace("/", "_").replace("\\", "_").strip()
    if len(safe) == 0:
        safe = "unknown_device"
    return f"{safe}.csv"


def _default_device_history_path() -> Path:
    return Path(DEFAULT_PROBING_HISTORY_DIR) / _device_history_filename()


def _compute_probe_budget_bytes(memory_cap_gib: float | None) -> Tuple[int, int, int, int, int, int]:
    free_b, total_b = torch.cuda.mem_get_info()
    used_b = int(total_b - free_b)
    if memory_cap_gib is None:
        cap_bytes = int(total_b)
    else:
        cap_bytes = int(float(memory_cap_gib) * (1024**3))
        if cap_bytes <= 0:
            raise ValueError(f"probe_memory_cap_gib must be > 0, got {memory_cap_gib}")
    effective_budget_bytes = int(min(int(free_b), int(cap_bytes)))
    usable_budget_bytes = int(float(MEM_BUDGET_COE) * float(effective_budget_bytes))
    return int(used_b), int(free_b), int(total_b), int(cap_bytes), int(effective_budget_bytes), int(usable_budget_bytes)


def _is_cuda_oom_or_budget_error(exc: BaseException) -> bool:
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
        or ("hard memory budget violated" in msg)
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


SubsequenceAttentionFn = Callable[
    ...,
    Tuple[torch.Tensor, torch.Tensor],
]

def stream_cqsa(
    # tensor params: if Q/K/V are provided, ignore N/D/B/H/dtype.
    Q: torch.Tensor | None = None,
    K: torch.Tensor | None = None,
    V: torch.Tensor | None = None,
    # shape params
    N: int | None = None,
    D: int | None = None,
    B: int | None = None,
    H: int | None = None,
    dtype: torch.dtype = torch.float16,
    bwd: bool = False,
    # CQS params
    CQS_c: int = 7,
    CQS_interest_set: Sequence[int] = (0, 1, 3),
    # runtime params
    itr: int | None = None,
    n_cap: int = 1,
    memory_budget: float | None = None,
    # experiment params
    EXPMT_return_O: bool = True,
    EXPMT_save_log: bool = False,
    EXPMT_num_subseq: int | None = None,
    EXPMT_attention_kernel: str | SubsequenceAttentionFn | None = default_subsequence_attention,
) -> torch.Tensor | tuple[torch.Tensor, "StreamResult"] | "StreamResult":
    """
    Simplified Stream-CQSA API.

    - `itr` controls subsequence depth. If provided and OOM occurs, it is increased until fit.
    - `n_cap` is the admission cap: the number of subsequences allowed to reside on GPU at once.
    - `memory_budget` is a hard per-run GPU memory budget in GiB.
    - `EXPMT_num_subseq` limits total subsequences processed as the workload size.
    - Scheduling is always event-driven and merging is always performed on CPU.
    - Dimension-only runs always synthesize Q/K/V from the fixed dummy tensor at
      `stream_cqsa/dummy_matrix/N_100K_D_128.pt`.
    """
    qkv_mode = (Q is not None) or (K is not None) or (V is not None)
    if qkv_mode and not (Q is not None and K is not None and V is not None):
        raise ValueError("Provide either all of Q/K/V or none of them.")

    if qkv_mode:
        assert Q is not None and K is not None and V is not None
        if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
            raise ValueError("Q, K, V must be rank-4 tensors [B, H, N, D].")
        if Q.shape != K.shape or Q.shape != V.shape:
            raise ValueError(f"Q, K, V must have identical shape. Got {Q.shape}, {K.shape}, {V.shape}.")
        if Q.device != K.device or Q.device != V.device:
            raise ValueError("Q, K, V must be on the same device.")
        B_i, H_i, N_i, D_i = [int(x) for x in Q.shape]
        N_use, D_use, B_use, H_use = int(N_i), int(D_i), int(B_i), int(H_i)
        dtype_use = Q.dtype
    else:
        missing = [name for name, val in [("N", N), ("D", D), ("B", B), ("H", H)] if val is None]
        if missing:
            raise ValueError(f"When Q/K/V are not provided, require {', '.join(missing)}.")
        N_use, D_use, B_use, H_use = int(N), int(D), int(B), int(H)
        dtype_use = dtype
    if int(n_cap) < 1:
        raise ValueError(f"n_cap must be >= 1, got {n_cap}")
    if memory_budget is not None and float(memory_budget) <= 0.0:
        raise ValueError(f"memory_budget must be > 0, got {memory_budget}")
    c = int(CQS_c)
    max_itr_allowed = _max_itr_for_sequence_len(n_tokens=int(N_use), c=int(c))

    # Differentiable path: enable stream_cqsa(...).backward() when Q/K/V require gradients.
    # This path currently focuses on correctness and uses CUDA tensor ops for backward.
    if qkv_mode and torch.is_grad_enabled():
        assert Q is not None and K is not None and V is not None
        if bool(Q.requires_grad or K.requires_grad or V.requires_grad):
            if not bool(EXPMT_return_O):
                raise ValueError("Autograd path requires EXPMT_return_O=True.")
            if bool(EXPMT_save_log):
                print("[cqsa_stream] autograd path: EXPMT_save_log is ignored.", flush=True)
            if Q.device.type != "cuda":
                if not bool(bwd):
                    raise ValueError(
                        "Autograd path with CPU Q/K/V requires bwd=True (streamed subsequence execution)."
                    )
                print(
                    "[cqsa_stream] autograd path: CPU Q/K/V enabled for streamed fwd+bwd.",
                    flush=True,
                )
            itr_for_grad = (
                int(itr)
                if itr is not None
                else int(min(int(DEFAULT_PROBE_START_NUM_ITR), int(max_itr_allowed)))
            )
            if int(itr_for_grad) > int(max_itr_allowed):
                raise ValueError(
                    f"itr={itr_for_grad} exceeds max valid itr={max_itr_allowed} for N={int(N_use)} and c={int(c)}."
                )
            use_naive_backward = False
            if EXPMT_attention_kernel is not None:
                if callable(EXPMT_attention_kernel):
                    use_naive_backward = _attention_kernel_name(EXPMT_attention_kernel).strip().lower() == "custom_attn"
                else:
                    use_naive_backward = str(EXPMT_attention_kernel).strip().lower() == "custom_attn"
            if bool(use_naive_backward):
                print(
                    "[cqsa_stream] autograd path: using naive Python backward for custom_attn.",
                    flush=True,
                )
            # Current simplified runtime keeps subsequence parallelism fixed at 1.
            max_k_fwd = 1
            max_k_bwd = 1
            if bool(bwd):
                print(
                    f"[cqsa_stream] autograd path: streamed backward enabled "
                    f"(fwd_parallel={int(max_k_fwd)}, bwd_parallel={int(max_k_bwd)}).",
                    flush=True,
                )
            itr_try = int(itr_for_grad)
            while True:
                try:
                    out = stream_cqsa_autograd(
                        Q,
                        K,
                        V,
                        num_itr=int(itr_try),
                        c=int(CQS_c),
                        interest_set=tuple(int(x) for x in CQS_interest_set),
                        use_naive_backward=bool(use_naive_backward),
                        use_streamed_backward=bool(bwd),
                        max_parallel_subseq_fwd=int(max_k_fwd),
                        max_parallel_subseq_bwd=int(max_k_bwd),
                        schedule_mode="event",
                    )
                    break
                except RuntimeError as exc:
                    if not _is_cuda_oom_or_budget_error(exc):
                        raise
                    if int(c) <= 1:
                        raise RuntimeError(
                            "CUDA OOM in autograd path and c<=1, cannot reduce subsequence size via itr."
                        ) from exc
                    nxt_itr = int(itr_try + 1)
                    if int(nxt_itr) > int(max_itr_allowed):
                        raise RuntimeError(
                            f"CUDA OOM persists in autograd forward up to max itr={max_itr_allowed} "
                            f"(N={int(N_use)}, c={int(c)})."
                        ) from exc
                    print(
                        f"[OOM guardrail] CUDA OOM at itr={int(itr_try)}; retry with itr={int(nxt_itr)}.",
                        flush=True,
                    )
                    itr_try = int(nxt_itr)
                    torch.cuda.empty_cache()
            return out

    itr_request = None if itr is None else int(itr)
    if EXPMT_num_subseq is not None:
        needed_itr = 0
        while int(c ** max(0, needed_itr)) < int(EXPMT_num_subseq):
            needed_itr += 1
        if itr_request is None:
            itr_request = int(max(1, needed_itr))
        else:
            itr_request = int(max(int(itr_request), int(needed_itr)))
    if itr_request is not None:
        assert N_use >= (c ** itr_request), "Expected N >= c^itr (a chunk needs at least one token)."
    if EXPMT_attention_kernel is None:
        kernel_fn = default_subsequence_attention
    elif callable(EXPMT_attention_kernel):
        kernel_fn = EXPMT_attention_kernel
    else:
        key = str(EXPMT_attention_kernel).strip().lower()
        if key == "default_subsequence_attention":
            kernel_fn = default_subsequence_attention
        elif key == "custom_attn":
            kernel_fn = custom_attn
        else:
            raise ValueError(
                "Unknown EXPMT_attention_kernel. Use one of: "
                "default_subsequence_attention (callable) or "
                "'default_subsequence_attention' (string); "
                "custom_attn (callable) or 'custom_attn' (string)."
            )

    run_output_dir: Path | None
    if EXPMT_save_log:
        run_output_dir = _make_timestamped_run_dir(STREAM_CQSA_ROOT / "CQSA_stream_logs_single_fn")
        print(f"[cqsa_stream] output_dir={run_output_dir}", flush=True)
    else:
        run_output_dir = None

    itr_try = (
        int(itr_request)
        if itr_request is not None
        else int(min(int(DEFAULT_PROBE_START_NUM_ITR), int(max_itr_allowed)))
    )
    itr_first_attempt = int(itr_try)
    if int(itr_try) > int(max_itr_allowed):
        raise ValueError(
            f"itr={itr_try} exceeds max valid itr={max_itr_allowed} for N={int(N_use)} and c={int(c)}."
        )
    while True:
        runner = CQSAStreamRunner(
            N=int(N_use),
            D=int(D_use),
            B=int(B_use),
            H=int(H_use),
            c=int(CQS_c),
            interest_set=tuple(int(x) for x in CQS_interest_set),
            subseq_attention_fn=kernel_fn,
            dtype=dtype_use,
            input_std=0.1,
            seed=int(GLOBAL_STREAM_SEED),
        )
        try:
            result = runner.run(
                output_dir=run_output_dir,
                probe_max_parallel_subseq=1,
                probe_memory_cap_gib=(float(memory_budget) if memory_budget is not None else None),
                probe_num_itr=int(itr_try),
                use_probing_history=False,
                override_max_subsequence=None,
                dump_cuda_snapshot=bool(EXPMT_save_log),
                schedule_mode="event",
                merge_on_gpu=False,
                keep_final_output=bool(EXPMT_return_O),
                save_accumulators=False,
                max_rounds=None,
                num_subseq_limit=(int(EXPMT_num_subseq) if EXPMT_num_subseq is not None else None),
                print_every_rounds=10,
                save_log=bool(EXPMT_save_log),
                source_q=Q,
                source_k=K,
                source_v=V,
                print_gpu_mem_initial=bool(itr_try == int(itr_first_attempt)),
            )
            break
        except RuntimeError as exc:
            if not _is_cuda_oom_or_budget_error(exc):
                raise
            if int(c) <= 1:
                raise RuntimeError(
                    "CUDA OOM and c<=1, cannot reduce subsequence size via itr."
                ) from exc
            nxt_itr = int(itr_try + 1)
            if int(nxt_itr) > int(max_itr_allowed):
                raise RuntimeError(
                    f"CUDA OOM persists up to max itr={max_itr_allowed} (N={int(N_use)}, c={int(c)})."
                ) from exc
            print(
                f"[OOM guardrail] CUDA OOM at itr={int(itr_try)}; retry with itr={int(nxt_itr)}.",
                flush=True,
            )
            itr_try = int(nxt_itr)
            torch.cuda.empty_cache()

    if not EXPMT_return_O:
        return result

    if result.output_cpu is None:
        raise RuntimeError("Expected final output tensor but got None.")
    O_final = result.output_cpu.permute(0, 2, 1, 3).contiguous()  # [B,H,N,D], CPU
    if EXPMT_save_log:
        return O_final, result
    return O_final


def compare_O(
    O1: torch.Tensor,
    O2: torch.Tensor,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> bool:
    if O1.shape != O2.shape:
        return False
    if (not bool(torch.isfinite(O1).all().item())) or (not bool(torch.isfinite(O2).all().item())):
        return False
    return bool(torch.allclose(O1, O2, rtol=float(rtol), atol=float(atol)))


def _probe_history_headers() -> List[str]:
    return [
        "N",
        "D",
        "B",
        "H",
        "dtype",
        "c",
        "itr",
        "attn_kernel",
        "memory",
        "time",
    ]


def _resolve_probing_history_file(
    probing_history_file: str | Path | None,
    new_probing_history_file: bool | None,
) -> Path:
    if probing_history_file is None:
        base = _default_device_history_path()
    else:
        raw = Path(probing_history_file).expanduser().resolve()
        if raw.suffix == "":
            base = raw / _device_history_filename()
        else:
            base = raw

    base.parent.mkdir(parents=True, exist_ok=True)
    if not bool(new_probing_history_file):
        return base

    stem = base.stem
    suffix = base.suffix if base.suffix else ".csv"
    max_idx = 0
    for p in base.parent.glob(f"{stem}_*{suffix}"):
        tail = p.name[len(stem) + 1 : -len(suffix)]
        if tail.isdigit():
            max_idx = max(max_idx, int(tail))
    next_idx = max_idx + 1
    return base.parent / f"{stem}_{next_idx}{suffix}"


def _make_timestamped_run_dir(base_output_dir: str | Path) -> Path:
    base = Path(base_output_dir).expanduser().resolve()
    stamp = datetime.now().strftime("%m%d%y_%H%M%S")
    run_dir = base / stamp
    idx = 1
    while run_dir.exists():
        run_dir = base / f"{stamp}_{idx:02d}"
        idx += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


@dataclass
class StreamResult:
    itr_max: int
    max_subsequence: int
    num_paths_total: int
    num_paths_processed: int
    elapsed_s: float
    timing_breakdown_csv: str
    timeline_csv: str
    probe_attempts_csv: str
    probe_summary_txt: str
    cuda_snapshot_path: str | None
    output_cpu: torch.Tensor | None = None
    num_global_cpu: torch.Tensor | None = None
    den_global_cpu: torch.Tensor | None = None
    processed_paths: List[Tuple[int, ...]] | None = None
    timing_ms: Dict[str, float] | None = None
    timeline_rows: List[Dict[str, Any]] | None = None


class CQSAStreamRunner:
    """
    CQSA streaming runner.

    Parallelism model:
    - One subsequence = one CUDA kernel launch on one CUDA stream.
    - A "group" launches up to max_subsequence subsequences on distinct streams.
    - This is not identical to computing one larger subsequence: performance depends on
      SM occupancy, memory bandwidth, and kernel overlap. In practice it is usually
      faster than strict serial execution, but not exactly k-times faster.
    """

    def __init__(
        self,
        *,
        N: int,
        D: int,
        B: int = 1,
        H: int = 32,
        c: int = 7,
        interest_set: Sequence[int] = (0, 1, 3),
        subseq_attention_fn: SubsequenceAttentionFn | None = None,
        dtype: torch.dtype = torch.float16,
        input_std: float = 0.1,
        seed: int = 123,
    ) -> None:
        self.N = int(N)
        self.D = int(D)
        self.B = int(B)
        self.H = int(H)
        self.c = int(c)
        self.interest_set = tuple(int(x) for x in interest_set)
        self.dtype = dtype
        self.input_std = float(input_std)
        self.seed = int(seed)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required.")
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("dtype must be fp16 or bf16 for current CQSA CUDA kernel.")

        self.device_index = int(torch.cuda.current_device())
        self.device = torch.device(f"cuda:{self.device_index}")
        self.softmax_scale = 1.0 / math.sqrt(self.D)
        self.mask_engine = CQS_mask(interest_set=self.interest_set, c=self.c)
        self.subseq_attention_fn: SubsequenceAttentionFn = (
            default_subsequence_attention if subseq_attention_fn is None else subseq_attention_fn
        )
        self.attn_kernel_name = _attention_kernel_name(self.subseq_attention_fn)
        self.timeline_rows: List[Dict[str, Any]] = []
        self._t0 = time.time()
        # Synthetic QKV generation state (used only when source_q/source_k/source_v are not provided).
        self._cpu_gen = torch.Generator(device="cpu")
        self._cpu_gen.manual_seed(int(self.seed))
        self._dummy_base_cpu = torch.load(str(DUMMY_QKV_PATH), map_location="cpu")
        if not isinstance(self._dummy_base_cpu, torch.Tensor):
            raise TypeError(f"Dummy tensor at {DUMMY_QKV_PATH} is not a torch.Tensor")
        if tuple(self._dummy_base_cpu.shape) != (1, 1, 100_000, self.D):
            raise ValueError(
                f"Unexpected dummy tensor shape {tuple(self._dummy_base_cpu.shape)} at {DUMMY_QKV_PATH}; "
                f"expected (1, 1, 100000, {self.D})"
            )
        self._dummy_base_cpu = self._dummy_base_cpu.contiguous()

    def _log_mem(
        self,
        stage: str,
        *,
        round_idx: int = -1,
        paths_done: int = 0,
        active_subseq: int = 0,
        num_itr: int = -1,
        probe_attempt: int = -1,
        probe_status: str = "",
        probe_mem_gib: float = -1.0,
    ) -> None:
        try:
            free_b, total_b = torch.cuda.mem_get_info(self.device)
            used_b = int(total_b - free_b)
        except Exception:
            try:
                total_b = int(torch.cuda.get_device_properties(self.device).total_memory)
            except Exception:
                total_b = 0
            free_b = 0
            used_b = 0
        try:
            allocated_b = int(torch.cuda.memory_allocated(self.device))
            reserved_b = int(torch.cuda.memory_reserved(self.device))
            peak_allocated_b = int(torch.cuda.max_memory_allocated(self.device))
        except Exception:
            allocated_b = 0
            reserved_b = 0
            peak_allocated_b = 0
        self.timeline_rows.append(
            {
                "t_rel_s": float(time.time() - self._t0),
                "stage": stage,
                "round": int(round_idx),
                "paths_done": int(paths_done),
                "active_subseq": int(active_subseq),
                "num_itr": int(num_itr),
                "probe_attempt": int(probe_attempt),
                "probe_status": probe_status,
                "probe_mem_gib": float(probe_mem_gib),
                "gpu_used_gib": float(used_b) / (1024**3),
                "gpu_free_gib": float(free_b) / (1024**3),
                "gpu_total_gib": float(total_b) / (1024**3),
                "cuda_allocated_gib": float(allocated_b) / (1024**3),
                "cuda_reserved_gib": float(reserved_b) / (1024**3),
                "cuda_peak_allocated_gib": float(peak_allocated_b) / (1024**3),
            }
        )

    def _path_seed(self, path: Tuple[int, ...]) -> int:
        s = int(self.seed) & ((1 << 64) - 1)
        for x in path:
            s = (s * 6364136223846793005 + (1 + int(x))) & ((1 << 64) - 1)
        return int(s & 0x7FFFFFFF)

    def _make_qkv_cpu(
        self,
        local_size: int,
        path: Tuple[int, ...],
        slot_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build Q/K/V for dimension-only runs from the fixed dummy tensor
        `dummy_matrix/N_100K_D_128.pt` by tiling/slicing along sequence length.
        """
        _ = path, slot_idx  # retained for API compatibility / possible future path-dependent generation.
        lsz = int(local_size)
        base_seq = self._dummy_base_cpu[0, 0]  # [100K, D]
        repeat = int(math.ceil(float(lsz) / float(base_seq.shape[0])))
        tiled = base_seq.repeat((repeat, 1))[:lsz].contiguous()
        dummy_blhd = tiled.unsqueeze(0).unsqueeze(2).repeat(self.B, 1, self.H, 1).contiguous()
        q_buf = dummy_blhd.to(dtype=self.dtype).pin_memory()
        k_buf = dummy_blhd.clone().pin_memory()
        v_buf = dummy_blhd.clone().pin_memory()
        return q_buf, k_buf, v_buf

    def _probe(
        self,
        *,
        max_parallel_subseq: int,
        memory_cap_gib: float | None = None,
        start_num_itr: int = DEFAULT_PROBE_START_NUM_ITR,
        max_num_itr: int = DEFAULT_PROBE_MAX_NUM_ITR,
        dummy_mask: bool = False,
    ) -> Dict[str, Any]:
        probe = CQSAProbe(
            N=self.N,
            D=self.D,
            B=self.B,
            H=self.H,
            c=self.c,
            interest_set=self.interest_set,
            dtype=self.dtype,
            input_std=self.input_std,
            memory_cap_gib=memory_cap_gib,
            subseq_attention_fn=self.subseq_attention_fn,
        )
        result = probe.run(
            start_num_itr=int(start_num_itr),
            max_num_itr=int(max_num_itr),
            max_parallel_subseq=int(max_parallel_subseq),
        )
        if bool(dummy_mask):
            # For compatibility with callers that request dummy-mask probing only.
            # CQSAProbe.run() currently probes real-mask by default; fallback to a
            # fixed-itr manual probe path when dummy mode is requested.
            itr = int(result["itr_max"])
            fit, mem_delta, peak_delta, peak_abs = probe._probe_once(
                num_itr=itr,
                number_of_subsequence=1,
                dummy_mask=True,
            )
            if fit:
                result["single_subseq_mem_consumption_gib"] = float(mem_delta) / (1024**3)
                result["single_subseq_peak_delta_gib"] = float(peak_delta) / (1024**3)
                result["single_subseq_peak_used_gib"] = float(peak_abs) / (1024**3)
        return result

    def _load_matching_probe_history_rows(self, history_path: Path) -> List[Tuple[int, float]]:
        if not history_path.exists():
            return []
        try:
            with history_path.open("r", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            return []
        if len(rows) == 0:
            return []

        target_dtype = _dtype_to_str(self.dtype)
        target_kernel = str(self.attn_kernel_name)
        per_itr: Dict[int, float] = {}
        for row in rows:
            try:
                if int(row.get("N", -1)) != self.N:
                    continue
                if int(row.get("D", -1)) != self.D:
                    continue
                if int(row.get("B", -1)) != self.B:
                    continue
                if int(row.get("H", -1)) != self.H:
                    continue
                if str(row.get("dtype", "")) != target_dtype:
                    continue
                if int(row.get("c", -1)) != self.c:
                    continue
                if str(row.get("attn_kernel", "")).strip() != target_kernel:
                    continue
                itr = int(row.get("itr", -1))
                mem_raw = row.get("memory", row.get("memory_required", "nan"))
                mem_req = float(mem_raw)
                if itr < 0 or (not math.isfinite(mem_req)) or mem_req <= 0.0:
                    continue
                if itr not in per_itr:
                    per_itr[itr] = float(mem_req)
                else:
                    per_itr[itr] = min(float(per_itr[itr]), float(mem_req))
            except Exception:
                continue
        return sorted([(int(k), float(v)) for k, v in per_itr.items()], key=lambda x: int(x[0]))

    def _append_probe_history(
        self,
        history_path: Path,
        *,
        itr: int,
        memory_required_gib: float,
        time_s: float | str | None = None,
    ) -> None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        headers = _probe_history_headers()
        row = {
            "N": int(self.N),
            "D": int(self.D),
            "B": int(self.B),
            "H": int(self.H),
            "dtype": _dtype_to_str(self.dtype),
            "c": int(self.c),
            "itr": int(itr),
            "attn_kernel": str(self.attn_kernel_name),
            "memory": f"{float(memory_required_gib):.6f}",
            "time": (
                str(time_s)
                if isinstance(time_s, str)
                else (
                    f"{float(time_s):.6f}"
                    if (time_s is not None and math.isfinite(float(time_s)) and float(time_s) >= 0.0)
                    else ""
                )
            ),
        }
        write_header = (not history_path.exists()) or (history_path.stat().st_size == 0)
        with history_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _history_has_exact_entry(self, history_path: Path, *, itr: int) -> bool:
        if not history_path.exists():
            return False
        try:
            with history_path.open("r", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            return False
        if len(rows) == 0:
            return False

        target_dtype = _dtype_to_str(self.dtype)
        target_kernel = str(self.attn_kernel_name)
        target_itr = int(itr)
        for row in rows:
            try:
                if int(row.get("N", -1)) != self.N:
                    continue
                if int(row.get("D", -1)) != self.D:
                    continue
                if int(row.get("B", -1)) != self.B:
                    continue
                if int(row.get("H", -1)) != self.H:
                    continue
                if str(row.get("dtype", "")) != target_dtype:
                    continue
                if int(row.get("c", -1)) != self.c:
                    continue
                if str(row.get("attn_kernel", "")).strip() != target_kernel:
                    continue
                if int(row.get("itr", -1)) != target_itr:
                    continue
                mem_raw = row.get("memory", row.get("memory_required", "nan"))
                mem_req = float(mem_raw)
                if math.isfinite(mem_req) and mem_req > 0.0:
                    return True
            except Exception:
                continue
        return False

    def _estimate_memory_required_gib_for_itr(self, *, itr: int, memory_cap_gib: float | None = None) -> float:
        probe = CQSAProbe(
            N=self.N,
            D=self.D,
            B=self.B,
            H=self.H,
            c=self.c,
            interest_set=self.interest_set,
            dtype=self.dtype,
            input_std=self.input_std,
            memory_cap_gib=memory_cap_gib,
        )
        local_size = int(probe._local_size_for_path0(int(itr)))
        est_bytes = int(probe._estimate_single_subseq_bytes(int(local_size)))
        return float(est_bytes) / (1024**3)

    def _device_name(self) -> str:
        try:
            return str(torch.cuda.get_device_name(self.device_index))
        except Exception:
            return "unknown_device"

    def _memory_model_dir(self) -> Path:
        dtype_s = _dtype_to_str(self.dtype)
        dev = self._device_name().replace("/", "_").replace("\\", "_").strip()
        if len(dev) == 0:
            dev = "unknown_device"
        model_name = f"{dev}_D_{self.D}_B_{self.B}_H_{self.H}_{dtype_s}_{self.attn_kernel_name}"
        return Path(STREAM_CQSA_ROOT) / "memory_model" / model_name

    def _subseq_length_for_itr(self, itr: int) -> float:
        ratio = float(len(self.interest_set)) / float(self.c)
        return float(self.N) * (ratio ** float(max(0, int(itr))))

    def _matching_itr0_history_N_values(self, history_path: Path) -> List[int]:
        if (not history_path.exists()) or history_path.stat().st_size == 0:
            return []
        try:
            with history_path.open("r", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            return []
        target_dtype = _dtype_to_str(self.dtype)
        target_kernel = str(self.attn_kernel_name)
        vals: List[int] = []
        for row in rows:
            try:
                if int(row.get("D", -1)) != self.D:
                    continue
                if int(row.get("B", -1)) != self.B:
                    continue
                if int(row.get("H", -1)) != self.H:
                    continue
                if int(row.get("itr", -1)) != 0:
                    continue
                if str(row.get("dtype", "")) != target_dtype:
                    continue
                if str(row.get("attn_kernel", "")).strip() != target_kernel:
                    continue
                n = int(float(row.get("N", "nan")))
                m = float(row.get("memory", "nan"))
                if (not math.isfinite(m)) or m <= 0.0:
                    continue
                vals.append(int(n))
            except Exception:
                continue
        return sorted(set(vals))

    def _bootstrap_history_points_for_model(self, history_path: Path) -> List[int]:
        script_path = Path(STREAM_CQSA_ROOT) / "scripts" / "run_stream_cqsa.py"
        if not script_path.exists():
            return self._matching_itr0_history_N_values(history_path)

        n_targets = [1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000, 17000, 19000]
        dtype_s = _dtype_to_str(self.dtype)
        for i in range(len(n_targets)):
            while True:
                n_try = max(1, int(n_targets[i]))
                cmd = [
                    "python",
                    str(script_path),
                    "--service",
                    "--N",
                    str(n_try),
                    "--D",
                    str(self.D),
                    "--B",
                    str(self.B),
                    "--H",
                    str(self.H),
                    "--dtype",
                    str(dtype_s),
                    "--itrs",
                    "0",
                    "--EXPMT_attention_kernel",
                    str(self.attn_kernel_name),
                    "--PROBE_history_file",
                    str(history_path),
                ]
                proc = subprocess.run(cmd, cwd=str(STREAM_CQSA_ROOT), capture_output=True, text=True)
                out = f"{proc.stdout}\n{proc.stderr}".upper()
                is_oom = (proc.returncode != 0) or ("OOM" in out) or ("OUT_OF_MEMORY" in out)
                if is_oom:
                    if n_try <= 1:
                        break
                    for j in range(i, len(n_targets)):
                        n_targets[j] = max(1, int(n_targets[j]) // 2)
                    continue
                break

        return self._matching_itr0_history_N_values(history_path)

    def _ensure_memory_model(self, history_path: Path) -> Path:
        model_dir = self._memory_model_dir()
        model_pt = model_dir / "memory_fit.pt"
        if model_pt.exists():
            return model_pt

        model_dir.mkdir(parents=True, exist_ok=True)
        n_vals = self._matching_itr0_history_N_values(history_path)
        if len(n_vals) < 10:
            n_vals = self._bootstrap_history_points_for_model(history_path)

        if len(n_vals) < 10:
            raise RuntimeError(
                "Unable to build memory model: need >=10 probing points at itr=0 for filter "
                f"D={self.D}, B={self.B}, H={self.H}, dtype={_dtype_to_str(self.dtype)}, "
                f"attn_kernel={self.attn_kernel_name}. Current recorded N values: {n_vals}. "
                "Please run probing manually with ROOT/scripts/run_stream_cqsa.py --service ... and retry."
            )

        fitting_mem(
            profile_path=history_path,
            filter={
                "D": int(self.D),
                "B": int(self.B),
                "H": int(self.H),
                "itr": 0,
                "dtype": str(_dtype_to_str(self.dtype)),
                "attn_kernel": str(self.attn_kernel_name),
            },
            save_model=model_dir,
            degree="auto",
        )
        if not model_pt.exists():
            raise RuntimeError(f"Memory model fitting did not create model file: {model_pt}")
        return model_pt

    def _infer_probe_pair_from_model(
        self,
        *,
        history_path: Path,
        memory_avail_gib: float,
        probe_num_itr: int | None,
        probe_max_parallel_subseq: int | None,
    ) -> Tuple[int, int, float]:
        model_pt = self._ensure_memory_model(history_path)

        def _pred_mem(itr: int) -> float:
            subseq_len = self._subseq_length_for_itr(int(itr))
            mem = float(pred_mem(model_pt, float(subseq_len)))
            if (not math.isfinite(mem)) or mem <= 0.0:
                raise RuntimeError(
                    f"Invalid memory prediction from model {model_pt} at itr={itr}: {mem}"
                )
            return float(mem)

        if probe_num_itr is not None and probe_max_parallel_subseq is not None:
            itr = int(probe_num_itr)
            mem = _pred_mem(itr)
            k = max(1, min(int(probe_max_parallel_subseq), int(self.c ** max(0, itr))))
            return int(itr), int(k), float(mem)

        if probe_num_itr is not None:
            itr = int(probe_num_itr)
            mem = _pred_mem(itr)
            k = int(math.floor(float(memory_avail_gib) / float(mem)))
            k = max(1, min(int(self.c ** max(0, itr)), int(k)))
            return int(itr), int(k), float(mem)

        # No itr provided: choose smallest itr that fits one subsequence in available memory.
        itr = int(DEFAULT_PROBE_START_NUM_ITR)
        mem = _pred_mem(itr)
        while mem > float(memory_avail_gib) and itr < int(DEFAULT_PROBE_MAX_NUM_ITR):
            itr += 1
            mem = _pred_mem(itr)

        k = int(math.floor(float(memory_avail_gib) / float(mem)))
        k = max(1, min(int(self.c ** max(0, itr)), int(k)))
        return int(itr), int(k), float(mem)

    def run(
        self,
        *,
        output_dir: str | Path | None,
        probe_max_parallel_subseq: int | None = None,
        probe_memory_cap_gib: float | None = None,
        probe_num_itr: int | None = None,
        probing_history_file: str | Path | None = None,
        new_probing_history_file: bool | None = None,
        use_probing_history: bool = True,
        schedule_mode: str = "event",
        override_max_subsequence: int | None = None,
        dump_cuda_snapshot: bool = True,
        merge_on_gpu: bool = False,
        keep_final_output: bool = False,
        save_accumulators: bool = False,
        max_rounds: int | None = None,
        num_subseq_limit: int | None = None,
        print_every_rounds: int = 10,
        save_log: bool = True,
        print_gpu_mem_initial: bool = True,
        source_q: torch.Tensor | None = None,
        source_k: torch.Tensor | None = None,
        source_v: torch.Tensor | None = None,
        qkv_generation_mode: str = "constant",
        return_internal_state: bool = False,
        return_timing_ms: bool = False,
        return_timeline_rows: bool = False,
    ) -> StreamResult:
        outdir: Path | None = None
        if bool(save_log):
            if output_dir is None:
                raise ValueError("output_dir is required when save_log=True.")
            outdir = Path(output_dir).expanduser().resolve()
            outdir.mkdir(parents=True, exist_ok=True)

        use_external_qkv = (source_q is not None) or (source_k is not None) or (source_v is not None)
        if use_external_qkv and not (source_q is not None and source_k is not None and source_v is not None):
            raise ValueError("source_q/source_k/source_v must be all provided or all None.")
        if use_external_qkv:
            assert source_q is not None and source_k is not None and source_v is not None
            if source_q.shape != source_k.shape or source_q.shape != source_v.shape:
                raise ValueError(
                    f"source_q/source_k/source_v shape mismatch: {source_q.shape}, {source_k.shape}, {source_v.shape}."
                )
            if source_q.ndim != 4:
                raise ValueError("source_q/source_k/source_v must be rank-4 [B,H,N,D].")
            b0, h0, n0, d0 = [int(x) for x in source_q.shape]
            if b0 != self.B or h0 != self.H or n0 != self.N or d0 != self.D:
                raise ValueError(
                    "source_q/source_k/source_v shape must match runner dimensions "
                    f"[B,H,N,D]=[{self.B},{self.H},{self.N},{self.D}], got [{b0},{h0},{n0},{d0}]."
                )
            if source_q.device != source_k.device or source_q.device != source_v.device:
                raise ValueError("source_q/source_k/source_v must be on the same device.")

        schedule_mode_n = str(schedule_mode).strip().lower()
        if schedule_mode_n != "event":
            print(f"[cqsa_stream] forcing event scheduler (got schedule_mode='{schedule_mode}')", flush=True)
            schedule_mode_n = "event"
        if bool(merge_on_gpu):
            print("[cqsa_stream] forcing CPU merge path (merge_on_gpu is disabled).", flush=True)
            merge_on_gpu = False

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        snapshot_started = False
        snapshot_path: Path | None = None
        try:
            if bool(save_log) and dump_cuda_snapshot and hasattr(torch.cuda.memory, "_record_memory_history"):
                torch.cuda.memory._record_memory_history(
                    enabled="all",
                    context="all",
                    stacks="all",
                    max_entries=100_000,
                    clear_history=True,
                )
                snapshot_started = True
        except Exception:
            snapshot_started = False

        t_start = time.time()
        t_perf_start = time.perf_counter()
        timing_ms: Dict[str, float] = {
            "probe": 0.0,
            "mask_gen": 0.0,
            "qkv_cpu_gen": 0.0,
            "h2d": 0.0,
            "compute": 0.0,
            "sync": 0.0,
            "d2h": 0.0,
            "merge": 0.0,
            "total": 0.0,
        }
        self._log_mem("run_start")
        self._log_mem("probe_start")
        history_path: Path | None = None
        used_b, free_b, total_b, cap_b, effective_budget_bytes, usable_budget_bytes = _compute_probe_budget_bytes(
            probe_memory_cap_gib
        )
        memory_avail_gib = float(effective_budget_bytes) / (1024**3)
        device_name = self._device_name()
        if bool(print_gpu_mem_initial):
            print(
                "[cqsa_stream] gpu_mem_initial "
                f"device='{device_name}' "
                f"used_gib={float(used_b)/(1024**3):.3f} "
                f"free_gib={float(free_b)/(1024**3):.3f} "
                f"total_gib={float(total_b)/(1024**3):.3f} "
                f"memory_cap_gib={float(cap_b)/(1024**3):.3f} "
                f"effective_budget_gib={float(effective_budget_bytes)/(1024**3):.3f} "
                f"usable_budget_gib={float(usable_budget_bytes)/(1024**3):.3f} "
                f"(MEM_BUDGET_COE={float(MEM_BUDGET_COE):.3f})",
                flush=True,
            )

        itr_max = int(probe_num_itr) if probe_num_itr is not None else int(DEFAULT_PROBE_START_NUM_ITR)
        if int(itr_max) < 0:
            raise ValueError(f"probe_num_itr/itr must be >= 0, got {itr_max}")
        max_subsequence = 1
        attempts: List[ProbeAttempt] = []
        probe_source = "disabled"
        probe_result: Dict[str, Any] = {
            "N": self.N,
            "D": self.D,
            "B": self.B,
            "H": self.H,
            "c": self.c,
            "interest_set": list(self.interest_set),
            "itr_max": int(itr_max),
            "recommended_parallel_subseq": int(max_subsequence),
            "single_subseq_mem_consumption_gib": None,
            "attempts": attempts,
        }
        if probe_memory_cap_gib is not None:
            est_mem_gib = float(
                self._estimate_memory_required_gib_for_itr(
                    itr=int(itr_max),
                    memory_cap_gib=float(probe_memory_cap_gib),
                )
            )
            probe_result["single_subseq_mem_consumption_gib"] = float(est_mem_gib)
            if float(est_mem_gib) > float(memory_avail_gib):
                raise RuntimeError(
                    "Hard memory budget violated before launch: "
                    f"estimated_single_subseq_gib={est_mem_gib:.3f} > available_budget_gib={memory_avail_gib:.3f} "
                    f"(itr={int(itr_max)})."
                )

        self._log_mem("probe_done")

        if override_max_subsequence is not None and int(override_max_subsequence) > 0:
            print(
                "[cqsa_stream] ignoring override_max_subsequence because subseq parallelism is fixed at 1.",
                flush=True,
            )
        all_paths: List[Tuple[int, ...]]
        if itr_max <= 0:
            all_paths = [tuple()]
        else:
            all_paths = list(product(range(self.c), repeat=itr_max))
        num_paths_total = len(all_paths)
        print(
            f"[cqsa_stream] probe_done source={probe_source} itr_max={itr_max} max_subsequence={max_subsequence} "
            f"num_paths_total={num_paths_total}",
            flush=True,
        )

        num_global_cpu = None
        den_global_cpu = None
        num_global_gpu = None
        den_global_gpu = None
        if merge_on_gpu:
            num_global_gpu = torch.zeros((self.B, self.N, self.H, self.D), dtype=torch.float32, device=self.device)
            den_global_gpu = torch.zeros((self.B, self.H, self.N), dtype=torch.float32, device=self.device)
        else:
            num_global_cpu = torch.zeros((self.B, self.N, self.H, self.D), dtype=torch.float32, device="cpu")
            den_global_cpu = torch.zeros((self.B, self.H, self.N), dtype=torch.float32, device="cpu")

        streams = [torch.cuda.Stream(device=self.device) for _ in range(max_subsequence)]
        inflight: List[Dict[str, Any] | None] = [None for _ in range(max_subsequence)]
        next_path_idx = 0
        paths_done = 0
        processed_paths: List[Tuple[int, ...]] = []
        max_rounds_i: int | None = None
        round1_paths = 0
        oom_downscale_count = 0
        target_change_reasons: List[str] = []
        if num_subseq_limit is not None:
            target_paths = min(int(num_paths_total), max(0, int(num_subseq_limit)))
            if target_paths < num_paths_total:
                print(
                    f"[cqsa_stream] num_subseq_limit={num_subseq_limit} => target_paths={target_paths}/{num_paths_total}",
                    flush=True,
                )
        elif max_rounds is None:
            target_paths = int(num_paths_total)
        else:
            max_rounds_i = int(max_rounds)
            if max_rounds_i <= 0:
                target_paths = min(int(num_paths_total), int(max_subsequence))
            else:
                target_paths = min(int(num_paths_total), int(max_rounds_i) * int(max_subsequence))
            if target_paths < num_paths_total:
                print(
                    f"[cqsa_stream] max_rounds={max_rounds} => target_paths={target_paths}/{num_paths_total}",
                    flush=True,
                )
        progress_bar = None

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
                print("[cqsa_stream] tqdm not installed; ETA progress bar disabled.", flush=True)

        def _progress_update(delta: int = 1) -> None:
            if int(delta) <= 0:
                return
            _ensure_progress_bar()
            if progress_bar is not None:
                total = progress_bar.total
                step = int(delta)
                if total is not None:
                    remaining = int(total) - int(progress_bar.n)
                    if remaining <= 0:
                        return
                    step = min(int(step), int(remaining))
                    if step <= 0:
                        return
                progress_bar.update(int(step))

        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

        def _is_cuda_oom(exc: BaseException) -> bool:
            oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
            cur: BaseException | None = exc
            while cur is not None:
                if oom_type is not None and isinstance(cur, oom_type):
                    return True
                cur = cur.__cause__
            msg = str(exc).lower()
            return (
                ("out of memory" in msg)
                or ("cuda_oom" in msg)
                or ("cuda out of memory" in msg)
                or ("cuda oom" in msg)
                or (("oom" in msg) and ("cuda" in msg))
            )

        def _set_target_paths(new_target_paths: int, reason: str) -> None:
            nonlocal target_paths, progress_bar
            old_target = int(target_paths)
            target_paths = max(int(paths_done), min(int(num_paths_total), int(new_target_paths)))
            if progress_bar is not None:
                progress_bar.total = int(target_paths)
                progress_bar.refresh()
            if int(target_paths) != int(old_target):
                target_change_reasons.append(str(reason))
                print(
                    f"[cqsa_stream] target_paths update ({reason}): {old_target} -> {target_paths}",
                    flush=True,
                )

        def _round_based_target_for_k(k: int) -> int:
            if num_subseq_limit is not None:
                return min(int(num_paths_total), max(0, int(num_subseq_limit)))
            if max_rounds_i is None:
                return int(num_paths_total)
            if max_rounds_i <= 0:
                return min(int(num_paths_total), max(1, int(k)))
            return min(int(num_paths_total), int(max_rounds_i) * int(k))

        def _recompute_target_paths_for_new_parallel(*, reason: str) -> None:
            new_target = _round_based_target_for_k(int(max_subsequence))
            _set_target_paths(int(new_target), reason=reason)

        def _decrease_parallelism_on_oom(path: Tuple[int, ...], slot_idx: int) -> bool:
            nonlocal max_subsequence, oom_downscale_count
            old_k = int(max_subsequence)
            if old_k <= 1:
                return False
            max_subsequence = int(old_k - 1)
            oom_downscale_count += 1
            _recompute_target_paths_for_new_parallel(reason="oom_downscale")
            print(
                "[cqsa_stream] CUDA OOM on subsequence launch: "
                f"reduce max_subsequence {old_k} -> {max_subsequence}; "
                f"requeue path={path} slot={slot_idx}",
                flush=True,
            )
            self._log_mem(
                "parallel_readjust_down_oom",
                paths_done=paths_done,
                active_subseq=sum(1 for t in inflight if t is not None),
                num_itr=itr_max,
            )
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return True

        def _launch_subsequence(path: Tuple[int, ...], slot_idx: int) -> str:
            nonlocal paths_done
            t0_mask = time.perf_counter()
            mask_one = self.mask_engine.gen_mask(
                N=self.N,
                num_itr=itr_max,
                quorum_idx=list(path),
                interest_set=self.interest_set,
                c=self.c,
                include_trace=False,
            )
            timing_ms["mask_gen"] += (time.perf_counter() - t0_mask) * 1000.0

            token_ids_np = np.asarray(mask_one["token_ids"], dtype=np.int64)
            local_size = int(mask_one["local_size"])
            if local_size <= 0 or token_ids_np.size == 0:
                paths_done += 1
                _progress_update(1)
                return "skipped"

            t0_qkv = time.perf_counter()
            token_ids_cpu = torch.from_numpy(token_ids_np.astype(np.int64, copy=False)).to(torch.long)
            src_q_cuda: torch.Tensor | None = None
            src_k_cuda: torch.Tensor | None = None
            src_v_cuda: torch.Tensor | None = None
            if use_external_qkv:
                assert source_q is not None and source_k is not None and source_v is not None
                src_device = source_q.device
                if src_device.type == "cpu":
                    q_cpu = source_q.index_select(2, token_ids_cpu).permute(0, 2, 1, 3).contiguous()
                    k_cpu = source_k.index_select(2, token_ids_cpu).permute(0, 2, 1, 3).contiguous()
                    v_cpu = source_v.index_select(2, token_ids_cpu).permute(0, 2, 1, 3).contiguous()
                    q_cpu = q_cpu.pin_memory()
                    k_cpu = k_cpu.pin_memory()
                    v_cpu = v_cpu.pin_memory()
                else:
                    # Keep source tensors and gather on the launch stream to avoid
                    # cross-stream hazards (default-stream gather + non-default compute).
                    src_q_cuda = source_q
                    src_k_cuda = source_k
                    src_v_cuda = source_v
                    q_cpu = None
                    k_cpu = None
                    v_cpu = None
            else:
                q_cpu, k_cpu, v_cpu = self._make_qkv_cpu(
                    local_size,
                    path,
                    slot_idx,
                )
            timing_ms["qkv_cpu_gen"] += (time.perf_counter() - t0_qkv) * 1000.0
            token_ids_gpu = None
            if merge_on_gpu:
                token_ids_gpu = token_ids_cpu.to(self.device, non_blocking=True)

            s = streams[slot_idx]
            ev_h2d_start = torch.cuda.Event(enable_timing=True)
            ev_h2d_end = torch.cuda.Event(enable_timing=True)
            ev_compute_start = torch.cuda.Event(enable_timing=True)
            ev_compute_end = torch.cuda.Event(enable_timing=True)
            try:
                with torch.cuda.stream(s):
                    ev_h2d_start.record()
                    if src_q_cuda is not None and src_k_cuda is not None and src_v_cuda is not None:
                        token_ids_src = token_ids_cpu.to(self.device, non_blocking=True)
                        q = src_q_cuda.index_select(2, token_ids_src).permute(0, 2, 1, 3).contiguous()
                        k = src_k_cuda.index_select(2, token_ids_src).permute(0, 2, 1, 3).contiguous()
                        v = src_v_cuda.index_select(2, token_ids_src).permute(0, 2, 1, 3).contiguous()
                    elif q_cpu is not None and q_cpu.device == self.device:
                        assert k_cpu is not None and v_cpu is not None
                        q, k, v = q_cpu, k_cpu, v_cpu
                    else:
                        assert q_cpu is not None and k_cpu is not None and v_cpu is not None
                        q = q_cpu.to(self.device, non_blocking=True)
                        k = k_cpu.to(self.device, non_blocking=True)
                        v = v_cpu.to(self.device, non_blocking=True)
                    ev_h2d_end.record()
                    ev_compute_start.record()
                    try:
                        num_i, den_i = self.subseq_attention_fn(
                            q=q,
                            k=k,
                            v=v,
                            cqs_mask=mask_one,
                            softmax_scale=float(self.softmax_scale),
                        )
                    except TypeError as exc:
                        raise TypeError(
                            "subseq_attention_fn must accept keyword args: q, k, v, cqs_mask, softmax_scale"
                        ) from exc

                    if not torch.is_tensor(num_i) or not torch.is_tensor(den_i):
                        raise TypeError("subseq_attention_fn must return (Num_i, Den_i) as torch.Tensor pair.")
                    if num_i.device != self.device or den_i.device != self.device:
                        raise RuntimeError("subseq_attention_fn must return tensors on the same CUDA device.")
                    expected_num_shape = (self.B, local_size, self.H, self.D)
                    expected_den_shape = (self.B, self.H, local_size)
                    if tuple(num_i.shape) != expected_num_shape:
                        raise RuntimeError(
                            f"subseq_attention_fn returned Num_i with shape {tuple(num_i.shape)}, "
                            f"expected {expected_num_shape}."
                        )
                    if tuple(den_i.shape) != expected_den_shape:
                        raise RuntimeError(
                            f"subseq_attention_fn returned Den_i with shape {tuple(den_i.shape)}, "
                            f"expected {expected_den_shape}."
                        )

                    num_i = torch.nan_to_num(num_i.float(), nan=0.0, posinf=0.0, neginf=0.0)
                    den_i = torch.nan_to_num(den_i.float(), nan=0.0, posinf=0.0, neginf=0.0)
                    ev_compute_end.record()
            except RuntimeError as exc:
                if _is_cuda_oom(exc):
                    return "oom_requeue"
                raise

            inflight[slot_idx] = {
                "slot": slot_idx,
                "path": path,
                "token_ids_cpu": token_ids_cpu,
                "token_ids_gpu": token_ids_gpu,
                "num_i_gpu": num_i,
                "den_i_gpu": den_i,
                "ev_h2d_start": ev_h2d_start,
                "ev_h2d_end": ev_h2d_end,
                "ev_compute_start": ev_compute_start,
                "ev_compute_end": ev_compute_end,
                "tensors_gpu": [q, k, v],
            }
            return "launched"

        def _drain_slot(slot_idx: int) -> bool:
            nonlocal paths_done
            task = inflight[slot_idx]
            if task is None:
                return False

            timing_ms["h2d"] += float(task["ev_h2d_start"].elapsed_time(task["ev_h2d_end"]))
            timing_ms["compute"] += float(task["ev_compute_start"].elapsed_time(task["ev_compute_end"]))

            token_ids_cpu = task["token_ids_cpu"]
            if merge_on_gpu:
                assert num_global_gpu is not None and den_global_gpu is not None
                t0_merge = time.perf_counter()
                token_ids_gpu = task["token_ids_gpu"]
                num_global_gpu.index_add_(1, token_ids_gpu, task["num_i_gpu"])
                den_global_gpu.index_add_(2, token_ids_gpu, task["den_i_gpu"])
                torch.cuda.synchronize(self.device)
                timing_ms["merge"] += (time.perf_counter() - t0_merge) * 1000.0
            else:
                assert num_global_cpu is not None and den_global_cpu is not None
                t0_d2h = time.perf_counter()
                num_i_cpu = task["num_i_gpu"].cpu()
                den_i_cpu = task["den_i_gpu"].cpu()
                timing_ms["d2h"] += (time.perf_counter() - t0_d2h) * 1000.0

                t0_merge = time.perf_counter()
                num_global_cpu.index_add_(1, token_ids_cpu, num_i_cpu)
                den_global_cpu.index_add_(2, token_ids_cpu, den_i_cpu)
                timing_ms["merge"] += (time.perf_counter() - t0_merge) * 1000.0

            paths_done += 1
            processed_paths.append(tuple(task["path"]))
            _progress_update(1)
            inflight[slot_idx] = None
            task.clear()
            return True

        if schedule_mode_n == "event":
            completion_q: queue.Queue[int] = queue.Queue()
            worker_queues: List[queue.Queue[Any]] = [queue.Queue(maxsize=1) for _ in range(max_subsequence)]
            stop_token = object()

            def _slot_worker(slot_idx: int) -> None:
                torch.cuda.set_device(self.device_index)
                q = worker_queues[slot_idx]
                while True:
                    item = q.get()
                    if item is stop_token:
                        return
                    ev = item["ev_compute_end"]
                    ev.synchronize()
                    completion_q.put(slot_idx)

            worker_threads = [
                threading.Thread(target=_slot_worker, args=(i,), daemon=True) for i in range(max_subsequence)
            ]
            for t in worker_threads:
                t.start()

            self._log_mem(
                "pipeline_start",
                round_idx=0,
                paths_done=0,
                active_subseq=0,
                num_itr=itr_max,
            )

            while paths_done < target_paths or any(task is not None for task in inflight):
                while next_path_idx < target_paths:
                    free_slot = next((i for i, task in enumerate(inflight[:max_subsequence]) if task is None), None)
                    if free_slot is None:
                        break
                    path = all_paths[next_path_idx]
                    next_path_idx += 1
                    launch_status = _launch_subsequence(path, free_slot)
                    if launch_status == "launched":
                        worker_queues[free_slot].put({"ev_compute_end": inflight[free_slot]["ev_compute_end"]})
                        self._log_mem(
                            "subseq_dispatch",
                            round_idx=max(0, int(paths_done // max(1, max_subsequence))),
                            paths_done=paths_done,
                            active_subseq=sum(1 for t in inflight if t is not None),
                            num_itr=itr_max,
                        )
                    elif launch_status == "oom_requeue":
                        next_path_idx = max(0, int(next_path_idx) - 1)
                        if not _decrease_parallelism_on_oom(path, free_slot):
                            if progress_bar is not None:
                                progress_bar.close()
                                progress_bar = None
                            raise RuntimeError(
                                f"CUDA OOM at max_subsequence=1 while launching path={path} (itr={itr_max})."
                            )
                        break

                if not any(task is not None for task in inflight):
                    break
                t0_sync = time.perf_counter()
                wait_slot = int(completion_q.get())
                timing_ms["sync"] += (time.perf_counter() - t0_sync) * 1000.0
                self._log_mem(
                    "pipeline_wait",
                    round_idx=max(0, int(paths_done // max(1, max_subsequence))),
                    paths_done=paths_done,
                    active_subseq=sum(1 for t in inflight if t is not None),
                    num_itr=itr_max,
                )
                if not _drain_slot(wait_slot):
                    continue

                self._log_mem(
                    "subseq_done",
                    round_idx=max(0, int(paths_done // max(1, max_subsequence))),
                    paths_done=paths_done,
                    active_subseq=sum(1 for t in inflight if t is not None),
                    num_itr=itr_max,
                )

            for q in worker_queues:
                q.put(stop_token)
            for t in worker_threads:
                t.join(timeout=1.0)
        else:
            round_idx = 0
            while paths_done < target_paths:
                launched_slots: List[int] = []
                self._log_mem(
                    "round_start",
                    round_idx=round_idx,
                    paths_done=paths_done,
                    active_subseq=0,
                    num_itr=itr_max,
                )

                while len(launched_slots) < max_subsequence and next_path_idx < target_paths:
                    slot_idx = len(launched_slots)
                    path = all_paths[next_path_idx]
                    next_path_idx += 1
                    launch_status = _launch_subsequence(path, slot_idx)
                    if launch_status == "launched":
                        launched_slots.append(slot_idx)
                    elif launch_status == "oom_requeue":
                        next_path_idx = max(0, int(next_path_idx) - 1)
                        if not _decrease_parallelism_on_oom(path, slot_idx):
                            if progress_bar is not None:
                                progress_bar.close()
                                progress_bar = None
                            raise RuntimeError(
                                f"CUDA OOM at max_subsequence=1 while launching path={path} (itr={itr_max})."
                            )
                        break

                if len(launched_slots) == 0:
                    if next_path_idx >= target_paths:
                        break
                    continue

                self._log_mem(
                    "round_launch_done",
                    round_idx=round_idx,
                    paths_done=paths_done,
                    active_subseq=len(launched_slots),
                    num_itr=itr_max,
                )

                t0_sync = time.perf_counter()
                for slot_idx in launched_slots:
                    streams[slot_idx].synchronize()
                timing_ms["sync"] += (time.perf_counter() - t0_sync) * 1000.0
                self._log_mem(
                    "round_synced",
                    round_idx=round_idx,
                    paths_done=paths_done,
                    active_subseq=len(launched_slots),
                    num_itr=itr_max,
                )

                for slot_idx in launched_slots:
                    _drain_slot(slot_idx)

                self._log_mem(
                    "round_accum_done",
                    round_idx=round_idx,
                    paths_done=paths_done,
                    active_subseq=len(launched_slots),
                    num_itr=itr_max,
                )

                torch.cuda.empty_cache()
                self._log_mem(
                    "round_cache_cleared",
                    round_idx=round_idx,
                    paths_done=paths_done,
                    active_subseq=0,
                    num_itr=itr_max,
                )

                round_idx += 1

        if progress_bar is not None:
            progress_bar.close()

        if target_paths < num_paths_total:
            if num_subseq_limit is not None:
                print(
                    f"[cqsa_stream] reached num_subseq_limit={num_subseq_limit}; stopped after {paths_done}/{num_paths_total} paths.",
                    flush=True,
                )
            else:
                print(
                    f"[cqsa_stream] reached max_rounds={max_rounds}; stopped after {paths_done}/{num_paths_total} paths.",
                    flush=True,
                )
        if max_rounds_i is not None or num_subseq_limit is not None:
            print(
                (
                    "[cqsa_stream] path_budget_summary: "
                    + (
                        f"num_subseq_limit={int(num_subseq_limit)}, "
                        if num_subseq_limit is not None
                        else f"max_rounds={max_rounds_i}, "
                    )
                    + f"round1_paths={int(round1_paths)}, "
                    + f"oom_downscales={int(oom_downscale_count)}, "
                    + f"final_max_subsequence={int(max_subsequence)}, "
                    + f"target_paths={int(target_paths)}, "
                    + f"paths_done={int(paths_done)}, "
                    + f"reasons={target_change_reasons if len(target_change_reasons)>0 else ['initial']}"
                ),
                flush=True,
            )

        output_cpu = None
        if keep_final_output:
            if merge_on_gpu:
                assert num_global_gpu is not None and den_global_gpu is not None
                Den_global = den_global_gpu.transpose(1, 2).unsqueeze(-1).clamp_min(1e-12)
                output_gpu = num_global_gpu / Den_global
                valid = den_global_gpu.transpose(1, 2).unsqueeze(-1) > 0
                output_gpu = torch.where(valid, output_gpu, torch.zeros_like(output_gpu))
                output_cpu = output_gpu.cpu()
                del output_gpu
            else:
                assert num_global_cpu is not None and den_global_cpu is not None
                Den_global = den_global_cpu.transpose(1, 2).unsqueeze(-1).clamp_min(1e-12)
                output_cpu = num_global_cpu / Den_global
                valid = den_global_cpu.transpose(1, 2).unsqueeze(-1) > 0
                output_cpu = torch.where(valid, output_cpu, torch.zeros_like(output_cpu))

        self._log_mem("run_done", paths_done=paths_done, num_itr=itr_max)
        elapsed_s = float(time.time() - t_start)

        timing_ms["total"] = (time.perf_counter() - t_perf_start) * 1000.0
        peak_cuda_allocated_gib = float(torch.cuda.max_memory_allocated(self.device)) / (1024**3)
        print(f"[cqsa_stream] peak_cuda_allocated_gib={peak_cuda_allocated_gib:.3f}", flush=True)
        end_to_end_core_path_time_ms = timing_ms["total"] - timing_ms["probe"] - timing_ms["qkv_cpu_gen"]
        total_optional_time_ms = timing_ms["probe"] + timing_ms["qkv_cpu_gen"]
        total_communication_time_ms = timing_ms["h2d"] + timing_ms["sync"] + timing_ms["d2h"]
        total_core_computation_time_ms = timing_ms["mask_gen"] + timing_ms["compute"] + timing_ms["merge"]
        timing_row = {
            "probe": float(timing_ms["probe"]),
            "qkv_cpu_gen": float(timing_ms["qkv_cpu_gen"]),
            "mask_gen": float(timing_ms["mask_gen"]),
            "h2d": float(timing_ms["h2d"]),
            "compute": float(timing_ms["compute"]),
            "sync": float(timing_ms["sync"]),
            "d2h": float(timing_ms["d2h"]),
            "merge": float(timing_ms["merge"]),
            "total optional time": float(total_optional_time_ms),
            "total communication time": float(total_communication_time_ms),
            "total core computation time": float(total_core_computation_time_ms),
            "end-to-end core path time": float(end_to_end_core_path_time_ms),
            "peak_cuda_allocated_gib": float(peak_cuda_allocated_gib),
        }

        timeline_csv = ""
        timing_breakdown_csv = ""
        probe_attempts_csv = ""
        probe_summary_txt = ""
        if bool(save_log):
            assert outdir is not None
            # Save timeline and probe artifacts.
            timeline_csv_p = outdir / "cuda_memory_timeline.csv"
            with timeline_csv_p.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.timeline_rows[0].keys()) if self.timeline_rows else [])
                if self.timeline_rows:
                    writer.writeheader()
                    writer.writerows(self.timeline_rows)

            timing_breakdown_csv_p = outdir / "timing_breakdown.csv"
            with timing_breakdown_csv_p.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(timing_row.keys()))
                writer.writeheader()
                writer.writerow(timing_row)

            if snapshot_started and hasattr(torch.cuda.memory, "_dump_snapshot"):
                snapshot_path = outdir / "cuda_memory_snapshot.pickle"
                try:
                    torch.cuda.memory._dump_snapshot(str(snapshot_path))
                except Exception:
                    snapshot_path = None

            timeline_csv = str(timeline_csv_p)
            timing_breakdown_csv = str(timing_breakdown_csv_p)
            probe_attempts_csv = ""
            probe_summary_txt = ""
        if snapshot_started and hasattr(torch.cuda.memory, "_record_memory_history"):
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass

        # Optional output dump only when explicitly requested.
        if bool(save_log) and output_cpu is not None:
            assert outdir is not None
            torch.save(output_cpu, outdir / "final_output_cpu.pt")
        if bool(save_log) and save_accumulators:
            assert outdir is not None
            if merge_on_gpu:
                assert num_global_gpu is not None and den_global_gpu is not None
                torch.save(num_global_gpu.cpu(), outdir / "accum_num_cpu.pt")
                torch.save(den_global_gpu.cpu(), outdir / "accum_den_cpu.pt")
            else:
                assert num_global_cpu is not None and den_global_cpu is not None
                torch.save(num_global_cpu, outdir / "accum_num_cpu.pt")
                torch.save(den_global_cpu, outdir / "accum_den_cpu.pt")

        return StreamResult(
            itr_max=int(itr_max),
            max_subsequence=int(max_subsequence),
            num_paths_total=int(num_paths_total),
            num_paths_processed=int(paths_done),
            elapsed_s=float(elapsed_s),
            timing_breakdown_csv=str(timing_breakdown_csv),
            timeline_csv=str(timeline_csv),
            probe_attempts_csv=str(probe_attempts_csv),
            probe_summary_txt=str(probe_summary_txt),
            cuda_snapshot_path=(str(snapshot_path) if snapshot_path is not None else None),
            output_cpu=output_cpu,
            num_global_cpu=(
                (
                    num_global_gpu.detach().cpu()
                    if (merge_on_gpu and num_global_gpu is not None)
                    else num_global_cpu
                )
                if bool(return_internal_state)
                else None
            ),
            den_global_cpu=(
                (
                    den_global_gpu.detach().cpu()
                    if (merge_on_gpu and den_global_gpu is not None)
                    else den_global_cpu
                )
                if bool(return_internal_state)
                else None
            ),
            processed_paths=(processed_paths if bool(return_internal_state) else None),
            timing_ms=(dict(timing_ms) if bool(return_timing_ms) else None),
            timeline_rows=(list(self.timeline_rows) if bool(return_timeline_rows) else None),
        )


def run_scqsa_stream(
    *,
    N: int,
    D: int,
    B: int = 1,
    H: int = 32,
    c: int = 7,
    interest_set: Sequence[int] = (0, 1, 3),
    subseq_attention_fn: SubsequenceAttentionFn | None = None,
    dtype: torch.dtype = torch.float16,
    input_std: float = 0.1,
    seed: int = 123,
    output_dir: str | Path = "./CQSA_stream_logs",
    probe_max_parallel_subseq: int | None = None,
    probe_memory_cap_gib: float | None = None,
    probe_num_itr: int | None = None,
    probing_history_file: str | Path | None = None,
    new_probing_history_file: bool | None = None,
    schedule_mode: str = "event",
    override_max_subsequence: int | None = None,
    dump_cuda_snapshot: bool = True,
    merge_on_gpu: bool = False,
    keep_final_output: bool = False,
    save_accumulators: bool = False,
    max_rounds: int | None = None,
    print_every_rounds: int = 10,
) -> StreamResult:
    """
    Single-call SCQSA entry point.

    Probe selection priority:
    1) User inputs: probe_num_itr + probe_max_parallel_subseq
    2) Probe history cache match
    3) Live probing

    Subsequence plugin:
    - `subseq_attention_fn` can override per-subsequence attention.
    - It must take `cqs_mask` and return `(Num_i, Den_i)`.
    """
    runner = CQSAStreamRunner(
        N=int(N),
        D=int(D),
        B=int(B),
        H=int(H),
        c=int(c),
        interest_set=tuple(int(x) for x in interest_set),
        subseq_attention_fn=subseq_attention_fn,
        dtype=dtype,
        input_std=float(input_std),
        seed=int(seed),
    )
    run_output_dir = _make_timestamped_run_dir(output_dir)
    print(f"[cqsa_stream] output_dir={run_output_dir}", flush=True)
    return runner.run(
        output_dir=run_output_dir,
        probe_max_parallel_subseq=(
            int(probe_max_parallel_subseq) if probe_max_parallel_subseq is not None else None
        ),
        probe_memory_cap_gib=(float(probe_memory_cap_gib) if probe_memory_cap_gib is not None else None),
        probe_num_itr=(int(probe_num_itr) if probe_num_itr is not None else None),
        probing_history_file=probing_history_file,
        new_probing_history_file=new_probing_history_file,
        schedule_mode=str(schedule_mode),
        override_max_subsequence=override_max_subsequence,
        dump_cuda_snapshot=bool(dump_cuda_snapshot),
        merge_on_gpu=bool(merge_on_gpu),
        keep_final_output=bool(keep_final_output),
        save_accumulators=bool(save_accumulators),
        max_rounds=max_rounds,
        print_every_rounds=int(print_every_rounds),
        save_log=True,
        source_q=None,
        source_k=None,
        source_v=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CQSA streaming runner with probe + memory timeline.")
    parser.add_argument("--N", type=int, default=1_000_000)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--c", type=int, default=7)
    parser.add_argument("--interest-set", type=int, nargs="+", default=[0, 1, 3])
    parser.add_argument("--input-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--probe-max-parallel-subseq", type=int, default=None)
    parser.add_argument(
        "--probe-memory-cap-gib",
        type=float,
        default=None,
        help="Optional attention-memory budget in GiB for probing (default: full GPU, bounded by current free memory).",
    )
    parser.add_argument("--probe-num-itr", type=int, default=None)
    parser.add_argument(
        "--probing-history-file",
        type=Path,
        default=None,
        help="Optional probing history CSV path. Default is Probing_history/{device_name}.csv.",
    )
    parser.add_argument(
        "--new-probing-history-file",
        action="store_true",
        help="Create and use a new indexed probing history file (e.g., probing_history_1.csv).",
    )
    parser.add_argument(
        "--schedule-mode",
        type=str,
        default="event",
        choices=["event", "round"],
        help="Scheduling mode for subsequence execution.",
    )
    parser.add_argument("--override-max-subsequence", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./CQSA_stream_logs"),
    )
    parser.add_argument("--keep-final-output", action="store_true")
    parser.add_argument("--save-accumulators", action="store_true")
    parser.add_argument("--merge-on-gpu", action="store_true")
    parser.add_argument("--no-snapshot", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--print-every-rounds", type=int, default=10)
    args = parser.parse_args()

    result = run_scqsa_stream(
        N=args.N,
        D=args.D,
        B=args.B,
        H=args.H,
        c=args.c,
        interest_set=tuple(args.interest_set),
        dtype=torch.float16,
        input_std=float(args.input_std),
        seed=int(args.seed),
        output_dir=args.output_dir,
        probe_max_parallel_subseq=args.probe_max_parallel_subseq,
        probe_memory_cap_gib=args.probe_memory_cap_gib,
        probe_num_itr=args.probe_num_itr,
        probing_history_file=args.probing_history_file,
        new_probing_history_file=bool(args.new_probing_history_file),
        schedule_mode=str(args.schedule_mode),
        override_max_subsequence=args.override_max_subsequence,
        dump_cuda_snapshot=not bool(args.no_snapshot),
        merge_on_gpu=bool(args.merge_on_gpu),
        keep_final_output=bool(args.keep_final_output),
        save_accumulators=bool(args.save_accumulators),
        max_rounds=args.max_rounds,
        print_every_rounds=int(args.print_every_rounds),
    )

    print("CQSA stream run complete.")
    print(f"itr_max={result.itr_max}")
    print(f"max_subsequence={result.max_subsequence}")
    print(f"num_paths_processed={result.num_paths_processed}/{result.num_paths_total}")
    print(f"elapsed_s={result.elapsed_s:.3f}")
    print(f"timing_breakdown_csv={result.timing_breakdown_csv}")
    print(f"timeline_csv={result.timeline_csv}")
    print(f"probe_attempts_csv={result.probe_attempts_csv}")
    print(f"probe_summary_txt={result.probe_summary_txt}")
    print(f"cuda_snapshot_path={result.cuda_snapshot_path}")


if __name__ == "__main__":
    main()
