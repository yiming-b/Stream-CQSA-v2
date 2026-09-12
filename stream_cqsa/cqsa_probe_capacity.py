from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

from .config import MEM_BUDGET_COE
from .cqs_mask import CQS_mask
from .interface import flash_attn_func_cqs_group_bits


def _group_runs_to_group_bits(local_size: int, group_runs: Sequence[Sequence[Tuple[int, int]]]) -> np.ndarray:
    bits = np.zeros((local_size,), dtype=np.int64)
    for bit_id, runs in enumerate(group_runs):
        if bit_id >= 63:
            raise ValueError("Too many unique group runs for int64 bit encoding.")
        bit = np.int64(1) << np.int64(bit_id)
        for s, e in runs:
            si, ei = int(s), int(e)
            if ei > si:
                bits[si:ei] |= bit
    return bits


def _live_mem() -> Tuple[int, int, int]:
    free_b, total_b = torch.cuda.mem_get_info()
    used_b = int(total_b - free_b)
    return used_b, int(free_b), int(total_b)


@dataclass
class ProbeAttempt:
    num_itr: int
    number_of_subsequence: int
    memory_consumption_gb: str
    status: str


ProbeAttentionFn = Callable[..., Tuple[torch.Tensor, torch.Tensor]]


class CQSAProbe:
    def __init__(
        self,
        *,
        N: int,
        D: int,
        B: int,
        H: int,
        c: int = 7,
        interest_set: Sequence[int] = (0, 1, 3),
        dtype: torch.dtype = torch.float16,
        input_std: float = 0.1,
        memory_cap_gib: float | None = None,
        memory_cap_gb: float | None = None,
        subseq_attention_fn: ProbeAttentionFn | None = None,
    ) -> None:
        self.N = int(N)
        self.D = int(D)
        self.B = int(B)
        self.H = int(H)
        self.c = int(c)
        self.interest_set = tuple(int(x) for x in interest_set)
        self.dtype = dtype
        self.input_std = float(input_std)
        # Backward compatibility:
        # - prefer memory_cap_gib
        # - accept legacy memory_cap_gb (same GiB unit semantics in code paths here)
        if memory_cap_gib is not None and memory_cap_gb is not None:
            if float(memory_cap_gib) != float(memory_cap_gb):
                raise ValueError(
                    "Both memory_cap_gib and memory_cap_gb were provided with different values; "
                    "please provide only one."
                )
        cap_val = memory_cap_gib if memory_cap_gib is not None else memory_cap_gb
        self.memory_cap_gib = None if cap_val is None else float(cap_val)
        self.softmax_scale = 1.0 / math.sqrt(self.D)
        self.mask_engine = CQS_mask(interest_set=self.interest_set, c=self.c)
        self.subseq_attention_fn = subseq_attention_fn

    def _dtype_nbytes(self) -> int:
        return int(torch.empty((), dtype=self.dtype).element_size())

    def _local_size_for_path0(self, num_itr: int) -> int:
        qidx = [0] * int(num_itr)
        mask = self.mask_engine.gen_mask(
            N=self.N,
            num_itr=int(num_itr),
            quorum_idx=qidx,
            interest_set=self.interest_set,
            c=self.c,
            include_trace=False,
        )
        return int(mask["local_size"])

    def _local_size_for_path0_fast(self, num_itr: int) -> int:
        """
        Fast exact local-size computation for quorum path [0] * num_itr.
        Avoids building token_ids / group_runs when probing-only speed is preferred.
        """
        cur_len = int(self.N)
        if int(num_itr) <= 0:
            return cur_len
        c = int(self.c)
        interest = tuple(int(x) for x in self.interest_set)
        for _ in range(int(num_itr)):
            q, r = divmod(cur_len, c)
            # first r chunks have size q+1, others q
            sizes = [q + 1 if i < r else q for i in range(c)]
            # path0 => subseq_i = 0 => selected chunks are exactly interest_set
            nxt = 0
            for ch in interest:
                if 0 <= ch < c:
                    nxt += int(sizes[ch])
            cur_len = int(nxt)
        return int(cur_len)

    def _estimate_single_subseq_bytes(self, local_size: int) -> int:
        """
        Analytic memory estimate for one subsequence runtime in stream path.
        This intentionally includes a safety margin for kernel workspace and
        allocator behavior.
        """
        lsz = int(local_size)
        if lsz <= 0:
            return 0

        n_elem = int(self.B) * int(lsz) * int(self.H) * int(self.D)
        dtype_b = self._dtype_nbytes()

        # Persistent per-subsequence tensors around kernel call.
        bytes_qkv = 3 * n_elem * dtype_b                      # q,k,v
        bytes_out = n_elem * dtype_b                          # out
        bytes_num = n_elem * 4                                # num (fp32)
        bytes_lse = int(self.B) * int(self.H) * int(lsz) * 4  # lse (fp32)
        bytes_den = int(self.B) * int(self.H) * int(lsz) * 4  # den (fp32)
        bytes_group_bits = int(lsz) * 8                       # int64 mask bits

        base_bytes = bytes_qkv + bytes_out + bytes_num + bytes_lse + bytes_den + bytes_group_bits

        # Safety headroom for transient kernels / allocator fragmentation.
        safety_mul = 1.35
        safety_add = 256 * 1024 * 1024  # 256 MiB
        return int(base_bytes * safety_mul + safety_add)

    def _mask_for_path0(self, num_itr: int, *, dummy_mask: bool = False) -> Tuple[int, torch.Tensor]:
        if bool(dummy_mask):
            local_size = int(self._local_size_for_path0_fast(int(num_itr)))
            bits = torch.zeros((local_size,), device="cuda", dtype=torch.int64).contiguous()
            return local_size, bits

        qidx = [0] * int(num_itr)
        mask = self.mask_engine.gen_mask(
            N=self.N,
            num_itr=int(num_itr),
            quorum_idx=qidx,
            interest_set=self.interest_set,
            c=self.c,
            include_trace=False,
        )
        local_size = int(mask["local_size"])
        bits_np = _group_runs_to_group_bits(local_size, mask["group_runs"])
        bits = torch.from_numpy(bits_np).to(device="cuda", dtype=torch.int64).contiguous()
        return local_size, bits

    def _cqs_mask_for_path0(self, num_itr: int, *, dummy_mask: bool = False) -> Dict[str, Any]:
        if bool(dummy_mask):
            local_size = int(self._local_size_for_path0_fast(int(num_itr)))
            return {"local_size": int(local_size), "group_runs": []}
        qidx = [0] * int(num_itr)
        mask = self.mask_engine.gen_mask(
            N=self.N,
            num_itr=int(num_itr),
            quorum_idx=qidx,
            interest_set=self.interest_set,
            c=self.c,
            include_trace=False,
        )
        return mask

    def _probe_once(
        self,
        num_itr: int,
        number_of_subsequence: int,
        *,
        dummy_mask: bool = False,
        return_timing: bool = False,
    ) -> Tuple[bool, int, int, int] | Tuple[bool, int, int, int, Dict[str, float]]:
        """
        Returns:
            (fit_without_oom, mem_delta_bytes, peak_delta_bytes, peak_live_used_bytes)
            If return_timing=True, appends a timing dict with keys:
              - qkv_gen_ms
        """
        if self.subseq_attention_fn is None:
            local_size, group_bits = self._mask_for_path0(num_itr, dummy_mask=bool(dummy_mask))
            cqs_mask = None
        else:
            cqs_mask = self._cqs_mask_for_path0(num_itr, dummy_mask=bool(dummy_mask))
            local_size = int(cqs_mask["local_size"])
            group_bits = None
        if local_size <= 0:
            return True, 0, 0, 0

        streams = [torch.cuda.Stream() for _ in range(max(1, int(number_of_subsequence)))]
        staged: List[torch.Tensor] = []
        qkv_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        peak_used = 0

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        used_before, _, _ = _live_mem()
        peak_used = max(peak_used, int(used_before))

        try:
            for i in range(len(streams)):
                s = streams[i]
                with torch.cuda.stream(s):
                    # Probe with constant-valued Q/K/V tensors to avoid expensive random-matrix generation.
                    # Three constants are sampled (one per Q/K/V) once per subsequence launch.
                    ev_qkv_start = torch.cuda.Event(enable_timing=True)
                    ev_qkv_end = torch.cuda.Event(enable_timing=True)
                    ev_qkv_start.record()
                    q_const = float(torch.rand((), device="cpu").item())
                    k_const = float(torch.rand((), device="cpu").item())
                    v_const = float(torch.rand((), device="cpu").item())
                    q = torch.empty(self.B, local_size, self.H, self.D, device="cuda", dtype=self.dtype).fill_(q_const)
                    k = torch.empty(self.B, local_size, self.H, self.D, device="cuda", dtype=self.dtype).fill_(k_const)
                    v = torch.empty(self.B, local_size, self.H, self.D, device="cuda", dtype=self.dtype).fill_(v_const)
                    ev_qkv_end.record()
                    qkv_events.append((ev_qkv_start, ev_qkv_end))
                    if self.subseq_attention_fn is None:
                        assert group_bits is not None
                        out, lse, _ = flash_attn_func_cqs_group_bits(
                            q,
                            k,
                            v,
                            group_bits,
                            dropout_p=0.0,
                            softmax_scale=float(self.softmax_scale),
                            causal=False,
                            return_attn_probs=True,
                        )
                        # Build Num_i numerator and Den_i denominator to match probe target.
                        den = torch.exp(lse.float())  # [B, H, L]
                        num = out.float() * den.transpose(1, 2).unsqueeze(-1)  # [B, L, H, D]
                        del out, lse
                    else:
                        assert cqs_mask is not None
                        try:
                            num, den = self.subseq_attention_fn(
                                q=q,
                                k=k,
                                v=v,
                                cqs_mask=cqs_mask,
                                softmax_scale=float(self.softmax_scale),
                            )
                        except TypeError as exc:
                            raise TypeError(
                                "subseq_attention_fn must accept keyword args: q, k, v, cqs_mask, softmax_scale"
                            ) from exc
                        if not torch.is_tensor(num) or not torch.is_tensor(den):
                            raise TypeError("subseq_attention_fn must return (Num_i, Den_i) as torch.Tensor pair.")
                        if num.device != q.device or den.device != q.device:
                            raise RuntimeError("subseq_attention_fn must return tensors on CUDA device.")
                        expected_num_shape = (self.B, local_size, self.H, self.D)
                        expected_den_shape = (self.B, self.H, local_size)
                        if tuple(num.shape) != expected_num_shape:
                            raise RuntimeError(
                                f"subseq_attention_fn returned Num_i with shape {tuple(num.shape)}, expected {expected_num_shape}."
                            )
                        if tuple(den.shape) != expected_den_shape:
                            raise RuntimeError(
                                f"subseq_attention_fn returned Den_i with shape {tuple(den.shape)}, expected {expected_den_shape}."
                            )
                    num = torch.nan_to_num(num.float(), nan=0.0, posinf=0.0, neginf=0.0)
                    den = torch.nan_to_num(den.float(), nan=0.0, posinf=0.0, neginf=0.0)

                staged.extend([q, k, v, den, num])
                used, _, _ = _live_mem()
                peak_used = max(peak_used, int(used))

            for s in streams:
                s.synchronize()
            qkv_gen_ms = 0.0
            for ev_s, ev_e in qkv_events:
                try:
                    qkv_gen_ms += float(ev_s.elapsed_time(ev_e))
                except Exception:
                    pass
            used_after, _, _ = _live_mem()
            peak_used = max(peak_used, int(used_after))
            mem_delta = max(0, int(used_after) - int(used_before))
            peak_delta = max(0, int(peak_used) - int(used_before))
            if bool(return_timing):
                return True, int(mem_delta), int(peak_delta), int(peak_used), {"qkv_gen_ms": float(qkv_gen_ms)}
            return True, int(mem_delta), int(peak_delta), int(peak_used)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                peak_delta = max(0, int(peak_used) - int(used_before))
                if bool(return_timing):
                    return False, int(peak_delta), int(peak_delta), int(peak_used), {"qkv_gen_ms": 0.0}
                return False, int(peak_delta), int(peak_delta), int(peak_used)
            raise
        finally:
            staged.clear()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def run(
        self,
        *,
        start_num_itr: int = 1,
        max_num_itr: int = 12,
        max_parallel_subseq: int = 64,
    ) -> Dict[str, Any]:
        if int(start_num_itr) < 0:
            raise ValueError(f"start_num_itr must be >= 0, got {start_num_itr}")
        if int(max_num_itr) < int(start_num_itr):
            raise ValueError(
                f"max_num_itr must be >= start_num_itr, got max_num_itr={max_num_itr}, start_num_itr={start_num_itr}"
            )

        used0, free0, total0 = _live_mem()
        attempts: List[ProbeAttempt] = []

        # Memory budget for attention probing.
        # By default: cap is full GPU memory; effective budget is limited by current free memory.
        if self.memory_cap_gib is None:
            cap_bytes = int(total0)
        else:
            cap_bytes = int(float(self.memory_cap_gib) * (1024**3))
            if cap_bytes <= 0:
                raise ValueError(f"memory_cap_gib must be > 0, got {self.memory_cap_gib}")
        effective_budget_bytes = int(min(int(free0), int(cap_bytes)))
        analytic_target_bytes = int(float(MEM_BUDGET_COE) * float(effective_budget_bytes))

        # 1) Analytic estimate chooses a good initial itr.
        local_size_est_by_itr: Dict[int, int] = {}
        est_mem_by_itr: Dict[int, int] = {}
        itr_guess: int | None = None
        for itr in range(int(start_num_itr), int(max_num_itr) + 1):
            lsz = self._local_size_for_path0(itr)
            local_size_est_by_itr[int(itr)] = int(lsz)
            est_b = self._estimate_single_subseq_bytes(lsz)
            est_mem_by_itr[int(itr)] = int(est_b)
            if est_b <= analytic_target_bytes and itr_guess is None:
                itr_guess = int(itr)
        if itr_guess is None:
            itr_guess = int(max_num_itr)

        # 2) Validate estimate by real run; adjust upward on OOM.
        itr_max = None
        mem_single = None
        peak_delta_single = None
        peak_abs_single = None
        validated_itr = int(itr_guess)
        while validated_itr <= int(max_num_itr):
            fit, mem_delta, peak_delta, peak_abs = self._probe_once(num_itr=validated_itr, number_of_subsequence=1)
            if not fit:
                attempts.append(
                    ProbeAttempt(
                        num_itr=int(validated_itr),
                        number_of_subsequence=1,
                        memory_consumption_gb="OOM",
                        status="OOM",
                    )
                )
                validated_itr += 1
                continue

            attempts.append(
                ProbeAttempt(
                    num_itr=int(validated_itr),
                    number_of_subsequence=1,
                    memory_consumption_gb=f"{peak_delta / (1024**3):.3f}",
                    status="OK",
                )
            )
            itr_max = int(validated_itr)
            # Use peak-delta (not end-state delta) as the conservative single-subsequence memory metric.
            mem_single = int(peak_delta)
            peak_delta_single = int(peak_delta)
            peak_abs_single = int(peak_abs)
            break

        if itr_max is None or mem_single is None:
            raise RuntimeError(
                f"No fitting num_itr found up to max_num_itr={int(max_num_itr)} "
                f"(initial analytic guess was itr={int(itr_guess)})."
            )

        # User-requested direct estimate:
        # max_parallel ~= floor(MEM_BUDGET_COE * effective_budget / single_subseq_peak_delta)
        usable_mem = float(MEM_BUDGET_COE) * float(effective_budget_bytes)
        estimated_k = int(math.floor(usable_mem / float(max(1, mem_single))))
        estimated_k = max(1, min(int(max_parallel_subseq), estimated_k))

        est_peak = float(estimated_k) * float(mem_single)
        remaining_ratio_est = max(0.0, float(effective_budget_bytes - est_peak) / float(max(1, effective_budget_bytes)))

        return {
            "N": self.N,
            "D": self.D,
            "B": self.B,
            "H": self.H,
            "c": self.c,
            "interest_set": list(self.interest_set),
            "total_gpu_mem_gib": float(total0) / (1024**3),
            "initial_used_gpu_mem_gib": float(used0) / (1024**3),
            "initial_free_gpu_mem_gib": float(free0) / (1024**3),
            "memory_cap_gib": (float(cap_bytes) / (1024**3)),
            "effective_probe_budget_gib": float(effective_budget_bytes) / (1024**3),
            "mem_budget_coe": float(MEM_BUDGET_COE),
            "analytic_target_gib": float(analytic_target_bytes) / (1024**3),
            "probe_method": "analytic_estimate_then_validate",
            "initial_itr_guess": int(itr_guess),
            "initial_itr_guess_local_size": int(local_size_est_by_itr.get(int(itr_guess), -1)),
            "initial_itr_guess_est_mem_gib": float(est_mem_by_itr.get(int(itr_guess), 0)) / (1024**3),
            "itr_max": int(itr_max),
            "single_subseq_mem_consumption_gib": float(mem_single) / (1024**3),
            "single_subseq_peak_delta_gib": float(peak_delta_single) / (1024**3),
            "single_subseq_peak_used_gib": float(peak_abs_single) / (1024**3),
            "max_parallel_subseq_fit": int(estimated_k),  # kept for backward compatibility
            "recommended_parallel_subseq": int(estimated_k),
            "best_peak_used_gib": float(est_peak) / (1024**3),
            "remaining_ratio": float(remaining_ratio_est),
            "parallel_estimation_rule": "floor(MEM_BUDGET_COE * min(free_gpu_mem, memory_cap) / single_subseq_peak_delta)",
            "attempts": attempts,
            # Backward-compatible aliases
            "total_gpu_mem_gb": float(total0) / (1024**3),
            "initial_used_gpu_mem_gb": float(used0) / (1024**3),
            "initial_free_gpu_mem_gb": float(free0) / (1024**3),
            "memory_cap_gb": (float(cap_bytes) / (1024**3)),
            "effective_probe_budget_gb": float(effective_budget_bytes) / (1024**3),
            "analytic_target_gb": float(analytic_target_bytes) / (1024**3),
            "initial_itr_guess_est_mem_gb": float(est_mem_by_itr.get(int(itr_guess), 0)) / (1024**3),
            "single_subseq_mem_consumption_gb": float(mem_single) / (1024**3),
            "single_subseq_peak_delta_gb": float(peak_delta_single) / (1024**3),
            "single_subseq_peak_used_gb": float(peak_abs_single) / (1024**3),
            "best_peak_used_gb": float(est_peak) / (1024**3),
        }


def _save_attempts_csv(path: Path, attempts: Sequence[ProbeAttempt]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_itr", "number_of_subsequence", "memory_consumption_gb", "status"])
        for a in attempts:
            w.writerow([a.num_itr, a.number_of_subsequence, a.memory_consumption_gb, a.status])


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe CQSA subsequence fit and parallel capacity.")
    parser.add_argument("--N", type=int, default=1_000_000)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--c", type=int, default=7)
    parser.add_argument("--interest-set", type=int, nargs="+", default=[0, 1, 3])
    parser.add_argument("--max-num-itr", type=int, default=12)
    parser.add_argument("--start-num-itr", type=int, default=1)
    parser.add_argument("--max-parallel-subseq", type=int, default=64)
    parser.add_argument("--memory-cap-gb", type=float, default=None)
    parser.add_argument("--input-std", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./CQSA_probe_results"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for probing.")

    probe = CQSAProbe(
        N=args.N,
        D=args.D,
        B=args.B,
        H=args.H,
        c=args.c,
        interest_set=tuple(args.interest_set),
        dtype=torch.float16,
        input_std=float(args.input_std),
        memory_cap_gb=args.memory_cap_gb,
    )
    result = probe.run(
        start_num_itr=args.start_num_itr,
        max_num_itr=args.max_num_itr,
        max_parallel_subseq=args.max_parallel_subseq,
    )

    outdir = args.output_dir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    _save_attempts_csv(outdir / "probe_table.csv", result["attempts"])

    # Save summary (attempt list is in CSV).
    summary = {k: v for k, v in result.items() if k != "attempts"}
    (outdir / "probe_summary.txt").write_text(
        "\n".join(
            [
                f"N={summary['N']}, D={summary['D']}, B={summary['B']}, H={summary['H']}",
                f"c={summary['c']}, interest_set={summary['interest_set']}",
                f"total_gpu_mem_gb={summary['total_gpu_mem_gb']:.3f}, initial_used_gpu_mem_gb={summary['initial_used_gpu_mem_gb']:.3f}",
                f"itr_max={summary['itr_max']}",
                f"single_subseq_peak_used_gb={summary.get('single_subseq_peak_used_gb', summary['best_peak_used_gb']):.3f}",
                f"max_parallel_subseq_fit={summary['max_parallel_subseq_fit']}",
                f"recommended_parallel_subseq={summary['recommended_parallel_subseq']}",
                f"best_peak_used_gb={summary['best_peak_used_gb']:.3f}",
                f"remaining_ratio={summary['remaining_ratio']:.4f}",
                f"parallel_estimation_rule={summary.get('parallel_estimation_rule', 'n/a')}",
            ]
        )
        + "\n"
    )

    print("Probing finished.")
    print(f"itr_max={summary['itr_max']}")
    print(f"max_parallel_subseq_fit={summary['max_parallel_subseq_fit']}")
    print(f"recommended_parallel_subseq={summary['recommended_parallel_subseq']}")
    print(f"best_peak_used_gb={summary['best_peak_used_gb']:.3f}")
    print(f"table_csv={outdir / 'probe_table.csv'}")
    print(f"summary_txt={outdir / 'probe_summary.txt'}")


if __name__ == "__main__":
    main()
