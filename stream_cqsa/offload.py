from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _nbytes(tensors: Iterable[torch.Tensor]) -> int:
    return int(sum(int(t.numel()) * int(t.element_size()) for t in tensors))


@dataclass
class TransferTracker:
    """Track host/device transfer time and logical tensor bytes."""

    device: torch.device
    device_to_host_s: float = 0.0
    host_to_device_s: float = 0.0
    device_to_host_bytes: int = 0
    host_to_device_bytes: int = 0
    t_prepare_s: float = 0.0
    t_compute_s: float = 0.0
    t_cpu_merge_s: float = 0.0
    t_finalize_s: float = 0.0
    t_gpu_flush_s: float = 0.0

    def time_stage(self, key: str, fn):
        _sync(self.device)
        t0 = time.perf_counter()
        out = fn()
        _sync(self.device)
        elapsed_s = time.perf_counter() - t0
        if key == "t_prepare_ms":
            self.t_prepare_s += elapsed_s
        elif key == "t_compute_ms":
            self.t_compute_s += elapsed_s
        elif key == "t_cpu_merge_ms":
            self.t_cpu_merge_s += elapsed_s
        elif key == "t_finalize_ms":
            self.t_finalize_s += elapsed_s
        elif key == "t_gpu_flush_ms":
            self.t_gpu_flush_s += elapsed_s
        else:
            raise ValueError(f"Unknown timing stage '{key}'")
        return out

    def flush_cuda_cache(self) -> None:
        if self.device.type != "cuda":
            return
        _sync(self.device)
        t0 = time.perf_counter()
        torch.cuda.empty_cache()
        _sync(self.device)
        self.t_gpu_flush_s += time.perf_counter() - t0

    def to_cpu(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not tensors:
            return tuple()
        if self.device.type != "cuda":
            return tuple(t.to("cpu") for t in tensors)
        self.device_to_host_bytes += _nbytes(tensors)
        _sync(self.device)
        t0 = time.perf_counter()
        out = tuple(t.to("cpu") for t in tensors)
        _sync(self.device)
        self.device_to_host_s += time.perf_counter() - t0
        return out

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device == self.device:
            return tensor
        if self.device.type != "cuda":
            return tensor.to(self.device)
        self.host_to_device_bytes += _nbytes((tensor,))
        _sync(self.device)
        t0 = time.perf_counter()
        out = tensor.to(self.device)
        _sync(self.device)
        self.host_to_device_s += time.perf_counter() - t0
        return out

    def as_dict(self) -> dict[str, float]:
        d2h_ms = float(self.device_to_host_s) * 1000.0
        h2d_ms = float(self.host_to_device_s) * 1000.0
        return {
            "communication_ms": d2h_ms + h2d_ms,
            "device_to_host_ms": d2h_ms,
            "host_to_device_ms": h2d_ms,
            "t_d2h_ms": d2h_ms,
            "t_h2d_ms": h2d_ms,
            "t_prepare_ms": float(self.t_prepare_s) * 1000.0,
            "t_compute_ms": float(self.t_compute_s) * 1000.0,
            "t_cpu_merge_ms": float(self.t_cpu_merge_s) * 1000.0,
            "t_finalize_ms": float(self.t_finalize_s) * 1000.0,
            "t_gpu_flush_ms": float(self.t_gpu_flush_s) * 1000.0,
            "device_to_host_transfer_gib": float(self.device_to_host_bytes) / float(1024**3),
            "host_to_device_transfer_gib": float(self.host_to_device_bytes) / float(1024**3),
        }
