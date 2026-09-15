"""
The one entry point: ``stream_cqsa.attention`` with the signature of
``torch.nn.functional.scaled_dot_product_attention``.

    out = stream_cqsa.attention(q, k, v, is_causal=True)

* below the memory boundary it IS the monolithic call (SDPA / FlashAttention);
* above it, the planner picks the decomposition (depth, quorum set, accumulator
  placement, host streaming, concurrency) from the free memory and the call
  runs exactly, on the best kernel available (native wave kernel, CUDA
  extension, or the Triton kernels when nothing is compiled);
* gradients flow when the inputs require them;
* ``verbose=True`` (or ``CQSA_VERBOSE=1``) shows what is happening.

``patch_sdpa()`` swaps this in for ``F.scaled_dot_product_attention`` so an
existing model runs unchanged; ``estimate(N, ...)`` is the dry run that says
what a call would cost before it is made.
"""
from __future__ import annotations

import contextlib
import os
import warnings
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .progress import verbose_enabled

__all__ = ["attention", "estimate", "patch_sdpa", "unpatch_sdpa", "patched_sdpa", "kernels_available"]


def kernels_available() -> dict[str, bool]:
    """Which attention kernels this environment can run."""
    from . import interface as I
    out = {"cuda_extension": I.cqsa_cuda is not None, "cuda_extension_noncausal": I.cqsa_cuda_noncausal is not None}
    try:
        from .native_wave import native_available
        out["native_wave"] = native_available()
    except Exception:
        out["native_wave"] = False
    try:
        import triton  # noqa: F401
        from . import triton_kernel  # noqa: F401
        out["triton"] = True
    except Exception:
        out["triton"] = False
    try:
        import flash_attn  # noqa: F401
        out["flash_attn"] = True
    except Exception:
        out["flash_attn"] = False
    return out


def _check_inputs(q, k, v, attn_mask, dropout_p, enable_gqa):
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"stream_cqsa.attention expects q/k/v as [B, H, N, D]; got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if attn_mask is not None:
        raise NotImplementedError("stream_cqsa.attention supports the plain and the causal (is_causal=True) cases; an explicit "
                                  "attn_mask is not supported. For FlexAttention-style masks use stream_cqsa.adapters.flex_inner.")
    if dropout_p:
        raise NotImplementedError("stream_cqsa.attention does not support dropout (dropout_p must be 0).")
    if enable_gqa or k.shape[1] != q.shape[1]:
        raise NotImplementedError("stream_cqsa.attention needs the same number of heads for q and k/v (no GQA); repeat k/v heads first.")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k and v must have the same shape [B, H, N, D] (cross-attention with N_q != N_k is not supported)")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"stream_cqsa.attention needs fp16 or bf16 inputs (got {q.dtype}); cast with .half() or .bfloat16(). "
                        "Accumulation is fp32 internally regardless.")
    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError("q, k and v must share a dtype")
    if not (q.device == k.device == v.device):
        raise ValueError("q, k and v must be on the same device")


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask=None, dropout_p: float = 0.0,
              is_causal: bool = False, scale: Optional[float] = None, enable_gqa: bool = False, *,
              verbose: Optional[bool] = None, hardware=None, kernel: str = "auto", plan_only: bool = False,
              return_plan: bool = False, **overrides) -> torch.Tensor:
    """
    Exact attention that always fits. Drop-in for ``F.scaled_dot_product_attention``
    (same positional signature; masks, dropout and GQA are rejected with a message).

    q/k/v ``[B, H, N, D]`` fp16/bf16, on a device or in host memory. Returns the
    output in the inputs' dtype and device. Differentiable when the inputs require
    gradients. ``kernel``: "auto" (native wave kernel when built and applicable,
    else the CUDA extension, else Triton), "wave" (wave engine, CUDA kernel when built else Triton),
    "wave-cuda", "wave-triton", "cuda", "triton".
    ``overrides`` (itr=, c=, interest_set=, max_parallel=, ...) pin engine settings.
    ``plan_only=True`` returns the plan without computing anything.
    """
    _INSIDE[0] += 1
    try:
        return _attention(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa, verbose=verbose, hardware=hardware,
                          kernel=kernel, plan_only=plan_only, return_plan=return_plan, **overrides)
    finally:
        _INSIDE[0] -= 1


def _attention(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa, *, verbose, hardware, kernel, plan_only,
               return_plan, **overrides):
    from .autoconfig import plan as _plan, hardware_from_dict, detect_hardware
    _check_inputs(q, k, v, attn_mask, dropout_p, enable_gqa)
    B, H, N, D = q.shape
    if scale is None:
        scale = float(D) ** -0.5
    on_cuda = q.device.type == "cuda"
    out_device = q.device
    hw = hardware_from_dict(hardware) if isinstance(hardware, dict) else (hardware or detect_hardware())
    direction = "bwd" if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad) else "fwd"
    p = _plan(N=N, B=B, H=H, D=D, dtype=q.dtype, causal=bool(is_causal), hardware=hw, direction=direction,
              allow_distributed=False)
    if plan_only:
        return p
    vb = verbose_enabled(verbose)

    # ---- monolithic: the call fits, so it IS the normal kernel ------------------
    if p.mode == "mono" and not overrides:
        if vb:
            print(f"Stream-CQSA: N={N} fits ({p.est_peak_gib:.1f} GiB) -- monolithic call, no decomposition", flush=True)
        qd, kd, vd = ((t if on_cuda else t.to("cuda", non_blocking=True)) for t in (q, k, v))
        try:
            out = _TRUE_SDPA(qd, kd, vd, is_causal=bool(is_causal), scale=scale)
            out = out if on_cuda else out.to(q.device)
            return (out, p) if return_plan else out
        except Exception as exc:                                     # noqa: BLE001
            if not _is_oom(exc):
                raise
            if vb:
                print("Stream-CQSA: the monolithic call ran out of memory after all; decomposing", flush=True)
            del qd, kd, vd
            torch.cuda.empty_cache()
            p = _plan(N=N, B=B, H=H, D=D, dtype=q.dtype, causal=bool(is_causal), hardware=detect_hardware(),
                      direction=direction, allow_distributed=False)
            if p.mode == "mono":     # the planner still thinks it fits: force a decomposition
                p.mode, p.itr, p.c, p.interest_set = "cqsa", 1, 7, (0, 1, 3)

    # ---- decomposed ------------------------------------------------------------
    kw = p.engine_kwargs() if p.mode != "mono" else dict(itr=1, c=7, interest_set=(0, 1, 3))
    if on_cuda and kw.get("stream_from_host"):
        # The inputs are on the device. Prefer the fastest configuration that leaves them
        # there; stream from the host only if nothing device-resident fits (then a pinned
        # host copy is made -- the caller's device tensors stay where they are).
        dev_ok = [c for c in p.candidates if c["ok"] and c["mode"] == "cqsa" and not c["stream_from_host"]]
        if dev_ok:
            c0 = min(dev_ok, key=lambda c: c["time"])
            kw = dict(itr=int(c0["itr"]), c=int(c0["c"]), interest_set=tuple(c0["interest_set"]),
                      low_memory=(c0["acc"] == "cpu"), accumulate_on_gpu=(c0["acc"] == "gpu"),
                      stream_from_host=False, max_parallel=int(c0["n_par"]), shared_chunks=False)
        elif direction == "bwd":
            # Copying the inputs to the host would sever the autograd graph. Keep them on the
            # device, park the accumulator in host memory, and let the engine escalate the
            # depth on its own if the device is still short.
            if vb:
                print("Stream-CQSA: no device-resident configuration fits the estimate; keeping Q/K/V on the device with a "
                      "host accumulator and escalating on demand", flush=True)
            kw.update(stream_from_host=False, low_memory=True, accumulate_on_gpu=False, shared_chunks=False)
        else:
            if vb:
                print("Stream-CQSA: no device-resident configuration fits; copying Q/K/V to pinned host memory and streaming", flush=True)
            if direction == "bwd":
                q, k, v = (t.to("cpu") for t in (q, k, v))                 # differentiable copies: the graph stays intact
            else:
                q, k, v = (t.detach().to("cpu").pin_memory() for t in (q, k, v))
            on_cuda = False
    kw.update(overrides)
    if not kw.get("stream_from_host"):
        kw["shared_chunks"] = False
    use_wave = _pick_wave(kernel, kw, is_causal, q, direction)
    if vb:
        print(f"Stream-CQSA: {p.reason}", flush=True)
    if use_wave:
        from .native_wave import wave_attention, wave_forward
        wkw = dict(itr=int(kw.get("itr", 1)), c=int(kw.get("c", 7)), interest_set=tuple(kw.get("interest_set", (0, 1, 3))))
        wkern = WAVE_KERNELS.get(kernel, "auto")
        try:
            if direction == "bwd":
                out = wave_attention(q, k, v, causal=bool(is_causal), scale=scale, verbose=verbose, **wkw)
            else:
                acc_gpu = not (kw.get("low_memory") or kw.get("accumulate_on_gpu") is False)
                out, _ = wave_forward(q, k, v, causal=bool(is_causal), scale=scale, verbose=verbose,
                                      accumulate_on_gpu=acc_gpu, kernel=wkern, **wkw)
        except Exception as exc:                                     # noqa: BLE001
            if not _is_oom(exc) or kernel in WAVE_KERNELS:
                raise
            # the wave engine keeps its accumulator on the device; the classic engine can offload it
            if vb:
                print("Stream-CQSA: the wave engine ran out of memory; retrying on the streaming engine", flush=True)
            torch.cuda.empty_cache()
            use_wave = False
    if not use_wave:
        from .native_autograd import stream_cqsa_attn
        from .stable_stream import stream_cqsa_forward
        if kernel in ("triton", "wave-triton"):
            os.environ["CQSA_FORWARD"] = "triton"; os.environ["CQSA_BACKWARD"] = "triton"
        host = not on_cuda
        if direction == "bwd":
            out = stream_cqsa_attn(q, k, v, causal=bool(is_causal), scale=scale, verbose=verbose,
                                   itr=kw.get("itr", "auto"), c=kw.get("c", 7), interest_set=kw.get("interest_set", (0, 1, 3)),
                                   stream_from_host=bool(kw.get("stream_from_host", host)),
                                   accumulate_on_gpu=bool(kw.get("accumulate_on_gpu", True)),
                                   max_parallel=kw.get("max_parallel"))
        else:
            kw.setdefault("stream_from_host", host)
            # leave the fp32 result where the accumulator is; _deliver casts and moves it
            out, _ = stream_cqsa_forward(q, k, v, causal=bool(is_causal), scale=scale, verbose=verbose, out_device="acc", **kw)
    out = _deliver(out, out_device, q.dtype, vb)
    return (out, p) if return_plan else out


def _deliver(out: torch.Tensor, device, dtype, vb: bool) -> torch.Tensor:
    """Hand the result over in the caller's dtype and on the caller's device.

    The engines return fp32; the cast needs room for a second copy. When the device
    cannot take it, the cast is done in host memory and the result moved back; if
    even the half-size result does not fit next to what the caller keeps on the
    device, it is returned in host memory with a warning rather than failing after
    the work is done."""
    if out.dtype == dtype and out.device == device:
        return out
    try:
        return out.to(device=device, dtype=dtype)
    except Exception as exc:                                         # noqa: BLE001
        if not _is_oom(exc):
            raise
    host = out.to("cpu") if out.device.type != "cpu" else out
    del out
    torch.cuda.empty_cache()
    host = host.to(dtype)
    if device.type == "cpu":
        return host
    try:
        return host.to(device)
    except Exception as exc:                                         # noqa: BLE001
        if not _is_oom(exc):
            raise
    warnings.warn(f"Stream-CQSA: the result ({host.numel() * host.element_size() / 2**30:.1f} GiB) does not fit on {device} "
                  f"next to what is already there; returning it in host memory.", RuntimeWarning, stacklevel=3)
    return host


WAVE_KERNELS = {"wave": "auto", "wave-cuda": "cuda", "wave-triton": "triton"}


def _pick_wave(kernel: str, kw: dict, is_causal: bool, q: torch.Tensor, direction: str) -> bool:
    if kernel in WAVE_KERNELS:
        # the wave backward runs on the CUDA kernel only: a Triton-only wave request for a
        # backward goes to the classic engine's Triton kernels
        if direction != "fwd" and kernel == "wave-triton":
            return False
        return True
    if kernel in ("cuda", "triton"):
        return False
    if kernel != "auto":
        raise ValueError(f"kernel must be 'auto', 'wave', 'wave-cuda', 'wave-triton', 'cuda' or 'triton' (got {kernel!r})")
    try:
        from .native_wave import native_available
        if not native_available():
            return False
    except Exception:
        return False
    # the wave forward can accumulate on the host (accumulate_on_gpu=False); the wave backward
    # still keeps its fp32 gradients on the device, so host-accumulator backwards use the classic engine
    if direction != "fwd" and (kw.get("low_memory") or kw.get("accumulate_on_gpu") is False):
        return False
    from .native_wave import native_supports
    return native_supports(q.dtype, int(q.shape[-1]))      # depends on the kernel set the extension was built with


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------
def estimate(N: int, B: int = 1, H: int = 8, D: int = 64, dtype=torch.float16, causal: bool = True,
             hardware=None, direction: str = "both", print_table: bool = True) -> dict[str, Any]:
    """
    What would ``attention`` do for this shape on this machine, and what would it
    cost? Prints (and returns) the monolithic estimate, the chosen decomposition
    and the runner-up configurations, for the forward and the backward.
    """
    from .autoconfig import plan as _plan, hardware_from_dict, detect_hardware, _cname, GIB
    hw = hardware_from_dict(hardware) if isinstance(hardware, dict) else (hardware or detect_hardware())
    res: dict[str, Any] = {"N": N, "B": B, "H": H, "D": D, "dtype": str(dtype), "causal": causal, "hardware": hw.summary()}
    dirs = ("fwd", "bwd") if direction == "both" else (direction,)
    lines = [f"Stream-CQSA estimate for N={N:,} B={B} H={H} D={D} {str(dtype).replace('torch.', '')} {'causal' if causal else 'non-causal'}",
             f"  hardware: {hw.summary()}"]
    itemsize = torch.tensor([], dtype=dtype).element_size()
    for d in dirs:
        p = _plan(N=N, B=B, H=H, D=D, dtype=dtype, causal=causal, hardware=hw, direction=d, allow_distributed=False)
        mono = p.candidates[0]
        feas = sorted([c for c in p.candidates if c["ok"] and c["mode"] != "mono"], key=lambda c: c["time"])[:3]
        r = dict(plan=p.name(), est_time_s=p.est_time_s, est_peak_gib=p.est_peak_gib, est_host_gib=p.est_host_gib,
                 mono_fits=bool(mono["ok"]), mono_peak_gib=mono["peak"], mono_time_s=mono["time"], reason=p.reason,
                 runners_up=[dict(config=_cname(c), time_s=c["time"], peak_gib=c["peak"], host_gib=c["host"]) for c in feas],
                 feasible=any(c["ok"] for c in p.candidates))
        res[d] = r
        label = "forward" if d == "fwd" else "backward (fwd+bwd step)"
        lines.append(f"  {label}:")
        lines.append(f"    monolithic FlashAttention: {'fits' if mono['ok'] else 'does NOT fit'} ({mono['peak']:.1f} GiB), ~{mono['time']:.2f} s")
        if p.mode == "mono":
            lines.append(f"    -> monolithic call")
        elif r["feasible"]:
            lines.append(f"    -> Stream-CQSA {p.name()}: ~{p.est_time_s:.1f} s, {p.est_peak_gib:.1f} GiB on the device"
                         + (f", {p.est_host_gib:.0f} GiB host" if p.est_host_gib else ""))
            for c in feas[1:]:
                lines.append(f"       also: {_cname(c)} ~{c['time']:.1f} s, {c['peak']:.1f} GiB")
        else:
            floor = (4 + 4) * N * B * H * D / GIB
            lines.append(f"    -> nothing fits: the fp32 output + accumulator alone are {floor:.1f} GiB on the device; "
                         f"reduce B/H/D, add host memory, or use more devices")
    if print_table:
        print("\n".join(lines), flush=True)
    res["text"] = "\n".join(lines)
    return res


# ---------------------------------------------------------------------------
# monkeypatch F.scaled_dot_product_attention
# ---------------------------------------------------------------------------
_ORIGINAL_SDPA = None
_PATCH_MIN_TOKENS = 65536
_TRUE_SDPA = F.scaled_dot_product_attention          # captured at import, before any patch
_INSIDE = [0]                                          # re-entrancy: calls made by the engines go to the original


def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    orig = _ORIGINAL_SDPA or _TRUE_SDPA
    supported = (attn_mask is None and not dropout_p and not enable_gqa and query.dim() == 4
                 and query.dtype in (torch.float16, torch.bfloat16) and key.shape == query.shape == value.shape
                 and query.shape[-2] >= _PATCH_MIN_TOKENS and _INSIDE[0] == 0)
    if not supported:
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                    **({"enable_gqa": enable_gqa} if enable_gqa else {}))
    return attention(query, key, value, is_causal=is_causal, scale=scale)


def patch_sdpa(min_tokens: int = 65536) -> None:
    """
    Route ``torch.nn.functional.scaled_dot_product_attention`` through
    ``stream_cqsa.attention`` for the calls it handles (4-D fp16/bf16, no mask,
    no dropout, no GQA, N >= ``min_tokens``); everything else goes to the original.
    Below the memory boundary the routed call is still the monolithic kernel, so
    the patch changes nothing until a call would have run out of memory.
    """
    global _ORIGINAL_SDPA, _PATCH_MIN_TOKENS
    _PATCH_MIN_TOKENS = int(min_tokens)
    if _ORIGINAL_SDPA is None:
        _ORIGINAL_SDPA = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = _patched_sdpa
        torch.nn.functional.scaled_dot_product_attention = _patched_sdpa


def unpatch_sdpa() -> None:
    global _ORIGINAL_SDPA
    if _ORIGINAL_SDPA is not None:
        F.scaled_dot_product_attention = _ORIGINAL_SDPA
        torch.nn.functional.scaled_dot_product_attention = _ORIGINAL_SDPA
        _ORIGINAL_SDPA = None


@contextlib.contextmanager
def patched_sdpa(min_tokens: int = 65536):
    """``with stream_cqsa.patched_sdpa(): model(...)``"""
    was = _ORIGINAL_SDPA is not None
    patch_sdpa(min_tokens)
    try:
        yield
    finally:
        if not was:
            unpatch_sdpa()
