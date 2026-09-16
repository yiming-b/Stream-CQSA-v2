"""
``python -m stream_cqsa.doctor`` / ``stream_cqsa.doctor()``: what this machine
can run, and how far.

Reports the software (torch, CUDA, Triton, flash-attn), the kernels this
package can use (CUDA extension, native wave kernel, Triton), the devices and
host memory, the calibration cache, and the largest sequence length the
monolithic kernel and Stream-CQSA can handle here (forward and backward, by the
planner's memory model). ``check=True`` also runs a small exactness check of
``stream_cqsa.attention`` against ``F.scaled_dot_product_attention``.
"""
from __future__ import annotations

import os
import platform
import sys
from typing import Any

import torch


def _gib(b: float) -> str:
    return f"{b / 2**30:.1f} GiB"


def _largest_feasible(hw, *, B, H, D, dtype, causal, direction, mono: bool) -> int:
    """Largest power-of-two N (up to 2^26) that fits, per the planner."""
    from .autoconfig import plan
    best = 0
    for e in range(12, 27):
        N = 1 << e
        p = plan(N=N, B=B, H=H, D=D, dtype=dtype, causal=causal, hardware=hw, direction=direction, allow_distributed=False)
        ok = p.candidates[0]["ok"] if mono else any(c["ok"] for c in p.candidates)
        if ok:
            best = N
        else:
            break
    return best


def doctor(*, check: bool = True, B: int = 1, H: int = 8, D: int = 64, dtype=torch.float16, causal: bool = True,
           print_report: bool = True) -> dict[str, Any]:
    from .api import kernels_available
    from . import __version__
    rep: dict[str, Any] = {"stream_cqsa": __version__, "python": platform.python_version(), "torch": torch.__version__,
                           "cuda": torch.version.cuda, "platform": platform.platform()}
    lines = [f"stream_cqsa {__version__} doctor", f"  python {rep['python']}  torch {rep['torch']}  CUDA {rep['cuda']}  ({platform.machine()})"]
    try:
        import triton
        rep["triton"] = triton.__version__
    except Exception:
        rep["triton"] = None
    try:
        import flash_attn
        rep["flash_attn"] = getattr(flash_attn, "__version__", "?")
    except Exception:
        rep["flash_attn"] = None
    lines.append(f"  triton {rep['triton'] or 'not installed'}  flash-attn {rep['flash_attn'] or 'not installed'}")

    ks = kernels_available()
    rep["kernels"] = ks
    from . import interface as I
    def _where(mod):
        return os.path.basename(getattr(mod, "__file__", "")) if mod is not None else "-"
    lines.append("  kernels:")
    lines.append(f"    CUDA extension (causal / non-causal): {'yes' if ks['cuda_extension'] else 'no'} {_where(I.cqsa_cuda)} / "
                 f"{'yes' if ks['cuda_extension_noncausal'] else 'no'} {_where(I.cqsa_cuda_noncausal)}")
    lines.append(f"    native wave kernel (cqsa_native):     {'yes' if ks['native_wave'] else 'no (build native/ to enable; optional)'}")
    triton_hint = "pip install triton-windows" if sys.platform == "win32" else "pip install triton"
    lines.append(f"    Triton kernels (no build):            {'yes' if ks['triton'] else f'no ({triton_hint})'}")
    if not (ks["cuda_extension"] or ks["triton"]):
        lines.append(f"    !! no kernel available: {triton_hint} (stream-cqsa >= 2.2.2 installs it automatically), "
                     "or install the CUDA wheels from the release page (Linux only)")
    if not torch.cuda.is_available():
        lines.append("  no CUDA device visible")
        rep["devices"] = []
        if print_report:
            print("\n".join(lines), flush=True)
        return rep

    from .autoconfig import detect_hardware, _cost_model_cache_path, load_cost_model
    hw = detect_hardware()
    rep["hardware"] = hw.summary()
    lines.append("  devices:")
    for d in hw.devices:
        free, total = torch.cuda.mem_get_info(torch.device(d.name))
        lines.append(f"    {d.name}: {d.model}, {_gib(free)} free of {_gib(total)} (budget {_gib(d.budget_bytes)})")
    lines.append(f"  host: {_gib(hw.host_budget_bytes)} budget, {hw.n_cpu} cpus")
    cm = load_cost_model()
    rep["calibrated"] = cm is not None
    lines.append(f"  cost model: {'calibrated for this GPU (' + _cost_model_cache_path() + ')' if cm else 'defaults (run stream_cqsa.calibrate(save=True) to fit this machine, ~1 min)'}")

    rep["limits"] = {}
    lines.append(f"  largest N (B={B} H={H} D={D} {str(dtype).replace('torch.', '')}, {'causal' if causal else 'non-causal'}) by the planner's memory model:")
    for direction, label in (("fwd", "forward"), ("bwd", "forward+backward")):
        m = _largest_feasible(hw, B=B, H=H, D=D, dtype=dtype, causal=causal, direction=direction, mono=True)
        s = _largest_feasible(hw, B=B, H=H, D=D, dtype=dtype, causal=causal, direction=direction, mono=False)
        rep["limits"][direction] = dict(monolithic=m, stream_cqsa=s)
        lines.append(f"    {label:18s} monolithic up to N={m:,}   Stream-CQSA up to N={s:,}")

    if check and (ks["cuda_extension"] or ks["triton"]):
        from .api import attention
        import torch.nn.functional as F
        dev = torch.device(hw.devices[0].name)
        g = torch.Generator(device="cpu").manual_seed(0)
        N = 4096
        q, k, v = (torch.randn(1, H, N, D, generator=g, dtype=torch.float32).to(dev, dtype) for _ in range(3))
        ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), is_causal=causal)
        o_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        o_cqsa = attention(q, k, v, is_causal=causal, itr=1)          # forced decomposition
        e_s = ((o_sdpa.double() - ref).norm() / ref.norm()).item()
        e_c = ((o_cqsa.double() - ref).norm() / ref.norm()).item()
        rep["check"] = dict(N=N, err_sdpa=e_s, err_stream_cqsa=e_c, ok=e_c <= 1.5 * e_s + 1e-6)
        lines.append(f"  check (N={N}, decomposed vs float64): Stream-CQSA {e_c:.1e}, SDPA {e_s:.1e} -> {'ok' if rep['check']['ok'] else 'MISMATCH'}")
    if print_report:
        print("\n".join(lines), flush=True)
    rep["text"] = "\n".join(lines)
    return rep


def _main():
    r = doctor(check="--no-check" not in sys.argv)
    sys.exit(0 if r.get("check", {}).get("ok", True) else 1)


if __name__ == "__main__":
    _main()
