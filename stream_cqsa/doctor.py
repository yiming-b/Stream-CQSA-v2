"""
``python -m stream_cqsa.doctor`` / ``stream_cqsa.doctor()``: what this machine
can run, and how far.

Reports the software (torch, CUDA, Triton, flash-attn), the kernels this
package can use (CUDA extension, native wave kernel, Triton), the devices and
host memory, the calibration cache, and what fits on this machine: the largest
sequence length one monolithic kernel call handles (device memory) and the
largest Stream-CQSA handles under this machine's host RAM (forward and
backward, by the planner's memory model). Stream-CQSA has no limit of its own;
the host has to hold the sequence and the accumulators. ``check=True`` also
runs a small exactness check of
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


def _largest_feasible(hw, *, B, H, D, dtype, causal, direction, mono: bool):
    """(largest power-of-two N up to 2^30 that fits per the planner, what binds at the next size).
    Stream-CQSA bounds the *device* footprint; the host must still hold the sequence and the fp32
    accumulators / gradients, so the host budget is what eventually binds."""
    from .autoconfig import plan
    best, binds = 0, ""
    for e in range(12, 31):
        N = 1 << e
        p = plan(N=N, B=B, H=H, D=D, dtype=dtype, causal=causal, hardware=hw, direction=direction, allow_distributed=False,
                 out_bytes_per_el=0)
        cs = [p.candidates[0]] if mono else [c for c in p.candidates if c["mode"] != "mono"]
        if any(c["ok"] for c in cs):
            best = N
            continue
        if not mono and cs:
            dev_budget = hw.devices[0].budget_bytes / 2**30; host_budget = hw.host_budget_bytes / 2**30
            fit_dev = [c for c in cs if c["peak"] <= dev_budget]        # configurations the device could hold
            fit_host = [c for c in cs if c["host"] <= host_budget]      # configurations the host could hold
            need_host = min(c["host"] for c in fit_dev) if fit_dev else None    # host RAM the device-feasible ones need
            need_dev = min(c["peak"] for c in fit_host) if fit_host else None   # device memory the host-feasible ones need
            host_msg = f"N={N:,} needs {need_host:.0f} GiB of host RAM ({host_budget:.0f} available)" if need_host is not None else f"N={N:,}: no configuration fits the device budget"
            dev_msg = f"{need_dev:.1f} GiB on the device ({dev_budget:.1f} available)" if need_dev is not None else "no configuration fits host RAM"
            if need_host is not None and need_host > host_budget and (need_dev is None or need_dev <= dev_budget):
                binds = f"limited by host RAM: {host_msg}"
            elif need_dev is not None and need_dev > dev_budget and (need_host is None or need_host <= host_budget):
                binds = f"limited by the device budget: N={N:,} needs {dev_msg}"
            else:
                binds = f"limited by host RAM and the device budget: {host_msg}; {dev_msg}"
        break
    return best, binds


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
    lines.append(f"  what fits on this machine (B={B} H={H} D={D} {str(dtype).replace('torch.', '')}, {'causal' if causal else 'non-causal'}; "
                 "planner's memory model; inputs and outputs in host memory; powers of two):")
    for direction, label in (("fwd", "forward"), ("bwd", "forward+backward")):
        m, _ = _largest_feasible(hw, B=B, H=H, D=D, dtype=dtype, causal=causal, direction=direction, mono=True)
        s, binds = _largest_feasible(hw, B=B, H=H, D=D, dtype=dtype, causal=causal, direction=direction, mono=False)
        rep["limits"][direction] = dict(monolithic=m, stream_cqsa=s, stream_cqsa_bound=binds)
        lines.append(f"    {label:18s} one monolithic kernel call: up to N={m:,} (device memory)")
        lines.append(f"    {'':18s} Stream-CQSA:                up to N={s:,}" + (f" -- {binds}" if binds else ""))
    lines.append("  Stream-CQSA has no sequence-length limit of its own: its device footprint stays bounded at any N. The sequence,")
    lines.append("  the fp32 accumulators and the gradients live in host RAM, and that budget (80% of free RAM here) is what sets")
    lines.append("  the numbers above; more host RAM, or a smaller model of the host footprint, moves them.")

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
