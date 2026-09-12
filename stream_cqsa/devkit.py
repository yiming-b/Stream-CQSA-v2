"""
Developer kit: measure an attention kernel inside the Stream-CQSA framework
against its monolithic call, for exactness and performance.

    from stream_cqsa.devkit import compare_kernels, quick_bench, run_config

Two ways in:

* ``compare_kernels(inner_fn, mono_fn, ...)`` -- you supply the kernel that
  runs on one CQS subproblem and the monolithic kernel it is supposed to
  reproduce; you get back whether the decomposed result matches the monolithic
  one (bit-identical / within its own rounding / not), the error of each
  against a float64 reference, and time + peak memory of both.

  Inner-kernel contract (the engine's own):
      inner_fn(q_i, k_i, v_i, group_bits, *, causal, scale, **ignored)
          -> (out_i [B, L, H, D] float32, lse_i [B, L, H] float32)
  where q_i/k_i/v_i are ``[B, L, H, D]`` (token-major) in the input dtype and
  ``group_bits`` is an int64 ``[L]`` vector: pair (row, col) is KEPT iff
  ``bits[row] & bits[col] == 0`` (and, if causal, col <= row). ``lse_i`` must be
  the per-row log-sum-exp of the kept scores (``-inf`` for rows with no kept
  pair). The extension's kernel also accepts block summaries
  (``blk_or/blk_and``); a kernel that does not take them is called without.

  Monolithic contract: ``mono_fn(q, k, v) -> out [B, H, N, D]`` (any float
  dtype), or ``(out, lse)``.

* ``quick_bench(...)`` -- the standard sweep: FlashAttention-2 / SDPA
  monolithic versus Stream-CQSA at the given depths, accumulator placements
  and parallelism, all on the same random inputs, with accuracy against
  float64 on sampled rows, time and peak device memory; and the configuration
  the "maximise memory use under the budget unless something is faster AND
  smaller" rule would pick.

Accuracy at large N uses the paper's sampled float64 reference: R query rows
against all N keys, streamed in tiles (O(R*N) memory), so it works at any N.
"""
from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Sequence

import torch

GIB = float(1 << 30)


# ---------------------------------------------------------------------------
# float64 reference on sampled rows
# ---------------------------------------------------------------------------

def reference_rows(q, k, v, rows, *, causal: bool, scale: float, tile: int = 4096,
                   device: str | torch.device = "cuda"):
    """Exact float64 attention for query `rows` against all keys. [B,H,R,D]."""
    dev = torch.device(device)
    B, H, N, D = q.shape
    rows = rows.to(q.device)
    qs = q.index_select(2, rows).to(dev, torch.float64)
    R = qs.shape[2]
    acc = torch.zeros(B, H, R, D, dtype=torch.float64, device=dev)
    m = torch.full((B, H, R), float("-inf"), dtype=torch.float64, device=dev)
    l = torch.zeros(B, H, R, dtype=torch.float64, device=dev)
    rd = rows.to(dev)
    for s in range(0, N, tile):
        e = min(s + tile, N)
        ks = k[:, :, s:e].to(dev, torch.float64)
        vs = v[:, :, s:e].to(dev, torch.float64)
        sc = (qs @ ks.transpose(-1, -2)) * scale
        if causal:
            cols = torch.arange(s, e, device=dev)
            sc = sc.masked_fill(cols[None, None, None, :] > rd[None, None, :, None], float("-inf"))
        mn = torch.maximum(m, sc.amax(-1))
        fin = torch.isfinite(mn)
        ms = torch.where(fin, mn, torch.zeros_like(mn))
        corr = torch.where(fin & torch.isfinite(m), torch.exp(m - ms), torch.zeros_like(m))
        p = torch.where(fin.unsqueeze(-1), torch.exp(sc - ms.unsqueeze(-1)), torch.zeros_like(sc))
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        acc = acc * corr.unsqueeze(-1) + p @ vs
        l = l * corr + p.sum(-1)
        m = torch.where(fin, mn, m)
        del ks, vs, sc, p
    return acc / l.clamp_min(torch.finfo(torch.float64).tiny).unsqueeze(-1)


def sample_rows(N: int, R: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    R = min(R, N)
    return torch.randperm(N, generator=g)[:R].sort().values


def accuracy_vs_fp64(out, q, k, v, *, causal: bool, scale: float, rows: torch.Tensor,
                     ref_rows: torch.Tensor | None = None) -> dict[str, float]:
    """Relative Frobenius and max-abs error of `out` on `rows` against float64."""
    if ref_rows is None:
        ref_rows = reference_rows(q, k, v, rows, causal=causal, scale=scale)
    o = out.index_select(2, rows.to(out.device)).to(ref_rows.device, torch.float64)
    d = o - ref_rows
    return dict(rel_fro=(d.norm() / ref_rows.norm().clamp_min(1e-30)).item(),
                max_abs=d.abs().max().item(),
                max_rel=(d.abs().max() / ref_rows.abs().max().clamp_min(1e-30)).item())


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.to(torch.float64); b = b.to(a.device, torch.float64)
    d = a - b
    return dict(rel_fro=(d.norm() / b.norm().clamp_min(1e-30)).item(),
                max_abs=d.abs().max().item(),
                bit_identical=bool(torch.equal(a, b)))


# ---------------------------------------------------------------------------
# timing + memory
# ---------------------------------------------------------------------------

def _drain(device):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def measure(fn: Callable[[], Any], *, device, reps: int = 2, warmup: int = 1) -> tuple[Any, dict[str, float]]:
    """Run `fn` warmup+reps times; return (last result, {s, reps_s, peak_gib, workspace_gib})."""
    device = torch.device(device)
    times = []
    result = None
    peak = 0.0
    work = 0.0
    for i in range(warmup + reps):
        _drain(device)
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
        t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
        pk = torch.cuda.max_memory_allocated(device)
        peak = max(peak, pk / GIB)
        work = max(work, (pk - base) / GIB)
    return result, dict(s=min(times), reps_s=[round(t, 4) for t in times], peak_gib=peak, workspace_gib=work)


# ---------------------------------------------------------------------------
# configurations
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """One way of computing attention. mode: 'mono' | 'cqsa' | 'cqsa_dist'."""
    mode: str = "cqsa"
    itr: int = 1
    acc: str = "gpu"                 # 'gpu' | 'cpu'  (where the accumulator lives)
    stream_from_host: bool = False   # Q/K/V resident on the host
    n_par: int = 1                   # subproblems in flight
    shared_chunks: bool | None = None  # None: True when itr==1 and stream_from_host
    world: int = 1                   # devices (cqsa_dist)
    mono_backend: str = "flash"      # 'flash' (flash-attn) | 'sdpa'
    label: str = ""

    def name(self) -> str:
        if self.label:
            return self.label
        if self.mode == "mono":
            return f"mono[{self.mono_backend}]"
        s = f"cqsa itr={self.itr} acc={self.acc} npar={self.n_par}"
        if self.stream_from_host:
            s += " host"
        if self.mode == "cqsa_dist":
            s += f" x{self.world}"
        return s


def _mono_fn(backend: str, causal: bool, scale: float) -> Callable:
    if backend == "flash":
        from flash_attn import flash_attn_func

        def f(q, k, v):
            return flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                   softmax_scale=scale, causal=causal).transpose(1, 2)
        return f
    import torch.nn.functional as F

    def g(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
    return g


def run_config(q, k, v, cfg: Config, *, causal: bool, scale: float | None = None,
               inner: Callable | None = None, device=None, allow_escalation: bool = False,
               return_info: bool = False):
    """Compute attention for `cfg`. q/k/v [B,H,N,D]; returns out [B,H,N,D] (input dtype for mono, fp32 for cqsa)."""
    B, H, N, D = q.shape
    scale = float(D) ** -0.5 if scale is None else float(scale)
    device = torch.device(device or (q.device if q.is_cuda else "cuda"))
    if cfg.mode == "mono":
        qq, kk, vv = (t.to(device, non_blocking=True) for t in (q, k, v))
        out = _mono_fn(cfg.mono_backend, causal, scale)(qq, kk, vv)
        return (out, {}) if return_info else out
    from .stable_stream import stream_cqsa_forward
    if cfg.stream_from_host:
        qq, kk, vv = (t if t.device.type == "cpu" else t.cpu() for t in (q, k, v))
    else:
        qq, kk, vv = (t.to(device, non_blocking=True) for t in (q, k, v))
    shared = cfg.shared_chunks if cfg.shared_chunks is not None else (cfg.stream_from_host and cfg.itr == 1)
    kw = dict(itr=int(cfg.itr), causal=causal, scale=scale, inner=inner,
              stream_from_host=bool(cfg.stream_from_host), low_memory=(cfg.acc == "cpu"),
              accumulate_on_gpu=(cfg.acc == "gpu"), max_parallel=int(cfg.n_par),
              allow_escalation=allow_escalation, shared_chunks=bool(shared))
    if cfg.mode == "cqsa_dist":
        from .distributed import dist_stream_cqsa_forward   # optional module (next/distributed)
        out, info = dist_stream_cqsa_forward(qq, kk, vv, **kw)
    else:
        out, info = stream_cqsa_forward(qq, kk, vv, **kw)
    return (out, info) if return_info else out


# ---------------------------------------------------------------------------
# compare_kernels
# ---------------------------------------------------------------------------

@dataclass
class KernelReport:
    N: int
    B: int
    H: int
    D: int
    dtype: str
    causal: bool
    itr: int
    cqsa_vs_mono: dict = field(default_factory=dict)
    mono_vs_fp64: dict = field(default_factory=dict)
    cqsa_vs_fp64: dict = field(default_factory=dict)
    verdict: str = ""
    perf_mono: dict = field(default_factory=dict)
    perf_cqsa: dict = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)

    def __str__(self):
        f = lambda d, kk: d.get(kk, float("nan"))
        lines = [
            f"compare_kernels: N={self.N} B={self.B} H={self.H} D={self.D} {self.dtype} causal={self.causal} itr={self.itr}",
            f"  exactness : {self.verdict}",
            f"    CQSA(inner) vs monolithic : rel {f(self.cqsa_vs_mono,'rel_fro'):.2e}  max|d| {f(self.cqsa_vs_mono,'max_abs'):.2e}  bit-identical={self.cqsa_vs_mono.get('bit_identical')}",
            f"    monolithic  vs float64    : rel {f(self.mono_vs_fp64,'rel_fro'):.2e}  max|d| {f(self.mono_vs_fp64,'max_abs'):.2e}",
            f"    CQSA(inner) vs float64    : rel {f(self.cqsa_vs_fp64,'rel_fro'):.2e}  max|d| {f(self.cqsa_vs_fp64,'max_abs'):.2e}",
            f"  performance: monolithic {f(self.perf_mono,'s'):.4f} s, peak {f(self.perf_mono,'peak_gib'):.2f} GiB | "
            f"CQSA {f(self.perf_cqsa,'s'):.4f} s, peak {f(self.perf_cqsa,'peak_gib'):.2f} GiB  "
            f"(ratio {f(self.perf_cqsa,'s') / max(f(self.perf_mono,'s'), 1e-12):.2f}x time, "
            f"{f(self.perf_cqsa,'peak_gib') / max(f(self.perf_mono,'peak_gib'), 1e-12):.2f}x memory)",
        ]
        return "\n".join(lines)


def compare_kernels(inner_fn: Callable, mono_fn: Callable | None = None, *,
                    N: int, B: int = 1, H: int = 8, D: int = 64, dtype=torch.float16,
                    causal: bool = True, itr: int = 1, cfg: Config | None = None,
                    seed: int = 0, acc_rows: int = 256, reps: int = 2, device="cuda",
                    q=None, k=None, v=None, scale: float | None = None,
                    rounding_tolerance: float = 4.0, verbose: bool = True,
                    reference: str | Callable | None = "attention") -> KernelReport:
    """
    Plug `inner_fn` into Stream-CQSA and compare with `mono_fn` (default:
    flash-attn). `cfg` overrides itr/acc/n_par/residency (mode is forced to cqsa).

    Verdict: 'bit-identical' if the decomposed and monolithic outputs agree
    exactly; 'exact (within rounding)' if their difference is <= rounding_tolerance
    times the monolithic kernel's own error against float64 -- i.e. the
    decomposition adds nothing beyond re-rounding; otherwise 'NOT exact', with
    the numbers. An approximate kernel is judged the same way: what matters is
    whether the decomposed path reproduces the monolithic path of the same
    kernel, not whether the kernel itself is exact.

    `reference`: "attention" (default) = plain softmax attention in float64 on
    sampled rows -- right for any kernel that computes standard attention;
    a callable `ref(q, k, v, rows) -> float64 [B,H,R,D]` for kernels that
    compute something else (a biased or windowed attention); or None, in which
    case the verdict uses a fixed rounding floor for the input dtype
    (fp16 1e-3, bf16 8e-3, fp32 1e-6 relative) instead of the float64 gap.
    """
    device = torch.device(device)
    scale = float(D) ** -0.5 if scale is None else float(scale)
    if q is None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        q, k, v = (torch.randn(B, H, N, D, generator=g, dtype=torch.float32).to(dtype) for _ in range(3))
    B, H, N, D = q.shape
    cfg = cfg or Config(mode="cqsa", itr=itr)
    cfg.mode = "cqsa"
    cfg.itr = int(itr if cfg.itr is None else cfg.itr)
    mono_fn = mono_fn or _mono_fn("flash", causal, scale)

    def mono():
        qq, kk, vv = (t.to(device, non_blocking=True) for t in (q, k, v))
        r = mono_fn(qq, kk, vv)
        return r[0] if isinstance(r, (tuple, list)) else r
    out_m, perf_m = measure(mono, device=device, reps=reps)
    out_c, perf_c = measure(lambda: run_config(q, k, v, cfg, causal=causal, scale=scale, inner=inner_fn, device=device),
                            device=device, reps=reps)
    rows = sample_rows(N, acc_rows, seed)
    if reference == "attention":
        ref = reference_rows(q, k, v, rows, causal=causal, scale=scale, device=device)
    elif callable(reference):
        ref = reference(q, k, v, rows).to(torch.float64)
    else:
        ref = None
    nan = dict(rel_fro=float("nan"), max_abs=float("nan"), max_rel=float("nan"))
    rep = KernelReport(N=N, B=B, H=H, D=D, dtype=str(dtype).replace("torch.", ""), causal=causal, itr=cfg.itr,
                       cqsa_vs_mono=diff_stats(out_c.float(), out_m.float()),
                       mono_vs_fp64=accuracy_vs_fp64(out_m, q, k, v, causal=causal, scale=scale, rows=rows, ref_rows=ref) if ref is not None else dict(nan),
                       cqsa_vs_fp64=accuracy_vs_fp64(out_c, q, k, v, causal=causal, scale=scale, rows=rows, ref_rows=ref) if ref is not None else dict(nan),
                       perf_mono=perf_m, perf_cqsa=perf_c)
    floor = {torch.float16: 1e-3, torch.bfloat16: 8e-3}.get(q.dtype, 1e-6)
    gap = rep.mono_vs_fp64["rel_fro"] if ref is not None else float("nan")
    if rep.cqsa_vs_mono["bit_identical"]:
        rep.verdict = "bit-identical"
    elif ref is None:
        rep.verdict = (f"exact (within the {str(q.dtype).replace('torch.','')} rounding floor {floor:.0e}: {rep.cqsa_vs_mono['rel_fro']:.1e})"
                       if rep.cqsa_vs_mono["rel_fro"] <= floor else
                       f"NOT exact: decomposed differs from monolithic by {rep.cqsa_vs_mono['rel_fro']:.1e} (> rounding floor {floor:.0e})")
    elif rep.cqsa_vs_mono["rel_fro"] <= rounding_tolerance * max(gap, 1e-12):
        rep.verdict = (f"exact (within rounding: {rep.cqsa_vs_mono['rel_fro']:.1e} vs the monolithic kernel's "
                       f"own {rep.mono_vs_fp64['rel_fro']:.1e} error against float64)")
    else:
        rep.verdict = (f"NOT exact: decomposed differs from monolithic by {rep.cqsa_vs_mono['rel_fro']:.1e} "
                       f"(> {rounding_tolerance:g}x the monolithic error {rep.mono_vs_fp64['rel_fro']:.1e})")
    if verbose:
        print(rep)
    return rep


# ---------------------------------------------------------------------------
# quick_bench
# ---------------------------------------------------------------------------

def default_configs(*, host_ok: bool = True, max_itr: int = 2, n_pars: Sequence[int] = (1, 2, 4)) -> list[Config]:
    cfgs = [Config(mode="mono", mono_backend="flash"), Config(mode="mono", mono_backend="sdpa")]
    for itr in range(1, max_itr + 1):
        for acc in ("gpu", "cpu"):
            for host in ((False, True) if host_ok else (False,)):
                for n_par in n_pars:
                    if acc == "cpu" and not host:
                        continue   # host accumulator with device-resident inputs is never the point
                    cfgs.append(Config(mode="cqsa", itr=itr, acc=acc, stream_from_host=host, n_par=n_par))
    return cfgs


def select_by_rule(rows: list[dict], budget_gib: float | None, time_key: str = "s",
                   mem_key: str = "peak_gib", tol: float = 0.02) -> dict | None:
    """
    The user's rule: use as much memory as the budget allows, unless another
    configuration is both faster and smaller. That is the Pareto frontier of
    (time, memory) among feasible rows, and on that frontier the choice is the
    fastest point (ties within `tol` broken toward less memory).
    """
    feas = [r for r in rows if r.get("ok") and (budget_gib is None or r[mem_key] <= budget_gib)]
    if not feas:
        return None
    front = [r for r in feas if not any((o[time_key] < r[time_key]) and (o[mem_key] < r[mem_key]) for o in feas)]
    best_t = min(r[time_key] for r in front)
    near = [r for r in front if r[time_key] <= best_t * (1 + tol)]
    return min(near, key=lambda r: r[mem_key])


def quick_bench(*, N: int, B: int = 1, H: int = 8, D: int = 64, dtype=torch.float16, causal: bool = True,
                configs: Sequence[Config] | None = None, budget_gib: float | None = None,
                inner: Callable | None = None, seed: int = 0, acc_rows: int = 256, reps: int = 2,
                device="cuda", verbose: bool = True) -> dict:
    """Run every configuration on the same inputs; return {'rows': [...], 'pick': row}."""
    device = torch.device(device)
    scale = float(D) ** -0.5
    g = torch.Generator(device="cpu").manual_seed(seed)
    q, k, v = (torch.randn(B, H, N, D, generator=g, dtype=torch.float32).to(dtype).pin_memory() for _ in range(3))
    rows_idx = sample_rows(N, acc_rows, seed)
    ref = reference_rows(q, k, v, rows_idx, causal=causal, scale=scale, device=device)
    configs = list(configs) if configs is not None else default_configs()
    rows = []
    if verbose:
        print(f"quick_bench N={N} B={B} H={H} D={D} {str(dtype).replace('torch.','')} causal={causal}"
              f"  device={torch.cuda.get_device_name(device)}  budget={budget_gib}")
        print(f"{'config':>34} {'time s':>9} {'peak GiB':>9} {'rel err vs fp64':>16} {'note':>6}")
    for cfg in configs:
        r = dict(config=cfg.name(), cfg=asdict(cfg), ok=False)
        try:
            out, perf = measure(lambda: run_config(q, k, v, cfg, causal=causal, scale=scale, inner=inner, device=device),
                                device=device, reps=reps)
            r.update(perf); r["ok"] = True
            r.update({f"acc_{kk}": vv for kk, vv in accuracy_vs_fp64(out, q, k, v, causal=causal, scale=scale,
                                                                   rows=rows_idx, ref_rows=ref).items()})
            del out
        except torch.cuda.OutOfMemoryError:
            r["note"] = "OOM"
        except Exception as e:  # report, keep going
            r["note"] = f"fail: {type(e).__name__}: {str(e)[:80]}"
        _drain(device)
        if verbose:
            if r["ok"]:
                print(f"{r['config']:>34} {r['s']:9.3f} {r['peak_gib']:9.2f} {r['acc_rel_fro']:16.2e}")
            else:
                print(f"{r['config']:>34} {'--':>9} {'--':>9} {'--':>16} {r.get('note','')}")
        rows.append(r)
    pick = select_by_rule(rows, budget_gib)
    if verbose and pick:
        print(f"pick (rule: max memory under budget unless faster AND smaller): {pick['config']}  "
              f"{pick['s']:.3f} s, {pick['peak_gib']:.2f} GiB")
    return dict(N=N, B=B, H=H, D=D, dtype=str(dtype), causal=causal, budget_gib=budget_gib, rows=rows, pick=pick)
