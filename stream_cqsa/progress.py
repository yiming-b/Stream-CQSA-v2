"""
Progress reporting for Stream-CQSA calls (``verbose=True`` or ``CQSA_VERBOSE=1``).

One line announcing that Stream-CQSA is running and how the call was decomposed,
a tqdm-style bar over the subproblems (or waves) with the elapsed time and an
estimate of the time remaining, and a closing line with the total. The estimate
is the usual rate extrapolation over the units done so far; before the first
unit finishes, the banner carries the planner's cost-model prediction instead.

Uses tqdm when it is installed and a plain single-line bar otherwise; everything
goes to stderr, so stdout stays clean for the caller.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional


def verbose_enabled(verbose: Optional[bool]) -> bool:
    """``verbose=None`` defers to the CQSA_VERBOSE environment variable."""
    if verbose is None:
        return os.environ.get("CQSA_VERBOSE", "0").strip().lower() not in ("", "0", "false", "no", "off")
    return bool(verbose)


def _fmt_s(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m{int(s):02d}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m"


def _fmt_n(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1_000_000 else (f"{n / 1e3:.0f}K" if n >= 10_000 else str(n))


class Progress:
    """A bar over ``total`` units of work (subproblems, or packed tokens of waves)."""

    def __init__(self, total: int, *, enabled: bool, desc: str = "Stream-CQSA", unit: str = "subproblem",
                 banner: Optional[str] = None, expected_s: Optional[float] = None, stream=None):
        self.enabled = bool(enabled)
        self.total = int(total)
        self.done = 0
        self.unit = unit
        self.t0 = time.perf_counter()
        self.stream = stream or sys.stderr
        self._bar = None
        if not self.enabled:
            return
        if banner:
            line = f"Stream-CQSA: {banner}"
            if expected_s is not None:
                line += f" | expected ~{_fmt_s(expected_s)} (cost model)"
            print(line, file=self.stream, flush=True)
        try:
            from tqdm import tqdm
            self._bar = tqdm(total=self.total, desc=desc, unit=unit, dynamic_ncols=True, leave=True,
                             file=self.stream, mininterval=0.2)
        except Exception:      # tqdm absent: plain bar
            self._bar = None
            self._last = -1.0
            self._draw()

    # ---- plain fallback -------------------------------------------------
    def _draw(self, final: bool = False):
        el = time.perf_counter() - self.t0
        frac = self.done / max(1, self.total)
        eta = (el / frac - el) if self.done else float("nan")
        bar = "#" * int(30 * frac) + "-" * (30 - int(30 * frac))
        msg = f"\r  [{bar}] {self.done}/{self.total} {self.unit}s  {_fmt_s(el)} elapsed"
        msg += f", ~{_fmt_s(eta)} left" if self.done and not final else ""
        print(msg + ("\n" if final else ""), end="", file=self.stream, flush=True)

    # ---- api -----------------------------------------------------------
    def set_total(self, total: int):
        self.total = int(total)
        if self._bar is not None:
            self._bar.total = self.total
            self._bar.refresh()

    def update(self, n: int = 1):
        self.done += int(n)
        if not self.enabled:
            return
        if self._bar is not None:
            self._bar.update(int(n))
        else:
            now = time.perf_counter()
            if now - self._last > 0.2 or self.done >= self.total:
                self._last = now
                self._draw()

    def close(self, note: str = ""):
        if not self.enabled:
            return
        el = time.perf_counter() - self.t0
        if self._bar is not None:
            self._bar.close()
        else:
            self._draw(final=True)
        print(f"Stream-CQSA: done in {_fmt_s(el)}" + (f" ({note})" if note else ""), file=self.stream, flush=True)


def describe_call(*, what: str, N: int, B: int, H: int, D: int, c: int, itr: int, n_tasks: int,
                  causal: bool, device: str, extra: str = "") -> str:
    """The banner line: what is being computed and how it was decomposed."""
    L = "".join([f"{what} of N={_fmt_n(N)} tokens (B={B}, H={H}, D={D}, {'causal' if causal else 'non-causal'}) ",
                 f"decomposed over c={c} at depth itr={itr}: {n_tasks} subproblems on {device}"])
    return L + (f" | {extra}" if extra else "")


def expected_seconds(*, N: int, B: int, H: int, D: int, itr: int, c: int, causal: bool,
                     direction: str = "fwd", acc: str = "gpu", stream_from_host: bool = False,
                     n_par: int = 1) -> Optional[float]:
    """Cost-model prediction for the banner; None if the planner is unavailable."""
    try:
        from .autoconfig import CostModel, QUORUM_SETS
        m = CostModel()
        l = len(QUORUM_SETS.get(int(c), (0, 1, 3)))
        return float(m.cqsa_time(int(N), int(B), int(H), int(D), int(itr), bool(causal), acc,
                                 bool(stream_from_host), int(n_par), c=int(c), l=l, direction=direction))
    except Exception:
        return None
