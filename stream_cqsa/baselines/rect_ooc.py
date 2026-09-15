"""
Bounded-memory rectangular out-of-core attention: the exact OOC rival of Stream-CQSA.

Canonical Q/K/V (and, for the backward, O, dO, lse) live in host memory; the device holds
one query tile, one key/value window (two with double buffering), the tile's accumulators
and FlashAttention-2's workspace. Every tile pair is an ordinary FlashAttention-2 call:

* forward: ``_flash_attn_forward`` on (q tile, kv window); partial outputs are merged with the
  fp32 log-sum-exp update (stable online softmax across windows), the last window of a causal
  tile is bottom-right aligned so the mask is exactly the global causal mask;
* backward: ``_flash_attn_backward`` on (q tile, kv window) with the **global** lse and the
  global O, so the kernel's P = exp(S - lse) are the true probabilities and its
  Delta = rowsum(dO * O) is the true one: the per-pair dQ/dK/dV are exact partial sums that
  only need adding (fp32) over pairs. Independently normalised per-tile backwards would be wrong.

Two schedules share the code: ``schedule="fixed"`` (R2: deterministic budget-based equal tiles,
query-outer traversal, one staging buffer) and ``schedule="adaptive"`` (R3: independent q/kv tile
sizes chosen from a bounded candidate set probed on the first call, pinned double buffers with
prefetch on a copy stream, kv-outer traversal for the backward so every K/V window is read once).
Bytes moved, kernel calls, per-phase device-stream time (summed, may overlap) and wall time are
returned in the stats. Layout: ``[B, H, N, D]`` head-major like the rest of the package; the
host copies are made token-major once (counted as packing).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

try:
    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
except Exception as _e:  # pragma: no cover
    _flash_attn_forward = _flash_attn_backward = None
    _IMPORT_ERROR = _e


@dataclass
class OOCStats:
    schedule: str = "fixed"
    traversal: str = "q_outer"
    tile_q: int = 0
    tile_k: int = 0
    n_pairs: int = 0
    n_kernels: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    host_pack_bytes: int = 0
    probe_s: float = 0.0             # cold: adaptive tile probing (part of wall_s)
    probe_cached: bool = False
    phases_s: dict = field(default_factory=dict)   # summed device-stream durations, may overlap
    wall_s: float = 0.0
    base_alloc_bytes: int = 0        # allocated by this process before the call (not the rival's)
    base_reserved_bytes: int = 0
    free_at_start_bytes: int = 0     # driver-free device memory at call start
    peak_alloc_bytes: int = 0
    peak_reserved_bytes: int = 0
    budget_bytes: int = 0
    budget_breached: bool = False    # (peak allocated - baseline) > budget
    pinned_bytes: int = 0

    def as_dict(self):
        d = dict(self.__dict__)
        d["phases_s"] = dict(self.phases_s)
        return d


def _nbytes(*ts) -> int:
    return sum(int(t.numel()) * t.element_size() for t in ts if t is not None)


class _Phase:
    """Device-stream timer: cuda events on the given stream, resolved at close()."""

    def __init__(self):
        self.ev: dict[str, list] = {}

    def start(self, name, stream):
        e = torch.cuda.Event(enable_timing=True); e.record(stream)
        return (name, e)

    def stop(self, tok, stream):
        e = torch.cuda.Event(enable_timing=True); e.record(stream)
        self.ev.setdefault(tok[0], []).append((tok[1], e))

    def totals(self):
        torch.cuda.synchronize()
        return {k: sum(a.elapsed_time(b) for a, b in v) / 1e3 for k, v in self.ev.items()}


def tile_budget_bytes(T_q: int, T_k: int, H: int, D: int, itemsize: int, *, direction: str, double: bool) -> int:
    """Device bytes a (T_q, T_k) pair needs, inputs double-buffered if `double`."""
    Dr = -(-D // 32) * 32
    hd = H * D
    stage = 2 if double else 1
    if direction == "fwd":
        q = T_q * hd * itemsize * stage
        kv = 2 * T_k * hd * itemsize * stage
        acc = T_q * hd * 4 + T_q * H * 4 * 2            # fp32 acc + lse (+ merge temporary)
        work = T_q * hd * itemsize + T_q * H * 4 + T_q * hd * 4   # kernel out, lse, fp32 cast
        return q + kv + acc + work
    # backward: q, dO, O16 tiles + lse; K, V window; dq/dk/dv outputs; FA-2 dq_accum fp32 + softmax_d;
    # fp32 accumulators (dq for q-outer, dk/dv for kv-outer: take the larger)
    qside = 3 * T_q * hd * itemsize * stage + T_q * H * 4 * stage
    kv = 2 * T_k * hd * itemsize * stage
    outs = (T_q + 2 * T_k) * hd * itemsize
    work = T_q * H * Dr * 4 + T_q * H * 4
    accum = max(T_q * hd * 4, 2 * T_k * hd * 4) + max(T_q, T_k) * hd * 4   # accumulator + fp32 cast temp
    return qside + kv + outs + work + accum


def plan_tiles(N: int, H: int, D: int, itemsize: int, budget_bytes: int, *, direction: str,
               double: bool, fraction: float = 0.6, aspect: float = 1.0, align: int = 1024) -> tuple[int, int]:
    """Largest equal-ish tiles (T_k = aspect * T_q) with pair bytes <= fraction * budget; multiples of `align`."""
    lo, hi = align, max(align, -(-N // align) * align)
    best = align
    while lo <= hi:
        mid = (lo + hi) // 2 // align * align
        T_q = max(align, mid); T_k = min(N, max(align, int(T_q * aspect) // align * align))
        if tile_budget_bytes(min(T_q, N), T_k, H, D, itemsize, direction=direction, double=double) <= fraction * budget_bytes:
            best = T_q; lo = mid + align
        else:
            hi = mid - align
    T_q = min(best, N); T_k = min(N, max(align, int(best * aspect) // align * align))
    return int(T_q), int(T_k)


class RectOOC:
    """Rectangular OOC attention under a device budget. See the module docstring."""

    _tune_cache: dict = {}

    def __init__(self, *, budget_bytes: int, device="cuda", schedule: str = "fixed", tile_q: Optional[int] = None,
                 tile_k: Optional[int] = None, traversal: str = "auto", probe_max: int = 6, align: int = 1024):
        if _flash_attn_forward is None:
            raise RuntimeError(f"flash-attn is required for the rectangular OOC rival: {_IMPORT_ERROR}")
        self.budget = int(budget_bytes)
        self.device = torch.device(device)
        self.schedule = schedule
        self.double = schedule == "adaptive"
        self.tile_q, self.tile_k = tile_q, tile_k
        self.traversal = traversal
        self.probe_max = int(probe_max)
        self.align = int(align)
        self.copy_stream = torch.cuda.Stream(device=self.device) if self.double else None
        self._pin_events: dict = {}

    # ------------------------------------------------------------------ helpers
    def _tiles(self, N, H, D, itemsize, direction, q_tm=None, k_tm=None, v_tm=None, extra=None, causal=True, scale=1.0):
        """(T_q, T_k, traversal, probe_s, cached) for this call."""
        if self.tile_q and self.tile_k:
            return int(self.tile_q), int(self.tile_k), self._trav(direction), 0.0, False
        if self.schedule == "fixed":
            T_q, T_k = plan_tiles(N, H, D, itemsize, self.budget, direction=direction, double=False)
            return T_q, T_k, "q_outer", 0.0, False
        key = (N, H, D, itemsize, direction, bool(causal), self.budget)
        if key in RectOOC._tune_cache:
            T_q, T_k, trav = RectOOC._tune_cache[key]
            return T_q, T_k, trav, 0.0, True
        # bounded probe: candidates from the budget at three sizes x two aspects, one pair each
        t0 = time.perf_counter()
        cands = []
        for frac in (0.6, 0.3, 0.15):
            for aspect in (1.0, 2.0):
                T_q, T_k = plan_tiles(N, H, D, itemsize, self.budget, direction=direction, double=True, fraction=frac, aspect=aspect)
                if (T_q, T_k) not in cands:
                    cands.append((T_q, T_k))
        cands = cands[:self.probe_max]
        best, best_rate = None, float("inf")
        for T_q, T_k in cands:
            rate = self._probe_pair(T_q, T_k, H, D, itemsize, direction, scale)
            if rate < best_rate:
                best, best_rate = (T_q, T_k), rate
        trav = self._trav(direction)
        RectOOC._tune_cache[key] = (best[0], best[1], trav)
        torch.cuda.synchronize(self.device)
        return best[0], best[1], trav, time.perf_counter() - t0, False

    def _trav(self, direction):
        if self.traversal != "auto":
            return self.traversal
        return "q_outer" if (self.schedule == "fixed" or direction == "fwd") else "kv_outer"

    def _probe_pair(self, T_q, T_k, H, D, itemsize, direction, scale):
        """Seconds per (query token x key token) for one tile pair, inputs already on the device."""
        dev = self.device; dt = torch.float16 if itemsize == 2 else torch.bfloat16
        q = torch.randn(1, T_q, H, D, device=dev, dtype=dt); k = torch.randn(1, T_k, H, D, device=dev, dtype=dt); v = torch.randn_like(k)
        def one():
            if direction == "fwd":
                _flash_attn_forward(q, k, v, 0.0, scale, False, -1, -1, 0.0, None, False)
            else:
                o, lse, _, _ = _flash_attn_forward(q, k, v, 0.0, scale, False, -1, -1, 0.0, None, False)
                _flash_attn_backward(torch.randn_like(o), q, k, v, o, lse, None, None, None, 0.0, scale, False, -1, -1, 0.0, None, False)
        one(); torch.cuda.synchronize(dev)
        t0 = time.perf_counter(); one(); torch.cuda.synchronize(dev); dt_ = time.perf_counter() - t0
        del q, k, v
        return dt_ / (T_q * T_k)

    def _stage(self, src_view, buf, dst, ph, stats):
        """host view -> pinned buf -> device dst (on the copy stream if double-buffered).
        The pinned buffer is reused: wait for its previous asynchronous H2D before writing into it."""
        n = src_view.shape[0]
        ev = self._pin_events.get(buf.data_ptr())
        if ev is not None:
            ev.synchronize()
        b = buf[:n]; b.copy_(src_view)                       # host pack (strided read into pinned)
        stats.host_pack_bytes += _nbytes(b)
        stream = self.copy_stream or torch.cuda.current_stream(self.device)
        tok = ph.start("h2d", stream)
        with torch.cuda.stream(stream):
            dst[:n].copy_(b, non_blocking=True)
        ph.stop(tok, stream)
        ev = torch.cuda.Event(); ev.record(stream); self._pin_events[buf.data_ptr()] = ev
        stats.h2d_bytes += _nbytes(b)
        if self.copy_stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.copy_stream)
        return dst[:n]

    def _windows(self, s, e, N, T_k, causal):
        """Key windows [ks, ke) that query tile [s, e) needs (clipped to e when causal)."""
        end = e if causal else N
        return [(ks, min(ks + T_k, end)) for ks in range(0, end, T_k)]

    @staticmethod
    def _subpairs(s, e, ks, ke, causal):
        """(q_lo, q_hi, k_hi, causal_flag) sub-pairs of query tile [s, e) x key window [ks, ke).
        Causal: the diagonal block (queries [lo, hi) x keys [ks, hi), bottom-right aligned so that
        key <= query is exactly the global mask) and the block below it (queries [ke, e) see all
        keys of the window). Non-causal: the whole tile x window."""
        if not causal:
            return [(s, e, ke, False)]
        out = []
        lo, hi = max(s, ks), min(e, ke)
        if lo < hi:
            out.append((lo, hi, hi, True))
        if ke <= e and max(s, ke) < e:
            out.append((max(s, ke), e, ke, False))
        return out

    # ------------------------------------------------------------------ forward
    def forward(self, q, k, v, *, causal: bool = True, scale: Optional[float] = None):
        """q/k/v host [B, H, N, D] fp16/bf16 -> (out fp32 host [B, H, N, D], lse fp32 host [B, H, N], stats)."""
        assert not q.is_cuda, "the rival's canonical inputs live in host memory"
        B, H, N, D = q.shape
        itemsize = q.element_size(); dt = q.dtype; dev = self.device
        scale = float(D ** -0.5 if scale is None else scale)
        stats = OOCStats(schedule=self.schedule, budget_bytes=self.budget)
        t_wall = time.perf_counter()
        torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev)
        stats.base_alloc_bytes = torch.cuda.memory_allocated(dev); stats.base_reserved_bytes = torch.cuda.memory_reserved(dev)
        stats.free_at_start_bytes = torch.cuda.mem_get_info(dev)[0]
        ph = _Phase()
        T_q, T_k, trav, probe_s, cached = self._tiles(N, H, D, itemsize, "fwd", causal=causal, scale=scale)
        stats.tile_q, stats.tile_k, stats.traversal, stats.probe_s, stats.probe_cached = T_q, T_k, trav, probe_s, cached
        # token-major host copies (packing), pinned staging buffers
        t0 = time.perf_counter()
        q_tm, k_tm, v_tm = (t.transpose(1, 2).contiguous() for t in (q, k, v))
        stats.host_pack_bytes += _nbytes(q_tm, k_tm, v_tm)
        stats.phases_s["pack_host_s"] = time.perf_counter() - t0
        nbuf = 2 if self.double else 1
        pin_q = torch.empty((T_q, H, D), dtype=dt).pin_memory()
        pin_k = [torch.empty((T_k, H, D), dtype=dt).pin_memory() for _ in range(nbuf)]
        pin_v = [torch.empty((T_k, H, D), dtype=dt).pin_memory() for _ in range(nbuf)]
        pin_o = torch.empty((T_q, H, D), dtype=torch.float32).pin_memory()
        stats.pinned_bytes = _nbytes(pin_q, pin_o, *pin_k, *pin_v)
        out = torch.empty((B, N, H, D), dtype=torch.float32)          # host, token-major; returned transposed
        lse_out = torch.empty((B, H, N), dtype=torch.float32)
        dq_ = [torch.empty((T_q, H, D), device=dev, dtype=dt) for _ in range(1)]
        dk_ = [torch.empty((T_k, H, D), device=dev, dtype=dt) for _ in range(nbuf)]
        dv_ = [torch.empty((T_k, H, D), device=dev, dtype=dt) for _ in range(nbuf)]
        cur = torch.cuda.current_stream(dev)
        for b in range(B):
            for s in range(0, N, T_q):
                e = min(N, s + T_q); Sq = e - s
                qd = self._stage(q_tm[b, s:e], pin_q, dq_[0], ph, stats)
                acc = torch.zeros((Sq, H, D), device=dev, dtype=torch.float32)
                lse = torch.full((Sq, H, 1), float("-inf"), device=dev, dtype=torch.float32)
                for wi, (ks, ke) in enumerate(self._windows(s, e, N, T_k, causal)):
                    slot = wi % nbuf
                    kd = self._stage(k_tm[b, ks:ke], pin_k[slot], dk_[slot], ph, stats)
                    vd = self._stage(v_tm[b, ks:ke], pin_v[slot], dv_[slot], ph, stats)
                    for lo, hi, khi, cflag in self._subpairs(s, e, ks, ke, causal):
                        tok = ph.start("kernel", cur)
                        o, l, _, _ = _flash_attn_forward(qd[lo - s:hi - s][None], kd[:khi - ks][None], vd[:khi - ks][None],
                                                         0.0, scale, bool(cflag), -1, -1, 0.0, None, False)
                        ph.stop(tok, cur); stats.n_kernels += 1; stats.n_pairs += 1
                        tok = ph.start("merge", cur)
                        l = l[0].transpose(0, 1).unsqueeze(-1)            # [sq, H, 1]
                        a_ = acc[lo - s:hi - s]; m_ = lse[lo - s:hi - s]
                        new = torch.logaddexp(m_, l)
                        a_.mul_(torch.exp(m_ - new)).add_(o[0].float() * torch.exp(l - new))
                        m_.copy_(new)
                        ph.stop(tok, cur)
                        del o, l, new
                    if self.double:
                        self.copy_stream.wait_stream(cur)             # do not overwrite a slot the kernel still reads
                tok = ph.start("d2h", cur)
                pin_o[:Sq].copy_(acc, non_blocking=True)
                ph.stop(tok, cur); cur.synchronize()
                out[b, s:e].copy_(pin_o[:Sq]); lse_out[b, :, s:e].copy_(lse.squeeze(-1).transpose(0, 1).cpu())
                stats.d2h_bytes += _nbytes(pin_o[:Sq]) + Sq * H * 4
                del acc, lse
        torch.cuda.synchronize(dev)
        stats.phases_s.update(ph.totals())
        stats.wall_s = time.perf_counter() - t_wall
        stats.peak_alloc_bytes = torch.cuda.max_memory_allocated(dev); stats.peak_reserved_bytes = torch.cuda.max_memory_reserved(dev)
        stats.budget_breached = (stats.peak_alloc_bytes - stats.base_alloc_bytes) > self.budget
        del dq_, dk_, dv_, pin_q, pin_k, pin_v, pin_o; self._pin_events.clear()
        return out.transpose(1, 2), lse_out, stats

    # ------------------------------------------------------------------ backward
    def backward(self, q, k, v, out, dout, lse, *, causal: bool = True, scale: Optional[float] = None):
        """Host q/k/v/dout [B,H,N,D] (same dtype), out [B,H,N,D] (fp32 or dtype), lse [B,H,N] fp32
        -> (dq, dk, dv) host in the input dtype (accumulated in fp32), stats."""
        assert not q.is_cuda
        B, H, N, D = q.shape
        itemsize = q.element_size(); dt = q.dtype; dev = self.device
        scale = float(D ** -0.5 if scale is None else scale)
        stats = OOCStats(schedule=self.schedule, budget_bytes=self.budget)
        t_wall = time.perf_counter()
        torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev)
        stats.base_alloc_bytes = torch.cuda.memory_allocated(dev); stats.base_reserved_bytes = torch.cuda.memory_reserved(dev)
        stats.free_at_start_bytes = torch.cuda.mem_get_info(dev)[0]
        ph = _Phase()
        T_q, T_k, trav, probe_s, cached = self._tiles(N, H, D, itemsize, "bwd", causal=causal, scale=scale)
        stats.tile_q, stats.tile_k, stats.traversal, stats.probe_s, stats.probe_cached = T_q, T_k, trav, probe_s, cached
        t0 = time.perf_counter()
        q_tm, k_tm, v_tm, do_tm = (t.transpose(1, 2).contiguous() for t in (q, k, v, dout))
        o_tm = out.transpose(1, 2).to(dt).contiguous()                  # the kernel wants O in the input dtype
        lse_c = lse.contiguous()
        stats.host_pack_bytes += _nbytes(q_tm, k_tm, v_tm, do_tm, o_tm)
        stats.phases_s["pack_host_s"] = time.perf_counter() - t0
        dq_h = torch.zeros((B, N, H, D), dtype=torch.float32); dk_h = torch.zeros_like(dq_h); dv_h = torch.zeros_like(dq_h)
        nbuf = 2 if self.double else 1
        pin_q, pin_do, pin_o = ([torch.empty((T_q, H, D), dtype=dt).pin_memory() for _ in range(nbuf)] for _ in range(3))
        pin_lse = [torch.empty((H, T_q), dtype=torch.float32).pin_memory() for _ in range(nbuf)]
        pin_k, pin_v = ([torch.empty((T_k, H, D), dtype=dt).pin_memory() for _ in range(nbuf)] for _ in range(2))
        pin_g = torch.empty((max(T_q, T_k), H, D), dtype=torch.float32).pin_memory()
        stats.pinned_bytes = _nbytes(pin_g, *pin_q, *pin_do, *pin_o, *pin_lse, *pin_k, *pin_v)
        bq = [torch.empty((T_q, H, D), device=dev, dtype=dt) for _ in range(nbuf)]
        bdo = [torch.empty((T_q, H, D), device=dev, dtype=dt) for _ in range(nbuf)]
        bo = [torch.empty((T_q, H, D), device=dev, dtype=dt) for _ in range(nbuf)]
        blse = [torch.empty((H, T_q), device=dev, dtype=torch.float32) for _ in range(nbuf)]
        bk = [torch.empty((T_k, H, D), device=dev, dtype=dt) for _ in range(nbuf)]
        bv = [torch.empty((T_k, H, D), device=dev, dtype=dt) for _ in range(nbuf)]
        cur = torch.cuda.current_stream(dev)

        def pair(qd, dod, od, lsed, kd, vd, last):
            lsed = lsed.contiguous()                                  # [H, Sq]: the kernel wants it contiguous
            tok = ph.start("kernel", cur)
            dq_p = torch.empty_like(qd); dk_p = torch.empty_like(kd); dv_p = torch.empty_like(vd)
            _flash_attn_backward(dod[None], qd[None], kd[None], vd[None], od[None], lsed[None], dq_p[None], dk_p[None], dv_p[None],
                                 0.0, scale, bool(last), -1, -1, 0.0, None, False)
            ph.stop(tok, cur); stats.n_kernels += 1; stats.n_pairs += 1
            return dq_p, dk_p, dv_p

        def d2h_add(dst_h, dev_t, s):
            n = dev_t.shape[0]
            tok = ph.start("d2h", cur)
            pin_g[:n].copy_(dev_t, non_blocking=True)
            ph.stop(tok, cur); cur.synchronize()
            t1 = time.perf_counter()
            dst_h[s:s + n] += pin_g[:n]
            stats.phases_s["merge_host_s"] = stats.phases_s.get("merge_host_s", 0.0) + (time.perf_counter() - t1)
            stats.d2h_bytes += _nbytes(pin_g[:n])

        for b in range(B):
            if trav == "q_outer":
                for s in range(0, N, T_q):
                    e = min(N, s + T_q); Sq = e - s
                    qd = self._stage(q_tm[b, s:e], pin_q[0], bq[0], ph, stats)
                    dod = self._stage(do_tm[b, s:e], pin_do[0], bdo[0], ph, stats)
                    od = self._stage(o_tm[b, s:e], pin_o[0], bo[0], ph, stats)
                    lsed = self._stage(lse_c[b, :, s:e].transpose(0, 1), pin_lse[0].transpose(0, 1), blse[0].transpose(0, 1), ph, stats).transpose(0, 1)
                    dq_acc = torch.zeros((Sq, H, D), device=dev, dtype=torch.float32)
                    for wi, (ks, ke) in enumerate(self._windows(s, e, N, T_k, causal)):
                        slot = wi % nbuf
                        kd = self._stage(k_tm[b, ks:ke], pin_k[slot], bk[slot], ph, stats)
                        vd = self._stage(v_tm[b, ks:ke], pin_v[slot], bv[slot], ph, stats)
                        for lo, hi, khi, cflag in self._subpairs(s, e, ks, ke, causal):
                            a, z = lo - s, hi - s
                            dq_p, dk_p, dv_p = pair(qd[a:z], dod[a:z], od[a:z], lsed[:, a:z], kd[:khi - ks], vd[:khi - ks], cflag)
                            tok = ph.start("merge", cur); dq_acc[a:z] += dq_p.float(); ph.stop(tok, cur)
                            d2h_add(dk_h[b], dk_p.float(), ks); d2h_add(dv_h[b], dv_p.float(), ks)
                            del dq_p, dk_p, dv_p
                        if self.double:
                            self.copy_stream.wait_stream(cur)
                    d2h_add(dq_h[b], dq_acc, s); del dq_acc
            else:   # kv_outer: each K/V window read once; dK/dV accumulate on the device
                for ks in range(0, N, T_k):
                    ke = min(N, ks + T_k); Sk = ke - ks
                    kd = self._stage(k_tm[b, ks:ke], pin_k[0], bk[0], ph, stats)
                    vd = self._stage(v_tm[b, ks:ke], pin_v[0], bv[0], ph, stats)
                    dk_acc = torch.zeros((Sk, H, D), device=dev, dtype=torch.float32); dv_acc = torch.zeros_like(dk_acc)
                    s0 = (ks // T_q) * T_q if causal else 0
                    for ti, s in enumerate(range(s0, N, T_q)):
                        e = min(N, s + T_q)
                        subs = self._subpairs(s, e, ks, ke, causal)
                        if not subs:
                            continue
                        slot = ti % nbuf
                        qd = self._stage(q_tm[b, s:e], pin_q[slot], bq[slot], ph, stats)
                        dod = self._stage(do_tm[b, s:e], pin_do[slot], bdo[slot], ph, stats)
                        od = self._stage(o_tm[b, s:e], pin_o[slot], bo[slot], ph, stats)
                        lsed = self._stage(lse_c[b, :, s:e].transpose(0, 1), pin_lse[slot].transpose(0, 1), blse[slot].transpose(0, 1), ph, stats).transpose(0, 1)
                        for lo, hi, khi, cflag in subs:
                            a, z = lo - s, hi - s
                            dq_p, dk_p, dv_p = pair(qd[a:z], dod[a:z], od[a:z], lsed[:, a:z], kd[:khi - ks], vd[:khi - ks], cflag)
                            tok = ph.start("merge", cur); dk_acc[:khi - ks] += dk_p.float(); dv_acc[:khi - ks] += dv_p.float(); ph.stop(tok, cur)
                            d2h_add(dq_h[b], dq_p.float(), lo)
                            del dq_p, dk_p, dv_p
                        if self.double:
                            self.copy_stream.wait_stream(cur)
                    d2h_add(dk_h[b], dk_acc, ks); d2h_add(dv_h[b], dv_acc, ks); del dk_acc, dv_acc
        torch.cuda.synchronize(dev)
        stats.phases_s.update(ph.totals())
        stats.wall_s = time.perf_counter() - t_wall
        stats.peak_alloc_bytes = torch.cuda.max_memory_allocated(dev); stats.peak_reserved_bytes = torch.cuda.max_memory_reserved(dev)
        stats.budget_breached = (stats.peak_alloc_bytes - stats.base_alloc_bytes) > self.budget
        del bq, bdo, bo, blse, bk, bv, pin_q, pin_do, pin_o, pin_lse, pin_k, pin_v, pin_g; self._pin_events.clear()
        return dq_h.transpose(1, 2).to(dt), dk_h.transpose(1, 2).to(dt), dv_h.transpose(1, 2).to(dt), stats
