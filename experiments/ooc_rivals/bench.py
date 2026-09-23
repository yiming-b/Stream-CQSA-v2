"""
Automatic Stream-CQSA versus exact out-of-core rivals under a device memory budget.

    python bench.py run    --out results/ooc_rivals --budgets 10,20,40,80 --N 262144,...,4194304 --directions fwd,bwd
    python bench.py worker '<json>'                       (one measurement in its own process; prints RESULT ...)
    python bench.py report --out results/ooc_rivals       (tables, figures, report.md)

Methods (all start from host-resident fp16 Q/K/V (and dO) and end with host outputs / gradients):
  fa2                GPU-resident FlashAttention-2 (R1): H2D inputs, one flash_attn_func call (+ its autograd
                     backward for "bwd"), D2H. OOM recorded. A kernel-only reference (inputs already on the
                     device, no transfers) is measured AFTER the timed runs, outside the timed region, labelled.
  rect_ooc           bounded-memory rectangular OOC attention, fixed schedule (R2)
  rect_ooc_adaptive  the same with adaptive tiles, double buffering, kv-outer backward (R3)
  cqsa_auto          stream_cqsa.attention() in its automatic mode with the budget as the hardware
                     description: the planner picks depth / accumulator placement / residency /
                     concurrency, the router picks the engine (classic / wave) and kernel; retries and
                     escalations are the library's own and are inside the timed call.

Timed region (harness v2, 2026-09-23). Every timed region is "host inputs -> host outputs" and contains
exactly one execution of the method: for fa2 the H2D copies, one forward call (plus one backward), the D2H
copies; for the OOC methods their own call(s). Input preparation is UNTIMED and gives every method its
inputs in the host layout it consumes without a copy:
  * Q/K/V (and dO) are stored token-major ([B, N, H, D] contiguous, FlashAttention's layout) for fa2 and
    both rectangular rivals (they pack to exactly this layout; with it their pack step is a no-op) and
    Q/K/V for Stream-CQSA (its streaming engines gather token-major; head-major storage costs them a copy).
    The [B, H, N, D] logical shape the package APIs take is the zero-copy transposed view of that storage.
  * dO for Stream-CQSA stays head-major ([B, H, N, D] contiguous): its backward makes one token-major copy
    of dO itself either way and computes rowsum(dO*O) on head-major slices, so head-major is its cheaper
    layout (a token-major dO would cost it two copies).
  * The values are identical in every layout (generated head-major from seed 0 as in harness v1, then
    re-laid out), so rows of both harness versions see the same data.
  * Casts a method needs to meet the output contract (host output in the input dtype) stay inside the
    timed region: the rectangular rivals return fp32 outputs and pay their fp32 -> fp16 cast; Stream-CQSA
    pays its own inside attention(); FA-2 returns fp16.
Harness v1 (grid jobs 13952418-25) differed: the fa2 forward arm ran flash_attn_func twice inside the timed
region (result + kernel-only) and transposed Q/K/V on the host there (FA-2 forward ~2x inflated); all
arms received head-major inputs and paid a host transpose (rect: its pack_host_s phase, 0.54 s of 132 s at
4M, <0.5%; Stream-CQSA: the same copy; the cqsa bwd arm also cloned Q/K/V inside the timed region). Rows
carry "harness": "v1" (absent) or "v2".

Host memory (why the 4M backward rows of v1 died at --mem=96G): v1 compared gradients in float64 (three
full fp64 copies per gradient: 48 GiB at 4M), kept the cold run's outputs alive through the warm runs and
loaded the gradient reference whole. v2 checks correctness right after the cold run and frees the outputs
before the warm runs, compares in chunks of the sequence (fp64 per chunk only), counts NaN/Inf in chunks,
memory-maps the gradient reference, and records the worker's peak RSS (VmHWM) over the whole worker
(host_rss_peak_gib, includes input generation) and per stage after input generation: the cold timed run
(host_rss_cold_run_gib), the correctness check (host_rss_check_gib; the memory-mapped reference's resident
file pages count here but are reclaimable), the warm timed runs (host_rss_warm_run_gib), plus the RSS
holding the inputs (host_rss_inputs_gib) and the input-generation peak (host_rss_gen_peak_gib). A row whose cold run completed but a warm run failed keeps
cold_s and names the failed repetition (failed_rep).

Budget: torch.cuda.set_per_process_memory_fraction(budget / total) caps the allocator; the planner
receives the same budget; baseline / free / peak allocated / peak reserved are recorded and a breach
(peak allocated - baseline > budget) is flagged. "bwd" rows time the forward (with its saved state)
plus the backward, as in the paper; forward-only rows are "fwd". Correctness: forward against float64
on sampled rows at every N; gradients against float64 at N <= 16K (unit tests) and against the
gradients of the first method that finished the same N (FlashAttention-2 where it fits), pairwise above.
"""
import argparse, fcntl, gc, json, os, subprocess, sys, time, resource, platform, traceback
import torch

B, H, D = 1, 8, 64
DEFAULT_BUDGETS = "10,20,40,80"
DEFAULT_N = "262144,524288,1048576,2097152,4194304"
METHODS = ["fa2", "rect_ooc", "rect_ooc_adaptive", "cqsa_auto"]
ACC_ROWS = 128
HARNESS = "v2"
CHUNK = 1 << 16                     # tokens per chunk in the host-side comparisons (256 MiB per fp64 temporary)


def gib(x): return x / 2**30


# --------------------------------------------------------------------------- host memory probes
def _vm(field):
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith(field + ":"):
                return int(ln.split()[1]) * 1024
    except OSError:
        pass
    return None


def reset_peak_rss():
    """Reset VmHWM to the current RSS (Linux clear_refs '5'); False if not permitted."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------- inputs
def _token_major(t):
    """Same values, token-major storage, [B, H, N, D] logical view (zero-copy transpose of [B, N, H, D])."""
    return t.transpose(1, 2).contiguous().transpose(1, 2)


def make_inputs(N, method, direction, seed=0):
    """q, k, v, dout as [B, H, N, D] views in the method's native host layout (see the module docstring).
    Generated head-major in fp32 from seed 0 exactly as harness v1 did, one tensor at a time, then re-laid out."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(4):                                         # q, k, v, dout: the same generator sequence as v1
        if i == 3 and direction == "fwd":
            break                                              # dO is the last draw: skipping it leaves q, k, v unchanged
        t = torch.randn(B, H, N, D, generator=g, dtype=torch.float32).to(torch.float16)
        hm_ok = (i == 3 and method == "cqsa_auto")             # Stream-CQSA's dO path: head-major is its cheap layout
        out.append(t if hm_ok else _token_major(t))
        del t
    if direction == "fwd":
        out.append(None)
    return tuple(out)


def layout_of(t):
    if t is None:
        return None
    return "head-major" if t.is_contiguous() else ("token-major" if t.transpose(1, 2).is_contiguous() else "strided")


# --------------------------------------------------------------------------- correctness (bounded host memory)
def ref_rows(q, k, v, rows, *, causal, scale, tile=8192):
    """float64 attention for the sampled query rows, streaming keys on the device."""
    dev = torch.device("cuda")
    qr = q[0, :, rows].to(dev).double()                          # [H, R, D]
    N = q.shape[2]
    m = torch.full((H, len(rows)), float("-inf"), device=dev, dtype=torch.float64)
    l = torch.zeros((H, len(rows)), device=dev, dtype=torch.float64)
    acc = torch.zeros((H, len(rows), D), device=dev, dtype=torch.float64)
    rows_t = torch.as_tensor(rows, device=dev)
    for s in range(0, N, tile):
        e = min(N, s + tile)
        kt = k[0, :, s:e].to(dev).double(); vt = v[0, :, s:e].to(dev).double()
        sc = torch.einsum("hrd,hkd->hrk", qr, kt) * scale
        if causal:
            sc = sc.masked_fill(torch.arange(s, e, device=dev)[None, None, :] > rows_t[None, :, None], float("-inf"))
        mn = torch.maximum(m, sc.amax(-1)); w = torch.exp(m - mn)
        p = torch.exp(sc - mn[..., None])
        acc = acc * w[..., None] + torch.einsum("hrk,hkd->hrd", p, vt); l = l * w + p.sum(-1); m = mn
    return (acc / l[..., None]).cpu()                            # [H, R, D]


def rel_rows(out, ref, rows):
    got = out[0, :, rows].double().cpu()
    return float((got - ref).norm() / ref.norm().clamp_min(1e-300))


def rel(a, b, chunk=CHUNK):
    """||a - b|| / ||b|| over [B, H, N, D] tensors, chunked along N (fp64 per chunk only)."""
    num = den = 0.0
    for s in range(0, a.shape[2], chunk):
        x = a[:, :, s:s + chunk].double(); y = b[:, :, s:s + chunk].double()
        num += float(((x - y) ** 2).sum()); den += float((y ** 2).sum())
        del x, y
    return (num ** 0.5) / max(den ** 0.5, 1e-300)


def count_nonfinite(t, chunk=CHUNK):
    return int(sum(int((~torch.isfinite(t[:, :, s:s + chunk])).sum()) for s in range(0, t.shape[2], chunk)))


def fp16_copy(t, chunk=CHUNK):
    """Contiguous fp16 copy of a [B, H, N, D] tensor, built chunk by chunk (no full-size fp32 temporary)."""
    if t.dtype == torch.float16:
        return t
    o = torch.empty(t.shape, dtype=torch.float16)
    for s in range(0, t.shape[2], chunk):
        o[:, :, s:s + chunk].copy_(t[:, :, s:s + chunk])
    return o


# --------------------------------------------------------------------------- worker
def worker(cfg):
    method, N, direction, budget_gib, reps = cfg["method"], cfg["N"], cfg["direction"], cfg["budget_gib"], cfg["reps"]
    dev = torch.device("cuda")
    torch.set_num_threads(min(8, os.cpu_count() or 8))       # host-side copies/adds: 80 OpenMP threads are pathological on shared nodes
    total = torch.cuda.get_device_properties(0).total_memory
    budget = int(budget_gib * 2**30)
    torch.cuda.set_per_process_memory_fraction(min(1.0, budget / total))
    causal = True; scale = D ** -0.5
    q, k, v, dout = make_inputs(N, method, direction, cfg.get("seed", 0))
    rows = sorted(set(int(x) for x in torch.linspace(0, N - 1, ACC_ROWS).round().tolist()))
    r = dict(cfg, harness=HARNESS, gpu=torch.cuda.get_device_name(0), total_gib=gib(total), host=platform.node(),
             torch=torch.__version__, cuda=torch.version.cuda, src=os.environ.get("CQSA_SRC", os.path.dirname(os.path.abspath(__file__))),
             slurm_job=os.environ.get("SLURM_JOB_ID"), status="ok", error="", error_where=None, times_s=[], cold_s=None, warm_s=None,
             peak_alloc_gib=None, peak_reserved_gib=None, base_alloc_gib=None, free_start_gib=None, budget_breached=None,
             h2d_bytes=None, d2h_bytes=None, bytes_kind=None, n_kernels=None, phases=None, choices={}, err_fwd=None,
             err_grad_vs_ref=None, grad_ref_method=None, nan_inf=0, host_rss_peak_gib=None, host_rss_gen_peak_gib=None, host_rss_cold_run_gib=None, host_rss_check_gib=None, host_rss_warm_run_gib=None,
             host_rss_inputs_gib=None, pinned_gib=None, kernel_only_s=None, kernel_only_reps=None, retries=None,
             input_layout={"q": layout_of(q), "k": layout_of(k), "v": layout_of(v), "dout": layout_of(dout)})
    r["host_rss_inputs_gib"] = gib(_vm("VmRSS") or 0)
    r["host_rss_gen_peak_gib"] = gib(max(_vm("VmHWM") or 0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))
    rss_reset = reset_peak_rss()                             # also resets ru_maxrss: every stage is read separately
    prep = prepare(method, direction, q, k, v, dout)
    try:
        for rep in range(reps + 1):                          # rep 0 = cold (includes any JIT / probing), then warm
            r["failed_rep"] = rep                            # cleared when every repetition completed
            reset_leaves(prep)
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated(); free0 = torch.cuda.mem_get_info()[0]
            t0 = time.perf_counter()
            res = run_once(method, direction, prep, causal, scale, budget, dev)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            pk = torch.cuda.max_memory_allocated(); r["peak_alloc_gib"] = max(r["peak_alloc_gib"] or 0, gib(pk))
            r["peak_reserved_gib"] = max(r["peak_reserved_gib"] or 0, gib(torch.cuda.max_memory_reserved()))
            r["budget_breached"] = bool((r["budget_breached"] or False) or (pk - base > budget))
            if rep == 0:
                r["cold_s"] = dt; r["base_alloc_gib"] = gib(base); r["free_start_gib"] = gib(free0)
                for key_ in ("h2d_bytes", "d2h_bytes", "bytes_kind", "n_kernels", "phases", "choices", "retries", "pinned_gib"):
                    if key_ in res: r[key_] = res[key_]
                if rss_reset:
                    r["host_rss_cold_run_gib"] = gib(_vm("VmHWM") or 0); reset_peak_rss()
                check(r, res, q, k, v, rows, causal, scale, direction, method, cfg)   # untimed; then the outputs are freed
                if rss_reset:
                    r["host_rss_check_gib"] = gib(_vm("VmHWM") or 0)
            else:
                r["times_s"].append(dt)
            for key_ in ("out", "dq", "dk", "dv"):
                res.pop(key_, None)
            del res
            reset_leaves(prep)
            if rep == 0 and rss_reset:
                gc.collect(); reset_peak_rss()
        r["failed_rep"] = None
        r["warm_s"] = sorted(r["times_s"])[len(r["times_s"]) // 2] if r["times_s"] else None
        if rss_reset and reps > 0:
            r["host_rss_warm_run_gib"] = gib(_vm("VmHWM") or 0)
        if method == "fa2":
            gc.collect(); torch.cuda.empty_cache()
            r["kernel_only_s"], r["kernel_only_reps"] = fa2_kernel_only(direction, prep, causal, scale, dev, r["warm_s"])
    except Exception as exc:                                        # noqa: BLE001
        msg = str(exc)
        r["status"] = "oom" if ("out of memory" in msg.lower() or "OutOfMemory" in type(exc).__name__) else "error"
        r["error"] = f"{type(exc).__name__}: {msg[:300]}"
        tb = traceback.extract_tb(exc.__traceback__)
        r["error_where"] = [f"{os.path.relpath(f.filename, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))}:{f.lineno} {f.name}" for f in tb[-4:]]
        r["peak_alloc_gib"] = gib(torch.cuda.max_memory_allocated())
        if method == "cqsa_auto" and not r.get("choices"):
            r["choices"] = cqsa_plan_only(prep, direction, causal, scale, budget)
    r["host_rss_peak_gib"] = max([gib(_vm("VmHWM") or 0)] + [r[k_] for k_ in ("host_rss_gen_peak_gib", "host_rss_cold_run_gib",
                                  "host_rss_check_gib", "host_rss_warm_run_gib") if r.get(k_)])
    return r


def check(r, res, q, k, v, rows, causal, scale, direction, method, cfg):
    """Correctness of the cold run's outputs (untimed), in bounded host memory."""
    ref = ref_rows(q, k, v, rows, causal=causal, scale=scale)
    out = res["out"]
    r["nan_inf"] = count_nonfinite(out)
    r["err_fwd"] = rel_rows(out, ref, rows)
    if direction != "bwd":
        return
    for g_ in ("dq", "dk", "dv"):
        r["nan_inf"] += count_nonfinite(res[g_])
    gref_path = cfg.get("grad_ref")
    if gref_path and os.path.exists(gref_path):
        gref = torch.load(gref_path, mmap=True)                   # 12 GiB at 4M: paged in chunk by chunk
        r["err_grad_vs_ref"] = [rel(res[g_], gref[g_]) for g_ in ("dq", "dk", "dv")]
        r["grad_ref_method"] = gref.get("method")
        del gref
    elif gref_path:
        # first method to finish this N writes the gradient reference the others are compared with
        torch.save({**{g_: fp16_copy(res[g_]) for g_ in ("dq", "dk", "dv")}, "method": method}, gref_path)
        r["grad_ref_method"] = method + " (this row is the reference)"


# --------------------------------------------------------------------------- methods
def prepare(method, direction, q, k, v, dout):
    """Untimed: everything a method needs that is not part of its execution (layouts, autograd leaves)."""
    p = dict(q=q, k=k, v=v, dout=dout)
    if method == "fa2":
        p.update(qt=q.transpose(1, 2), kt=k.transpose(1, 2), vt=v.transpose(1, 2),          # [B, N, H, D] contiguous: no copy
                 dot=None if dout is None else dout.transpose(1, 2))
        assert all(t.is_contiguous() for t in (p["qt"], p["kt"], p["vt"]))
    if method == "cqsa_auto" and direction == "bwd":
        p["leaves"] = tuple(t.detach().requires_grad_(True) for t in (q, k, v))            # views of the same storage
    return p


def reset_leaves(p):
    for t in p.get("leaves", ()):
        t.grad = None


def run_once(method, direction, p, causal, scale, budget, dev):
    q, k, v, dout = p["q"], p["k"], p["v"], p["dout"]
    if method == "fa2":
        from flash_attn import flash_attn_func
        res = {}
        qt, kt, vt = p["qt"], p["kt"], p["vt"]
        if direction == "fwd":
            qd, kd, vd = (t.to(dev) for t in (qt, kt, vt))
            o = flash_attn_func(qd, kd, vd, causal=causal, softmax_scale=scale)
            res["out"] = o.to("cpu").transpose(1, 2)                                        # host [B, N, H, D] storage
            res["h2d_bytes"] = 3 * qt.numel() * 2; res["d2h_bytes"] = o.numel() * 2
            del qd, kd, vd, o
        else:
            qd, kd, vd = (t.to(dev).requires_grad_(True) for t in (qt, kt, vt))
            o = flash_attn_func(qd, kd, vd, causal=causal, softmax_scale=scale)
            dq, dk, dv = torch.autograd.grad(o, [qd, kd, vd], p["dot"].to(dev))
            res["out"] = o.detach().to("cpu").transpose(1, 2)
            res["dq"], res["dk"], res["dv"] = (g.to("cpu").transpose(1, 2) for g in (dq, dk, dv))
            res["h2d_bytes"] = 4 * qt.numel() * 2; res["d2h_bytes"] = 4 * qt.numel() * 2
            del qd, kd, vd, o, dq, dk, dv
        res["bytes_kind"] = "counted"; res["n_kernels"] = 1 if direction == "fwd" else 2
        return res
    if method in ("rect_ooc", "rect_ooc_adaptive"):
        from stream_cqsa.baselines.rect_ooc import RectOOC
        r_ = RectOOC(budget_bytes=budget, schedule="fixed" if method == "rect_ooc" else "adaptive")
        out, lse, st = r_.forward(q, k, v, causal=causal, scale=scale)
        res = {"h2d_bytes": st.h2d_bytes, "d2h_bytes": st.d2h_bytes, "n_kernels": st.n_kernels,
               "phases": {"fwd": st.phases_s}, "bytes_kind": "counted",
               "choices": {"fwd": dict(tile_q=st.tile_q, tile_k=st.tile_k, traversal=st.traversal, probe_s=st.probe_s, probe_cached=st.probe_cached)}}
        if direction == "bwd":
            dq, dk, dv, sb = r_.backward(q, k, v, out, dout, lse, causal=causal, scale=scale)
            res.update(dq=dq, dk=dk, dv=dv); res["h2d_bytes"] += sb.h2d_bytes; res["d2h_bytes"] += sb.d2h_bytes
            res["n_kernels"] += sb.n_kernels; res["phases"]["bwd"] = sb.phases_s
            res["choices"]["bwd"] = dict(tile_q=sb.tile_q, tile_k=sb.tile_k, traversal=sb.traversal, probe_s=sb.probe_s, probe_cached=sb.probe_cached)
        # the output contract is fp16: the rival returns fp32 and pays the cast here (timed). In a "bwd" row the cast
        # comes after the backward, whose fp32 gradient accumulators are gone by then (4 GiB less host peak at 4M)
        res["out"] = out.half(); del out, lse
        res["pinned_gib"] = max(st.pinned_bytes, sb.pinned_bytes if direction == "bwd" else 0) / 2**30
        return res
    if method == "cqsa_auto":
        from stream_cqsa import attention
        hw = cqsa_hardware(budget)
        if direction == "fwd":
            out, pl, info = attention(q, k, v, is_causal=causal, scale=scale, hardware=hw, return_plan=True, return_info=True)
            res = {"out": out}
        else:
            qg, kg, vg = p["leaves"]
            out, pl, info = attention(qg, kg, vg, is_causal=causal, scale=scale, hardware=hw, return_plan=True, return_info=True)
            out.backward(dout)
            res = {"out": out.detach(), "dq": qg.grad, "dk": kg.grad, "dv": vg.grad}
            del out
        info = {k_: (v_ if isinstance(v_, (int, float, str, bool, list, dict, type(None))) else str(v_)) for k_, v_ in info.items()}
        res["choices"] = plan_dict(pl); res["choices"]["info"] = info
        res["retries"] = info.get("oom_retries")
        res["n_kernels"] = info.get("n_subproblems")
        # data movement: modelled from the plan (the engine is not byte-instrumented): streamed inputs
        # per subproblem + fp32 output back when the accumulator is on the host
        itr = int(info.get("itr", 0) or 0); c = int(info.get("c", 7) or 7); l = len(info.get("interest_set", (0, 1, 3)) or (0, 1, 3))
        N = q.shape[2]
        if info.get("engine") == "monolithic":
            h2d, d2h = 3 * N * H * D * 2, N * H * D * 2
        else:
            h2d = (l ** itr) * N * H * D * 2 * 3 if info.get("stream_from_host", True) or info.get("host_resident", True) else 3 * N * H * D * 2
            d2h = N * H * D * 4 if not info.get("accumulate_on_gpu", True) else N * H * D * 2
        if direction == "bwd":
            h2d *= 2; d2h += 3 * N * H * D * 2
        res["h2d_bytes"], res["d2h_bytes"], res["bytes_kind"] = int(h2d), int(d2h), "modelled"
        return res
    raise ValueError(method)


def cqsa_hardware(budget):
    from stream_cqsa.autoconfig import hardware_from_dict
    return hardware_from_dict({"cuda:0": f"{budget}", "host": f"{int(os.environ.get('CQSA_HOST_BUDGET_GIB', '200')) << 30}"})


def plan_dict(pl):
    return {"plan": pl.name(), "reason": getattr(pl, "reason", ""), "est_time_s": getattr(pl, "est_time_s", None),
            "est_peak_gib": getattr(pl, "est_peak_gib", None), "est_host_gib": getattr(pl, "est_host_gib", None),
            "engine_kwargs": ({k_: (list(v_) if isinstance(v_, tuple) else v_) for k_, v_ in pl.engine_kwargs().items()}
                              if getattr(pl, "mode", "mono") != "mono" else None)}


def cqsa_plan_only(p, direction, causal, scale, budget):
    """The plan attention() would run (recorded for failed rows; the call's own plan is lost with the exception)."""
    try:
        from stream_cqsa import attention
        args = p["leaves"] if direction == "bwd" else (p["q"], p["k"], p["v"])
        pl = attention(*args, is_causal=causal, scale=scale, hardware=cqsa_hardware(budget), plan_only=True)
        d = plan_dict(pl); d["recorded"] = "plan_only after the failure (same planner inputs)"
        return d
    except Exception as exc:                                        # noqa: BLE001
        return {"plan_only_error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def fa2_kernel_only(direction, p, causal, scale, dev, warm_s):
    """Resident kernel-only reference: inputs already on the device, no transfers, outside the timed region."""
    from flash_attn import flash_attn_func
    qd, kd, vd = (t.to(dev) for t in (p["qt"], p["kt"], p["vt"]))
    dod = p["dot"].to(dev) if direction == "bwd" else None
    n = 3 if (warm_s or 0) < 60 else 1
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        if direction == "fwd":
            o = flash_attn_func(qd, kd, vd, causal=causal, softmax_scale=scale)
        else:
            qd.requires_grad_(True); kd.requires_grad_(True); vd.requires_grad_(True)
            o = flash_attn_func(qd, kd, vd, causal=causal, softmax_scale=scale)
            g = torch.autograd.grad(o, [qd, kd, vd], dod); del g
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        del o
    del qd, kd, vd, dod
    return sorted(ts)[len(ts) // 2], n


# --------------------------------------------------------------------------- driver
def append_row(path, r):
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(r) + "\n"); f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def run(a):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "results.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            if r.get("harness") == HARNESS:              # rows of an older harness are re-measured, not skipped
                done.add((r["method"], r["N"], r["direction"], r["budget_gib"]))
    budgets = [float(x) for x in a.budgets.split(",")]; Ns = [int(x) for x in a.N.split(",")]
    methods = a.methods.split(","); dirs = a.directions.split(",")
    gref_dir = os.path.join(a.out, "grad_ref"); os.makedirs(gref_dir, exist_ok=True)
    for budget in budgets:
        for N in Ns:
            for direction in dirs:
                for method in methods:
                    key = (method, N, direction, budget)
                    if key in done: continue
                    cfg = dict(method=method, N=N, direction=direction, budget_gib=budget, reps=a.reps, seed=0)
                    if direction == "bwd":
                        cfg["grad_ref"] = os.path.join(gref_dir, f"N{N}.pt")      # written by the first method that finishes
                    t0 = time.time()
                    proc = subprocess.run([sys.executable, "-P", os.path.abspath(__file__), "worker", json.dumps(cfg)],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=a.timeout)
                    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
                    if line is None:
                        err = (proc.stderr or "")[-600:]
                        r = dict(cfg, harness=HARNESS, status="error", error=f"worker exited {proc.returncode}: {err}",
                                 slurm_job=os.environ.get("SLURM_JOB_ID"))
                    else:
                        r = json.loads(line[7:])
                    r["worker_wall_s"] = time.time() - t0
                    append_row(path, r)
                    st = r["status"]; t = r.get("warm_s") or r.get("cold_s")
                    print(f"budget {budget:5.1f} GiB N={N:8d} {direction} {method:18s}: {st:5s} "
                          + (f"{t:8.2f} s (cold {r['cold_s']:.2f}) peak {r['peak_alloc_gib']:.2f} GiB err {r['err_fwd']:.1e} "
                             f"rss {r.get('host_rss_peak_gib') or 0:.1f} GiB (cold run {r.get('host_rss_cold_run_gib') or 0:.1f}, check {r.get('host_rss_check_gib') or 0:.1f}, warm {r.get('host_rss_warm_run_gib') or 0:.1f})"
                             + (f" kernel-only {r['kernel_only_s']:.2f} s" if r.get("kernel_only_s") else "")
                             if st == "ok" else r["error"][:160] + f" rss {r.get('host_rss_peak_gib') or 0:.1f} GiB"), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["run", "worker", "report"]); ap.add_argument("arg", nargs="?")
    ap.add_argument("--out", default="results/ooc_rivals"); ap.add_argument("--budgets", default=DEFAULT_BUDGETS)
    ap.add_argument("--N", default=DEFAULT_N); ap.add_argument("--directions", default="fwd,bwd")
    ap.add_argument("--methods", default=",".join(METHODS)); ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=6 * 3600)
    a = ap.parse_args()
    if a.mode == "worker":
        print("RESULT " + json.dumps(worker(json.loads(a.arg))), flush=True)
    elif a.mode == "run":
        run(a)
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from report import report
        report(a.out)


if __name__ == "__main__":
    main()
