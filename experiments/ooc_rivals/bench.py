"""
Automatic Stream-CQSA versus exact out-of-core rivals under a device memory budget.

    python bench.py run    --out results/ooc_rivals --budgets 10,20,40,80 --N 262144,...,4194304 --directions fwd,bwd
    python bench.py worker '<json>'                       (one measurement in its own process; prints RESULT ...)
    python bench.py report --out results/ooc_rivals       (tables, figures, report.md)

Methods (all start from host-resident fp16 Q/K/V and end with host outputs / gradients):
  fa2                GPU-resident FlashAttention-2 (R1): H2D inputs, kernel, D2H; OOM recorded. Also a
                     kernel-only reference (device-resident, no transfers) where it fits, labelled as such.
  rect_ooc           bounded-memory rectangular OOC attention, fixed schedule (R2)
  rect_ooc_adaptive  the same with adaptive tiles, double buffering, kv-outer backward (R3)
  cqsa_auto          stream_cqsa.attention() in its automatic mode with the budget as the hardware
                     description: the planner picks depth / accumulator placement / residency /
                     concurrency, the router picks the engine (classic / wave) and kernel; retries and
                     escalations are the library's own and are inside the timed call.

Budget: torch.cuda.set_per_process_memory_fraction(budget / total) caps the allocator; the planner
receives the same budget; baseline / free / peak allocated / peak reserved are recorded and a breach
(peak allocated - baseline > budget) is flagged. "bwd" rows time the forward (with its saved state)
plus the backward, as in the paper; forward-only rows are "fwd". Correctness: forward against float64
on sampled rows at every N; gradients against float64 at N <= 16K (unit tests) and against
FlashAttention-2's fp16 gradients where FA-2 fits, pairwise between methods above that.
"""
import argparse, gc, json, os, subprocess, sys, time, resource, platform
import torch

B, H, D = 1, 8, 64
DEFAULT_BUDGETS = "10,20,40,80"
DEFAULT_N = "262144,524288,1048576,2097152,4194304"
METHODS = ["fa2", "rect_ooc", "rect_ooc_adaptive", "cqsa_auto"]
ACC_ROWS = 128


def gib(x): return x / 2**30


def make_qkv(N, seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(B, H, N, D, generator=g, dtype=torch.float32).to(torch.float16) for _ in range(4))   # q, k, v, dout


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


def rel(a, b):
    a = a.double().cpu(); b = b.double().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


# --------------------------------------------------------------------------- worker
def worker(cfg):
    method, N, direction, budget_gib, reps = cfg["method"], cfg["N"], cfg["direction"], cfg["budget_gib"], cfg["reps"]
    dev = torch.device("cuda")
    torch.set_num_threads(min(8, os.cpu_count() or 8))       # host-side copies/adds: 80 OpenMP threads are pathological on shared nodes
    total = torch.cuda.get_device_properties(0).total_memory
    budget = int(budget_gib * 2**30)
    torch.cuda.set_per_process_memory_fraction(min(1.0, budget / total))
    causal = True; scale = D ** -0.5
    q, k, v, dout = make_qkv(N, cfg.get("seed", 0))
    rows = sorted(set(int(x) for x in torch.linspace(0, N - 1, ACC_ROWS).round().tolist()))
    r = dict(cfg, gpu=torch.cuda.get_device_name(0), total_gib=gib(total), host=platform.node(), torch=torch.__version__,
             cuda=torch.version.cuda, status="ok", error="", times_s=[], cold_s=None, warm_s=None,
             peak_alloc_gib=None, peak_reserved_gib=None, base_alloc_gib=None, free_start_gib=None, budget_breached=None,
             h2d_bytes=None, d2h_bytes=None, bytes_kind=None, n_kernels=None, phases=None, choices={}, err_fwd=None,
             err_grad_vs_ref=None, grad_ref_method=None, nan_inf=0, host_rss_peak_gib=None, kernel_only_s=None, retries=None)
    outs = {}
    try:
        for rep in range(reps + 1):                          # rep 0 = cold (includes any JIT / probing), then warm
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated(); free0 = torch.cuda.mem_get_info()[0]
            t0 = time.perf_counter()
            res = run_once(method, direction, q, k, v, dout, causal, scale, budget, dev)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            if rep == 0:
                r["cold_s"] = dt; outs = res
                r["base_alloc_gib"] = gib(base); r["free_start_gib"] = gib(free0)
            else:
                r["times_s"].append(dt)
            pk = torch.cuda.max_memory_allocated(); r["peak_alloc_gib"] = max(r["peak_alloc_gib"] or 0, gib(pk))
            r["peak_reserved_gib"] = max(r["peak_reserved_gib"] or 0, gib(torch.cuda.max_memory_reserved()))
            r["budget_breached"] = bool((r["budget_breached"] or False) or (pk - base > budget))
            if rep > 0:
                for key_ in ("out", "dq", "dk", "dv"):
                    res.pop(key_, None)
            del res
        r["warm_s"] = sorted(r["times_s"])[len(r["times_s"]) // 2] if r["times_s"] else None
        for key_ in ("h2d_bytes", "d2h_bytes", "bytes_kind", "n_kernels", "phases", "choices", "retries", "kernel_only_s"):
            if key_ in outs: r[key_] = outs[key_]
        # correctness
        ref = ref_rows(q, k, v, rows, causal=causal, scale=scale)
        out = outs["out"]
        r["nan_inf"] = int((~torch.isfinite(out.float())).sum().item())
        r["err_fwd"] = rel_rows(out, ref, rows)
        if direction == "bwd":
            for g_ in ("dq", "dk", "dv"):
                r["nan_inf"] += int((~torch.isfinite(outs[g_].float())).sum().item())
            gref_path = cfg.get("grad_ref")
            if gref_path and os.path.exists(gref_path):
                gref = torch.load(gref_path)
                r["err_grad_vs_ref"] = [rel(outs[g_], gref[g_]) for g_ in ("dq", "dk", "dv")]
                r["grad_ref_method"] = gref.get("method")
            elif gref_path:
                # first method to finish this N writes the gradient reference the others are compared with
                # (FlashAttention-2 where it fits, since it runs first; else the first OOC method)
                torch.save({**{g_: outs[g_].half() for g_ in ("dq", "dk", "dv")}, "method": method}, gref_path)
                r["grad_ref_method"] = method + " (this row is the reference)"
    except Exception as exc:                                        # noqa: BLE001
        msg = str(exc)
        r["status"] = "oom" if ("out of memory" in msg.lower() or "OutOfMemory" in type(exc).__name__) else "error"
        r["error"] = f"{type(exc).__name__}: {msg[:300]}"
        r["peak_alloc_gib"] = gib(torch.cuda.max_memory_allocated())
    r["host_rss_peak_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    return r


def run_once(method, direction, q, k, v, dout, causal, scale, budget, dev):
    if method == "fa2":
        from flash_attn import flash_attn_func
        res = {}
        qt, kt, vt = (t.transpose(1, 2).contiguous() for t in (q, k, v))       # FA-2 wants [B, N, H, D]
        if direction == "fwd":
            qd, kd, vd = (t.to(dev, non_blocking=True) for t in (qt, kt, vt))
            o = flash_attn_func(qd, kd, vd, causal=causal, softmax_scale=scale)
            res["out"] = o.transpose(1, 2).to("cpu")
            res["h2d_bytes"] = 3 * qt.numel() * 2; res["d2h_bytes"] = o.numel() * 2
            del qd, kd, vd
            torch.cuda.synchronize(); t0 = time.perf_counter()
            qd, kd, vd = (t.to(dev) for t in (qt, kt, vt)); torch.cuda.synchronize()
            t1 = time.perf_counter(); o2 = flash_attn_func(qd, kd, vd, causal=causal, softmax_scale=scale); torch.cuda.synchronize()
            res["kernel_only_s"] = time.perf_counter() - t1
            del qd, kd, vd, o2, o
        else:
            qd, kd, vd = (t.to(dev).requires_grad_(True) for t in (qt, kt, vt))
            o = flash_attn_func(qd, kd, vd, causal=causal, softmax_scale=scale)
            dq, dk, dv = torch.autograd.grad(o, [qd, kd, vd], dout.transpose(1, 2).contiguous().to(dev))
            res["out"] = o.detach().transpose(1, 2).to("cpu")
            res["dq"], res["dk"], res["dv"] = (g.transpose(1, 2).to("cpu") for g in (dq, dk, dv))
            res["h2d_bytes"] = 4 * qt.numel() * 2; res["d2h_bytes"] = 4 * qt.numel() * 2
            del qd, kd, vd, o, dq, dk, dv
        res["bytes_kind"] = "counted"; res["n_kernels"] = 1 if direction == "fwd" else 2
        return res
    if method in ("rect_ooc", "rect_ooc_adaptive"):
        from stream_cqsa.baselines.rect_ooc import RectOOC
        r_ = RectOOC(budget_bytes=budget, schedule="fixed" if method == "rect_ooc" else "adaptive")
        out, lse, st = r_.forward(q, k, v, causal=causal, scale=scale)
        res = {"out": out.half(), "h2d_bytes": st.h2d_bytes, "d2h_bytes": st.d2h_bytes, "n_kernels": st.n_kernels,
               "phases": {"fwd": st.phases_s}, "bytes_kind": "counted",
               "choices": {"fwd": dict(tile_q=st.tile_q, tile_k=st.tile_k, traversal=st.traversal, probe_s=st.probe_s, probe_cached=st.probe_cached)}}
        if direction == "bwd":
            dq, dk, dv, sb = r_.backward(q, k, v, out, dout, lse, causal=causal, scale=scale)
            res.update(dq=dq, dk=dk, dv=dv); res["h2d_bytes"] += sb.h2d_bytes; res["d2h_bytes"] += sb.d2h_bytes
            res["n_kernels"] += sb.n_kernels; res["phases"]["bwd"] = sb.phases_s
            res["choices"]["bwd"] = dict(tile_q=sb.tile_q, tile_k=sb.tile_k, traversal=sb.traversal, probe_s=sb.probe_s, probe_cached=sb.probe_cached)
        return res
    if method == "cqsa_auto":
        from stream_cqsa import attention
        from stream_cqsa.autoconfig import hardware_from_dict
        hw = hardware_from_dict({"cuda:0": f"{budget}", "host": f"{int(os.environ.get('CQSA_HOST_BUDGET_GIB', '200')) << 30}"})
        t0 = time.perf_counter()
        if direction == "fwd":
            out, p, info = attention(q, k, v, is_causal=causal, scale=scale, hardware=hw, return_plan=True, return_info=True)
            res = {"out": out}
        else:
            qg, kg, vg = (t.clone().requires_grad_(True) for t in (q, k, v))
            out, p, info = attention(qg, kg, vg, is_causal=causal, scale=scale, hardware=hw, return_plan=True, return_info=True)
            out.backward(dout)
            res = {"out": out.detach(), "dq": qg.grad, "dk": kg.grad, "dv": vg.grad}
        info = {k_: (v_ if isinstance(v_, (int, float, str, bool, list, dict, type(None))) else str(v_)) for k_, v_ in info.items()}
        res["choices"] = {"plan": p.name(), "reason": getattr(p, "reason", ""), "est_time_s": getattr(p, "est_time_s", None),
                          "est_peak_gib": getattr(p, "est_peak_gib", None), "info": info}
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


# --------------------------------------------------------------------------- driver
def run(a):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "results.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); done.add((r["method"], r["N"], r["direction"], r["budget_gib"]))
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
                    proc = subprocess.run([sys.executable, os.path.abspath(__file__), "worker", json.dumps(cfg)],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=a.timeout)
                    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
                    if line is None:
                        err = (proc.stderr or "")[-600:]
                        r = dict(cfg, status="error", error=f"worker exited {proc.returncode}: {err}")
                    else:
                        r = json.loads(line[7:])
                    with open(path, "a") as f:
                        f.write(json.dumps(r) + "\n")
                    st = r["status"]; t = r.get("warm_s") or r.get("cold_s")
                    print(f"budget {budget:5.1f} GiB N={N:8d} {direction} {method:18s}: {st:5s} "
                          + (f"{t:8.2f} s (cold {r['cold_s']:.2f}) peak {r['peak_alloc_gib']:.2f} GiB err {r['err_fwd']:.1e}" if st == "ok" else r["error"][:100]), flush=True)


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
        from report import report
        report(a.out)


if __name__ == "__main__":
    main()
