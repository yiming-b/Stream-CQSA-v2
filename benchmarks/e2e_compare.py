"""
End-to-end forward comparison, one GPU, identical inputs:

  sdpa      torch SDPA (default backend), Q/K/V on device         [OOMs past ~1M]
  fa2       upstream flash-attn 2.8.3, Q/K/V on device            [OOMs past ~8M]
  shipped   Stream-CQSA as shipped (packages/stream-cqsa, cqsa_cuda), acc=CPU
  next      Stream-CQSA next/ engine (pipelined merge, shared_chunks) + kernel
            variant --kernel (default cqsa_cuda_next_v3), acc=CPU

Every arm is a subprocess with its own PYTHONPATH so the shipped and next
packages never share an interpreter. Stream-CQSA arms read Q/K/V from pinned
host memory (stream_from_host=True, low_memory=True), which is the paper's
recovery configuration; sdpa/fa2 hold Q/K/V on the device. All arms are checked
against fa2's output at N <= 1M (rel. Frobenius error).

    sbatch next/slurm/run1_test.slurm next/bench/e2e_compare.py --N 262144 1048576 2097152
"""
import argparse, json, os, subprocess, sys, time

DEV = "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev"

def arm(name, N, itr, reps, kernel, npar):
    import torch, gc
    B, H, D = 1, 8, 64
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, N, D, dtype=torch.float16).pin_memory() for _ in range(3))
    import socket
    r = dict(arm=name, N=N, host=socket.gethostname(), loadavg_before=os.getloadavg()[0], ncpu=os.cpu_count())
    def timeit(fn):
        ts = []
        for i in range(reps + 1):
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t0 = time.perf_counter(); out = fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        r["s"] = min(ts[1:]); r["reps_s"] = [round(t, 3) for t in ts]; r["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        r["loadavg_after"] = os.getloadavg()[0]
        return out
    try:
        if name in ("sdpa", "fa2"):
            qd, kd, vd = (t.cuda() for t in (q, k, v))
            if name == "sdpa":
                import torch.nn.functional as F
                out = timeit(lambda: F.scaled_dot_product_attention(qd, kd, vd, is_causal=True))
            else:
                from flash_attn import flash_attn_func
                out = timeit(lambda: flash_attn_func(qd.transpose(1, 2), kd.transpose(1, 2), vd.transpose(1, 2), causal=True).transpose(1, 2))
            out = out.float().cpu()
        else:
            from stream_cqsa.stable_stream import stream_cqsa_forward
            import stream_cqsa.interface as I
            r["kernel"] = os.path.basename(I.cqsa_cuda.__file__)
            kw = dict(itr=itr, causal=True, stream_from_host=True, low_memory=True, allow_escalation=False, max_parallel=npar)
            if name == "next":
                kw.update(shared_chunks=True)
            out = timeit(lambda: stream_cqsa_forward(q, k, v, **kw)[0]).float().cpu()
        ref = f"{DEV}/next/logs/e2e_ref_{N}.pt"
        if name == "fa2":
            torch.save(out, ref)
        elif os.path.exists(ref):
            o2 = torch.load(ref); r["rel_err_vs_fa2"] = ((out - o2).norm() / o2.norm()).item()
    except torch.cuda.OutOfMemoryError as e:
        r["oom"] = True
    print("RESULT " + json.dumps(r), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs=3, metavar=("NAME", "N", "ITR"))
    ap.add_argument("--N", type=int, nargs="+", default=[262144, 1048576])
    ap.add_argument("--itr", type=int, default=1)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--npar", type=int, default=1)
    ap.add_argument("--kernel", default="cqsa_cuda_next_v3")
    ap.add_argument("--arms", nargs="+", default=["fa2", "sdpa", "shipped", "next"])
    ap.add_argument("--out", default="e2e_compare.json")
    a = ap.parse_args()
    if a.arm:
        return arm(a.arm[0], int(a.arm[1]), int(a.arm[2]), a.reps, a.kernel, a.npar)
    import torch
    print(f"GPU {torch.cuda.get_device_name(0)}  B=1 H=8 D=64 fp16 causal, itr={a.itr}, Stream-CQSA acc=CPU n_par={a.npar}, next kernel={a.kernel}")
    base_env = dict(os.environ)
    envs = {
        "shipped": {**base_env, "PYTHONPATH": f"{DEV}/packages/stream-cqsa", "CQSA_CUDA_MODULE": "cqsa_cuda", "CQSA_FWD_MERGE_ASYNC": "0"},
        "next":    {**base_env, "PYTHONPATH": f"{DEV}/next/pkg:" + ":".join(f"{DEV}/next/kernel/{v}" for v in ("base", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11")),
                    "CQSA_CUDA_MODULE": a.kernel},
    }
    rows = []
    print(f"{'N':>8} | " + " ".join(f"{x:>14}" for x in a.arms) + " | rel.err vs fa2 (shipped, next)")
    for N in a.N:
        res = {}
        for name in a.arms:
            env = envs.get(name, base_env)
            cmd = [sys.executable, __file__, "--arm", name, str(N), str(a.itr), "--reps", str(a.reps), "--npar", str(a.npar), "--kernel", a.kernel]
            p = subprocess.run(cmd, capture_output=True, text=True, env=env)
            l = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
            if not l:
                print(f"  {name} FAILED: {p.stderr[-800:]}"); res[name] = dict(arm=name, N=N, failed=True); continue
            res[name] = json.loads(l[-1][7:])
        def cell(n):
            x = res.get(n, {})
            return f"{'OOM':>14}" if x.get("oom") else f"{'fail':>14}" if x.get("failed") else f"{x['s']:8.2f}s/{x['peak_gib']:4.1f}G"
        errs = ", ".join(f"{res[n].get('rel_err_vs_fa2', float('nan')):.1e}" for n in ("shipped", "next") if n in res)
        loads = " ".join(f"{n}:{res[n].get('loadavg_before', float('nan')):.0f}->{res[n].get('loadavg_after', float('nan')):.0f}" for n in a.arms if n in res)
        print(f"{N:>8} | " + " ".join(cell(n) for n in a.arms) + f" | {errs}   [host {next(iter(res.values())).get('host','?')} load {loads}]", flush=True)
        rows.append(dict(N=N, **{n: res[n] for n in res}))
    out = f"{DEV}/next/logs/{a.out}"; json.dump(rows, open(out, "w"), indent=1); print("wrote", out)

if __name__ == "__main__":
    main()
