"""
Track 2 follow-up: with the pipelined merge on, what do the remaining knobs buy?

  shared_chunks   contiguous chunk DMA instead of a CPU row-gather per task
  cpu_threads     torch intra-op threads for the host merge (default 8)
  n_par           1 or 2 (4 was measured worse)

    sbatch next/slurm/run1_test.slurm next/bench/pipeline_opts.py --N 262144 1048576
"""
import argparse, json, os, sys, time, gc, subprocess

def worker(N, npar, shared, threads, reps):
    import torch
    from stream_cqsa.stable_stream import stream_cqsa_forward, TraceRecorder
    B, H, D = 1, 8, 64
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, N, D, dtype=torch.float16) for _ in range(3))
    kw = dict(itr=1, causal=True, stream_from_host=True, low_memory=True, allow_escalation=False,
              max_parallel=npar, shared_chunks=bool(shared), cpu_threads=threads)
    ts = []; stages = {}
    for rep in range(reps):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        tr = TraceRecorder(enabled=(rep == reps - 1))
        t0 = time.perf_counter()
        out, info = stream_cqsa_forward(q, k, v, trace=tr, **kw)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
        if tr.enabled:
            stages = {k2: round(v2) for k2, v2 in info.get("stage_totals_ms", {}).items() if k2 != "wait"}
    ref_ok = None
    if N <= 262144:
        import torch.nn.functional as F
        ref = F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True).float()
        ref_ok = ((out.float() - ref).norm() / ref.norm()).item()
    print("RESULT " + json.dumps(dict(N=N, npar=npar, shared=shared, threads=threads, ms=min(ts),
                                      stages=stages, rel_err=ref_ok)), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=5, type=int, metavar=("N", "NPAR", "SHARED", "THREADS", "REPS"))
    ap.add_argument("--N", type=int, nargs="+", default=[262144])
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    if a.worker:
        return worker(*a.worker)
    import torch
    print(f"GPU {torch.cuda.get_device_name(0)}  cpus={os.cpu_count()}  pipelined merge ON, acc=CPU, itr=1 causal")
    rows = []
    for N in a.N:
        print(f"\n=== N={N} ===")
        print(f"{'n_par':>5} {'shared':>7} {'threads':>8} {'ms':>9}  {'stages':<55} {'rel_err':>8}")
        for npar in (1, 2):
            for shared in (0, 1):
                for threads in (8, 16):
                    out = subprocess.run([sys.executable, __file__, "--worker", str(N), str(npar), str(shared), str(threads), str(a.reps)],
                                         capture_output=True, text=True)
                    l = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
                    if not l:
                        print(f"{npar:>5} {shared:>7} {threads:>8}  FAILED: {out.stderr.strip().splitlines()[-1][:110] if out.stderr.strip() else '?'}"); continue
                    r = json.loads(l[-1][7:]); rows.append(r)
                    st = " ".join(f"{k}={v}" for k, v in r["stages"].items())
                    err = "-" if r["rel_err"] is None else f"{r['rel_err']:.1e}"
                    print(f"{npar:>5} {shared:>7} {threads:>8} {r['ms']:9.1f}  {st:<55} {err:>8}", flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "pipeline_opts.json")
    json.dump(rows, open(out, "w"), indent=1); print("\nwrote", out)

if __name__ == "__main__":
    main()
