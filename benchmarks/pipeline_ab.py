"""
A/B the pipelined host-accumulator forward against the shipped blocking one.

The switch (CQSA_FWD_MERGE_ASYNC) is read at import, so every arm is a fresh
subprocess. Arms are interleaved (flag 0, flag 1, flag 0, ...) at each setting
so noise on a shared node lands on both.

    python next/bench/pipeline_ab.py --N 65536 --npar 1 2 4 --reps 3 --check
    sbatch next/slurm/run1_test.slurm next/bench/pipeline_ab.py --N 262144 1048576 4194304 --npar 1 2 4
"""
import argparse, json, os, subprocess, sys, time

def worker(flag, N, npar, reps, check, acc):
    os.environ["CQSA_FWD_MERGE_ASYNC"] = str(flag)
    import torch, gc
    import torch.nn.functional as F
    from stream_cqsa.stable_stream import stream_cqsa_forward, _FWD_MERGE_ASYNC
    assert _FWD_MERGE_ASYNC == bool(flag)
    B, H, D = 1, 8, 64
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, N, D, dtype=torch.float16) for _ in range(3))
    kw = dict(itr=1, causal=True, stream_from_host=True, low_memory=(acc == "cpu"),
              allow_escalation=False, max_parallel=npar)
    ts = []
    for _ in range(reps):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, info = stream_cqsa_forward(q, k, v, **kw)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    r = dict(flag=flag, N=N, npar=npar, acc=acc, ms=sorted(ts)[len(ts) // 2], n_par_used=info["n_parallel"])
    if check:
        ref = F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True).float().cpu()
        out = out.float().cpu()          # engine returns the output on the device
        r["rel_err_vs_sdpa"] = ((out - ref).norm() / ref.norm()).item()
        # itr=2 too, and the lse
        out2, info2 = stream_cqsa_forward(q, k, v, **{**kw, "itr": 2})
        r["rel_err_itr2"] = ((out2.float().cpu() - ref).norm() / ref.norm()).item()
        r["lse_finite"] = bool(torch.isfinite(info["lse"]).all())
        r["untouched"] = int(info["untouched_tokens"])
        torch.save(out, f"/tmp/cqsa_ab_out_{flag}_{N}_{npar}.pt")
    print("RESULT " + json.dumps(r), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=5, metavar=("FLAG", "N", "NPAR", "REPS", "ACC"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--N", type=int, nargs="+", default=[65536])
    ap.add_argument("--npar", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--acc", nargs="+", default=["cpu"])
    a = ap.parse_args()
    if a.worker:
        f, n, p, r, acc = a.worker
        return worker(int(f), int(n), int(p), int(r), a.check, acc)
    import torch
    print(f"GPU {torch.cuda.get_device_name(0)}   pipelined(1) vs shipped(0), itr=1 causal, Q/K/V on host")
    rows = []
    for acc in a.acc:
        for N in a.N:
            print(f"\n=== N={N} acc={acc} ===")
            print(f"{'npar':>5} {'shipped ms':>11} {'pipelined ms':>13} {'speedup':>8}" + ("   correctness" if a.check else ""))
            for p in a.npar:
                res = {}
                for flag in ((0, 1, 0, 1) if a.reps == 1 else (0, 1)):
                    cmd = [sys.executable, __file__, "--worker", str(flag), str(N), str(p), str(a.reps), acc] + (["--check"] if a.check else [])
                    out = subprocess.run(cmd, capture_output=True, text=True)
                    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
                    if not line:
                        print(f"  flag={flag} FAILED:\n{out.stderr[-1500:]}"); continue
                    r = json.loads(line[-1][7:])
                    res.setdefault(flag, []).append(r)
                if 0 in res and 1 in res:
                    m0 = min(x["ms"] for x in res[0]); m1 = min(x["ms"] for x in res[1])
                    extra = ""
                    if a.check:
                        r1 = res[1][0]; r0 = res[0][0]
                        o0 = torch.load(f"/tmp/cqsa_ab_out_0_{N}_{p}.pt"); o1 = torch.load(f"/tmp/cqsa_ab_out_1_{N}_{p}.pt")
                        same = ((o0 - o1).abs().max()).item()
                        extra = (f"   vs SDPA: shipped {r0['rel_err_vs_sdpa']:.2e} pipelined {r1['rel_err_vs_sdpa']:.2e}"
                                 f"  itr2 {r1['rel_err_itr2']:.2e}  |shipped-pipelined|max {same:.1e}"
                                 f"  untouched={r1['untouched']} lse_finite={r1['lse_finite']}")
                    print(f"{p:>5} {m0:11.1f} {m1:13.1f} {m0/m1:8.2f}x{extra}", flush=True)
                    rows.append(dict(N=N, acc=acc, npar=p, shipped_ms=m0, pipelined_ms=m1))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", f"pipeline_ab_{'_'.join(map(str,a.N))}.json")
    json.dump(rows, open(out, "w"), indent=1); print("\nwrote", out)

if __name__ == "__main__":
    main()
