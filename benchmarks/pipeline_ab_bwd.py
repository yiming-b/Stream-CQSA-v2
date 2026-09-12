"""
A/B the backward's asynchronous scatter (CQSA_SCATTER_ASYNC=1, the next/ default)
against the shipped synchronous one (=0), host accumulator, Q/K/V/dO/O on host.
Each arm is a subprocess (the switch is read at import). Checks dq/dk/dv of the
two arms against each other (expected: bit-identical for dk/dv, dq within the
kernel's own atomicAdd nondeterminism) and against a dense fp32 SDPA autograd
reference at N <= 64K.

    sbatch next/slurm/run1_test.slurm next/bench/pipeline_ab_bwd.py --N 262144 1048576 --npar 1 2
"""
import argparse, json, os, subprocess, sys, time

DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "pab_bwd_dump")

def worker(flag, N, npar, reps, check):
    os.environ["CQSA_SCATTER_ASYNC"] = str(flag)
    import torch, gc
    from stream_cqsa.stable_stream import stream_cqsa_forward, stream_cqsa_backward, _SCATTER_ASYNC
    assert _SCATTER_ASYNC == bool(flag)
    B, H, D = 1, 8, 64
    torch.manual_seed(0)
    q, k, v, dout = (torch.randn(B, H, N, D, dtype=torch.float16) for _ in range(4))
    fkw = dict(itr=1, causal=True, stream_from_host=True, low_memory=True, allow_escalation=False, max_parallel=1)
    out, info = stream_cqsa_forward(q, k, v, **fkw)
    out_h = out.float().cpu().to(q.dtype); lse = info["lse"].float().cpu()
    bkw = dict(itr=1, causal=True, stream_from_host=True, accumulate_on_gpu=False, allow_escalation=False, max_parallel=npar)
    ts = []
    for _ in range(reps):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        dq, dk, dv = stream_cqsa_backward(q, k, v, dout, out_h, lse, **bkw)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    r = dict(flag=flag, N=N, npar=npar, ms=min(ts))
    if check:
        os.makedirs(DUMP, exist_ok=True)
        torch.save((dq.float().cpu(), dk.float().cpu(), dv.float().cpu()), f"{DUMP}/g_{flag}_{N}_{npar}.pt")
        if N <= 65536:
            qq, kk, vv = (t.cuda().float().requires_grad_(True) for t in (q, k, v))
            ref = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, is_causal=True)
            ref.backward(dout.cuda().float())
            r["rel_err_vs_sdpa"] = {n: ((g.float().cpu() - rg.grad.cpu()).norm() / rg.grad.cpu().norm()).item()
                                    for n, g, rg in (("dq", dq, qq), ("dk", dk, kk), ("dv", dv, vv))}
    print("RESULT " + json.dumps(r), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=4, metavar=("FLAG", "N", "NPAR", "REPS"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--N", type=int, nargs="+", default=[65536])
    ap.add_argument("--npar", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    if a.worker:
        f, n, p, r = a.worker
        return worker(int(f), int(n), int(p), int(r), a.check)
    import torch
    print(f"GPU {torch.cuda.get_device_name(0)}   backward: async scatter(1) vs shipped sync(0), itr=1 causal, acc=CPU, all operands on host")
    rows = []
    for N in a.N:
        print(f"\n=== N={N} ===")
        print(f"{'npar':>5} {'shipped ms':>11} {'async ms':>10} {'speedup':>8}" + ("   correctness" if a.check else ""))
        for p in a.npar:
            res = {}
            for flag in (0, 1):
                cmd = [sys.executable, __file__, "--worker", str(flag), str(N), str(p), str(a.reps)] + (["--check"] if a.check else [])
                out = subprocess.run(cmd, capture_output=True, text=True)
                line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
                if not line:
                    print(f"  flag={flag} FAILED:\n{out.stderr[-1500:]}"); continue
                res[flag] = json.loads(line[-1][7:])
            if 0 in res and 1 in res:
                m0, m1 = res[0]["ms"], res[1]["ms"]; extra = ""
                if a.check:
                    g0 = torch.load(f"{DUMP}/g_0_{N}_{p}.pt"); g1 = torch.load(f"{DUMP}/g_1_{N}_{p}.pt")
                    diffs = [((x - y).norm() / y.norm()).item() for x, y in zip(g1, g0)]
                    extra = f"   async vs shipped rel: dq {diffs[0]:.1e} dk {diffs[1]:.1e} dv {diffs[2]:.1e}"
                    if "rel_err_vs_sdpa" in res[1]:
                        e = res[1]["rel_err_vs_sdpa"]; extra += f"   vs SDPA fp32: dq {e['dq']:.1e} dk {e['dk']:.1e} dv {e['dv']:.1e}"
                print(f"{p:>5} {m0:11.1f} {m1:10.1f} {m0/m1:8.2f}x{extra}", flush=True)
                rows.append(dict(N=N, npar=p, shipped_ms=m0, async_ms=m1))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", f"pipeline_ab_bwd_{'_'.join(map(str, a.N))}.json")
    json.dump(rows, open(out, "w"), indent=1); print("\nwrote", out)

if __name__ == "__main__":
    main()
