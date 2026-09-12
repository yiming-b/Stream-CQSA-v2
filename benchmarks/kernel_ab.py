"""
Three-way kernel A/B at identical subproblem shapes, in one job:

  shipped   cqsa_cuda            the .so that produced the paper's numbers
  base      cqsa_cuda_next_base  the same source rebuilt here (toolchain check)
  v1        cqsa_cuda_next_v1    forward register fix

plus upstream FA-2 as the floor. Each module is a fresh subprocess (the
extension is chosen at import), interleaved per shape so drift lands on all
three. Also checks v1 against shipped bit-for-bit: the change moves where the
tile verdict is computed, not what it is, so outputs must match exactly.

    sbatch next/slurm/run1_test.slurm next/bench/kernel_ab.py
"""
import argparse, json, os, subprocess, sys

MODS = [("shipped", "cqsa_cuda"), ("base", "cqsa_cuda_next_base"), ("v1", "cqsa_cuda_next_v1"),
        ("v2", "cqsa_cuda_next_v2"), ("v3", "cqsa_cuda_next_v3"), ("v4", "cqsa_cuda_next_v4"), ("v5", "cqsa_cuda_next_v5"), ("v6", "cqsa_cuda_next_v6"), ("v7", "cqsa_cuda_next_v7"), ("v8", "cqsa_cuda_next_v8"), ("v9", "cqsa_cuda_next_v9"), ("v10", "cqsa_cuda_next_v10"), ("v11", "cqsa_cuda_next_v11")]
DUMP = os.environ.get("KAB_DUMP_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "kab_dump"))

def worker(mod, N, causal, iters, dump):
    os.environ["CQSA_CUDA_MODULE"] = mod
    import torch
    from stream_cqsa.interface import flash_attn_func_cqs_group_bits, flash_attn_func, CQSA_CUDA_MODULE, cqsa_cuda, cqs_block_summaries, CQS_BLK_SIZE
    from stream_cqsa.reference import group_bits_for_path
    assert CQSA_CUDA_MODULE == mod and cqsa_cuda is not None, f"{mod} not loaded"
    H, D, dtype = 8, 64, torch.float16
    scale = D ** -0.5
    ids_np, bits_np = group_bits_for_path(N, (0,), sorted_gather=True)
    L = int(ids_np.shape[0])
    bits = torch.as_tensor(bits_np, device="cuda", dtype=torch.int64)
    zero = torch.zeros_like(bits)
    # Block summaries are shape-only and the engine builds them once per task
    # (build_tasks_cached). Passing them in keeps the per-call host work
    # (bits.cpu() sync + numpy reduce + h2d) out of the timed loop; without
    # this the earlier kernel_baseline.py numbers for zero/mask were inflated.
    so = {k: v.cuda() for k, v in zip(("cqs_blk_or", "cqs_blk_and"), cqs_block_summaries(bits, CQS_BLK_SIZE))}
    sz = {k: v.cuda() for k, v in zip(("cqs_blk_or", "cqs_blk_and"), cqs_block_summaries(zero, CQS_BLK_SIZE))}
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, L, H, D, device="cuda", dtype=dtype) for _ in range(3))
    def t(fn, it=iters, wu=3):
        for _ in range(wu): fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(it): fn()
        e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1) / it
    r = dict(mod=mod, N=N, L=L, causal=causal)
    r["plain"] = t(lambda: flash_attn_func(q, k, v, softmax_scale=scale, causal=causal))
    r["zero"] = t(lambda: flash_attn_func_cqs_group_bits(q, k, v, zero, softmax_scale=scale, causal=causal, **sz))
    r["mask"] = t(lambda: flash_attn_func_cqs_group_bits(q, k, v, bits, softmax_scale=scale, causal=causal, **so))
    if dump:
        out = flash_attn_func_cqs_group_bits(q, k, v, bits, softmax_scale=scale, causal=causal, fp32_out=True, **so)
        os.makedirs(DUMP, exist_ok=True)
        torch.save(out.cpu(), f"{DUMP}/kab_{mod}_{N}_{int(causal)}.pt")
    print("RESULT " + json.dumps(r), flush=True)

def fa2(N, causal, iters):
    import torch
    from flash_attn import flash_attn_func
    from stream_cqsa.reference import group_bits_for_path
    H, D, dtype = 8, 64, torch.float16
    L = int(group_bits_for_path(N, (0,), sorted_gather=True)[0].shape[0])
    q, k, v = (torch.randn(1, L, H, D, device="cuda", dtype=dtype) for _ in range(3))
    for _ in range(3): flash_attn_func(q, k, v, softmax_scale=D**-0.5, causal=causal)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters): flash_attn_func(q, k, v, softmax_scale=D**-0.5, causal=causal)
    e1.record(); torch.cuda.synchronize()
    fa = e0.elapsed_time(e1) / iters
    # torch's own FlashAttention-2 build (stock kernels) via SDPA, same layout [B,H,L,D].
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    qb, kb, vb = (t.transpose(1, 2) for t in (q, k, v))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for _ in range(3): F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal, scale=D**-0.5)
        torch.cuda.synchronize(); e0.record()
        for _ in range(iters): F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal, scale=D**-0.5)
        e1.record(); torch.cuda.synchronize()
    print("RESULT " + json.dumps(dict(mod="fa2", N=N, causal=causal, plain=fa, sdpa_flash=e0.elapsed_time(e1)/iters)), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=4, metavar=("MOD", "N", "CAUSAL", "ITERS"))
    ap.add_argument("--fa2", nargs=3, metavar=("N", "CAUSAL", "ITERS"))
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--N", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--mods", nargs="+", default=["shipped", "base", "v1"])
    ap.add_argument("--out", default="kernel_ab.json")
    a = ap.parse_args()
    if a.worker:
        m, n, c, it = a.worker; return worker(m, int(n), c == "1", int(it), a.dump)
    if a.fa2:
        n, c, it = a.fa2; return fa2(int(n), c == "1", int(it))
    import torch
    modmap = dict(MODS)
    print(f"GPU {torch.cuda.get_device_name(0)}  H=8 D=64 fp16, L=3N/7, ms/call")
    rows = []
    for causal in (True, False):
        print(f"\n=== causal={causal} ===")
        print(f"{'N':>7} {'fa2':>7} {'sdpaF':>7} | " + " ".join(f"{m+'_plain':>9} {m+'_zero':>9} {m+'_mask':>9}" for m in a.mods)
              + " | " + " ".join(f"{m+'/ship':>8} {m+'|diff|':>9}" for m in a.mods if m != "shipped"))
        for N in a.N:
            res = {}
            out = subprocess.run([sys.executable, __file__, "--fa2", str(N), "1" if causal else "0", str(a.iters)], capture_output=True, text=True)
            l = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
            fa_row = json.loads(l[-1][7:]) if l else {}
            fa = fa_row.get("plain", float("nan")); sdpa = fa_row.get("sdpa_flash", float("nan"))
            for rep in range(2):
                for m in a.mods:
                    cmd = [sys.executable, __file__, "--worker", modmap[m], str(N), "1" if causal else "0", str(a.iters)] + (["--dump"] if rep == 0 else [])
                    out = subprocess.run(cmd, capture_output=True, text=True)
                    l = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
                    if not l:
                        print(f"  {m} FAILED: {out.stderr[-600:]}"); continue
                    r = json.loads(l[-1][7:])
                    prev = res.get(m)
                    res[m] = r if prev is None else {**r, **{k: min(prev[k], r[k]) for k in ("plain", "zero", "mask")}}
            keys = ("plain", "zero", "mask")
            cells = " ".join(" ".join(f"{res[m][k]:9.2f}" for k in keys) if m in res else " ".join(f"{'--':>9}" for _ in keys) for m in a.mods)
            comp = []
            for m in a.mods:
                if m == "shipped": continue
                ratio = diff = float("nan")
                if "shipped" in res and m in res:
                    ratio = res[m]["mask"] / res["shipped"]["mask"]
                    a0 = torch.load(f"{DUMP}/kab_cqsa_cuda_{N}_{int(causal)}.pt"); a1 = torch.load(f"{DUMP}/kab_{modmap[m]}_{N}_{int(causal)}.pt")
                    diff = (a0 - a1).abs().max().item()
                    res[m]["maxdiff_vs_shipped"] = diff
                comp.append(f"{ratio:8.3f} {diff:9.2e}")
            print(f"{N:>7} {fa:7.2f} {sdpa:7.2f} | {cells} | {' '.join(comp)}", flush=True)
            rows.append(dict(N=N, causal=causal, fa2=fa, sdpa_flash=sdpa, **{f"{m}_{k}": res[m][k] for m in res for k in res[m] if k in keys + ("maxdiff_vs_shipped",)}))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", a.out)
    json.dump(rows, open(out, "w"), indent=1); print("\nwrote", out)

if __name__ == "__main__":
    main()
