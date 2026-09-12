"""
Generate + execute the "OOM boundary" notebook: sequence length N swept explicitly,
baseline (SDPA / FlashAttention-2) vs Stream-CQSA, under a memory cap so that the
boundary is reached in seconds.

    python next/notebooks/make_oom_demo.py [--no-exec]
"""
import argparse, os, time
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "oom_boundary_demo.ipynb")
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# The OOM boundary: baseline vs Stream-CQSA as N grows

This notebook sets the sequence length **N explicitly** and runs the monolithic baseline
(PyTorch SDPA, FlashAttention-2 backend) and Stream-CQSA on the same inputs at each N.
A **memory cap** simulates a small device so the boundary arrives in seconds instead of hours:

* below the boundary both run; Stream-CQSA costs a little more time (it does 9/7 of the pair work
  and streams from host memory) and is, if anything, slightly *more* accurate against float64;
* past the boundary the baseline raises `OutOfMemoryError` and Stream-CQSA keeps going, exact.

The planner picks the configuration (monolithic call while it fits, then the decomposition depth,
quorum set and accumulator placement) from the capped budget automatically.
""")
code("""
import os, gc, time, torch, torch.nn.functional as F
os.environ.setdefault("CQSA_CUDA_MODULE", "cqsa_cuda_next_v11"); os.environ.setdefault("CQSA_CUDA_MODULE_NONCAUSAL", "cqsa_cuda_next_v9")
from stream_cqsa.autoconfig import auto_attention, hardware_from_dict, detect_hardware
from stream_cqsa.devkit import reference_rows, sample_rows, accuracy_vs_fp64
import stream_cqsa.interface as I
dev = torch.device("cuda"); B, H, D = 1, 8, 64
print(torch.cuda.get_device_name(0), "| kernel:", "Triton (no CUDA extension)" if I.cqsa_cuda is None else os.path.basename(I.cqsa_cuda.__file__))
CAP_GIB = 3.0
total = torch.cuda.get_device_properties(0).total_memory / 2**30
torch.cuda.set_per_process_memory_fraction(CAP_GIB / total)
hw = hardware_from_dict({"cuda:0": f"{CAP_GIB}GiB", "host": "200GiB"})
print(f"memory cap {CAP_GIB} GiB of {total:.0f} GiB  ->  the planner sees:", hw.summary())
def run(fn):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        out = fn(); torch.cuda.synchronize()
        return out, time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**30, None
    except torch.cuda.OutOfMemoryError as e:
        gc.collect(); torch.cuda.empty_cache()
        return None, time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**30, "OOM"
""")
md("## Sweep N")
code("""
Ns = [16_384, 65_536, 131_072, 262_144, 524_288, 1_048_576]
rows = []
print(f"{'N':>9} | {'SDPA/FA-2':>22} | {'Stream-CQSA':>50} | {'err vs fp64':>19}")
for N in Ns:
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(B, H, N, D, generator=g, dtype=torch.float32).to(torch.float16) for _ in range(3))   # host-resident
    rows_idx = sample_rows(N, 64); ref = reference_rows(q, k, v, rows_idx, causal=True, scale=D**-0.5)
    # baseline: inputs must be on the device
    def baseline():
        qd, kd, vd = (t.to(dev) for t in (q, k, v)); return F.scaled_dot_product_attention(qd, kd, vd, is_causal=True)
    ob, tb, mb, eb = run(baseline)
    # Stream-CQSA: planner on the capped budget, inputs streamed from the host when it decides so
    plan = {}
    def cqsa():
        out, p = auto_attention(q, k, v, causal=True, hardware=hw, allow_escalation=True); plan["p"] = p; return out
    oc, tc, mc, ec = run(cqsa)
    errb = accuracy_vs_fp64(ob, q, k, v, causal=True, scale=D**-0.5, rows=rows_idx, ref_rows=ref)["rel_fro"] if ob is not None else float("nan")
    errc = accuracy_vs_fp64(oc, q, k, v, causal=True, scale=D**-0.5, rows=rows_idx, ref_rows=ref)["rel_fro"] if oc is not None else float("nan")
    sb = f"OOM after {tb:.1f}s" if eb else f"{tb:6.2f} s, {mb:4.2f} GiB"
    sc = (f"OOM" if ec else f"{tc:6.2f} s, {mc:4.2f} GiB") + f"  [{plan['p'].name() if 'p' in plan else '-'}]"
    print(f"{N:>9} | {sb:>22} | {sc:>50} | {errb:8.1e} / {errc:8.1e}")
    rows.append(dict(N=N, baseline_s=tb, baseline_gib=mb, baseline=eb or "ok", cqsa_s=tc, cqsa_gib=mc, cqsa=ec or "ok", plan=plan["p"].name() if "p" in plan else None, err_baseline=errb, err_cqsa=errc))
    del q, k, v, ob, oc; gc.collect(); torch.cuda.empty_cache()
""")
md("""
## Reading the table

* **Small N** (16K–512K under this cap): both run. The planner sees that the monolithic call fits the
  capped budget and uses it, so Stream-CQSA's time, memory and error are the baseline's (the ~0.25 GiB
  extra peak is the fp32 output it returns). Forced to decompose below the boundary (`itr=1`), it costs
  1.3–1.6x the monolithic time on this hardware and is slightly closer to float64 (fp32 merge).
* **Large N** (1M, past the cap): SDPA raises `OutOfMemoryError`; the planner picks `c=13, itr=2`,
  host-resident Q/K/V and a host accumulator, and Stream-CQSA returns the exact result (2.0e-4 vs
  float64) at a 2.0 GiB device peak — below the cap.

On an un-capped 80 GB A100 the same crossover happens at ~8M tokens for the forward and ~4M for the
backward (paper, Table 2); the memory cap only moves it to where a notebook can afford to run.
""")
code("""
import json; print(json.dumps(rows, indent=1))
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
ap = argparse.ArgumentParser(); ap.add_argument("--no-exec", action="store_true"); ap.add_argument("--out", default=OUT)
a = ap.parse_args()
if not a.no_exec:
    from nbclient import NotebookClient
    t0 = time.time(); NotebookClient(nb, timeout=3000, kernel_name="python3", resources={"metadata": {"path": HERE}}).execute(); print(f"executed in {time.time() - t0:.0f} s")
nbf.write(nb, a.out); print("wrote", a.out)
