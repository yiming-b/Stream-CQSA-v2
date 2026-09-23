# Paper re-run with the v2 package (A100 80GB PCIe)

The paper's single-GPU experiments (one causal attention layer, B=1 H=8 D=64, fp16 and bf16, forward and
forward+backward, N = 8K ... 16.8M) re-run with the v2 engine: FlashAttention-2, SDPA (default / flash /
mem-efficient) and Stream-CQSA with the accumulator on the device (acc=GPU) or on the host (acc=CPU).

| file | content |
|---|---|
| `tables_native.md` | tables for the native CUDA kernel set (classic engine, v11 causal / v9 non-causal build) |
| `tables_triton.md` | the same experiment with the Triton kernels (`CQSA_FORWARD=triton CQSA_BACKWARD=triton`), its own table |
| `results_native.jsonl`, `results_triton.jsonl` | raw rows (388 each), frozen copies of the dev tree's `outputs/next/paper{,_triton}/results/results.jsonl` |
| `meta_*_last_job.json` | harness metadata of the last job that appended to each file |
| `make_tables.py` | regenerates both tables (`python results/paper_rerun/make_tables.py`) |

**Hardware.** Every job ran on an NVIDIA A100 80GB PCIe (79.3 GiB; name printed in each job log;
`--constraint=gpu80`), torch 2.10.0+cu130, flash-attn 2.8.3, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
(`docs/LOG.md`'s interim section of this re-run labels the GPU "A100-SXM4-80GB"; the job logs say PCIe.)

**Protocol.** Harness `experiments/paper/run_paper_experiment.py` (dev tree), `--isolate` (every
measurement in a fresh process, inputs regenerated from seed 0), 2 timed repetitions after 1 warm-up,
`--itr-list auto1` (Stream-CQSA's automatic depth with a floor of 1: always decomposed, so the cost of the
decomposition below the OOM boundary is measured, as in the paper), forward accuracy against float64 on
128 sampled rows. Residency differs by design (as in the paper): the baselines run on device-resident
Q/K/V; both Stream-CQSA arms keep Q/K/V in host memory and stream them (acc=GPU: fp32 accumulator on the
device; acc=CPU: on the host too). The ratio column is therefore not an equal-work kernel ratio; the
matched-residency comparison under device budgets is `results/ooc_rivals/`.

**Jobs** (rows accumulate; the last row per (method, dtype, direction, N) wins). Native set: cqsa_p_small
13847746, cqsa_p_accuracy 13847748, cqsa_p_large 13847752 / 13872743, cqsa_p_xl 13847754 / 13872745,
cqsa_p_mid 13847750 / 13872741, cqsa_p_mids 13888445, supplements cqsa_p_mids2 13908223 and cqsa_p_mids3
13915115. Triton set: the matching cqsa_pt_* jobs, supplements 13908224 and 13915470. Some earlier jobs
ended FAILED / CANCELLED part-way (host memory, time limit); the supplements re-ran the missing rows.

**Not-run cells** are marked "not run" with the reason as a footnote in each table, never as OOM:
FlashAttention-2 / SDPA at 16.8M forward and 8.4M+ forward+backward (device need, linear from the largest
completed N, exceeds 79.3 GiB), the mem-efficient SDPA backward from 4.2M on (it took 3839 s at 2.1M
against 76.5 s at 1.05M -- a 50x jump for 2x N -- so 4.2M was not attempted), Stream-CQSA acc=GPU
forward+backward at 16.8M (three fp32 gradient accumulators alone are 96 GiB), and bf16 beyond 8.4M
forward / 4.2M forward+backward.

**Reading the tables.** Below the OOM boundary Stream-CQSA pays for its decomposition: native kernel,
fp16, 1.12-1.73x FlashAttention-2's time forward (falling with N: 1.73x at 262K, 1.19x at 4.2M, 1.12x at
8.4M) and 1.97-2.30x forward+backward (the v11 backward kernel; the ultimate branch's v12b backward is
2.48x faster than v11 on A100-SXM4, not yet in a release). It is the only method that completes 16.8M forward
and 8.4M / 16.8M forward+backward on this GPU (acc=CPU: 32.0 GiB at 16.8M forward, 38.4 GiB at 16.8M
forward+backward). Its forward error against float64 is lower than the monolithic fp16 call's at every N
(e.g. 8.4e-4 vs 1.9e-3 at 4.2M). The Triton set is 1.33-1.72x forward and 1.79-1.95x forward+backward: its forward is slower than the
native kernel's (178.8 vs 153.2 s at 4.2M, acc=GPU) but its backward is faster (forward+backward 732.2 vs
799.2 s), i.e. the v11 CUDA backward was the weak kernel of the release, which is what v12b addresses.
