#!/bin/bash
#SBATCH --job-name=ooc_bwd4m_v2
#SBATCH --account=akey
#SBATCH --qos=gpu-short
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=120G
#SBATCH --gres=gpu:a100:1 --constraint="gpu80&pcie"
#SBATCH --time=14:00:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v2/results/ooc_rivals/logs/%x_%j.out
# Harness v2 re-run of the N=4.2M forward+backward rows of every method at every budget (10/20/40/80 GiB);
# the harness v1 rows died of host OOM at --mem=96G (MaxRSS 100.66 GB).
# Host memory estimate at 4.2M (1 KiB/token = one fp16 [N,H,D] tensor = 4 GiB):
#   inputs q,k,v,dO                                    16 GiB   (generation peak 24: + dO fp32 8 + fp16 4 - done)
#   rect_ooc(_adaptive): out fp32 8 + O fp16 4 + dQ/dK/dV fp32 24 + their fp16 casts 12 = 48, + pinned staging
#     (plan_tiles at 4.2M: 2.6 / 5.2 / 10.5 / 20.9 GiB fixed, <= 3.4 / 6.8 / 13.7 / 27.3 adaptive at 10/20/40/80 GiB)
#     + CUDA/torch/flash-attn ~2 -> 69 + pinned: 72-99 GiB (1M on della-vis1: 21.7 GiB incl. 1.4 pinned, i.e.
#     18.4 KiB/token + 2 GiB: 4.2M -> 79 + pinned before the harness moved the fp16 output cast after the backward)
#   cqsa_auto (acc=CPU, host-resident): out fp32 8 + O fp16 4 + dO token-major 4 + dQ/dK/dV fp32 24 + fp16 12 +
#     staging ~5 + 2 -> ~75 GiB (1M: 19.3 GiB cold run); monolithic at 80 GiB ~35 GiB; fa2 at 80 GiB ~34 GiB
#   correctness check: inputs 16 + outputs 16 + mmap'd reference 12 (reclaimable file pages) = 44 GiB
#   max ~99 GiB (rect adaptive, 80 GiB budget) -> 120G (x1.21)
# Submit from a frozen snapshot:
#   SNAP=$(bash slurm/ooc_rivals/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP slurm/ooc_rivals/bwd4m_v2.slurm
set -uo pipefail
: "${CQSA_SRC:?submit with --export=ALL,CQSA_SRC=<snapshot dir from slurm/ooc_rivals/snapshot.sh>}"
KERN=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-snapshots/kernels-ooc-rivals-2026-09-23   # frozen v11 / v9 / cqsa_native .so (MD5SUMS)
OUT=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v2/results/ooc_rivals
cd "$CQSA_SRC"
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/env.sh
export PYTHONPATH="$CQSA_SRC:$KERN"
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9
export CQSA_HOST_BUDGET_GIB=120          # the planner's host description = this job's --mem (v1 passed 200)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2; nproc; echo "src $CQSA_SRC kernels $KERN"
python -P -c "import stream_cqsa, stream_cqsa.interface as I, flash_attn; print(stream_cqsa.__file__, I.cqsa_cuda.__file__, flash_attn.__version__)"
python -P -u experiments/ooc_rivals/bench.py run --out "$OUT" --budgets 10,20,40,80 --N 4194304 --directions bwd --reps 3
