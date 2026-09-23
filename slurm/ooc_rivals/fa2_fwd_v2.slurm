#!/bin/bash
#SBATCH --job-name=ooc_fa2_fwd_v2
#SBATCH --account=akey
#SBATCH --qos=gpu-short
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G
#SBATCH --gres=gpu:a100:1 --constraint="gpu80&pcie"
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v2/results/ooc_rivals/logs/%x_%j.out
# Harness v2 re-run of every FA-2 forward row (budgets 10/20/40/80 GiB, N 256K..4M): one timed
# flash_attn_func call, token-major inputs prepared untimed, kernel-only reference measured after.
# Host memory: measured 4.89 GiB peak at 1M (della-vis1; input generation: q,k token-major + v fp32 +
# fp16 = 5 KiB/token); linear -> 19.6 GiB at 4.2M; x1.2 -> 24G.
# Submit from a frozen snapshot:
#   SNAP=$(bash slurm/ooc_rivals/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP slurm/ooc_rivals/fa2_fwd_v2.slurm
set -uo pipefail
: "${CQSA_SRC:?submit with --export=ALL,CQSA_SRC=<snapshot dir from slurm/ooc_rivals/snapshot.sh>}"
KERN=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-snapshots/kernels-ooc-rivals-2026-09-23   # frozen v11 / v9 / cqsa_native .so (MD5SUMS)
OUT=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v2/results/ooc_rivals
cd "$CQSA_SRC"
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/env.sh
export PYTHONPATH="$CQSA_SRC:$KERN"
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2; nproc; echo "src $CQSA_SRC kernels $KERN"
python -P -c "import stream_cqsa, stream_cqsa.interface as I, flash_attn; print(stream_cqsa.__file__, I.cqsa_cuda.__file__, flash_attn.__version__)"
python -P -u experiments/ooc_rivals/bench.py run --out "$OUT" --budgets 10,20,40,80 --directions fwd --methods fa2 --reps 3
