#!/bin/bash
#SBATCH --job-name=cqsa_profile
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --gres=gpu:a100:1 --constraint=gpu80
#SBATCH --time=00:59:00
#SBATCH --output=results/profile_sweep/%x_%j.out
set -uo pipefail
# run from the repository root: sbatch slurm/profile_sweep.slurm (edit ROOT for another checkout)
ROOT=${ROOT:-/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v2}
cd "$ROOT"
# cluster-specific environment (torch, triton, the compiled kernels); adapt to yours,
# or `pip install` the release wheels and drop the PYTHONPATH / CQSA_CUDA_MODULE lines
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/env.sh
export PYTHONPATH="$ROOT:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v11:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v9:${PYTHONPATH:-}"
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 CQSA_BACKWARD=cuda
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u benchmarks/profile_sweep.py run --out results/profile_sweep --reps 5 --warmup 1
python -u benchmarks/profile_sweep.py fit --out results/profile_sweep
