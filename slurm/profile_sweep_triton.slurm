#!/bin/bash
#SBATCH --job-name=cqsa_prof_triton
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --gres=gpu:a100:1 --constraint="sxm&gpu80"
#SBATCH --time=00:59:00
#SBATCH --output=results/profile_sweep/%x_%j.out
set -uo pipefail
cd /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev
source next/env_next.sh
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 CQSA_BACKWARD=cuda
export PYTHONPATH="${PYTHONPATH:-}:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native"   # cqsa_native (after the package: the build stub there shadows it)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -P -u benchmarks/profile_sweep.py run --out results/profile_sweep --reps 5 --warmup 1 --kernels triton
