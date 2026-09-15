#!/bin/bash
#SBATCH --job-name=ooc_b10_fwd
#SBATCH --account=akey
#SBATCH --qos=gpu-short
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --gres=gpu:a100:1 --constraint="gpu80&pcie"
#SBATCH --time=05:00:00
#SBATCH --output=results/ooc_rivals/logs/%x_%j.out
set -uo pipefail
cd /scratch/gpfs/AKEY/yb2807/Stream-CQSA-v2
# environment: torch 2.10 / CUDA 13 stack, the compiled kernels of the dev tree (classic causal / non-causal builds, wave)
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/env.sh
export PYTHONPATH="/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v2:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v11:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v9:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native"
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2; nproc
python -u experiments/ooc_rivals/bench.py run --out results/ooc_rivals --budgets 10 --directions fwd --reps 3
