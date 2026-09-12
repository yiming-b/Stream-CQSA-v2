#!/bin/bash
# Four-A100 runner for distributed work. torchrun spawns one rank per GPU.
#   sbatch next/slurm/run4.slurm <script.py> [args...]
#SBATCH --job-name=cqsa_next4
#SBATCH --account=akey
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=02:00:00
#SBATCH --output=next/logs/%x_%j.out
#SBATCH --error=next/logs/%x_%j.err
set -euo pipefail
cd /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev
source next/env_next.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CQSA_CUDA_MODULE=${CQSA_CUDA_MODULE:-cqsa_cuda_next_v11} CQSA_CUDA_MODULE_NONCAUSAL=${CQSA_CUDA_MODULE_NONCAUSAL:-cqsa_cuda_next_v9}
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
srun python -m torch.distributed.run --standalone --nproc_per_node=4 "$@"
