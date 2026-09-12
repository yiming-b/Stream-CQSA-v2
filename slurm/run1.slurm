#!/bin/bash
# Generic one-A100-80GB runner for next/ scripts.
#   sbatch next/slurm/run1.slurm <script.py> [args...]
#SBATCH --job-name=cqsa_next
#SBATCH --account=akey
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=next/logs/%x_%j.out
#SBATCH --error=next/logs/%x_%j.err
set -euo pipefail
cd /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev
source next/env_next.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "co-tenants: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
srun python -u "$@"
