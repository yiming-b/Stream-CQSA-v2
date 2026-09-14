#!/bin/bash
# The one-call API end to end on one A100-80GB.   sbatch slurm/quickstart.slurm   (from the repo root)
#SBATCH --job-name=cqsa_quickstart
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=80G
#SBATCH --gres=gpu:a100:1 --constraint=gpu80
#SBATCH --time=00:40:00
#SBATCH --output=results/quickstart/%x_%j.out
set -uo pipefail
# cluster-specific environment (torch, triton, the compiled kernels); adapt to yours
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/env.sh
export PYTHONPATH="$PWD:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native:${PYTHONPATH:-}"
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9
export PYTHONPATH="$PYTHONPATH:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v11:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v9"
python -P -m stream_cqsa.doctor
echo
python -P -u benchmarks/quickstart.py
