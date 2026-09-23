#!/bin/bash
#SBATCH --job-name=cqsa_layout_ab
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G
#SBATCH --gres=gpu:a100:1 --constraint="sxm&gpu80"
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v2/results/ultimate/layout_ab_%j.out
# Layout A/B (align=1 vs align=128) of the v11 CQS forward, backward dead-tile split, small-L split.
# Host tensors here are < 2 GiB; 24G covers torch itself (~6G) plus the task tables.
set -uo pipefail
cd /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev
source next/env_next.sh
export PYTHONPATH="${PYTHONPATH:-}:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native"   # cqsa_native (after the package: the build stub there shadows it)
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 CQSA_BACKWARD=cuda CQSA_CPU_THREADS=8
which python; hostname
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.total --format=csv,noheader
python -P -c "import stream_cqsa, torch; print(stream_cqsa.__file__, torch.__version__)"
python -P -u /scratch/gpfs/AKEY/yb2807/Stream-CQSA-v2/benchmarks/layout_ab.py
