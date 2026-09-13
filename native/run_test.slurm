#!/bin/bash
# Exactness tests then the benchmark, on one A100-80GB.   sbatch native/run_test.slurm  (from the repo root)
#SBATCH --job-name=native_test
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=120G
#SBATCH --gres=gpu:a100:1 --constraint=gpu80
#SBATCH --time=00:59:00
#SBATCH --output=results/native/%x_%j.out
set -uo pipefail
export PYTHONPATH="$PWD:$PWD/native:${PYTHONPATH:-}"
export CQSA_CUDA_MODULE=cqsa_cuda CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_nc   # the previous engine's kernels, for the comparison
python -P -u native/test_wave.py
echo "=== bench ==="
python -P -u native/bench_wave.py
