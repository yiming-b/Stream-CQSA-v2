#!/bin/bash
#SBATCH --job-name=cqsa_p_small
#SBATCH --account=akey
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/next/paper/slurm_logs/%x_%j.out
#SBATCH --error=outputs/next/paper/slurm_logs/%x_%j.err
set -euo pipefail
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/slurm/paper/common.sh
srun python experiments/paper/run_paper_experiment.py --n 8192,16384,32768,65536,131072,262144,524288,1048576,2097152 --dtypes fp16,bf16 --directions fwd,bwd --methods sdpa,sdpa_flash,sdpa_mem,flash,cqsa_accgpu,cqsa_acccpu --itr-list auto1 --isolate --reps 3 --warmup 1 --acc-rows 256 --out-dir "$RUN_DIR" --tag next_small
