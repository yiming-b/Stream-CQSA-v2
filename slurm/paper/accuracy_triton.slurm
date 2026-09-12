#!/bin/bash
#SBATCH --job-name=cqsa_pt_accuracy
#SBATCH --account=akey
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=outputs/next/paper/slurm_logs/%x_%j.out
#SBATCH --error=outputs/next/paper/slurm_logs/%x_%j.err
set -euo pipefail
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/slurm/paper/common_triton.sh
srun python experiments/paper/run_paper_experiment.py --preset accuracy --seeds 0,1,2,3,4 --methods sdpa,sdpa_flash,sdpa_mem,flash,cqsa_accgpu,cqsa_acccpu --out-dir "$RUN_DIR" --tag next_triton_accuracy
