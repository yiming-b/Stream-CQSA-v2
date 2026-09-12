#!/bin/bash
#SBATCH --job-name=cqsa_p_xl
#SBATCH --account=akey
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=23:00:00
#SBATCH --output=outputs/next/paper/slurm_logs/%x_%j.out
#SBATCH --error=outputs/next/paper/slurm_logs/%x_%j.err
set -euo pipefail
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/slurm/paper/common.sh
srun python experiments/paper/run_paper_experiment.py --n 16777216 --dtypes fp16 --directions bwd --methods cqsa_acccpu --itr-list auto --reps 1 --warmup 0 --acc-rows 64 --dense-acc-max-n 0 --cooldown-s 5 --out-dir "$RUN_DIR" --tag next_xl
