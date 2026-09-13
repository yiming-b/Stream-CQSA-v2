#!/bin/bash
#SBATCH --job-name=native_build
#SBATCH --account=akey
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=next/logs/build_%x_%j.out
#SBATCH --error=next/logs/build_%x_%j.err
set -euo pipefail
cd /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev
export MAX_JOBS=8 NVCC_THREADS=4
bash next/native/build.sh
