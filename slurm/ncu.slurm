#!/bin/bash
# ncu profile of one forward kernel launch per case (perf counters are blocked on the interactive node).
#   sbatch next/slurm/ncu.slurm <module>
#SBATCH --job-name=cqsa_ncu
#SBATCH --account=akey
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=next/logs/%x_%j.out
#SBATCH --error=next/logs/%x_%j.err
set -uo pipefail
cd /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev
source next/env_next.sh
MOD=${1:-cqsa_cuda_next_v3}
nvidia-smi --query-gpu=name --format=csv,noheader
CASES=(${CASES:-fa2:0 plain:0 zero:0 fa2:1 plain:1 zero:1 real:1})
for case in "${CASES[@]}"; do
  pat=${case%%:*}; causal=${case##*:}
  echo "=============== $MOD pattern=$pat causal=$causal"
  /usr/local/cuda-13.0/bin/ncu --nvtx --nvtx-include "measure/" -k regex:flash_fwd_kernel \
     --section SpeedOfLight --section Occupancy --section LaunchStats --section WarpStateStats --section SchedulerStats \
     --section MemoryWorkloadAnalysis --section InstructionStats \
     python next/bench/ncu_one.py $MOD $pat $causal 2>&1 | grep -v "^==PROF=="
done
