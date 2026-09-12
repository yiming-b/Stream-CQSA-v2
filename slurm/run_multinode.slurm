#!/bin/bash
# Two-node, 8-GPU runner: torch.distributed.run with a c10d rendezvous on node 0.
#   sbatch next/slurm/run_multinode.slurm next/distributed/test_dist.py 2097152 --itr 2 --acc gpu
#SBATCH --job-name=cqsa_dist8
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=00:59:00
#SBATCH --output=next/logs/%x_%j.out
#SBATCH --error=next/logs/%x_%j.err
set -euo pipefail
cd /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev
source next/env_next.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')  master: $MASTER"
srun python -m torch.distributed.run --nnodes=2 --nproc_per_node=4 \
     --rdzv_backend=c10d --rdzv_endpoint="$MASTER:29517" --rdzv_id="$SLURM_JOB_ID" "$@"
