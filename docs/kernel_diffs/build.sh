#!/bin/bash
# Build one variant in place. CPU only; ~15 min for the hdim64/128 fp16 set.
#   bash next/kernel/build.sh base
set -euo pipefail
NAME=$1
DEV=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev
source $DEV/env.sh
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export FLASH_ATTN_CUDA_ARCHS=80
export TORCH_CUDA_ARCH_LIST="8.0"
export CQSA_KERNEL_SET=${CQSA_KERNEL_SET:-a100_fp16_hdim64_128}
export MAX_JOBS=${MAX_JOBS:-8}
export NVCC_THREADS=${NVCC_THREADS:-4}
cd $DEV/next/kernel/$NAME
echo "build $NAME: kernel_set=$CQSA_KERNEL_SET MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS  $(date)"
t0=$(date +%s)
python setup.py build_ext --inplace 2>&1 | grep -v "^\s*$" | grep -iv "warning" | tail -20
echo "build $NAME finished in $(( $(date +%s) - t0 )) s  $(date)"
ls -la $DEV/next/kernel/$NAME/cqsa_cuda_next_${NAME}*.so
