#!/bin/bash
# Build the release wheel of Stream-CQSA-v2 on della: both CUDA extensions + the native wave
# kernel, torch 2.10 / CUDA 13, sm80 + sm90.   nohup bash next/native/wheel_build.sh &
set -euo pipefail
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/env.sh
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export FLASH_ATTN_CUDA_ARCHS="80;90" TORCH_CUDA_ARCH_LIST="8.0;9.0"
export CQSA_KERNEL_SET=common CQSA_BUILD_NATIVE=1
export MAX_JOBS=${MAX_JOBS:-8} NVCC_THREADS=${NVCC_THREADS:-4}
cd /scratch/gpfs/AKEY/yb2807/Stream-CQSA-v2
rm -rf dist && mkdir -p dist      # keep build/: finished objects are reused
echo "wheel build start $(date)"; t0=$(date +%s)
python setup.py bdist_wheel --dist-dir dist 2>&1 | grep -v "^\s*$" | grep -iv "warning" | tail -30
python -m wheel tags --build "1cu130torch210sm8090" --remove dist/*.whl
echo "wheel build finished in $(( $(date +%s) - t0 )) s  $(date)"; ls -la dist
