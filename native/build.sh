#!/bin/bash
# Build the native wave extension (cqsa_native) in place:  bash native/build.sh
# CQSA_KERNEL_SET: native_dev (hdim64 fp16 fwd+bwd, ~40 min on 8 cores) | common | full
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS:-80}
export CQSA_KERNEL_SET=${CQSA_KERNEL_SET:-native_dev}
export MAX_JOBS=${MAX_JOBS:-8} NVCC_THREADS=${NVCC_THREADS:-4}
cd "$HERE"
python setup.py build_ext --inplace
ls -la "$HERE"/cqsa_native*.so
