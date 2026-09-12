#!/bin/bash
# Create next/kernel/<name>/ : a buildable copy of the package's CUDA sources
# whose extension is named cqsa_cuda_next_<name>, so several kernel builds can
# coexist on PYTHONPATH and be selected per process with CQSA_CUDA_MODULE.
#   bash next/kernel/mk_variant.sh base      # unmodified source
#   bash next/kernel/mk_variant.sh v1        # then edit next/kernel/v1/csrc/...
set -euo pipefail
NAME=$1
DEV=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev
SRC=$DEV/packages/stream-cqsa
DST=$DEV/next/kernel/$NAME
mkdir -p "$DST"
rsync -a --exclude cutlass "$SRC/csrc/" "$DST/csrc/"
# cutlass is big and read-only: share it
[ -e "$DST/csrc/cutlass" ] || ln -s "$SRC/csrc/cutlass" "$DST/csrc/cutlass"
cp "$SRC/setup.py" "$DST/setup.py"
sed -i "s/name=\"cqsa_cuda\",/name=\"cqsa_cuda_next_${NAME}\",/" "$DST/setup.py"
grep -q "cqsa_cuda_next_${NAME}" "$DST/setup.py" || { echo "rename failed"; exit 1; }
echo "variant '$NAME' at $DST (extension cqsa_cuda_next_${NAME})"
