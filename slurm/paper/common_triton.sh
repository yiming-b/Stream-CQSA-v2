# shared preamble for the next/ paper re-runs (sourced by each job)
cd /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev
source next/env_next.sh
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RUN_DIR=outputs/next/paper_triton/results
export CQSA_FORWARD=triton CQSA_BACKWARD=triton
mkdir -p "$RUN_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import stream_cqsa.interface as I; print('kernels:', I.cqsa_cuda.__file__, I.cqsa_cuda_noncausal.__file__)"
