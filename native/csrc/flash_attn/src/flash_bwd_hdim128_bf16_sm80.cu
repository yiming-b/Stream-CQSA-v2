// Backward bf16 instantiations for the CQSA build.
// FlashAttention ships bf16 FORWARD kernels but no bf16 backward .cu files;
// run_mha_bwd_hdim128<T, Is_causal> is generic, so these are instantiations
// rather than new kernel code.
#include "namespace_config.h"
#include "flash_bwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_bwd_<cutlass::bfloat16_t, 128, false>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim128<cutlass::bfloat16_t, false>(params, stream);
}

template<>
void run_mha_bwd_<cutlass::bfloat16_t, 128, true>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim128<cutlass::bfloat16_t, true>(params, stream);
}

} // namespace FLASH_NAMESPACE
