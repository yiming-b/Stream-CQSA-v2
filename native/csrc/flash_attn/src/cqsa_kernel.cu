#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <torch/python.h>

#include <cstdint>

#include "namespace_config.h"

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_F32(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

namespace FLASH_NAMESPACE {

namespace {

__device__ __forceinline__ int map_local_to_global_idx(
    int l_idx,
    int num_segments,
    int g_start0,
    int l_start0,
    int seg_len0,
    int g_start1,
    int l_start1,
    int seg_len1,
    int g_start2,
    int l_start2,
    int seg_len2) {
    if (l_idx >= l_start0 && l_idx < l_start0 + seg_len0) {
        return g_start0 + (l_idx - l_start0);
    }
    if (num_segments > 1 && l_idx >= l_start1 && l_idx < l_start1 + seg_len1) {
        return g_start1 + (l_idx - l_start1);
    }
    if (num_segments > 2 && l_idx >= l_start2 && l_idx < l_start2 + seg_len2) {
        return g_start2 + (l_idx - l_start2);
    }
    return -1;
}

template <typename scalar_t>
__global__ void cqsa_accum_out_lse_stage_kernel(
    float* __restrict__ global_num,
    const scalar_t* __restrict__ out_sub,
    float* __restrict__ global_den,
    const float* __restrict__ lse_sub,
    int b,
    int n,
    int h,
    int d,
    int l_sub,
    int num_segments,
    int g_start0,
    int l_start0,
    int seg_len0,
    int g_start1,
    int l_start1,
    int seg_len1,
    int g_start2,
    int l_start2,
    int seg_len2) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(b) * l_sub * h * d;
    if (idx >= total) {
        return;
    }

    int64_t t = idx;
    const int d_idx = static_cast<int>(t % d);
    t /= d;
    const int h_idx = static_cast<int>(t % h);
    t /= h;
    const int l_idx = static_cast<int>(t % l_sub);
    const int b_idx = static_cast<int>(t / l_sub);
    const int g_idx = map_local_to_global_idx(
        l_idx,
        num_segments,
        g_start0,
        l_start0,
        seg_len0,
        g_start1,
        l_start1,
        seg_len1,
        g_start2,
        l_start2,
        seg_len2);
    if (g_idx < 0) {
        return;
    }

    const int64_t out_offset =
        ((((static_cast<int64_t>(b_idx) * l_sub) + l_idx) * h) + h_idx) * d + d_idx;
    const int64_t global_num_offset =
        ((((static_cast<int64_t>(b_idx) * n) + g_idx) * h) + h_idx) * d + d_idx;
    const int64_t lse_offset =
        ((static_cast<int64_t>(b_idx) * h) + h_idx) * l_sub + l_idx;
    const float den = __expf(lse_sub[lse_offset]);
    if (!(den > 0.0f) || !isfinite(den)) {
        return;
    }
    const float out_val = static_cast<float>(out_sub[out_offset]);
    if (!isfinite(out_val)) {
        return;
    }

    global_num[global_num_offset] += out_val * den;
    if (d_idx == 0) {
        const int64_t global_den_offset =
            ((static_cast<int64_t>(b_idx) * h) + h_idx) * n + g_idx;
        global_den[global_den_offset] += den;
    }
}

template <typename scalar_t>
__global__ void cqsa_accum_out_lse_index_kernel(
    float* __restrict__ global_num,
    const scalar_t* __restrict__ out_sub,
    float* __restrict__ global_den,
    const float* __restrict__ lse_sub,
    const int64_t* __restrict__ token_ids,
    int b,
    int n,
    int h,
    int d,
    int l_sub) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(b) * l_sub * h * d;
    if (idx >= total) {
        return;
    }

    int64_t t = idx;
    const int d_idx = static_cast<int>(t % d);
    t /= d;
    const int h_idx = static_cast<int>(t % h);
    t /= h;
    const int l_idx = static_cast<int>(t % l_sub);
    const int b_idx = static_cast<int>(t / l_sub);

    const int64_t g_idx_64 = token_ids[l_idx];
    if (g_idx_64 < 0 || g_idx_64 >= n) {
        return;
    }
    const int g_idx = static_cast<int>(g_idx_64);

    const int64_t out_offset =
        ((((static_cast<int64_t>(b_idx) * l_sub) + l_idx) * h) + h_idx) * d + d_idx;
    const int64_t global_num_offset =
        ((((static_cast<int64_t>(b_idx) * n) + g_idx) * h) + h_idx) * d + d_idx;
    const int64_t lse_offset =
        ((static_cast<int64_t>(b_idx) * h) + h_idx) * l_sub + l_idx;
    const float den = __expf(lse_sub[lse_offset]);
    if (!(den > 0.0f) || !isfinite(den)) {
        return;
    }
    const float out_val = static_cast<float>(out_sub[out_offset]);
    if (!isfinite(out_val)) {
        return;
    }

    atomicAdd(&global_num[global_num_offset], out_val * den);
    if (d_idx == 0) {
        const int64_t global_den_offset =
            ((static_cast<int64_t>(b_idx) * h) + h_idx) * n + g_idx;
        atomicAdd(&global_den[global_den_offset], den);
    }
}

inline void check_cuda_launch(const char* name) {
    const auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, name, " launch failed: ", cudaGetErrorString(err));
}

}  // namespace

void cqsa_accum_out_lse_stage_cuda(
    at::Tensor& global_num,
    const at::Tensor& out_sub,
    at::Tensor& global_den,
    const at::Tensor& lse_sub,
    const int64_t* global_starts,
    const int64_t* local_starts,
    const int64_t* segment_lens,
    int64_t num_segments) {
    if (num_segments <= 0) {
        return;
    }

    CHECK_CUDA(global_num);
    CHECK_CUDA(out_sub);
    CHECK_CUDA(global_den);
    CHECK_CUDA(lse_sub);
    CHECK_F32(global_num);
    CHECK_F32(global_den);
    CHECK_F32(lse_sub);
    CHECK_CONTIGUOUS(global_num);
    CHECK_CONTIGUOUS(out_sub);
    CHECK_CONTIGUOUS(global_den);
    CHECK_CONTIGUOUS(lse_sub);

    TORCH_CHECK(global_num.dim() == 4, "global_num must be [B, N, H, D]");
    TORCH_CHECK(out_sub.dim() == 4, "out_sub must be [B, L, H, D]");
    TORCH_CHECK(global_den.dim() == 3, "global_den must be [B, H, N]");
    TORCH_CHECK(lse_sub.dim() == 3, "lse_sub must be [B, H, L]");
    TORCH_CHECK(global_num.size(0) == out_sub.size(0), "batch mismatch for out");
    TORCH_CHECK(global_den.size(0) == lse_sub.size(0), "batch mismatch for lse");
    TORCH_CHECK(global_num.size(0) == global_den.size(0), "batch mismatch between num/den");
    TORCH_CHECK(global_num.size(2) == out_sub.size(2), "num_heads mismatch for out");
    TORCH_CHECK(global_den.size(1) == lse_sub.size(1), "num_heads mismatch for lse");
    TORCH_CHECK(global_num.size(2) == global_den.size(1), "num_heads mismatch between num/den");
    TORCH_CHECK(global_num.size(1) == global_den.size(2), "seq_len mismatch between num/den");
    TORCH_CHECK(out_sub.size(1) == lse_sub.size(2), "local seq_len mismatch between out_sub/lse_sub");
    TORCH_CHECK(
        out_sub.scalar_type() == at::kHalf || out_sub.scalar_type() == at::kBFloat16,
        "out_sub must be fp16 or bf16");
    TORCH_CHECK(num_segments <= 3, "num_segments must be <= 3");
    TORCH_CHECK(global_starts != nullptr, "global_starts must not be null");
    TORCH_CHECK(local_starts != nullptr, "local_starts must not be null");
    TORCH_CHECK(segment_lens != nullptr, "segment_lens must not be null");

    const int b = static_cast<int>(global_num.size(0));
    const int n = static_cast<int>(global_num.size(1));
    const int h = static_cast<int>(global_num.size(2));
    const int d = static_cast<int>(global_num.size(3));
    const int l_sub = static_cast<int>(out_sub.size(1));

    int g_start0 = 0;
    int l_start0 = 0;
    int seg_len0 = 0;
    int g_start1 = 0;
    int l_start1 = 0;
    int seg_len1 = 0;
    int g_start2 = 0;
    int l_start2 = 0;
    int seg_len2 = 0;

    for (int64_t i = 0; i < num_segments; ++i) {
        TORCH_CHECK(segment_lens[i] > 0, "segment length must be positive");
        TORCH_CHECK(global_starts[i] >= 0 && local_starts[i] >= 0, "segment start must be non-negative");
        TORCH_CHECK(global_starts[i] + segment_lens[i] <= global_num.size(1), "global segment out of range");
        TORCH_CHECK(local_starts[i] + segment_lens[i] <= out_sub.size(1), "local segment out of range");
        if (i == 0) {
            g_start0 = static_cast<int>(global_starts[i]);
            l_start0 = static_cast<int>(local_starts[i]);
            seg_len0 = static_cast<int>(segment_lens[i]);
        } else if (i == 1) {
            g_start1 = static_cast<int>(global_starts[i]);
            l_start1 = static_cast<int>(local_starts[i]);
            seg_len1 = static_cast<int>(segment_lens[i]);
        } else {
            g_start2 = static_cast<int>(global_starts[i]);
            l_start2 = static_cast<int>(local_starts[i]);
            seg_len2 = static_cast<int>(segment_lens[i]);
        }
    }

    const int n_segments = static_cast<int>(num_segments);

    constexpr int threads = 256;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int64_t total = static_cast<int64_t>(b) * l_sub * h * d;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    if (out_sub.scalar_type() == at::kHalf) {
        cqsa_accum_out_lse_stage_kernel<at::Half><<<blocks, threads, 0, stream>>>(
            global_num.data_ptr<float>(),
            out_sub.data_ptr<at::Half>(),
            global_den.data_ptr<float>(),
            lse_sub.data_ptr<float>(),
            b,
            n,
            h,
            d,
            l_sub,
            n_segments,
            g_start0,
            l_start0,
            seg_len0,
            g_start1,
            l_start1,
            seg_len1,
            g_start2,
            l_start2,
            seg_len2);
        check_cuda_launch("cqsa_accum_out_lse_stage_kernel_fp16");
    } else {
        cqsa_accum_out_lse_stage_kernel<at::BFloat16><<<blocks, threads, 0, stream>>>(
            global_num.data_ptr<float>(),
            out_sub.data_ptr<at::BFloat16>(),
            global_den.data_ptr<float>(),
            lse_sub.data_ptr<float>(),
            b,
            n,
            h,
            d,
            l_sub,
            n_segments,
            g_start0,
            l_start0,
            seg_len0,
            g_start1,
            l_start1,
            seg_len1,
            g_start2,
            l_start2,
            seg_len2);
        check_cuda_launch("cqsa_accum_out_lse_stage_kernel_bf16");
    }
}

void cqsa_accum_out_lse_index_cuda(
    at::Tensor& global_num,
    const at::Tensor& out_sub,
    at::Tensor& global_den,
    const at::Tensor& lse_sub,
    const at::Tensor& token_ids) {
    CHECK_CUDA(global_num);
    CHECK_CUDA(out_sub);
    CHECK_CUDA(global_den);
    CHECK_CUDA(lse_sub);
    CHECK_CUDA(token_ids);
    CHECK_F32(global_num);
    CHECK_F32(global_den);
    CHECK_F32(lse_sub);
    CHECK_CONTIGUOUS(global_num);
    CHECK_CONTIGUOUS(out_sub);
    CHECK_CONTIGUOUS(global_den);
    CHECK_CONTIGUOUS(lse_sub);
    CHECK_CONTIGUOUS(token_ids);

    TORCH_CHECK(global_num.dim() == 4, "global_num must be [B, N, H, D]");
    TORCH_CHECK(out_sub.dim() == 4, "out_sub must be [B, L, H, D]");
    TORCH_CHECK(global_den.dim() == 3, "global_den must be [B, H, N]");
    TORCH_CHECK(lse_sub.dim() == 3, "lse_sub must be [B, H, L]");
    TORCH_CHECK(token_ids.dim() == 1, "token_ids must be 1D");
    TORCH_CHECK(token_ids.scalar_type() == at::kLong, "token_ids must be int64");
    TORCH_CHECK(token_ids.numel() == out_sub.size(1), "token_ids length must match out_sub local length");
    TORCH_CHECK(global_num.size(0) == out_sub.size(0), "batch mismatch for out");
    TORCH_CHECK(global_den.size(0) == lse_sub.size(0), "batch mismatch for lse");
    TORCH_CHECK(global_num.size(0) == global_den.size(0), "batch mismatch between num/den");
    TORCH_CHECK(global_num.size(2) == out_sub.size(2), "num_heads mismatch for out");
    TORCH_CHECK(global_den.size(1) == lse_sub.size(1), "num_heads mismatch for lse");
    TORCH_CHECK(global_num.size(2) == global_den.size(1), "num_heads mismatch between num/den");
    TORCH_CHECK(global_num.size(1) == global_den.size(2), "seq_len mismatch between num/den");
    TORCH_CHECK(out_sub.size(1) == lse_sub.size(2), "local seq_len mismatch between out_sub/lse_sub");
    TORCH_CHECK(
        out_sub.scalar_type() == at::kHalf || out_sub.scalar_type() == at::kBFloat16,
        "out_sub must be fp16 or bf16");

    const int b = static_cast<int>(global_num.size(0));
    const int n = static_cast<int>(global_num.size(1));
    const int h = static_cast<int>(global_num.size(2));
    const int d = static_cast<int>(global_num.size(3));
    const int l_sub = static_cast<int>(out_sub.size(1));

    constexpr int threads = 256;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int64_t total = static_cast<int64_t>(b) * l_sub * h * d;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    if (out_sub.scalar_type() == at::kHalf) {
        cqsa_accum_out_lse_index_kernel<at::Half><<<blocks, threads, 0, stream>>>(
            global_num.data_ptr<float>(),
            out_sub.data_ptr<at::Half>(),
            global_den.data_ptr<float>(),
            lse_sub.data_ptr<float>(),
            token_ids.data_ptr<int64_t>(),
            b,
            n,
            h,
            d,
            l_sub);
        check_cuda_launch("cqsa_accum_out_lse_index_kernel_fp16");
    } else {
        cqsa_accum_out_lse_index_kernel<at::BFloat16><<<blocks, threads, 0, stream>>>(
            global_num.data_ptr<float>(),
            out_sub.data_ptr<at::BFloat16>(),
            global_den.data_ptr<float>(),
            lse_sub.data_ptr<float>(),
            token_ids.data_ptr<int64_t>(),
            b,
            n,
            h,
            d,
            l_sub);
        check_cuda_launch("cqsa_accum_out_lse_index_kernel_bf16");
    }
}

}  // namespace FLASH_NAMESPACE
