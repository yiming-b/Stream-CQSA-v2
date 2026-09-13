// next/native: batched, DETERMINISTIC recomposition kernels for wave mode.
//
// A wave is a set of subproblems whose (out_i, lse_i) -- or (dq_i, dk_i, dv_i)
// -- come out of one launch in the packed varlen layout. Every global token
// occurs in several of them. Instead of one merge per subproblem, the packed
// rows are sorted by global token once on the host side (stable, so rows of
// the same token keep subproblem order), and one kernel folds all of them
// into the accumulator in that fixed order. No atomics, so the result does
// not depend on scheduling, and the merge arithmetic is exactly the
// max-shifted update of StableAccumulator / FlashAttention's combine.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/python.h>
#include <cstdint>
#include <cmath>
#include "namespace_config.h"

namespace FLASH_NAMESPACE {
namespace {

#define WCHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define WCHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

// acc [Ntot, H, D] f32 (unnormalised, relative to exp(acc_m)), acc_l/acc_m [Ntot, H] f32.
// out_pack [T, H, D] f32, lse_pack [H, T] f32 (varlen layout).
// order [T] int64: packed rows sorted by global token; seg [U+1] int32: CSR over
// the U distinct tokens of this wave; uniq [U] int64: the token of each segment.
__global__ void wave_merge_kernel(
    float* __restrict__ acc, float* __restrict__ acc_l, float* __restrict__ acc_m,
    const float* __restrict__ out_pack, const float* __restrict__ lse_pack,
    const int64_t* __restrict__ order, const int* __restrict__ seg, const int64_t* __restrict__ uniq,
    const int U, const int H, const int D, const int64_t T, const int S) {
    const int u = blockIdx.x;
    const int h = blockIdx.y;
    if (u >= U) return;
    const int e0 = seg[u], e1 = seg[u + 1];
    if (e1 <= e0) return;
    const int64_t tok = uniq[u];
    const int64_t base = (tok * H + h);
    const float m_old = acc_m[base];
    // lse of packed row p: varlen layout [H, T] (S == 0) or uniform layout [W, H, S] (p = w*S + r)
    auto lse_at = [&](int64_t p) -> float {
        return S > 0 ? lse_pack[(p / S) * (int64_t)H * S + (int64_t)h * S + (p % S)] : lse_pack[(int64_t)h * T + p];
    };
    // pass 1: the new running max over the accumulator and every entry
    float M = m_old;
    for (int e = e0; e < e1; ++e) {
        const float lse = lse_at(order[e]);
        if (isfinite(lse)) { M = fmaxf(M, lse); }
    }
    if (!(M > -INFINITY)) return;   // nothing contributed to this token yet
    const float s_old = isfinite(m_old) ? __expf(m_old - M) : 0.f;
    // pass 2: rescale and fold, in entry order
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float a = acc[base * D + d] * s_old;
        for (int e = e0; e < e1; ++e) {
            const int64_t row = order[e];
            const float lse = lse_at(row);
            if (!isfinite(lse)) continue;
            const float w = __expf(lse - M);
            a += w * out_pack[(row * H + h) * D + d];
        }
        acc[base * D + d] = a;
    }
    if (threadIdx.x == 0) {
        float l = acc_l[base] * s_old;
        for (int e = e0; e < e1; ++e) {
            const float lse = lse_at(order[e]);
            if (isfinite(lse)) { l += __expf(lse - M); }
        }
        acc_l[base] = l;
        acc_m[base] = M;
    }
}

template <typename T_>
__device__ __forceinline__ float to_f32(T_ x);
template <> __device__ __forceinline__ float to_f32<float>(float x) { return x; }
template <> __device__ __forceinline__ float to_f32<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

// dst [Ntot, H, D] f32 += sum over the packed rows of each token, in order.
template <typename scalar_t>
__global__ void wave_scatter_add_kernel(
    float* __restrict__ dst, const scalar_t* __restrict__ src_pack,
    const int64_t* __restrict__ order, const int* __restrict__ seg, const int64_t* __restrict__ uniq,
    const int U, const int H, const int D) {
    const int u = blockIdx.x;
    const int h = blockIdx.y;
    if (u >= U) return;
    const int e0 = seg[u], e1 = seg[u + 1];
    if (e1 <= e0) return;
    const int64_t base = (uniq[u] * H + h);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float a = dst[base * D + d];
        for (int e = e0; e < e1; ++e) {
            a += to_f32<scalar_t>(src_pack[(order[e] * H + h) * D + d]);
        }
        dst[base * D + d] = a;
    }
}

}  // namespace

void wave_merge_cuda(at::Tensor acc, at::Tensor acc_l, at::Tensor acc_m,
                     const at::Tensor out_pack, const at::Tensor lse_pack,
                     const at::Tensor order, const at::Tensor seg, const at::Tensor uniq, const int S) {
    WCHECK_CUDA(acc); WCHECK_CUDA(out_pack); WCHECK_CUDA(lse_pack); WCHECK_CUDA(order); WCHECK_CUDA(seg); WCHECK_CUDA(uniq);
    WCHECK_CONTIG(acc); WCHECK_CONTIG(acc_l); WCHECK_CONTIG(acc_m); WCHECK_CONTIG(out_pack); WCHECK_CONTIG(lse_pack);
    WCHECK_CONTIG(order); WCHECK_CONTIG(seg); WCHECK_CONTIG(uniq);
    TORCH_CHECK(acc.dtype() == at::kFloat && out_pack.dtype() == at::kFloat && lse_pack.dtype() == at::kFloat, "fp32 expected");
    TORCH_CHECK(order.dtype() == at::kLong && uniq.dtype() == at::kLong && seg.dtype() == at::kInt, "index dtypes");
    TORCH_CHECK(acc.dim() == 3 && out_pack.dim() == 3, "acc/out_pack [N,H,D]");
    const int H = acc.size(1), D = acc.size(2);
    const int64_t T = out_pack.size(0);
    TORCH_CHECK(out_pack.size(1) == H && out_pack.size(2) == D, "shape mismatch");
    if (S > 0) {
        TORCH_CHECK(lse_pack.dim() == 3 && lse_pack.size(1) == H && lse_pack.size(2) == S && lse_pack.size(0) * (int64_t)S == T, "lse_pack [W,H,S]");
    } else {
        TORCH_CHECK(lse_pack.dim() == 2 && lse_pack.size(0) == H && lse_pack.size(1) == T, "lse_pack [H,T]");
    }
    const int U = uniq.numel();
    TORCH_CHECK(seg.numel() == U + 1, "seg must have U+1 entries");
    if (U == 0) return;
    const at::cuda::CUDAGuard guard(acc.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(U, H);
    const int threads = D >= 128 ? 128 : 64;
    wave_merge_kernel<<<grid, threads, 0, stream>>>(
        acc.data_ptr<float>(), acc_l.data_ptr<float>(), acc_m.data_ptr<float>(),
        out_pack.data_ptr<float>(), lse_pack.data_ptr<float>(),
        order.data_ptr<int64_t>(), seg.data_ptr<int>(), uniq.data_ptr<int64_t>(), U, H, D, T, S);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wave_scatter_add_cuda(at::Tensor dst, const at::Tensor src_pack,
                           const at::Tensor order, const at::Tensor seg, const at::Tensor uniq) {
    WCHECK_CUDA(dst); WCHECK_CUDA(src_pack); WCHECK_CONTIG(dst); WCHECK_CONTIG(src_pack);
    WCHECK_CONTIG(order); WCHECK_CONTIG(seg); WCHECK_CONTIG(uniq);
    TORCH_CHECK(dst.dtype() == at::kFloat, "dst must be fp32");
    TORCH_CHECK(dst.dim() == 3 && src_pack.dim() == 3 && dst.size(1) == src_pack.size(1) && dst.size(2) == src_pack.size(2), "[N,H,D] / [T,H,D]");
    const int H = dst.size(1), D = dst.size(2);
    const int U = uniq.numel();
    TORCH_CHECK(seg.numel() == U + 1, "seg must have U+1 entries");
    if (U == 0) return;
    const at::cuda::CUDAGuard guard(dst.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(U, H);
    const int threads = D >= 128 ? 128 : 64;
    if (src_pack.dtype() == at::kFloat) {
        wave_scatter_add_kernel<float><<<grid, threads, 0, stream>>>(dst.data_ptr<float>(), src_pack.data_ptr<float>(),
            order.data_ptr<int64_t>(), seg.data_ptr<int>(), uniq.data_ptr<int64_t>(), U, H, D);
    } else if (src_pack.dtype() == at::kHalf) {
        wave_scatter_add_kernel<__half><<<grid, threads, 0, stream>>>(dst.data_ptr<float>(), reinterpret_cast<const __half*>(src_pack.data_ptr()),
            order.data_ptr<int64_t>(), seg.data_ptr<int>(), uniq.data_ptr<int64_t>(), U, H, D);
    } else if (src_pack.dtype() == at::kBFloat16) {
        wave_scatter_add_kernel<__nv_bfloat16><<<grid, threads, 0, stream>>>(dst.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(src_pack.data_ptr()),
            order.data_ptr<int64_t>(), seg.data_ptr<int>(), uniq.data_ptr<int64_t>(), U, H, D);
    } else {
        TORCH_CHECK(false, "src_pack must be fp32, fp16 or bf16");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace FLASH_NAMESPACE
