/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>  // For at::Generator and at::PhiloxCudaState
#include "philox_unpack.cuh"  // For at::cuda::philox::unpack
#include <cuda_runtime_api.h>

#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "hardware_info.h"
#include "flash.h"
#include "static_switch.h"
#include <array>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace FLASH_NAMESPACE {

namespace {

constexpr int64_t kCQSANumChunk = 7;
constexpr std::array<int64_t, 3> kCQSAInterestSet = {0, 1, 3};

static inline void compute_chunk_layout_7(
    int64_t n_tokens,
    std::array<int64_t, kCQSANumChunk>& sizes,
    std::array<int64_t, kCQSANumChunk>& starts,
    std::array<int64_t, kCQSANumChunk>& ends) {
    const auto q = n_tokens / kCQSANumChunk;
    const auto r = n_tokens % kCQSANumChunk;
    const auto bound = kCQSANumChunk - r;
    for (int64_t i = 0; i < kCQSANumChunk; ++i) {
        sizes[i] = q + (i >= bound ? 1 : 0);
    }
    starts[0] = 0;
    for (int64_t i = 1; i < kCQSANumChunk; ++i) {
        starts[i] = starts[i - 1] + sizes[i - 1];
    }
    for (int64_t i = 0; i < kCQSANumChunk; ++i) {
        ends[i] = starts[i] + sizes[i];
    }
}

static inline void compute_chunk_layout_front(
    int64_t n_tokens,
    int64_t num_chunk,
    std::vector<int64_t>& starts,
    std::vector<int64_t>& ends) {
    starts.assign(static_cast<size_t>(num_chunk), 0);
    ends.assign(static_cast<size_t>(num_chunk), 0);
    const int64_t q = n_tokens / num_chunk;
    const int64_t r = n_tokens % num_chunk;
    int64_t offset = 0;
    for (int64_t i = 0; i < num_chunk; ++i) {
        const int64_t sz = q + (i < r ? 1 : 0);
        starts[static_cast<size_t>(i)] = offset;
        offset += sz;
        ends[static_cast<size_t>(i)] = offset;
    }
}

static inline std::array<int64_t, kCQSAInterestSet.size()> cqs_quorum_for_subseq(int64_t subseq_i) {
    std::array<int64_t, kCQSAInterestSet.size()> chunks{};
    for (int64_t j = 0; j < static_cast<int64_t>(kCQSAInterestSet.size()); ++j) {
        chunks[j] = (subseq_i + kCQSAInterestSet[static_cast<size_t>(j)]) % kCQSANumChunk;
    }
    return chunks;
}

static inline int64_t ipow_i64(int64_t base, int64_t exp) {
    int64_t out = 1;
    for (int64_t i = 0; i < exp; ++i) {
        TORCH_CHECK(out <= std::numeric_limits<int64_t>::max() / base, "num_itr too large (overflow)");
        out *= base;
    }
    return out;
}

static inline void build_path_state_and_group_bits(
    int64_t N,
    const std::vector<int64_t>& path,
    std::vector<int64_t>& token_ids,
    std::vector<int64_t>& group_bits) {
    token_ids.resize(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) {
        token_ids[static_cast<size_t>(i)] = i;
    }

    std::vector<std::vector<int16_t>> label_history;
    std::vector<std::array<int64_t, kCQSAInterestSet.size()>> quorum_history;
    label_history.reserve(path.size());
    quorum_history.reserve(path.size());

    for (size_t itr = 0; itr < path.size(); ++itr) {
        const int64_t subseq_i = path[itr];
        const int64_t cur_len = static_cast<int64_t>(token_ids.size());
        if (cur_len == 0) {
            break;
        }

        std::vector<int64_t> starts;
        std::vector<int64_t> ends;
        compute_chunk_layout_front(cur_len, kCQSANumChunk, starts, ends);
        const auto quorum_chunks = cqs_quorum_for_subseq(subseq_i);

        std::vector<int16_t> labels_cur(static_cast<size_t>(cur_len), 0);
        for (int64_t c = 0; c < kCQSANumChunk; ++c) {
            const int64_t s = starts[static_cast<size_t>(c)];
            const int64_t e = ends[static_cast<size_t>(c)];
            for (int64_t idx = s; idx < e; ++idx) {
                labels_cur[static_cast<size_t>(idx)] = static_cast<int16_t>(c);
            }
        }

        std::vector<int64_t> gather_idx;
        gather_idx.reserve(static_cast<size_t>(cur_len));
        for (int64_t j = 0; j < static_cast<int64_t>(kCQSAInterestSet.size()); ++j) {
            const int64_t c = quorum_chunks[static_cast<size_t>(j)];
            const int64_t s = starts[static_cast<size_t>(c)];
            const int64_t e = ends[static_cast<size_t>(c)];
            for (int64_t idx = s; idx < e; ++idx) {
                gather_idx.push_back(idx);
            }
        }

        for (auto& labels_prev : label_history) {
            std::vector<int16_t> reordered;
            reordered.reserve(gather_idx.size());
            for (const int64_t idx : gather_idx) {
                reordered.push_back(labels_prev[static_cast<size_t>(idx)]);
            }
            labels_prev.swap(reordered);
        }

        std::vector<int16_t> labels_new;
        labels_new.reserve(gather_idx.size());
        std::vector<int64_t> token_ids_new;
        token_ids_new.reserve(gather_idx.size());
        for (const int64_t idx : gather_idx) {
            labels_new.push_back(labels_cur[static_cast<size_t>(idx)]);
            token_ids_new.push_back(token_ids[static_cast<size_t>(idx)]);
        }
        label_history.push_back(std::move(labels_new));
        token_ids.swap(token_ids_new);
        quorum_history.push_back(quorum_chunks);
    }

    const int64_t local_size = static_cast<int64_t>(token_ids.size());
    group_bits.assign(static_cast<size_t>(local_size), 0);
    std::unordered_set<std::string> seen_groups;
    int64_t bit_id = 0;

    for (size_t itr = 0; itr < label_history.size(); ++itr) {
        const int16_t owner = static_cast<int16_t>(path[itr]);
        const auto& labels = label_history[itr];
        const auto& chunks = quorum_history[itr];
        for (int64_t j = 0; j < static_cast<int64_t>(chunks.size()); ++j) {
            const int16_t c = static_cast<int16_t>(chunks[static_cast<size_t>(j)]);
            if (c == owner) {
                continue;
            }

            std::vector<std::pair<int64_t, int64_t>> runs;
            for (int64_t pos = 0; pos < local_size;) {
                while (pos < local_size && labels[static_cast<size_t>(pos)] != c) {
                    ++pos;
                }
                if (pos >= local_size) {
                    break;
                }
                const int64_t s = pos;
                while (pos < local_size && labels[static_cast<size_t>(pos)] == c) {
                    ++pos;
                }
                runs.emplace_back(s, pos);
            }
            if (runs.empty()) {
                continue;
            }

            std::string key;
            key.reserve(runs.size() * 16);
            for (const auto& run : runs) {
                key.append(std::to_string(run.first));
                key.push_back('-');
                key.append(std::to_string(run.second));
                key.push_back('|');
            }
            if (!seen_groups.insert(key).second) {
                continue;
            }

            TORCH_CHECK(bit_id < 63, "CQSA mask has too many unique groups for int64 bit encoding");
            const int64_t bit = (int64_t(1) << bit_id);
            ++bit_id;
            for (const auto& run : runs) {
                for (int64_t pos = run.first; pos < run.second; ++pos) {
                    group_bits[static_cast<size_t>(pos)] |= bit;
                }
            }
        }
    }
}

} // namespace

void cqsa_accum_out_lse_stage_cuda(
    at::Tensor& global_num,
    const at::Tensor& out_sub,
    at::Tensor& global_den,
    const at::Tensor& lse_sub,
    const int64_t* global_starts,
    const int64_t* local_starts,
    const int64_t* segment_lens,
    int64_t num_segments);

void cqsa_accum_out_lse_index_cuda(
    at::Tensor& global_num,
    const at::Tensor& out_sub,
    at::Tensor& global_den,
    const at::Tensor& lse_sub,
    const at::Tensor& token_ids);

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool seqlenq_ngroups_swapped=false,
                      const bool unpadded_lse=false) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);
        if (seqlenq_ngroups_swapped) {
             params.q_batch_stride *= seqlen_q;
             params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
        TORCH_CHECK(softcap <= 0.0, "This flash attention build does not support softcap.");
    #endif
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
            "This flash attention build does not support local attention.");
    #endif

    params.is_seqlens_k_cumulative = true;

    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
    #endif

    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor out,
                      const at::Tensor dout,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *dq_accum_d,
                      void *dk_accum_d,
                      void *dv_accum_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool deterministic,
                      const bool unpadded_lse) {

    set_params_fprop(params,
                     b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     nullptr,
                     nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     false, // seqlenq_ngroups_swapped
                     unpadded_lse);

    // Set the pointers and strides.
    params.do_ptr = dout.data_ptr();
    params.do_row_stride = dout.stride(-3);
    params.do_head_stride = dout.stride(-2);
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.dq_row_stride = dq.stride(-3);
    params.dk_row_stride = dk.stride(-3);
    params.dv_row_stride = dv.stride(-3);
    params.dq_head_stride = dq.stride(-2);
    params.dk_head_stride = dk.stride(-2);
    params.dv_head_stride = dv.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = dout.stride(0);
        params.dq_batch_stride = dq.stride(0);
        params.dk_batch_stride = dk.stride(0);
        params.dv_batch_stride = dv.stride(0);
    }

    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;

    // Default off for every backward entry point; only mha_bwd flips it, and
    // only when the caller supplies dsoftmax_sum. Set here rather than at each
    // call site so a new entry point cannot inherit an uninitialised bool and
    // silently skip the preprocess pass.
    params.cqsa_ext_dpsum = false;

    params.deterministic = deterministic;
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
#ifdef CQSA_MINIMAL_FWD_KERNELS
#ifndef CQSA_MINIMAL_BOTH_DTYPES
    TORCH_CHECK(!params.is_bf16,
                "this CQSA build only supports fp16; rebuild with "
                "CQSA_KERNEL_SET=common (fp16+bf16) or =full.");
#endif
    TORCH_CHECK(!force_split_kernel, "CQSA minimal build does not support split-k forward.");
#ifdef CQSA_MINIMAL_HDIM64_NONCAUSAL_ONLY
    TORCH_CHECK(params.d == 64, "CQSA hdim64 non-causal build only supports head_dim=64.");
    TORCH_CHECK(!params.is_causal, "CQSA hdim64 non-causal build only supports non-causal forward.");
    run_mha_fwd_<cutlass::half_t, 64, false>(params, stream);
#else
    TORCH_CHECK(params.d == 64 || params.d == 128,
                "this CQSA build only supports head_dim=64 or 128; rebuild with "
                "CQSA_KERNEL_SET=full for other head dims.");
    FP16_SWITCH(!params.is_bf16, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
#ifdef CQSA_NATIVE_HDIM64_ONLY
            TORCH_CHECK(params.d == 64, "this native dev build only has head_dim=64 kernels");
            run_mha_fwd_<elem_type, 64, Is_causal>(params, stream);
#else
            if (params.d == 64) {
                run_mha_fwd_<elem_type, 64, Is_causal>(params, stream);
            } else {
                run_mha_fwd_<elem_type, 128, Is_causal>(params, stream);
            }
#endif
        });
    });
#endif
#else
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                    run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                } else {
                    run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                }
            });
        });
    });
#endif
}

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

std::tuple<at::Tensor, at::Tensor> set_params_splitkv(Flash_fwd_params &params, const int batch_size,
    const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
    const int head_size_rounded, const float p_dropout,
    const int num_splits, const int num_sm, struct c10::TensorOptions opts) {

    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
    params.num_splits = num_splits;
    at::Tensor softmax_lse_accum;
    at::Tensor out_accum;

    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        if (num_splits < 1) {
            // We multiply number of SMs by 2 to hard-code the fact that we're using 128 threads per block.
            params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, num_sm * 2, num_n_blocks, 128);
        }
        if (params.num_splits > 1) {
            softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
            out_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));
            params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
            params.oaccum_ptr = out_accum.data_ptr();
        }
        TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }

    return std::make_tuple(softmax_lse_accum, out_accum);
}

void set_params_alibi(Flash_fwd_params &params, std::optional<at::Tensor> &alibi_slopes_, int batch_size, int num_heads){
#ifdef FLASHATTENTION_DISABLE_ALIBI
    TORCH_CHECK(!alibi_slopes_.has_value(), "This flash attention build does not support alibi.");
    params.alibi_slopes_ptr = nullptr;
#else
    if (alibi_slopes_.has_value()) {
        auto alibi_slopes = alibi_slopes_.value();
        TORCH_CHECK(alibi_slopes.dtype() == torch::kFloat32, "ALiBi slopes must have dtype fp32");
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({num_heads}) || alibi_slopes.sizes() == torch::IntArrayRef({batch_size, num_heads}));
        params.alibi_slopes_ptr = alibi_slopes.data_ptr();
        params.alibi_slopes_batch_stride = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    } else {
        params.alibi_slopes_ptr = nullptr;
    }
#endif
}

std::vector<at::Tensor>
mha_fwd_impl(at::Tensor &q,         // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        std::optional<at::Generator> gen_,
        std::optional<at::Tensor> cqs_chunk_ends_,
        std::optional<int64_t> cqs_owner_chunk_,
        std::optional<at::Tensor> cqs_group_bits_,
        std::optional<at::Tensor> cqs_blk_or_ = std::nullopt,
        std::optional<at::Tensor> cqs_blk_and_ = std::nullopt,
        std::optional<int64_t> cqs_blk_size_ = std::nullopt,
        std::optional<at::Tensor> cqs_out_acc_ = std::nullopt,
        std::optional<at::Tensor> cqs_block_base_ = std::nullopt,
        std::optional<int64_t> cqs_seg_align_ = std::nullopt) {

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size % 8 == 0 && !alibi_slopes_.has_value();
    const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped) {
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size);
        if (seqlenq_ngroups_swapped) {
            out = out.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
        }
    } else {
        out = torch::empty_like(q);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    auto opts = q.options();

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
    }
    else {
        p = torch::empty({ 0 }, opts);
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap
                     );

    params.cqs_blk_or = nullptr;
    params.cqs_blk_and = nullptr;
    params.cqs_blk_size = 0;
    params.cqs_num_blocks = 0;
    params.cqs_block_base = nullptr;
    params.cqs_seg_align = 0;
    if (cqs_block_base_.has_value()) {
        auto bb = cqs_block_base_.value();
        CHECK_DEVICE(bb); CHECK_CONTIGUOUS(bb);
        TORCH_CHECK(bb.dtype() == torch::kInt32, "cqs_block_base must be int32");
        const int align = static_cast<int>(cqs_seg_align_.value_or(0));
        TORCH_CHECK(align > 0, "cqs_seg_align must be positive with cqs_block_base");
        TORCH_CHECK(bb.numel() >= (seqlen_q + align - 1) / align,
                    "cqs_block_base too short for seqlen_q");
        TORCH_CHECK(seqlen_q == seqlen_k,
                    "segmented input assumes the query and key subsequences coincide");
        params.cqs_block_base = bb.data_ptr<int>();
        params.cqs_seg_align = align;
    }
    params.cqs_out_acc_ptr = nullptr;
    if (cqs_out_acc_.has_value()) {
        auto acc_out = cqs_out_acc_.value();
        CHECK_DEVICE(acc_out);
        TORCH_CHECK(acc_out.dtype() == torch::kFloat32, "cqs_out_acc must be float32");
        TORCH_CHECK(acc_out.stride(-1) == 1, "cqs_out_acc must have contiguous last dimension");
        CHECK_SHAPE(acc_out, batch_size, seqlen_q, num_heads, head_size);
        // Same logical layout as `out`, so the o_* element strides apply as-is.
        TORCH_CHECK(acc_out.stride(0) == params.o_batch_stride
                    && acc_out.stride(1) == params.o_row_stride
                    && acc_out.stride(2) == params.o_head_stride,
                    "cqs_out_acc must have the same element strides as out");
        params.cqs_out_acc_ptr = acc_out.data_ptr();
    }
    TORCH_CHECK(
        !(cqs_chunk_ends_.has_value() && cqs_group_bits_.has_value()),
        "Provide either cqs_chunk_ends/cqs_owner_chunk or cqs_group_bits, not both");
    if (cqs_group_bits_.has_value()) {
        auto cqs_group_bits = cqs_group_bits_.value();
        CHECK_DEVICE(cqs_group_bits);
        CHECK_CONTIGUOUS(cqs_group_bits);
        TORCH_CHECK(cqs_group_bits.dtype() == torch::kInt64, "cqs_group_bits must be int64");
        TORCH_CHECK(cqs_group_bits.dim() == 1, "cqs_group_bits must be 1D");
        TORCH_CHECK(cqs_group_bits.numel() == seqlen_q, "cqs_group_bits length must match seqlen_q");
        TORCH_CHECK(seqlen_q == seqlen_k, "cqs_group_bits mode requires seqlen_q == seqlen_k");
        params.cqs_enabled = true;
        params.cqs_num_chunks = 0;
        params.cqs_owner_chunk = 0;
        params.cqs_chunk_ends = nullptr;
        params.cqs_group_bits = cqs_group_bits.data_ptr<int64_t>();
        if (cqs_blk_or_.has_value() && cqs_blk_and_.has_value()) {
            auto blk_or = cqs_blk_or_.value();
            auto blk_and = cqs_blk_and_.value();
            CHECK_DEVICE(blk_or); CHECK_CONTIGUOUS(blk_or);
            CHECK_DEVICE(blk_and); CHECK_CONTIGUOUS(blk_and);
            TORCH_CHECK(blk_or.dtype() == torch::kInt64 && blk_and.dtype() == torch::kInt64,
                        "cqs block summaries must be int64");
            TORCH_CHECK(blk_or.numel() == blk_and.numel(), "cqs block summaries must match in length");
            const int blk_size = static_cast<int>(cqs_blk_size_.value_or(0));
            TORCH_CHECK(blk_size > 0, "cqs_blk_size must be positive when summaries are provided");
            // next/ v5: the forward kernel's tile verdicts use a compile-time
            // block size (kCqsBlk in mask.h) so the index math is shifts.
            TORCH_CHECK(blk_size == 64, "cqs_blk_size must be 64 (CQS_BLK_SIZE) for this kernel build");
            TORCH_CHECK(blk_or.numel() >= (seqlen_q + blk_size - 1) / blk_size,
                        "cqs block summaries too short for seqlen_q");
            params.cqs_blk_or = blk_or.data_ptr<int64_t>();
            params.cqs_blk_and = blk_and.data_ptr<int64_t>();
            params.cqs_blk_size = blk_size;
            params.cqs_num_blocks = static_cast<int>(blk_or.numel());
        }
    } else if (cqs_chunk_ends_.has_value()) {
        auto cqs_chunk_ends = cqs_chunk_ends_.value();
        CHECK_DEVICE(cqs_chunk_ends);
        CHECK_CONTIGUOUS(cqs_chunk_ends);
        TORCH_CHECK(cqs_chunk_ends.dtype() == torch::kInt32, "cqs_chunk_ends must be int32");
        TORCH_CHECK(cqs_chunk_ends.dim() == 1, "cqs_chunk_ends must be 1D");
        TORCH_CHECK(cqs_chunk_ends.numel() >= 1, "cqs_chunk_ends must be non-empty");
        TORCH_CHECK(cqs_owner_chunk_.has_value(), "cqs_owner_chunk must be provided when cqs_chunk_ends is provided");
        const int owner = static_cast<int>(cqs_owner_chunk_.value());
        TORCH_CHECK(owner >= 0 && owner < cqs_chunk_ends.numel(), "cqs_owner_chunk out of range");
        params.cqs_enabled = true;
        params.cqs_num_chunks = static_cast<int>(cqs_chunk_ends.numel());
        params.cqs_owner_chunk = owner;
        params.cqs_chunk_ends = cqs_chunk_ends.data_ptr<int>();
        params.cqs_group_bits = nullptr;
    } else {
        params.cqs_enabled = false;
        params.cqs_num_chunks = 0;
        params.cqs_owner_chunk = 0;
        params.cqs_chunk_ends = nullptr;
        params.cqs_group_bits = nullptr;
    }

    // Keep references to these tensors to extend their lifetime
    at::Tensor softmax_lse_accum, out_accum;
    std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
        params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
        head_size_rounded, p_dropout, /*num_splits*/ 0, get_num_sm(get_current_device()), opts);

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    // Forward kernel will populate memory with the seed and offset.
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    if (p_dropout > 0.0)  {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    if (seqlenq_ngroups_swapped) {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
        q = q.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }
    return {out, softmax_lse, p, rng_state};
}

std::vector<at::Tensor>
mha_fwd(at::Tensor &q,         // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        std::optional<at::Generator> gen_) {
    return mha_fwd_impl(
        q,
        k,
        v,
        out_,
        alibi_slopes_,
        p_dropout,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        gen_,
        std::nullopt,
        std::nullopt,
        std::nullopt);
}

std::vector<at::Tensor>
mha_fwd_cqs(at::Tensor &q,         // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
            const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
            const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
            std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
            std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
            const float p_dropout,
            const float softmax_scale,
            bool is_causal,
            int window_size_left,
            int window_size_right,
            const float softcap,
            const bool return_softmax,
            std::optional<at::Generator> gen_,
            const at::Tensor &cqs_chunk_ends,
            int64_t cqs_owner_chunk) {
    auto cqs_chunk_ends_opt = std::optional<at::Tensor>(cqs_chunk_ends);
    auto cqs_owner_chunk_opt = std::optional<int64_t>(cqs_owner_chunk);
    return mha_fwd_impl(
        q,
        k,
        v,
        out_,
        alibi_slopes_,
        p_dropout,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        gen_,
        cqs_chunk_ends_opt,
        cqs_owner_chunk_opt,
        std::nullopt);
}

std::vector<at::Tensor>
mha_fwd_cqs_group_bits(at::Tensor &q,         // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
                       const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
                       const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
                       std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
                       std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                       const float p_dropout,
                       const float softmax_scale,
                       bool is_causal,
                       int window_size_left,
                       int window_size_right,
                       const float softcap,
                       const bool return_softmax,
                       std::optional<at::Generator> gen_,
                       const at::Tensor &cqs_group_bits,
                       std::optional<at::Tensor> cqs_blk_or_ = std::nullopt,
                       std::optional<at::Tensor> cqs_blk_and_ = std::nullopt,
                       std::optional<int64_t> cqs_blk_size_ = std::nullopt,
                       std::optional<at::Tensor> cqs_out_acc_ = std::nullopt,
                       std::optional<at::Tensor> cqs_block_base_ = std::nullopt,
                       std::optional<int64_t> cqs_seg_align_ = std::nullopt) {
    auto cqs_group_bits_opt = std::optional<at::Tensor>(cqs_group_bits);
    return mha_fwd_impl(
        q,
        k,
        v,
        out_,
        alibi_slopes_,
        p_dropout,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        gen_,
        std::nullopt,
        std::nullopt,
        cqs_group_bits_opt,
        cqs_blk_or_,
        cqs_blk_and_,
        cqs_blk_size_,
        cqs_out_acc_,
        cqs_block_base_,
        cqs_seg_align_);
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream);

std::vector<at::Tensor>
mha_bwd_cqs_group_bits(
    const at::Tensor &dout_num,      // [B, L, H, D], gradient wrt local numerator
    const at::Tensor &dden,          // [B, H, L], gradient wrt local denominator
    const at::Tensor &q,             // [B, L, H, D]
    const at::Tensor &k,             // [B, L, H, D]
    const at::Tensor &v,             // [B, L, H, D]
    const at::Tensor &cqs_group_bits,  // [L] int64
    const float softmax_scale) {

    CHECK_DEVICE(dout_num); CHECK_DEVICE(dden);
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cqs_group_bits);
    TORCH_CHECK(dout_num.dim() == 4, "dout_num must be [B,L,H,D]");
    TORCH_CHECK(dden.dim() == 3, "dden must be [B,H,L]");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be [B,L,H,D]");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q/k/v shape mismatch");
    TORCH_CHECK(dout_num.sizes() == q.sizes(), "dout_num shape must match q");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device() && q.device() == dout_num.device() && q.device() == dden.device(),
                "all tensors must be on the same CUDA device");
    TORCH_CHECK(cqs_group_bits.dim() == 1, "cqs_group_bits must be 1D");
    TORCH_CHECK(cqs_group_bits.numel() == q.size(1), "cqs_group_bits length must match local sequence length");
    TORCH_CHECK(dden.size(0) == q.size(0) && dden.size(1) == q.size(2) && dden.size(2) == q.size(1),
                "dden must have shape [B,H,L]");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16,
                "mha_bwd_cqs_group_bits supports fp16/bf16 q/k/v");

    auto bits = cqs_group_bits;
    if (bits.dtype() != torch::kInt64) {
        bits = bits.to(torch::kInt64);
    }
    if (bits.device() != q.device()) {
        bits = bits.to(q.device());
    }
    bits = bits.contiguous();

    // Otherwise the kernel will be launched from cuda:0 device.
    at::cuda::CUDAGuard device_guard{q.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    const auto q_dtype = q.dtype();

    // Run local forward once to obtain softmax_lse required by FA backward kernels.
    at::Tensor q_tmp = q;
    std::optional<at::Tensor> out_opt = std::nullopt;
    std::optional<at::Tensor> no_alibi = std::nullopt;
    std::optional<at::Generator> no_gen = std::nullopt;
    auto fwd_vec = mha_fwd_cqs_group_bits(
        q_tmp,
        k,
        v,
        out_opt,
        no_alibi,
        /*p_dropout=*/0.0f,
        softmax_scale,
        /*is_causal=*/false,
        /*window_size_left=*/-1,
        /*window_size_right=*/-1,
        /*softcap=*/0.0f,
        /*return_softmax=*/false,
        no_gen,
        bits
    );
    auto out = fwd_vec[0];
    auto softmax_lse = fwd_vec[1];

    // For cqsa_numden_mode in kernels, dO is expected to be pre-scaled:
    // dO_scaled = dNum * Den, where Den = exp(lse).
    auto den_tok = softmax_lse.transpose(1, 2).contiguous().exp().unsqueeze(-1);  // [B,L,H,1], fp32
    auto dout = (dout_num.to(torch::kFloat32) * den_tok).to(q_dtype).contiguous();

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    auto opts = q.options();
    at::Tensor dq = torch::empty_like(q);
    at::Tensor dk = torch::empty_like(k);
    at::Tensor dv = torch::empty_like(v);

    bool loop = true;
    at::Tensor dq_accum;
    if (loop) {
        dq_accum = torch::empty(
            {batch_size, seqlen_q_rounded, num_heads, head_size_rounded},
            opts.dtype(at::kFloat)
        );
        // In cqsa_numden_mode we skip the standard FA preprocess kernel that
        // normally clears dQ accumulation buffers. Explicitly zero dq_accum
        // here so dQ doesn't read uninitialized accumulation state.
        dq_accum.zero_();
    }

    // dsoftmax_sum stores row-wise dDen for CQSA Num/Den mode.
    auto dsoftmax_sum = torch::zeros(
        {batch_size, num_heads, seqlen_q_rounded},
        opts.dtype(at::kFloat)
    );
    dsoftmax_sum.narrow(/*dim=*/2, /*start=*/0, /*length=*/seqlen_q)
        .copy_(dden.to(torch::kFloat32));

    Flash_bwd_params params;
    set_params_dgrad(
        params,
        batch_size,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        seqlen_k_rounded,
        num_heads,
        num_heads_k,
        head_size,
        head_size_rounded,
        q,
        k,
        v,
        out,
        dout,
        dq,
        dk,
        dv,
        nullptr,
        nullptr,
        loop ? dq_accum.data_ptr() : nullptr,
        nullptr,
        nullptr,
        softmax_lse.data_ptr(),
        dsoftmax_sum.data_ptr(),
        /*p_dropout=*/0.0f,
        softmax_scale,
        /*window_size_left=*/-1,
        /*window_size_right=*/-1,
        /*softcap=*/0.0f,
        /*deterministic=*/false,
        /*unpadded_lse=*/false
    );
    params.dq_accum_split_stride = 0;
    params.cqsa_numden_mode = true;
    params.cqs_enabled = true;
    params.cqs_num_chunks = 0;
    params.cqs_owner_chunk = 0;
    params.cqs_chunk_ends = nullptr;
    params.cqs_group_bits = bits.data_ptr<int64_t>();

    if (seqlen_q > 0) {
        run_mha_bwd(params, stream);
    } else {
        dq.zero_();
        dk.zero_();
        dv.zero_();
    }

    return {dq, dk, dv};
}

std::vector<at::Tensor>
mha_fwd_cqsa_impl(at::Tensor &q,         // batch_size x seqlen x num_heads x head_size
                  const at::Tensor &k,   // batch_size x seqlen x num_heads_k x head_size
                  const at::Tensor &v,   // batch_size x seqlen x num_heads_k x head_size
                  const float p_dropout,
                  const float softmax_scale,
                  bool is_causal,
                  int64_t num_itr,
                  bool return_profile) {
    // NOTE: this entrypoint is intentionally specialized for speed:
    // - hardcoded CQSA layout parameters (num_chunk=7, interest_set=(0,1,3))
    // - supports iterative CQS splitting with num_chunk^num_itr subsequences
    // - no debug/probability outputs
    // - forward only
    using clock_t = std::chrono::steady_clock;
    const auto t_total_start = clock_t::now();
    double setup_ms = 0.0;
    double fork_ms = 0.0;
    double sync_ms = 0.0;
    double join_ms = 0.0;
    double finalize_ms = 0.0;

    at::cuda::CUDAGuard device_guard{q.device()};
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be rank-4");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "q/k/v dtype mismatch");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "batch mismatch");
    TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1), "seqlen mismatch");
    TORCH_CHECK(q.size(-1) == k.size(-1), "q/k head dim mismatch");
    TORCH_CHECK(k.size(-1) == v.size(-1), "k/v head dim mismatch");
    TORCH_CHECK(p_dropout == 0.f, "mha_fwd_cqsa currently supports dropout_p == 0 only");
    TORCH_CHECK(num_itr >= 0, "num_itr must be >= 0");

    const auto B = q.size(0);
    const auto N = q.size(1);
    const auto H = q.size(2);
    const auto D = q.size(3);

    auto accum_opts = q.options().dtype(torch::kFloat32);
    auto global_num = torch::zeros({B, N, H, D}, accum_opts);
    auto global_den = torch::zeros({B, H, N}, accum_opts);

    std::optional<at::Tensor> out_ = std::nullopt;
    std::optional<at::Tensor> alibi_slopes_ = std::nullopt;
    std::optional<at::Generator> gen_ = std::nullopt;

    struct CqsaStageResult {
        at::Tensor token_ids;
        at::Tensor cqs_group_bits;
        at::Tensor q_sub;
        at::Tensor k_sub;
        at::Tensor v_sub;
        at::Tensor out_sub;
        at::Tensor lse_sub;
        cudaEvent_t ready_event = nullptr;
    };

    const int64_t num_paths = ipow_i64(kCQSANumChunk, num_itr);
    int64_t num_streams = std::min<int64_t>(std::max<int64_t>(num_paths, 1), 4);
    if (const char* env_streams = std::getenv("CQSA_FUSED_STREAMS")) {
        try {
            const auto parsed = std::stoll(env_streams);
            if (parsed > 0) {
                num_streams = std::min<int64_t>(std::max<int64_t>(num_paths, 1), parsed);
            }
        } catch (...) {
        }
    }
    std::vector<c10::cuda::CUDAStream> streams;
    streams.reserve(num_streams);
    for (int64_t i = 0; i < num_streams; ++i) {
        streams.push_back(c10::cuda::getStreamFromPool(/*isHighPriority=*/false, q.get_device()));
    }

    const auto t_fork_start = clock_t::now();
    if (return_profile) {
        setup_ms = std::chrono::duration<double, std::milli>(t_fork_start - t_total_start).count();
    }

    std::vector<CqsaStageResult> staged_results;
    staged_results.reserve(static_cast<size_t>(num_paths));
    for (int64_t path_idx = 0; path_idx < num_paths; ++path_idx) {
        const auto stream = streams[path_idx % num_streams];
        c10::cuda::CUDAStreamGuard stream_guard(stream);

        std::vector<int64_t> path(static_cast<size_t>(num_itr), 0);
        int64_t tmp = path_idx;
        for (int64_t itr = num_itr - 1; itr >= 0; --itr) {
            path[static_cast<size_t>(itr)] = tmp % kCQSANumChunk;
            tmp /= kCQSANumChunk;
            if (itr == 0) {
                break;
            }
        }

        std::vector<int64_t> token_ids_host;
        std::vector<int64_t> group_bits_host;
        build_path_state_and_group_bits(N, path, token_ids_host, group_bits_host);
        if (token_ids_host.empty()) {
            continue;
        }

        auto token_ids = torch::tensor(
            token_ids_host,
            torch::TensorOptions().dtype(torch::kInt64).device(q.device()));
        auto cqs_group_bits = torch::tensor(
            group_bits_host,
            torch::TensorOptions().dtype(torch::kInt64).device(q.device()));

        auto q_sub = q.index_select(/*dim=*/1, token_ids);
        auto k_sub = k.index_select(/*dim=*/1, token_ids);
        auto v_sub = v.index_select(/*dim=*/1, token_ids);

        auto out_vec = mha_fwd_cqs_group_bits(
            q_sub,
            k_sub,
            v_sub,
            out_,
            alibi_slopes_,
            p_dropout,
            softmax_scale,
            is_causal,
            -1,
            -1,
            0.0f,
            false,
            gen_,
            cqs_group_bits);
        auto out_sub = out_vec[0];
        auto lse_sub = out_vec[1];
        cudaEvent_t ready_event = nullptr;
        {
            auto err = cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming);
            TORCH_CHECK(err == cudaSuccess, "cudaEventCreateWithFlags failed in mha_fwd_cqsa");
            err = cudaEventRecord(ready_event, stream.stream());
            TORCH_CHECK(err == cudaSuccess, "cudaEventRecord failed in mha_fwd_cqsa");
        }
        staged_results.push_back(CqsaStageResult{
            std::move(token_ids),
            std::move(cqs_group_bits),
            std::move(q_sub),
            std::move(k_sub),
            std::move(v_sub),
            std::move(out_sub),
            std::move(lse_sub),
            ready_event,
        });
    }
    if (return_profile) {
        const auto t_fork_end = clock_t::now();
        fork_ms = std::chrono::duration<double, std::milli>(t_fork_end - t_fork_start).count();
    }

    const auto t_sync_start = clock_t::now();
    auto current_stream = at::cuda::getCurrentCUDAStream().stream();
    for (const auto& stage : staged_results) {
        auto err = cudaStreamWaitEvent(current_stream, stage.ready_event, 0);
        TORCH_CHECK(err == cudaSuccess, "cudaStreamWaitEvent failed in mha_fwd_cqsa");
    }
    if (return_profile) {
        const auto t_sync_end = clock_t::now();
        sync_ms = std::chrono::duration<double, std::milli>(t_sync_end - t_sync_start).count();
    }

    const auto t_join_start = clock_t::now();
    for (const auto& stage : staged_results) {
        cqsa_accum_out_lse_index_cuda(
            global_num,
            stage.out_sub,
            global_den,
            stage.lse_sub,
            stage.token_ids);
    }
    for (auto& stage : staged_results) {
        if (stage.ready_event != nullptr) {
            auto err = cudaEventDestroy(stage.ready_event);
            TORCH_CHECK(err == cudaSuccess, "cudaEventDestroy failed in mha_fwd_cqsa");
            stage.ready_event = nullptr;
        }
    }
    if (return_profile) {
        auto err = cudaStreamSynchronize(current_stream);
        TORCH_CHECK(err == cudaSuccess, "cudaStreamSynchronize failed after CQSA join");
        const auto t_join_end = clock_t::now();
        join_ms = std::chrono::duration<double, std::milli>(t_join_end - t_join_start).count();
    }

    const auto t_finalize_start = clock_t::now();
    auto denom = global_den.transpose(1, 2).unsqueeze(-1).clamp_min(1e-12);
    auto out = (global_num / denom).to(q.scalar_type());
    auto covered = global_den > 0;
    out = torch::where(covered.transpose(1, 2).unsqueeze(-1), out, torch::zeros_like(out));
    if (!return_profile) {
        return {out, global_den};
    }

    const auto t_finalize_end = clock_t::now();
    finalize_ms = std::chrono::duration<double, std::milli>(t_finalize_end - t_finalize_start).count();
    const auto t_total_end = clock_t::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();
    auto timing = torch::tensor(
        {static_cast<float>(total_ms),
         static_cast<float>(setup_ms),
         static_cast<float>(fork_ms),
         static_cast<float>(sync_ms),
         static_cast<float>(join_ms),
         static_cast<float>(finalize_ms)},
        torch::TensorOptions().dtype(torch::kFloat32));
    return {out, global_den, timing};
}

std::vector<at::Tensor>
mha_fwd_cqsa(at::Tensor &q,         // batch_size x seqlen x num_heads x head_size
             const at::Tensor &k,   // batch_size x seqlen x num_heads_k x head_size
             const at::Tensor &v,   // batch_size x seqlen x num_heads_k x head_size
             const float p_dropout,
             const float softmax_scale,
             bool is_causal,
             int64_t num_itr) {
    return mha_fwd_cqsa_impl(q, k, v, p_dropout, softmax_scale, is_causal, num_itr, /*return_profile=*/false);
}

std::vector<at::Tensor>
mha_fwd_cqsa_profile(at::Tensor &q,         // batch_size x seqlen x num_heads x head_size
                     const at::Tensor &k,   // batch_size x seqlen x num_heads_k x head_size
                     const at::Tensor &v,   // batch_size x seqlen x num_heads_k x head_size
                     const float p_dropout,
                     const float softmax_scale,
                     bool is_causal,
                     int64_t num_itr) {
    return mha_fwd_cqsa_impl(q, k, v, p_dropout, softmax_scale, is_causal, num_itr, /*return_profile=*/true);
}

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               std::optional<const at::Tensor> &leftpad_k_, // batch_size
               std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool return_softmax,
               std::optional<at::Generator> gen_) {

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);

    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    int num_heads = sizes[1];
    const int head_size = sizes[2];
    const int num_heads_k = paged_KV ? k.size(2) : k.size(1);

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : k.size(0);
    const int page_block_size = !paged_KV ? 1 : k.size(1);
    TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");

    if (max_seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }  // causal=true is the same as causal=false in this case
    if (is_causal) { window_size_right = 0; }

    void *cu_seqlens_q_d = cu_seqlens_q.data_ptr();

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = max_seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size % 8 == 0 && !alibi_slopes_.has_value();
    const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped) {
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2).reshape({batch_size * ngroups, num_heads_k, head_size});
        max_seqlen_q = ngroups;
        num_heads = num_heads_k;
        cu_seqlens_q_d = nullptr;
    }

    const int total_q = q.sizes()[0];

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    if (!paged_KV) {
        const int total_k = k.size(0);
        CHECK_SHAPE(k, total_k, num_heads_k, head_size);
        CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    } else {
        CHECK_SHAPE(k, num_blocks, page_block_size, num_heads_k, head_size);
        CHECK_SHAPE(v, num_blocks, page_block_size, num_heads_k, head_size);
        CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    }

    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    if (seqused_k.has_value()){
        auto seqused_k_ = seqused_k.value();
        TORCH_CHECK(seqused_k_.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
        TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
        CHECK_SHAPE(seqused_k_, batch_size);
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, sizes[0], sizes[1], head_size);
        if (seqlenq_ngroups_swapped) {
            out = out.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2).reshape({batch_size * ngroups, num_heads_k, head_size});
        }
    } else {
        out = torch::empty_like(q);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    auto opts = q.options();
    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
    }
    else {
        p = torch::empty({ 0 }, opts);
    }

    if (zero_tensors) {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (return_softmax) {p.zero_();}
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k.data_ptr(),
                     seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     seqlenq_ngroups_swapped,
                     /*unpadded_lse*/true);
    params.total_q = total_q;

    if (paged_KV) {
        params.block_table = block_table.data_ptr<int>();
        params.block_table_batch_stride = block_table.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
    }
    params.page_block_size = page_block_size;
    // Keep references to these tensors to extend their lifetime
    at::Tensor softmax_lse_accum, out_accum;
    if (seqlenq_ngroups_swapped) {
        // Only apply split-k for decoding
        std::tie(softmax_lse_accum, out_accum) =
            set_params_splitkv(params, batch_size, num_heads, head_size,
                               max_seqlen_k, max_seqlen_q, head_size_rounded,
                               p_dropout, /*num_splits*/ 0, get_num_sm(get_current_device()), opts);
    }

    if (leftpad_k_.has_value()) {
        auto leftpad_k = leftpad_k_.value();
        TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
        TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
        CHECK_DEVICE(leftpad_k);
        CHECK_CONTIGUOUS(leftpad_k);
        CHECK_SHAPE(leftpad_k, batch_size);
        params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());
    }

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    // Forward kernel will populate memory with the seed and offset.
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    if (p_dropout > 0.0)  {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream, paged_KV);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    if (seqlenq_ngroups_swapped) {
        int64_t size_before[] = {batch_size, max_seqlen_q, num_heads_k, head_size};
        int64_t size_after[] = {batch_size, num_heads_k * max_seqlen_q, head_size};
        out = out.reshape(size_before).transpose(1, 2).reshape(size_after);
        q = q.reshape(size_before).transpose(1, 2).reshape(size_after);
        softmax_lse = softmax_lse.reshape({num_heads * max_seqlen_q, batch_size});
    }

    return {out, softmax_lse, p, rng_state};
}

#ifndef FLASHATTENTION_DISABLE_BACKWARD
void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
#ifdef CQSA_MINIMAL_FWD_KERNELS
#ifndef CQSA_MINIMAL_BOTH_DTYPES
    TORCH_CHECK(!params.is_bf16,
                "this CQSA build's backward only supports fp16; rebuild with "
                "CQSA_KERNEL_SET=common (fp16+bf16) or =full.");
#endif
#ifdef CQSA_MINIMAL_HDIM64_NONCAUSAL_ONLY
    TORCH_CHECK(params.d == 64, "CQSA hdim64 non-causal build backward only supports head_dim=64.");
    TORCH_CHECK(!params.is_causal, "CQSA hdim64 non-causal build backward is non-causal only.");
    run_mha_bwd_<cutlass::half_t, 64, false>(params, stream);
#else
    TORCH_CHECK(params.d == 64 || params.d == 128,
                "this CQSA build's backward only supports head_dim=64 or 128; "
                "rebuild with CQSA_KERNEL_SET=full for other head dims.");
    FP16_SWITCH(!params.is_bf16, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
#ifdef CQSA_NATIVE_HDIM64_ONLY
            TORCH_CHECK(params.d == 64, "this native dev build only has head_dim=64 kernels");
            run_mha_bwd_<elem_type, 64, Is_causal>(params, stream);
#else
            if (params.d == 64) {
                run_mha_bwd_<elem_type, 64, Is_causal>(params, stream);
            } else {
                run_mha_bwd_<elem_type, 128, Is_causal>(params, stream);
            }
#endif
        });
    });
#endif
#else
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_bwd_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        });
    });
#endif
}

std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x multiple_of(head_size_og, 8)
        const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,     // b x h x seqlen_q
        std::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
        std::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool deterministic,
        std::optional<at::Generator> gen_,
        std::optional<at::Tensor> &rng_state,
        // Optional CQS group-bit mask. With it set, the STANDARD softmax
        // backward runs on the masked pair set -- which is exact for
        // Stream-CQSA provided softmax_lse is the GLOBAL log-sum-exp, since
        // then p = exp(s - lse) <= 1 and no unshifted Num/Den is ever formed.
        std::optional<at::Tensor> cqs_group_bits_ = std::nullopt,
        // Block summaries of the group bits. Without these the mask's O(1)
        // tile test cannot run and every tile falls into the per-element
        // masking loop -- measured 9.1x slower than the stock kernel even with
        // an all-zero mask. The forward has taken these since the tile
        // early-out was added; the backward did not, which is why it was slow.
        std::optional<at::Tensor> cqs_blk_or_ = std::nullopt,
        std::optional<at::Tensor> cqs_blk_and_ = std::nullopt,
        std::optional<int64_t> cqs_blk_size_ = std::nullopt,
        // Precomputed rowsum(dO * O), [batch, heads, seqlen_q]. When given, the
        // preprocess pass is skipped and `out` is never read.
        std::optional<at::Tensor> dsoftmax_sum_ = std::nullopt){

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        TORCH_CHECK(false, "This flash attention build does not support backward.");
    #endif
    if (is_causal) { window_size_right = 0; }

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    bool is_dropout = p_dropout > 0.0;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size);

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        dv = torch::empty_like(v);
    }

    // bool loop = seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    bool loop = true;

    auto opts = q.options();
    auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;
    at::Tensor dk_accum, dv_accum;
    if (loop) {
        if (!deterministic) {
            dq_accum = torch::empty({batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        } else {
            const int nsplits = (get_num_sm(get_current_device()) + batch_size * num_heads - 1) / (batch_size * num_heads);
            dq_accum = torch::zeros({nsplits, batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        }
        // dk_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, opts.dtype(at::kFloat));
        // dv_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, opts.dtype(at::kFloat));
    }

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
        dv_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     dout, dq, dk_expanded, dv_expanded,
                     nullptr,
                     nullptr,
                     loop ? dq_accum.data_ptr() : nullptr,
                     // loop ? dk_accum.data_ptr() : nullptr,
                     // loop ? dv_accum.data_ptr() : nullptr,
                     nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     softmax_d.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     deterministic,
                     /*unpadded_lse*/false);

    // Standard softmax backward, optionally restricted to the CQS pair set.
    // Deliberately NOT cqsa_numden_mode: that path consumes dNum/dDen, which
    // requires Den = exp(lse) and overflows above lse ~= 88.7.
    params.cqsa_numden_mode = false;
    params.cqs_enabled = false;
    params.cqs_num_chunks = 0;
    params.cqs_owner_chunk = 0;
    params.cqs_chunk_ends = nullptr;
    params.cqs_group_bits = nullptr;
    params.cqs_blk_or = nullptr;
    params.cqs_blk_and = nullptr;
    params.cqs_blk_size = 0;
    params.cqs_num_blocks = 0;
    params.cqs_out_acc_ptr = nullptr;
    params.cqs_block_base = nullptr;
    params.cqs_seg_align = 0;
    if (cqs_group_bits_.has_value()) {
        auto bits = cqs_group_bits_.value();
        CHECK_DEVICE(bits); CHECK_CONTIGUOUS(bits);
        TORCH_CHECK(bits.dtype() == torch::kInt64, "cqs_group_bits must be int64");
        TORCH_CHECK(bits.dim() == 1 && bits.numel() == seqlen_q,
                    "cqs_group_bits must be [seqlen_q]");
        TORCH_CHECK(seqlen_q == seqlen_k, "CQS backward requires seqlen_q == seqlen_k");
        params.cqs_enabled = true;
        params.cqs_group_bits = bits.data_ptr<int64_t>();

        if (cqs_blk_or_.has_value() && cqs_blk_and_.has_value()) {
            auto blk_or = cqs_blk_or_.value();
            auto blk_and = cqs_blk_and_.value();
            CHECK_DEVICE(blk_or); CHECK_CONTIGUOUS(blk_or);
            CHECK_DEVICE(blk_and); CHECK_CONTIGUOUS(blk_and);
            TORCH_CHECK(blk_or.dtype() == torch::kInt64 && blk_and.dtype() == torch::kInt64,
                        "cqs block summaries must be int64");
            TORCH_CHECK(blk_or.numel() == blk_and.numel(),
                        "cqs_blk_or and cqs_blk_and must have equal length");
            const int blk_size = static_cast<int>(cqs_blk_size_.value_or(0));
            TORCH_CHECK(blk_size > 0, "cqs_blk_size must be positive when summaries are provided");
            // next/ v5: the forward kernel's tile verdicts use a compile-time
            // block size (kCqsBlk in mask.h) so the index math is shifts.
            TORCH_CHECK(blk_size == 64, "cqs_blk_size must be 64 (CQS_BLK_SIZE) for this kernel build");
            TORCH_CHECK((int64_t)blk_or.numel() * blk_size >= (int64_t)seqlen_q,
                        "cqs block summaries too short for seqlen_q");
            params.cqs_blk_or = blk_or.data_ptr<int64_t>();
            params.cqs_blk_and = blk_and.data_ptr<int64_t>();
            params.cqs_blk_size = blk_size;
            params.cqs_num_blocks = static_cast<int>(blk_or.numel());
        }
    }

    params.cqsa_ext_dpsum = false;
    if (dsoftmax_sum_.has_value()) {
        auto dps = dsoftmax_sum_.value();
        CHECK_DEVICE(dps);
        TORCH_CHECK(dps.dtype() == torch::kFloat32, "dsoftmax_sum must be fp32");
        TORCH_CHECK(dps.dim() == 3 && dps.size(0) == batch_size
                        && dps.size(1) == num_heads && dps.size(2) == seqlen_q,
                    "dsoftmax_sum must be [batch, heads, seqlen_q]");
        // The kernel strides dsoftmax_sum by seqlen_q_rounded, so it must live
        // in a rounded buffer even though only seqlen_q entries are defined.
        softmax_d.zero_();
        softmax_d.narrow(/*dim=*/2, /*start=*/0, /*length=*/seqlen_q)
            .copy_(dps.to(torch::kFloat32));
        // The preprocess pass being skipped is also what clears dq_accum.
        dq_accum.zero_();
        params.cqsa_ext_dpsum = true;
    }
    params.dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);

    auto launch = &run_mha_bwd;

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;

    if ( rng_state.has_value() ) {
        params.rng_state = reinterpret_cast<uint64_t*>(rng_state.value().data_ptr());
    } else if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
        auto seeds = at::cuda::philox::unpack(params.philox_args);
        params.rng_state[0] = std::get<0>(seeds);
        params.rng_state[1] = std::get<1>(seeds);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (seqlen_q > 0) {
        launch(params, stream);
    } else {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        at::sum_out(dk, at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
        at::sum_out(dv, at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
    }

    return { dq, dk, dv, softmax_d };
}

std::vector<at::Tensor>
mha_varlen_bwd(const at::Tensor &dout,  // total_q x num_heads, x head_size
               const at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &out,   // total_q x num_heads x head_size
               const at::Tensor &softmax_lse,    // h x total_q, softmax logsumexp
               std::optional<at::Tensor> &dq_,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &dk_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &dv_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               const int max_seqlen_q,
               const int max_seqlen_k,          // max sequence length to choose the kernel
               const float p_dropout,         // probability to drop
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool deterministic,
               std::optional<at::Generator> gen_,
               std::optional<at::Tensor> &rng_state) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        TORCH_CHECK(false, "This flash attention build does not support backward.");
    #endif
    if (is_causal) { window_size_right = 0; }

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    bool is_dropout = p_dropout > 0.0;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    const int total_q = sizes[0];
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int num_heads = sizes[1];
    const int head_size = sizes[2];
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    CHECK_SHAPE(out, total_q, num_heads, head_size);
    CHECK_SHAPE(dout, total_q, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, total_q, num_heads, head_size);
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, total_k, num_heads_k, head_size);
    } else {
        dv = torch::empty_like(v);
    }

    // bool loop = max_seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    bool loop = true;

    auto opts = q.options();
    auto softmax_d = torch::empty({num_heads, total_q + 128 * batch_size}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;
    if (loop) {
        // We don't want to allocate dq_accum of size (batch, seqlen_q_rounded, num_heads, head_size_rounded)
        // because that would be too large if there is a very long sequence and the rest of the sequences are short.
        // Instead, we allocate dq_accum of size (total_q + 128 * batch, num_heads, head_size_rounded).
        // Note that 128 is the max block size on the seqlen_q dimension.
        // For dQ, the i-th sequence is stored in indices from cu_seqlens[i] + 128 * i to
        // cu_seqlens[i + 1] * 128 * i - 1. This ensures that the i-th sequence and (i + 1)-th sequence will
        // be at least 128 apart. It's ok for us to do atomicAdds up to 128 rows beyond what we're normally
        // allowed to do. So we won't have to do any bound checking, and performance should stay the same.
        // Same holds for softmax_d, since LSE is stored in unpadded format.
        if (!deterministic) {
            dq_accum = torch::empty({total_q + 128 * batch_size, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        } else {
            const int nsplits = (get_num_sm(get_current_device()) + batch_size * num_heads - 1) / (batch_size * num_heads);
            dq_accum = torch::zeros({nsplits, total_q + 128 * batch_size, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        }
    }

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_expanded = torch::empty({total_k, num_heads, head_size}, opts);
        dv_expanded = torch::empty({total_k, num_heads, head_size}, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    if( zero_tensors ) {
        dq.zero_();
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     dout, dq, dk_expanded, dv_expanded,
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     loop ? dq_accum.data_ptr() : nullptr,
                     nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     softmax_d.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     deterministic,
                     /*unpadded_lse*/true);
    params.dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);
    params.total_q = total_q;

    auto launch = &run_mha_bwd;

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;

    if ( rng_state.has_value() ) {
        params.rng_state = reinterpret_cast<uint64_t*>(rng_state.value().data_ptr());
    } else if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
        auto seeds = at::cuda::philox::unpack(params.philox_args);
        params.rng_state[0] = std::get<0>(seeds);
        params.rng_state[1] = std::get<1>(seeds);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (max_seqlen_q > 0) {
        launch(params, stream);
    } else {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        at::sum_out(dk, at::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
        at::sum_out(dv, at::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
    }

    return { dq, dk, dv, softmax_d };
}

#endif

std::vector<at::Tensor>
mha_fwd_kvcache(at::Tensor &q,                 // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &vcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                std::optional<const at::Tensor> &k_, // batch_size x seqlen_knew x num_heads_k x head_size
                std::optional<const at::Tensor> &v_, // batch_size x seqlen_knew x num_heads_k x head_size
                std::optional<const at::Tensor> &seqlens_k_, // batch_size
                std::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                std::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                std::optional<const at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
                std::optional<const at::Tensor> &leftpad_k_, // batch_size
                std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
                std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits
                ) {

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(vcache.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        TORCH_CHECK(!cache_batch_idx_.has_value(), "Paged KVcache does not support cache_batch_idx");
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size_og = sizes[3];

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : kcache.size(0);
    const int page_block_size = !paged_KV ? 1 : kcache.size(1);
    TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");
    const int seqlen_k = !paged_KV ? kcache.size(1) : max_num_blocks_per_seq * page_block_size;
    const int num_heads_k = kcache.size(2);
    const int batch_size_c = !paged_KV ? kcache.size(0) : batch_size;
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size_og % 8 == 0 && !alibi_slopes_.has_value();
    if (seqlenq_ngroups_swapped) {
        const int ngroups = num_heads / num_heads_k;
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
    if (!paged_KV) {
        CHECK_SHAPE(kcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
        CHECK_SHAPE(vcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
    } else {
        CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(vcache, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    }

    at::Tensor q_padded, kcache_padded, vcache_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        kcache_padded = torch::nn::functional::pad(kcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        vcache_padded = torch::nn::functional::pad(vcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        kcache_padded = kcache;
        vcache_padded = vcache;
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
        if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
    } else {
        out = torch::empty_like(q_padded);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    auto opts = q.options();

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, kcache_padded, vcache_padded, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     /*p_d=*/nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap
                     );

    at::Tensor k, v, k_padded, v_padded;
    if (k_.has_value()) {
        TORCH_CHECK(v_.has_value(), "If key is supplied, value must also be passed in");
        TORCH_CHECK(seqlens_k_.has_value(), "If key is supplied, seqlens_k must also be passed in");
        TORCH_CHECK(seqlen_q <= seqlen_k, "If key is supplied, it must have seqlen <= the seqlen of the KV cache");
        k = k_.value();
        v = v_.value();
        TORCH_CHECK(k.dtype() == q_dtype, "Key must have the same dtype as query");
        TORCH_CHECK(v.dtype() == q_dtype, "Value must have the same dtype as query");
        CHECK_DEVICE(k); CHECK_DEVICE(v);
        TORCH_CHECK(k.stride(-1) == 1, "Key tensor must have contiguous last dimension");
        TORCH_CHECK(v.stride(-1) == 1, "Value tensor must have contiguous last dimension");
        int seqlen_knew = k.size(1);
        CHECK_SHAPE(k, batch_size, seqlen_knew, num_heads_k, head_size_og);
        CHECK_SHAPE(v, batch_size, seqlen_knew, num_heads_k, head_size_og);
        if (head_size_og % 8 != 0) {
            k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
            v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        } else {
            k_padded = k;
            v_padded = v;
        }
        params.seqlen_knew = seqlen_knew;
        params.knew_ptr = k_padded.data_ptr();
        params.vnew_ptr = v_padded.data_ptr();
        // All stride are in elements, not bytes.
        params.knew_batch_stride = k_padded.stride(0);
        params.vnew_batch_stride = v_padded.stride(0);
        params.knew_row_stride = k_padded.stride(-3);
        params.vnew_row_stride = v_padded.stride(-3);
        params.knew_head_stride = k_padded.stride(-2);
        params.vnew_head_stride = v_padded.stride(-2);
    }

    if (seqlens_k_.has_value()) {
        auto seqlens_k = seqlens_k_.value();
        TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
        CHECK_DEVICE(seqlens_k);
        CHECK_CONTIGUOUS(seqlens_k);
        CHECK_SHAPE(seqlens_k, batch_size);
        params.cu_seqlens_k = static_cast<int *>(seqlens_k.data_ptr());
    }
    params.is_seqlens_k_cumulative = !(seqlens_k_.has_value());
    if (leftpad_k_.has_value()) {
        TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
        auto leftpad_k = leftpad_k_.value();
        TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
        CHECK_DEVICE(leftpad_k);
        CHECK_CONTIGUOUS(leftpad_k);
        CHECK_SHAPE(leftpad_k, batch_size);
        params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());
    }

    if (rotary_cos_.has_value()) {
        TORCH_CHECK(k_.has_value(), "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided");
        auto rotary_cos = rotary_cos_.value();
        CHECK_DEVICE(rotary_cos);
        params.rotary_dim = rotary_cos.size(1) * 2;
        TORCH_CHECK(params.rotary_dim <= head_size, "rotary_dim must be <= headdim");
        TORCH_CHECK(params.rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
        const int seqlen_ro = rotary_cos.size(0);
        TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
        CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
        CHECK_CONTIGUOUS(rotary_cos);
        TORCH_CHECK(rotary_cos.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");

        TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
        auto rotary_sin = rotary_sin_.value();
        CHECK_DEVICE(rotary_sin);
        CHECK_SHAPE(rotary_sin, seqlen_ro, params.rotary_dim / 2);
        CHECK_CONTIGUOUS(rotary_sin);
        TORCH_CHECK(rotary_sin.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");
        params.rotary_cos_ptr = rotary_cos.data_ptr();
        params.rotary_sin_ptr = rotary_sin.data_ptr();
        params.is_rotary_interleaved = is_rotary_interleaved;
    } else {
        params.rotary_dim = 0;
    }

    if (cache_batch_idx_.has_value()) {
        auto cache_batch_idx = cache_batch_idx_.value();
        CHECK_DEVICE(cache_batch_idx);
        CHECK_CONTIGUOUS(cache_batch_idx);
        TORCH_CHECK(cache_batch_idx.scalar_type() == torch::kInt32, "cache_batch_idx must have dtype int32");
        params.cache_batch_idx = reinterpret_cast<int *>(cache_batch_idx.data_ptr());
    }

    // Keep references to these tensors to extend their lifetime
    at::Tensor softmax_lse_accum, out_accum;
    std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
        params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
        head_size_rounded, /*dropout*/ 0.f, num_splits, get_num_sm(get_current_device()), opts);

    if (paged_KV) {
        params.block_table = block_table.data_ptr<int>();
        params.block_table_batch_stride = block_table.stride(0);
    }
    params.page_block_size = page_block_size;


    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // Only split kernel supports appending to KV cache, or indexing to the cache with cache_batch_idx,
    // or paged KV cache
    run_mha_fwd(params, stream, /*force_split_kernel=*/k_.has_value() || cache_batch_idx_.has_value() || paged_KV);

    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) { out_.value().copy_(out); }
        if (k_.has_value()) {
            // It's expensive to copy the KV cache here for the case where head size not divisible by 8,
            // but we don't expect to get this case in practice. This is just so that the code works for that case.
            kcache.copy_(kcache_padded.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)}));
            vcache.copy_(vcache_padded.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)}));
        }
    }

    if (seqlenq_ngroups_swapped) {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }
    return {out, softmax_lse};
}

// ============================================================================
// next/native: WAVE mode -- several CQS subproblems in ONE launch.
//
// Layout is FlashAttention's varlen batch: subproblem w owns packed rows
// [cu_seqlens[w], cu_seqlens[w+1]) of every per-token array (bits, out, lse,
// dq/dk/dv ...). Q/K/V are either the packed wave itself, or -- with
// block_base -- any token-major [R, H, D] tensor (the ORIGINAL sequence, a
// chunk pool) that the subproblems address in place; block_base entries are
// relative to cu_seqlens[w], so q_ptr + cu_seqlens[w]*row_stride + delta lands
// on the global row. The forward hands over only the fp32 output and the lse.
// ============================================================================
void wave_merge_cuda(at::Tensor acc, at::Tensor acc_l, at::Tensor acc_m,
                     const at::Tensor out_pack, const at::Tensor lse_pack,
                     const at::Tensor order, const at::Tensor seg, const at::Tensor uniq, const int S);
void wave_scatter_add_cuda(at::Tensor dst, const at::Tensor src_pack,
                           const at::Tensor order, const at::Tensor seg, const at::Tensor uniq);

static void wave_check_tables(const at::Tensor &cu_seqlens, const at::Tensor &bits, const at::Tensor &blk_or,
                              const at::Tensor &blk_and, const at::Tensor &blk_cu, const int total_q) {
    CHECK_DEVICE(cu_seqlens); CHECK_CONTIGUOUS(cu_seqlens);
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32 && cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2, "cu_seqlens [W+1] int32");
    CHECK_DEVICE(bits); CHECK_CONTIGUOUS(bits);
    TORCH_CHECK(bits.dtype() == torch::kInt64 && bits.numel() == total_q, "bits must be [total] int64");
    CHECK_DEVICE(blk_or); CHECK_CONTIGUOUS(blk_or); CHECK_DEVICE(blk_and); CHECK_CONTIGUOUS(blk_and);
    TORCH_CHECK(blk_or.dtype() == torch::kInt64 && blk_and.dtype() == torch::kInt64 && blk_or.numel() == blk_and.numel(), "summaries");
    CHECK_DEVICE(blk_cu); CHECK_CONTIGUOUS(blk_cu);
    TORCH_CHECK(blk_cu.dtype() == torch::kInt32 && blk_cu.numel() == cu_seqlens.numel(), "blk_cu [W+1] int32");
}

// Uniform layout: W subproblems, each padded to S rows (S % 128 == 0). Plain batched
// launch (b = W, seqlen = S); tables at bidb * stride. q/k/v are either the packed wave
// [W*S, H, D] (batch stride S rows) or, with block_base [W * S/128] holding the row of
// every 128-block in the tensor the kernel reads, any [R, H, D] tensor (batch stride 0).
// Padding keys are hidden by the causal mask, so causal only.
std::vector<at::Tensor>
mha_fwd_wave_uniform(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                     const int S, const int W,
                     const at::Tensor &bits, const at::Tensor &blk_or, const at::Tensor &blk_and,
                     std::optional<at::Tensor> block_base_, const int seg_align,
                     const float softmax_scale, const bool is_causal,
                     std::optional<at::Tensor> out_acc_, std::optional<at::Tensor> lse_) {
    TORCH_CHECK(is_causal, "the uniform wave layout is causal only (padding keys rely on the causal mask)");
    TORCH_CHECK(S > 0 && S % 128 == 0 && W > 0, "uniform layout: S must be a positive multiple of 128");
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16, "fp16/bf16 only");
    TORCH_CHECK(k.dtype() == q_dtype && v.dtype() == q_dtype, "dtype mismatch");
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q/k/v must be [R, H, D]");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1, "last dim must be contiguous");
    const int num_heads = q.size(1);
    const int head_size = q.size(2);
    TORCH_CHECK(k.size(1) == num_heads && v.size(1) == num_heads, "wave mode: H_k must equal H");
    TORCH_CHECK(k.size(2) == head_size && v.size(2) == head_size, "head dim mismatch");
    TORCH_CHECK(head_size % 8 == 0 && head_size <= 256, "head dim");
    const int64_t total = (int64_t)W * S;
    const int nblk = S / 64, nbb = S / 128;
    CHECK_DEVICE(bits); CHECK_CONTIGUOUS(bits);
    TORCH_CHECK(bits.dtype() == torch::kInt64 && bits.numel() == total, "bits must be [W*S] int64");
    CHECK_DEVICE(blk_or); CHECK_CONTIGUOUS(blk_or); CHECK_DEVICE(blk_and); CHECK_CONTIGUOUS(blk_and);
    TORCH_CHECK(blk_or.dtype() == torch::kInt64 && blk_and.dtype() == torch::kInt64, "summaries int64");
    TORCH_CHECK(blk_or.numel() == (int64_t)W * nblk && blk_and.numel() == (int64_t)W * nblk, "summaries must be [W * S/64]");
    const bool in_place = block_base_.has_value();
    if (in_place) {
        auto bb = block_base_.value();
        CHECK_DEVICE(bb); CHECK_CONTIGUOUS(bb);
        TORCH_CHECK(bb.dtype() == torch::kInt32 && bb.numel() == (int64_t)W * nbb, "block_base must be [W * S/128] int32");
        TORCH_CHECK(seg_align == 128, "uniform layout: seg_align must be 128");
    } else {
        TORCH_CHECK(q.size(0) == total && k.size(0) == total && v.size(0) == total, "packed q/k/v must have W*S rows");
    }
    auto opts = q.options();
    at::Tensor out_acc;
    if (out_acc_.has_value()) {
        out_acc = out_acc_.value();
        CHECK_DEVICE(out_acc); CHECK_CONTIGUOUS(out_acc);
        TORCH_CHECK(out_acc.dtype() == torch::kFloat32, "out_acc fp32");
        CHECK_SHAPE(out_acc, total, num_heads, head_size);
    } else {
        out_acc = torch::empty({total, num_heads, head_size}, opts.dtype(at::kFloat));
    }
    at::Tensor softmax_lse;
    if (lse_.has_value()) {
        softmax_lse = lse_.value();
        CHECK_DEVICE(softmax_lse); CHECK_CONTIGUOUS(softmax_lse);
        TORCH_CHECK(softmax_lse.dtype() == torch::kFloat32, "lse fp32");
        CHECK_SHAPE(softmax_lse, W, num_heads, S);
    } else {
        softmax_lse = torch::empty({W, num_heads, S}, opts.dtype(at::kFloat));
    }
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    auto out4 = out_acc.view({W, S, num_heads, head_size});   // lends the batch/row/head strides
    Flash_fwd_params params;
    set_params_fprop(params, W, S, S, S, S, num_heads, num_heads, head_size, head_size_rounded,
                     q, k, v, out4, nullptr, nullptr, nullptr, nullptr, softmax_lse.data_ptr(),
                     0.f, softmax_scale, -1, 0, 0.f, /*seqlenq_ngroups_swapped=*/false, /*unpadded_lse=*/false);
    // batch stride: the packed wave advances S rows per subproblem; in place, every
    // subproblem reads the same tensor and the block map does the addressing.
    params.q_batch_stride = in_place ? 0 : (int64_t)S * params.q_row_stride;
    params.k_batch_stride = in_place ? 0 : (int64_t)S * params.k_row_stride;
    params.v_batch_stride = in_place ? 0 : (int64_t)S * params.v_row_stride;
    params.o_ptr = nullptr;                              // fp32 copy only
    params.cqs_out_acc_ptr = out_acc.data_ptr();
    params.cqs_enabled = true;
    params.cqs_num_chunks = 0; params.cqs_owner_chunk = 0; params.cqs_chunk_ends = nullptr;
    params.cqs_group_bits = bits.data_ptr<int64_t>();
    params.cqs_blk_or = blk_or.data_ptr<int64_t>();
    params.cqs_blk_and = blk_and.data_ptr<int64_t>();
    params.cqs_blk_size = 64;
    params.cqs_num_blocks = nblk;                        // per subproblem
    params.cqs_wave = true;
    params.cqs_blk_cu = nullptr; params.cqs_bb_cu = nullptr;
    params.cqs_wave_tok_stride = S;
    params.cqs_wave_blk_stride = nblk;
    params.cqs_wave_bb_stride = nbb;
    params.cqs_block_base = in_place ? block_base_.value().data_ptr<int>() : nullptr;
    params.cqs_seg_align = in_place ? seg_align : 0;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_fwd(params, stream);
    return {out_acc, softmax_lse};
}

std::vector<at::Tensor>
mha_fwd_wave(const at::Tensor &q,            // [R, H, D] token-major, last dim contiguous
             const at::Tensor &k,
             const at::Tensor &v,
             const at::Tensor &cu_seqlens,   // [W+1] int32
             const int max_seqlen,
             const int total_q,              // cu_seqlens[W]
             const at::Tensor &bits,         // [total] int64
             const at::Tensor &blk_or,       // packed 64-token summaries
             const at::Tensor &blk_and,
             const at::Tensor &blk_cu,       // [W+1] int32
             std::optional<at::Tensor> block_base_,   // packed int32, relative to cu_seqlens[w]
             std::optional<at::Tensor> bb_cu_,        // [W+1] int32
             const int seg_align,
             const float softmax_scale,
             const bool is_causal,
             std::optional<at::Tensor> out_acc_,      // [total, H, D] fp32 (allocated if absent)
             std::optional<at::Tensor> lse_,          // [H, total] fp32  (uniform: [W, H, S])
             const int uniform_S,                     // > 0: uniform layout (causal), W subproblems of S rows each
             const int uniform_W) {
    at::cuda::CUDAGuard device_guard{q.device()};
    if (uniform_S > 0) {
        return mha_fwd_wave_uniform(q, k, v, uniform_S, uniform_W, bits, blk_or, blk_and, block_base_, seg_align,
                                    softmax_scale, is_causal, out_acc_, lse_);
    }
    TORCH_CHECK(!is_causal, "causal wave launches use the uniform layout (uniform_S > 0); the varlen layout is non-causal only");
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16, "fp16/bf16 only");
    TORCH_CHECK(k.dtype() == q_dtype && v.dtype() == q_dtype, "dtype mismatch");
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q/k/v must be [R, H, D]");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1, "last dim must be contiguous");
    const int num_heads = q.size(1);
    const int head_size = q.size(2);
    TORCH_CHECK(k.size(1) == num_heads && v.size(1) == num_heads, "wave mode: H_k must equal H");
    TORCH_CHECK(k.size(2) == head_size && v.size(2) == head_size, "head dim mismatch");
    TORCH_CHECK(head_size % 8 == 0 && head_size <= 256, "head dim");
    const int W = cu_seqlens.numel() - 1;
    wave_check_tables(cu_seqlens, bits, blk_or, blk_and, blk_cu, total_q);
    if (!block_base_.has_value()) {
        TORCH_CHECK(q.size(0) == total_q && k.size(0) == total_q && v.size(0) == total_q, "packed q/k/v must have total rows");
    } else {
        auto bb = block_base_.value();
        CHECK_DEVICE(bb); CHECK_CONTIGUOUS(bb);
        TORCH_CHECK(bb.dtype() == torch::kInt32, "block_base must be int32");
        TORCH_CHECK(bb_cu_.has_value(), "bb_cu required with block_base");
        CHECK_DEVICE(bb_cu_.value()); CHECK_CONTIGUOUS(bb_cu_.value());
        TORCH_CHECK(bb_cu_.value().dtype() == torch::kInt32 && bb_cu_.value().numel() == W + 1, "bb_cu [W+1] int32");
        TORCH_CHECK(seg_align > 0 && seg_align % 128 == 0, "seg_align must be a positive multiple of 128");
    }

    auto opts = q.options();
    at::Tensor out_acc;
    if (out_acc_.has_value()) {
        out_acc = out_acc_.value();
        CHECK_DEVICE(out_acc); CHECK_CONTIGUOUS(out_acc);
        TORCH_CHECK(out_acc.dtype() == torch::kFloat32, "out_acc fp32");
        CHECK_SHAPE(out_acc, total_q, num_heads, head_size);
    } else {
        out_acc = torch::empty({total_q, num_heads, head_size}, opts.dtype(at::kFloat));
    }
    at::Tensor softmax_lse;
    if (lse_.has_value()) {
        softmax_lse = lse_.value();
        CHECK_DEVICE(softmax_lse); CHECK_CONTIGUOUS(softmax_lse);
        TORCH_CHECK(softmax_lse.dtype() == torch::kFloat32, "lse fp32");
        CHECK_SHAPE(softmax_lse, num_heads, total_q);
    } else {
        softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_rounded = round_multiple(max_seqlen, 128);

    Flash_fwd_params params;
    set_params_fprop(params, W, max_seqlen, max_seqlen, seqlen_rounded, seqlen_rounded,
                     num_heads, num_heads, head_size, head_size_rounded,
                     q, k, v, out_acc,                   // out_acc only lends its strides
                     cu_seqlens.data_ptr(), cu_seqlens.data_ptr(), nullptr,
                     nullptr, softmax_lse.data_ptr(),
                     0.f, softmax_scale, -1, is_causal ? 0 : -1, 0.f,
                     /*seqlenq_ngroups_swapped=*/false, /*unpadded_lse=*/true);
    params.total_q = total_q;
    params.o_ptr = nullptr;                              // fp32 copy only
    params.cqs_out_acc_ptr = out_acc.data_ptr();
    params.cqs_enabled = true;
    params.cqs_num_chunks = 0; params.cqs_owner_chunk = 0; params.cqs_chunk_ends = nullptr;
    params.cqs_group_bits = bits.data_ptr<int64_t>();
    params.cqs_blk_or = blk_or.data_ptr<int64_t>();
    params.cqs_blk_and = blk_and.data_ptr<int64_t>();
    params.cqs_blk_size = 64;
    params.cqs_num_blocks = static_cast<int>(blk_or.numel());
    params.cqs_wave = true;
    params.cqs_wave_tok_stride = 0; params.cqs_wave_blk_stride = 0; params.cqs_wave_bb_stride = 0;
    params.cqs_blk_cu = blk_cu.data_ptr<int>();
    params.cqs_block_base = block_base_.has_value() ? block_base_.value().data_ptr<int>() : nullptr;
    params.cqs_bb_cu = bb_cu_.has_value() ? bb_cu_.value().data_ptr<int>() : nullptr;
    params.cqs_seg_align = block_base_.has_value() ? seg_align : 0;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (max_seqlen > 0) { run_mha_fwd(params, stream); }
    return {out_acc, softmax_lse};
}

#ifndef FLASHATTENTION_DISABLE_BACKWARD
std::vector<at::Tensor>
mha_bwd_wave(const at::Tensor &dout,         // [total, H, D] packed
             const at::Tensor &q,            // [total, H, D] packed
             const at::Tensor &k,
             const at::Tensor &v,
             const at::Tensor &softmax_lse,  // [H, total] fp32, the GLOBAL lse gathered per packed token
             const at::Tensor &dpsum,        // [H, total + 128*W] fp32: rowsum(dO*O) at cu_seqlens[w] + 128*w
             const at::Tensor &cu_seqlens,   // [W+1] int32
             const int max_seqlen,
             const int total_q,
             const at::Tensor &bits,
             const at::Tensor &blk_or,
             const at::Tensor &blk_and,
             const at::Tensor &blk_cu,
             const float softmax_scale,
             const bool is_causal,
             std::optional<at::Tensor> dq_, std::optional<at::Tensor> dk_, std::optional<at::Tensor> dv_) {
    at::cuda::CUDAGuard device_guard{q.device()};
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16, "fp16/bf16 only");
    TORCH_CHECK(k.dtype() == q_dtype && v.dtype() == q_dtype && dout.dtype() == q_dtype, "dtype mismatch");
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse); CHECK_DEVICE(dpsum);
    TORCH_CHECK(q.dim() == 3, "q [total, H, D]");
    const int num_heads = q.size(1);
    const int head_size = q.size(2);
    TORCH_CHECK(head_size % 8 == 0 && head_size <= 256, "head dim");
    const int W = cu_seqlens.numel() - 1;
    wave_check_tables(cu_seqlens, bits, blk_or, blk_and, blk_cu, total_q);
    CHECK_SHAPE(q, total_q, num_heads, head_size); CHECK_SHAPE(k, total_q, num_heads, head_size);
    CHECK_SHAPE(v, total_q, num_heads, head_size); CHECK_SHAPE(dout, total_q, num_heads, head_size);
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1 && dout.stride(-1) == 1, "last dim contiguous");
    CHECK_CONTIGUOUS(softmax_lse); CHECK_SHAPE(softmax_lse, num_heads, total_q);
    TORCH_CHECK(softmax_lse.dtype() == torch::kFloat32 && dpsum.dtype() == torch::kFloat32, "lse/dpsum fp32");
    CHECK_CONTIGUOUS(dpsum); CHECK_SHAPE(dpsum, num_heads, total_q + 128 * W);

    auto opts = q.options();
    at::Tensor dq = dq_.has_value() ? dq_.value() : torch::empty_like(q);
    at::Tensor dk = dk_.has_value() ? dk_.value() : torch::empty_like(k);
    at::Tensor dv = dv_.has_value() ? dv_.value() : torch::empty_like(v);
    CHECK_SHAPE(dq, total_q, num_heads, head_size); CHECK_SHAPE(dk, total_q, num_heads, head_size); CHECK_SHAPE(dv, total_q, num_heads, head_size);
    TORCH_CHECK(dq.stride(-1) == 1 && dk.stride(-1) == 1 && dv.stride(-1) == 1, "grad last dim contiguous");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_rounded = round_multiple(max_seqlen, 128);
    // dq is accumulated in fp32 with atomics, per packed row, 128 rows of slack per subproblem
    // (see mha_varlen_bwd). The caller-supplied dpsum means the preprocess pass -- which would
    // have zeroed it -- is skipped, so zero it here.
    auto dq_accum = torch::zeros({total_q + 128 * W, num_heads, head_size_rounded}, opts.dtype(at::kFloat));

    Flash_bwd_params params;
    set_params_dgrad(params, W, max_seqlen, max_seqlen, seqlen_rounded, seqlen_rounded,
                     num_heads, num_heads, head_size, head_size_rounded,
                     q, k, v, dout /* stands in for `out`, never read */,
                     dout, dq, dk, dv,
                     cu_seqlens.data_ptr(), cu_seqlens.data_ptr(),
                     dq_accum.data_ptr(), nullptr, nullptr,
                     softmax_lse.data_ptr(), dpsum.data_ptr(),
                     0.f, softmax_scale, -1, is_causal ? 0 : -1, 0.f,
                     /*deterministic=*/false, /*unpadded_lse=*/true);
    params.dq_accum_split_stride = 0;
    params.total_q = total_q;
    params.cqsa_numden_mode = false;
    params.cqsa_ext_dpsum = true;
    params.cqs_enabled = true;
    params.cqs_num_chunks = 0; params.cqs_owner_chunk = 0; params.cqs_chunk_ends = nullptr;
    params.cqs_group_bits = bits.data_ptr<int64_t>();
    params.cqs_blk_or = blk_or.data_ptr<int64_t>();
    params.cqs_blk_and = blk_and.data_ptr<int64_t>();
    params.cqs_blk_size = 64;
    params.cqs_num_blocks = static_cast<int>(blk_or.numel());
    params.cqs_out_acc_ptr = nullptr;
    params.cqs_block_base = nullptr; params.cqs_seg_align = 0;
    params.cqs_wave = true;
    params.cqs_wave_tok_stride = 0; params.cqs_wave_blk_stride = 0; params.cqs_wave_bb_stride = 0;
    params.cqs_blk_cu = blk_cu.data_ptr<int>();
    params.cqs_bb_cu = nullptr;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (max_seqlen > 0) { run_mha_bwd(params, stream); }
    return {dq, dk, dv};
}
#endif

} // namespace FLASH_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    m.def("fwd_wave", &FLASH_NAMESPACE::mha_fwd_wave, "next/native: CQS forward, several subproblems per launch",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"), pybind11::arg("cu_seqlens"), pybind11::arg("max_seqlen"),
          pybind11::arg("total_q"), pybind11::arg("bits"), pybind11::arg("blk_or"), pybind11::arg("blk_and"), pybind11::arg("blk_cu"),
          pybind11::arg("block_base") = std::nullopt, pybind11::arg("bb_cu") = std::nullopt, pybind11::arg("seg_align") = 0,
          pybind11::arg("softmax_scale"), pybind11::arg("is_causal"),
          pybind11::arg("out_acc") = std::nullopt, pybind11::arg("lse") = std::nullopt,
          pybind11::arg("uniform_S") = 0, pybind11::arg("uniform_W") = 0);
#ifndef FLASHATTENTION_DISABLE_BACKWARD
    m.def("bwd_wave", &FLASH_NAMESPACE::mha_bwd_wave, "next/native: CQS backward (global lse), several subproblems per launch",
          pybind11::arg("dout"), pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"), pybind11::arg("softmax_lse"), pybind11::arg("dpsum"),
          pybind11::arg("cu_seqlens"), pybind11::arg("max_seqlen"), pybind11::arg("total_q"),
          pybind11::arg("bits"), pybind11::arg("blk_or"), pybind11::arg("blk_and"), pybind11::arg("blk_cu"),
          pybind11::arg("softmax_scale"), pybind11::arg("is_causal"),
          pybind11::arg("dq") = std::nullopt, pybind11::arg("dk") = std::nullopt, pybind11::arg("dv") = std::nullopt);
#endif
    m.def("wave_merge", &FLASH_NAMESPACE::wave_merge_cuda, "next/native: deterministic batched merge of (out, lse) into the accumulator",
          pybind11::arg("acc"), pybind11::arg("acc_l"), pybind11::arg("acc_m"), pybind11::arg("out_pack"), pybind11::arg("lse_pack"),
          pybind11::arg("order"), pybind11::arg("seg"), pybind11::arg("uniq"), pybind11::arg("S") = 0);
    m.def("wave_scatter_add", &FLASH_NAMESPACE::wave_scatter_add_cuda, "next/native: deterministic batched scatter-add");
    m.def("fwd", &FLASH_NAMESPACE::mha_fwd, "Forward pass");
    m.def("fwd_cqs", &FLASH_NAMESPACE::mha_fwd_cqs, "Forward pass with deterministic CQS mask");
    m.def("fwd_cqs_group_bits", &FLASH_NAMESPACE::mha_fwd_cqs_group_bits,
          "Forward pass with per-token CQS group-bit mask",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"), pybind11::arg("out_"), pybind11::arg("alibi_slopes_"),
          pybind11::arg("p_dropout"), pybind11::arg("softmax_scale"), pybind11::arg("is_causal"),
          pybind11::arg("window_size_left"), pybind11::arg("window_size_right"), pybind11::arg("softcap"),
          pybind11::arg("return_softmax"), pybind11::arg("gen_"), pybind11::arg("cqs_group_bits"),
          pybind11::arg("cqs_blk_or") = pybind11::none(), pybind11::arg("cqs_blk_and") = pybind11::none(),
          pybind11::arg("cqs_blk_size") = pybind11::none(),
          pybind11::arg("cqs_out_acc") = pybind11::none(),
          pybind11::arg("cqs_block_base") = pybind11::none(),
          pybind11::arg("cqs_seg_align") = pybind11::none());
    m.def("bwd_cqs_group_bits", &FLASH_NAMESPACE::mha_bwd_cqs_group_bits, "Backward pass with per-token CQS group-bit mask");
    m.def("fwd_cqsa", &FLASH_NAMESPACE::mha_fwd_cqsa, "Forward pass with fused CQSA (num_chunk=7, interest=(0,1,3), configurable num_itr)");
    m.def("fwd_cqsa_profile", &FLASH_NAMESPACE::mha_fwd_cqsa_profile, "Forward pass with fused CQSA + host timing + configurable num_itr");
    m.def("varlen_fwd", &FLASH_NAMESPACE::mha_varlen_fwd, "Forward pass (variable length)");
#ifndef FLASHATTENTION_DISABLE_BACKWARD
    m.def("bwd", &FLASH_NAMESPACE::mha_bwd, "Backward pass",
          pybind11::arg("dout"), pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
          pybind11::arg("out"), pybind11::arg("softmax_lse"),
          pybind11::arg("dq_"), pybind11::arg("dk_"), pybind11::arg("dv_"),
          pybind11::arg("alibi_slopes_"), pybind11::arg("p_dropout"),
          pybind11::arg("softmax_scale"), pybind11::arg("is_causal"),
          pybind11::arg("window_size_left"), pybind11::arg("window_size_right"),
          pybind11::arg("softcap"), pybind11::arg("deterministic"),
          pybind11::arg("gen_"), pybind11::arg("rng_state"),
          pybind11::arg("cqs_group_bits") = pybind11::none(),
          pybind11::arg("cqs_blk_or") = pybind11::none(),
          pybind11::arg("cqs_blk_and") = pybind11::none(),
          pybind11::arg("cqs_blk_size") = pybind11::none(),
          pybind11::arg("dsoftmax_sum") = pybind11::none());
    m.def("varlen_bwd", &FLASH_NAMESPACE::mha_varlen_bwd, "Backward pass (variable length)");
#endif
    m.def("fwd_kvcache", &FLASH_NAMESPACE::mha_fwd_kvcache, "Forward pass, with KV-cache");
}
