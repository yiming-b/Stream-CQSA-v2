/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"

#include <cuda.h>
#include <vector>

#include <ATen/cuda/CUDAGeneratorImpl.h> // For at::Generator and at::PhiloxCudaState

namespace FLASH_NAMESPACE {
constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the P matrix.
    void * __restrict__ p_ptr;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, total_q;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;
    int * __restrict__ leftpad_k;

    // If provided, the actual length of each k sequence.
    int * __restrict__ seqused_k;

    int *__restrict__ blockmask;

    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    // The cos and sin matrices for rotary embedding.
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;

    // The indices to index into the KV cache.
    int * __restrict__ cache_batch_idx;

    // Paged KV cache
    int * __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    // uint32_t p_dropout_in_uint;
    // uint16_t p_dropout_in_uint16_t;
    uint8_t p_dropout_in_uint8_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;
    float scale_softmax_rp_dropout;

    // Local window size
    int window_size_left, window_size_right;
    float softcap;

    // Random state.
    at::PhiloxCudaState philox_args;

    // Pointer to the RNG seed (idx 0) and offset (idx 1).
    uint64_t * rng_state;

    bool is_bf16;
    bool is_causal;

    // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
    // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
    bool is_seqlens_k_cumulative;

    bool is_rotary_interleaved;

    int num_splits;  // For split-KV version

    void * __restrict__ alibi_slopes_ptr;
    index_t alibi_slopes_batch_stride;

    bool unpadded_lse;  // For varlen paths: LSE is in [nheads, total_seqlen_q] format instead of [b, nheads, seqlen_q].
    bool seqlenq_ngroups_swapped;  // q has been transposed from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d).

    // Optional deterministic CQSA masking:
    // If enabled, for each query row we mask key positions in the same non-owner chunk.
    bool cqs_enabled;
    int cqs_num_chunks;
    int cqs_owner_chunk;
    int * __restrict__ cqs_chunk_ends;
    // Optional per-token group bitmask for general CQSA masks.
    // If non-null, mask(i, j) is true when (bits[i] & bits[j]) != 0.
    int64_t * __restrict__ cqs_group_bits;
    // Per-block summaries of cqs_group_bits, cqs_blk_size tokens per entry.
    // They let a tile decide in O(1) whether it contains any masked pair at all:
    //   (or[r] & or[c]) == 0   -> nothing in this tile is masked, take the fast path
    //   (and[r] & and[c]) != 0 -> every pair in this tile is masked
    // Without this, enabling CQS costs the generic masking path on every tile
    // even when the tile has nothing to mask.
    int64_t * __restrict__ cqs_blk_or;
    int64_t * __restrict__ cqs_blk_and;
    int cqs_blk_size;
    int cqs_num_blocks;
    // Optional fp32 copy of the (normalised) attention output, for Stream-CQSA
    // recomposition. The kernel already carries acc_o in fp32 registers; the
    // stock epilogue rounds it to Element on the way out, and that rounding is
    // ~83% of a subproblem's error. Writing the fp32 value costs one extra
    // store and removes it. Same logical [B, seqlen_q, H, D] layout as `out`,
    // so the o_* element strides apply unchanged.
    void * __restrict__ cqs_out_acc_ptr;
    // Segmented input. A CQS subsequence is a concatenation of contiguous runs
    // of the ORIGINAL sequence, so Q/K/V need not be gathered into a new buffer
    // at all -- the kernel can read the originals in place. With chunk
    // boundaries aligned to cqs_seg_align (and every tile size dividing it), a
    // tile never straddles a run boundary, so this reduces to one base-pointer
    // offset per tile:
    //     global_row(r) = cqs_block_base[r / align] + (r % align)
    // cqs_block_base has ceil(L / align) entries, L being the LOCAL length.
    // Null disables it and the kernel reads a pre-gathered buffer as before.
    int * __restrict__ cqs_block_base;
    int cqs_seg_align;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_bwd_params : public Flash_fwd_params {

    // The dO and dQKV matrices.
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;

    // To accumulate dQ
    void *__restrict__ dq_accum_ptr;
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;

    // // To accumulate dK and dV in case we're splitting the bwd along seqlen_q
    // dimension void *__restrict__ dk_accum_ptr; void *__restrict__
    // dv_accum_ptr;

    // The stride between rows of the dO, dQ, dK and dV matrices.
    // TD [2022-04-16]: We're using 32-bit indexing to save registers.
    // The code probably won't work for arrays larger than 2GB.
    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;

    // The pointer to the softmax d sum.
    void *__restrict__ dsoftmax_sum;

    // Stream-CQSA Num/Den backward mode:
    // dsoftmax_sum is treated as dDen (per row), and the kernel uses
    // dR = exp(score) * (dP + dDen_row) instead of standard softmax backward.
    bool cqsa_numden_mode;
    // Caller supplied dsoftmax_sum = rowsum(dO * O) already, so the preprocess
    // pass must be skipped (and must not overwrite it). Distinct from
    // cqsa_numden_mode, which also skips the preprocess but changes the dS math.
    bool cqsa_ext_dpsum;

    bool deterministic;
    index_t dq_accum_split_stride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream);

template<typename T, int Headdim, bool Is_causal> void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream);

}  // namespace FLASH_NAMESPACE
