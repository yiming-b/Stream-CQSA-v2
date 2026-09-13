/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once
#include "namespace_config.h"

#include <cute/tensor.hpp>
#include <cstdint>

namespace FLASH_NAMESPACE {

using namespace cute;

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor, const int max_seqlen_k,
                                  const int col_idx_offset_ = 0) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * 8;
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            const int col_idx = col_idx_base + j;
            if (col_idx >= max_seqlen_k) {
                // Without the "make_coord" we get wrong results
                #pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                }
            }
        }
    }
}

template <bool HasWSLeft=true, typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_local(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset,
                                        const int max_seqlen_q, const int warp_row_stride,
                                        const int window_size_left, const int window_size_right) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const int row_idx = row_idx_base + i * 8;
            const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
            const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;
                    if (col_idx >= col_idx_limit_right || (HasWSLeft && col_idx < col_idx_limit_left)) {
                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                    }
                }
            }
            // if (cute::thread0()) {
            //     printf("mi = %d, i = %d, row_idx = %d, max_seqlen_k = %d\n", mi, i, row_idx, max_seqlen_k);
            //     print(tensor(make_coord(i, mi), _));
            //     // print(tensor(_, j + nj * size<1, 0>(tensor)));
            // }
        }
    }
}

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_causal(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                         const int max_seqlen_k, const int row_idx_offset,
                                         const int max_seqlen_q, const int warp_row_stride) {
    // Causal masking is equivalent to local masking with window_size_left = infinity and window_size_right = 0
    apply_mask_local</*HasWSLeft=*/false>(tensor, col_idx_offset_, max_seqlen_k, row_idx_offset,
                                          max_seqlen_q, warp_row_stride, -1, 0);
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void apply_mask_causal_w_idx(
    Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &idx_rowcol,
    const int col_idx_offset_, const int max_seqlen_k, const int row_idx_offset)
{
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 2, "Only support 2D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(tensor) == size<0>(idx_rowcol));
    CUTE_STATIC_ASSERT_V(size<1>(tensor) == size<1>(idx_rowcol));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const int col_idx_limit = std::min(max_seqlen_k, 1 + row_idx_offset + get<0>(idx_rowcol(mi, 0)));
        #pragma unroll
        for (int ni = 0; ni < size<1, 1>(tensor); ++ni) {
            if (col_idx_offset_ + get<1>(idx_rowcol(0, ni)) >= col_idx_limit) {
                tensor(mi, ni) = -INFINITY;
            }
        }
        // if (cute::thread0()) {
        //     printf("ni = %d, j = %d, col_idx = %d, max_seqlen_k = %d\n", ni, j, col_idx, max_seqlen_k);
        //     print(tensor(_, make_coord(j, ni)));
        //     // print(tensor(_, j + ni * size<1, 0>(tensor)));
        // }
    }
}

// Range reductions over the block summaries, as free functions.
//
// The Mask struct also exposes these as methods, which is convenient but costly
// in a register-tight kernel: holding cqs_blk_or/cqs_blk_and/cqs_blk_size/
// cqs_num_blocks as members keeps them live for the kernel's entire lifetime,
// even though they are only read once per tile. The backward sits at the
// register ceiling (REG 253-255) where that is the difference between 120 B and
// 274 B of spill per thread -- measured 7x on the kernel. Callers that are
// tight on registers compute the tile state up front with these, then pass the
// answer to apply_mask and never hold the pointers.
__forceinline__ __device__ int64_t cqs_blk_range_or(const int64_t* blk_or, const int blk_size,
                                                    const int num_blocks,
                                                    const int start, const int extent) {
    if (blk_or == nullptr || blk_size <= 0) { return -1; }   // -1: "assume set"
    const int lo = start < 0 ? 0 : start;
    const int b0 = lo / blk_size;
    int b1 = (start + extent + blk_size - 1) / blk_size;
    if (b1 > num_blocks) { b1 = num_blocks; }
    int64_t acc = 0;
    for (int b = b0; b < b1; ++b) { acc |= blk_or[b]; }
    return acc;
}

// Is every pair in this tile guaranteed unmasked? Mirrors Mask::cqs_tile_is_clear.
// AND of the group bits across a token range, from the block summaries. Used to
// decide whether an entire tile is masked out. Partial edge blocks would make
// the AND describe tokens outside the range, so callers must pass block-aligned
// ranges; cqs_tile_masked enforces that.
__forceinline__ __device__ int64_t cqs_blk_range_and(const int64_t* blk_and, const int blk_size,
                                                     const int num_blocks,
                                                     const int start, const int extent) {
    const int lo = start < 0 ? 0 : start;
    const int b0 = lo / blk_size;
    int b1 = (start + extent + blk_size - 1) / blk_size;
    if (b1 > num_blocks) { b1 = num_blocks; }
    if (b0 >= b1) { return 0; }
    int64_t acc = ~int64_t(0);
    for (int b = b0; b < b1; ++b) { acc &= blk_and[b]; }
    return acc;
}

// True when every (row, col) pair in the tile shares a group bit, i.e. the whole
// tile is masked out and contributes nothing. The caller may then skip the
// tile's GEMMs entirely.
//
// Conservative in the safe direction: any condition it cannot verify returns
// false, so it can fail to skip a skippable tile but never skips a live one.
// Unlike the OR-based clear test, this needs block-aligned ranges -- a partial
// edge block's AND summary covers tokens outside the tile and could claim a
// mask that does not hold for the tile's own rows.
__forceinline__ __device__ bool cqs_tile_masked(const int64_t* blk_and, const int blk_size,
                                                const int num_blocks,
                                                const int row_base, const int row_extent,
                                                const int col_base, const int col_extent,
                                                const int max_seqlen_q, const int max_seqlen_k) {
    if (blk_and == nullptr || blk_size <= 0) { return false; }
    if ((row_base % blk_size) != 0 || (row_extent % blk_size) != 0) { return false; }
    if ((col_base % blk_size) != 0 || (col_extent % blk_size) != 0) { return false; }
    // A tile running past the sequence end contains out-of-range rows/cols whose
    // summaries say nothing useful.
    if (row_base + row_extent > max_seqlen_q) { return false; }
    if (col_base + col_extent > max_seqlen_k) { return false; }
    const int64_t and_row = cqs_blk_range_and(blk_and, blk_size, num_blocks, row_base, row_extent);
    if (and_row == 0) { return false; }
    return (and_row & cqs_blk_range_and(blk_and, blk_size, num_blocks, col_base, col_extent)) != 0;
}

__forceinline__ __device__ bool cqs_tile_clear(const int64_t* blk_or, const int blk_size,
                                               const int num_blocks,
                                               const int row_base, const int row_extent,
                                               const int col_base, const int col_extent) {
    if (blk_or == nullptr || blk_size <= 0) { return false; }
    const int64_t row_or = cqs_blk_range_or(blk_or, blk_size, num_blocks, row_base, row_extent);
    if (row_or == 0) { return true; }
    return (row_or & cqs_blk_range_or(blk_or, blk_size, num_blocks, col_base, col_extent)) == 0;
}

// next/ v4: the row half of both tile verdicts depends only on the CTA's row
// block, so it is evaluated once per CTA and only the column half per tile.
// cqs_tile_masked_rs / cqs_tile_clear_rs return exactly what cqs_tile_masked /
// cqs_tile_clear return for the same arguments (every early-out is preserved,
// just split between the two halves).
struct CqsRowSummary {
    int64_t row_or;
    int64_t row_and;
    bool    row_may_mask;   // cqs_tile_masked's row-side preconditions all hold and row_and != 0
    bool    row_clear;      // cqs_tile_clear's "row_or == 0" early-out
};

__forceinline__ __device__ CqsRowSummary cqs_row_summary(const int64_t* blk_or, const int64_t* blk_and,
                                                         const int blk_size, const int num_blocks,
                                                         const int row_base, const int row_extent,
                                                         const int max_seqlen_q) {
    CqsRowSummary r;
    r.row_or = 0; r.row_and = 0; r.row_may_mask = false; r.row_clear = false;
    if (blk_or != nullptr && blk_size > 0) {
        r.row_or = cqs_blk_range_or(blk_or, blk_size, num_blocks, row_base, row_extent);
        r.row_clear = (r.row_or == 0);
    }
    if (blk_and != nullptr && blk_size > 0
        && (row_base % blk_size) == 0 && (row_extent % blk_size) == 0
        && row_base + row_extent <= max_seqlen_q) {
        r.row_and = cqs_blk_range_and(blk_and, blk_size, num_blocks, row_base, row_extent);
        r.row_may_mask = (r.row_and != 0);
    }
    return r;
}

__forceinline__ __device__ bool cqs_tile_masked_rs(const CqsRowSummary& r, const int64_t* blk_and,
                                                   const int blk_size, const int num_blocks,
                                                   const int col_base, const int col_extent,
                                                   const int max_seqlen_k) {
    if (!r.row_may_mask) { return false; }
    if ((col_base % blk_size) != 0 || (col_extent % blk_size) != 0) { return false; }
    if (col_base + col_extent > max_seqlen_k) { return false; }
    return (r.row_and & cqs_blk_range_and(blk_and, blk_size, num_blocks, col_base, col_extent)) != 0;
}

__forceinline__ __device__ bool cqs_tile_clear_rs(const CqsRowSummary& r, const int64_t* blk_or,
                                                  const int blk_size, const int num_blocks,
                                                  const int col_base, const int col_extent) {
    if (blk_or == nullptr || blk_size <= 0) { return false; }
    if (r.row_clear) { return true; }
    return (r.row_or & cqs_blk_range_or(blk_or, blk_size, num_blocks, col_base, col_extent)) == 0;
}

// next/ v5: compile-time block size. The Python side always builds summaries
// with CQS_BLK_SIZE == 64 and flash_api.cpp rejects anything else, so the
// block-index math below is shifts and masks instead of runtime divisions
// (the runtime-blk_size helpers above cost ~8 integer divisions per verdict).
// The column half of both verdicts is a small register struct that the KV
// loop fetches one tile ahead, so the verdict at the top of an iteration is a
// register AND, never a dependent global-load chain.
static constexpr int kCqsBlk = 64;

struct CqsColSummary {
    int64_t col_or;
    int64_t col_and;
    bool    col_ok;         // cqs_tile_masked's column-side preconditions hold
};

__forceinline__ __device__ int64_t cqs_blk_range_or_c(const int64_t* blk_or, const int num_blocks,
                                                      const int start, const int extent) {
    const int lo = start < 0 ? 0 : start;
    const int b0 = lo / kCqsBlk;
    int b1 = (start + extent + kCqsBlk - 1) / kCqsBlk;
    if (b1 > num_blocks) { b1 = num_blocks; }
    int64_t acc = 0;
    for (int b = b0; b < b1; ++b) { acc |= blk_or[b]; }
    return acc;
}

__forceinline__ __device__ int64_t cqs_blk_range_and_c(const int64_t* blk_and, const int num_blocks,
                                                       const int start, const int extent) {
    const int lo = start < 0 ? 0 : start;
    const int b0 = lo / kCqsBlk;
    int b1 = (start + extent + kCqsBlk - 1) / kCqsBlk;
    if (b1 > num_blocks) { b1 = num_blocks; }
    if (b0 >= b1) { return 0; }
    int64_t acc = ~int64_t(0);
    for (int b = b0; b < b1; ++b) { acc &= blk_and[b]; }
    return acc;
}

// Same contract as cqs_row_summary, with blk_size == kCqsBlk.
__forceinline__ __device__ CqsRowSummary cqs_row_summary_c(const int64_t* blk_or, const int64_t* blk_and,
                                                           const int num_blocks,
                                                           const int row_base, const int row_extent,
                                                           const int max_seqlen_q) {
    CqsRowSummary r;
    r.row_or = 0; r.row_and = 0; r.row_may_mask = false; r.row_clear = false;
    if (blk_or != nullptr) {
        r.row_or = cqs_blk_range_or_c(blk_or, num_blocks, row_base, row_extent);
        r.row_clear = (r.row_or == 0);
    }
    if (blk_and != nullptr
        && (row_base % kCqsBlk) == 0 && (row_extent % kCqsBlk) == 0
        && row_base + row_extent <= max_seqlen_q) {
        r.row_and = cqs_blk_range_and_c(blk_and, num_blocks, row_base, row_extent);
        r.row_may_mask = (r.row_and != 0);
    }
    return r;
}

__forceinline__ __device__ CqsColSummary cqs_col_summary_c(const int64_t* blk_or, const int64_t* blk_and,
                                                           const int num_blocks,
                                                           const int col_base, const int col_extent,
                                                           const int max_seqlen_k) {
    CqsColSummary c;
    c.col_or = 0; c.col_and = 0; c.col_ok = false;
    // next/ v7 fast path: a full, block-aligned tile spanning exactly two
    // summary blocks (kBlockN == 2 * kCqsBlk) that lies inside the sequence.
    // Then b1 == b0 + 2 <= num_blocks, both range reductions are two adjacent
    // words, and each is one 16-byte load when the address allows it. This is
    // every tile but the ragged last one, and it returns exactly what the
    // general path below returns for the same arguments.
    if (col_extent == 2 * kCqsBlk && col_base >= 0 && (col_base % kCqsBlk) == 0
        && col_base + col_extent <= max_seqlen_k
        && (col_base / kCqsBlk) + 2 <= num_blocks) {
        const int b0 = col_base / kCqsBlk;
        if (blk_or != nullptr) {
            const int64_t* p = blk_or + b0;
            if ((reinterpret_cast<uintptr_t>(p) & 15) == 0) {
                const longlong2 w = *reinterpret_cast<const longlong2*>(p);
                c.col_or = w.x | w.y;
            } else {
                c.col_or = p[0] | p[1];
            }
        }
        if (blk_and != nullptr) {
            const int64_t* p = blk_and + b0;
            if ((reinterpret_cast<uintptr_t>(p) & 15) == 0) {
                const longlong2 w = *reinterpret_cast<const longlong2*>(p);
                c.col_and = w.x & w.y;
            } else {
                c.col_and = p[0] & p[1];
            }
            c.col_ok = true;
        }
        return c;
    }
    if (blk_or != nullptr) {
        c.col_or = cqs_blk_range_or_c(blk_or, num_blocks, col_base, col_extent);
    }
    if (blk_and != nullptr
        && (col_base % kCqsBlk) == 0 && (col_extent % kCqsBlk) == 0
        && col_base + col_extent <= max_seqlen_k) {
        c.col_and = cqs_blk_range_and_c(blk_and, num_blocks, col_base, col_extent);
        c.col_ok = true;
    }
    return c;
}

// == cqs_tile_masked(blk_and, 64, ...) for the same tile.
__forceinline__ __device__ bool cqs_tile_masked_c(const CqsRowSummary& r, const CqsColSummary& c) {
    return r.row_may_mask && c.col_ok && (r.row_and & c.col_and) != 0;
}

// == cqs_tile_clear(blk_or, 64, ...) for the same tile.
__forceinline__ __device__ bool cqs_tile_clear_c(const CqsRowSummary& r, const CqsColSummary& c,
                                                 const int64_t* blk_or) {
    if (blk_or == nullptr) { return false; }
    return r.row_clear || (r.row_or & c.col_or) == 0;
}

template <bool Is_causal, bool Is_local, bool Has_alibi>
struct Mask {

    const int max_seqlen_k, max_seqlen_q;
    const int window_size_left, window_size_right;
    const float alibi_slope;
    const bool cqs_enabled;
    const int cqs_num_chunks;
    const int cqs_owner_chunk;
    const int* cqs_chunk_ends;
    const int64_t* cqs_group_bits;
    const int64_t* cqs_blk_or;
    const int64_t* cqs_blk_and;
    const int cqs_blk_size;
    const int cqs_num_blocks;

    __forceinline__ __device__ Mask(const int max_seqlen_k, const int max_seqlen_q,
                                    const int window_size_left, const int window_size_right,
                                    const float alibi_slope=0.f,
                                    const bool cqs_enabled=false,
                                    const int cqs_num_chunks=0,
                                    const int cqs_owner_chunk=0,
                                    const int* cqs_chunk_ends=nullptr,
                                    const int64_t* cqs_group_bits=nullptr,
                                    const int64_t* cqs_blk_or=nullptr,
                                    const int64_t* cqs_blk_and=nullptr,
                                    const int cqs_blk_size=0,
                                    const int cqs_num_blocks=0)
        : max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , alibi_slope(!Has_alibi ? 0.0 : alibi_slope)
        , cqs_enabled(cqs_enabled)
        , cqs_num_chunks(cqs_num_chunks)
        , cqs_owner_chunk(cqs_owner_chunk)
        , cqs_chunk_ends(cqs_chunk_ends)
        , cqs_group_bits(cqs_group_bits)
        , cqs_blk_or(cqs_blk_or)
        , cqs_blk_and(cqs_blk_and)
        , cqs_blk_size(cqs_blk_size)
        , cqs_num_blocks(cqs_num_blocks) {
    };

    // OR of the group bits over a token range, from the block summaries.
    __forceinline__ __device__ int64_t cqs_range_or(const int start, const int extent) const {
        const int lo = start < 0 ? 0 : start;
        const int hi_tok = start + extent;
        const int b0 = lo / cqs_blk_size;
        int b1 = (hi_tok + cqs_blk_size - 1) / cqs_blk_size;
        if (b1 > cqs_num_blocks) { b1 = cqs_num_blocks; }
        int64_t acc = 0;
        for (int b = b0; b < b1; ++b) { acc |= cqs_blk_or[b]; }
        return acc;
    }

    // AND of the group bits over a token range, from the block summaries.
    __forceinline__ __device__ int64_t cqs_range_and(const int start, const int extent) const {
        const int lo = start < 0 ? 0 : start;
        const int hi_tok = start + extent;
        const int b0 = lo / cqs_blk_size;
        int b1 = (hi_tok + cqs_blk_size - 1) / cqs_blk_size;
        if (b1 > cqs_num_blocks) { b1 = cqs_num_blocks; }
        if (b1 <= b0) { return 0; }
        int64_t acc = ~int64_t(0);
        for (int b = b0; b < b1; ++b) { acc &= cqs_blk_and[b]; }
        return acc;
    }

    // True when EVERY pair inside this tile is CQS-masked. If some bit is set in
    // all rows and in all columns, then bits[r] & bits[c] is non-zero for every
    // (r, c), so the whole tile is dropped and its QK^T / PV GEMMs are pure
    // waste. Conservative: a false negative only costs the normal path.
    __forceinline__ __device__ bool cqs_tile_is_fully_masked(const int row_base, const int row_extent,
                                                             const int col_base, const int col_extent) const {
        if (!cqs_enabled || cqs_blk_and == nullptr || cqs_blk_size <= 0) { return false; }
        // Partial edge blocks would make the AND summary describe tokens outside
        // the tile, so only trust it on whole-block-aligned ranges.
        if ((row_base % cqs_blk_size) != 0 || (row_extent % cqs_blk_size) != 0) { return false; }
        if ((col_base % cqs_blk_size) != 0 || (col_extent % cqs_blk_size) != 0) { return false; }
        if (row_base + row_extent > max_seqlen_q || col_base + col_extent > max_seqlen_k) { return false; }
        const int64_t and_row = cqs_range_and(row_base, row_extent);
        if (and_row == 0) { return false; }
        return (and_row & cqs_range_and(col_base, col_extent)) != 0;
    }

    // True when no pair inside this tile can be CQS-masked, so the tile can use
    // the original FlashAttention fast paths untouched.
    __forceinline__ __device__ bool cqs_tile_is_clear(const int row_base, const int row_extent,
                                                      const int col_base, const int col_extent) const {
        if (cqs_blk_or == nullptr || cqs_blk_size <= 0) { return false; }
        const int64_t row_or = cqs_range_or(row_base, row_extent);
        if (row_or == 0) { return true; }
        return (row_or & cqs_range_or(col_base, col_extent)) == 0;
    }

    __forceinline__ __device__ int cqs_chunk_of(const int token_idx) const {
        if (!cqs_enabled || cqs_num_chunks <= 0 || cqs_chunk_ends == nullptr) {
            return -1;
        }
        int chunk_id = cqs_num_chunks - 1;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (i >= cqs_num_chunks) {
                break;
            }
            if (token_idx < cqs_chunk_ends[i]) {
                chunk_id = i;
                break;
            }
        }
        return chunk_id;
    }

    // Fetch a row's group bits once, so the inner column loop does not reload
    // them for every score element. Returns 0 when there is nothing to mask,
    // which callers use as an early-out for the whole row.
    __forceinline__ __device__ int64_t cqs_row_bits_of(const int row_idx) const {
        if (!cqs_enabled || cqs_group_bits == nullptr
            || row_idx < 0 || row_idx >= max_seqlen_q) {
            return 0;
        }
        return cqs_group_bits[row_idx];
    }

    // Row bits already in a register: one global load per element instead of two.
    __forceinline__ __device__ bool cqs_should_mask_row_bits(const int64_t row_bits,
                                                             const int col_idx) const {
        if (row_bits == 0 || col_idx < 0 || col_idx >= max_seqlen_k) {
            return false;
        }
        return (row_bits & cqs_group_bits[col_idx]) != 0;
    }

    __forceinline__ __device__ bool cqs_should_mask(const int row_idx, const int col_idx) const {
        if (!cqs_enabled || row_idx < 0 || col_idx < 0 || row_idx >= max_seqlen_q || col_idx >= max_seqlen_k) {
            return false;
        }
        if (cqs_group_bits != nullptr) {
            const int64_t row_bits = cqs_group_bits[row_idx];
            const int64_t col_bits = cqs_group_bits[col_idx];
            return (row_bits & col_bits) != 0;
        }
        const int row_chunk = cqs_chunk_of(row_idx);
        const int col_chunk = cqs_chunk_of(col_idx);
        if (row_chunk < 0 || col_chunk < 0) {
            return false;
        }
        return row_chunk != cqs_owner_chunk && col_chunk == row_chunk;
    }

    // Causal_mask: whether this particular iteration needs causal masking
    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor_,
                                               const int col_idx_offset_,
                                               const int row_idx_offset,
                                               const int warp_row_stride,
                                               const int tile_row_base=-1,
                                               const int tile_rows=0,
                                               const int tile_cols=0,
                                               // -1: run the tile test here (the
                                               // forward's path). 0/1: the caller
                                               // already knows, so this Mask need
                                               // not hold the summary pointers.
                                               const int cqs_tile_state=-1,
                                               // next/ v2: this tile's column
                                               // group-bits already staged in
                                               // shared memory, indexed by
                                               // col_idx - cqs_smem_col_base.
                                               const int64_t* cqs_smem_bits=nullptr,
                                               const int cqs_smem_col_base=0) {
        static_assert(!(Causal_mask && Is_local), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
        static constexpr bool Need_masking = Has_alibi || Causal_mask || Is_local || !Is_even_MN;
        // if (cute::thread0()) { printf("Has_alibi = %d, Causal_mask=%d, Is_local=%d, Is_even_MN = %d, Need_masking = %d\n", Has_alibi, Causal_mask, Is_local, Is_even_MN, Need_masking); }

        // O(1) tile test. If nothing in this tile can be masked, fall through to
        // the untouched FlashAttention paths -- otherwise merely *enabling* CQS
        // costs the generic masking loop on every tile, which measured ~4.3x
        // slower than plain FA even with an all-zero mask.
        const bool cqs_active = cqs_tile_state >= 0
            ? (cqs_tile_state != 0)
            : (cqs_enabled
               && !(tile_row_base >= 0
                    && cqs_tile_is_clear(tile_row_base, tile_rows, col_idx_offset_, tile_cols)));

        if constexpr (!Need_masking) {
            if (!cqs_active) {
                return;
            }
        }
        {
            // Reshape tensor_ from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
            Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
            // Do we need both row and column indices, or just column incides?
            static constexpr bool Col_idx_only = !(Has_alibi && !Is_causal) && !Is_local && !Causal_mask;
            const int lane_id = threadIdx.x % 32;
            const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
            if constexpr (Col_idx_only) {
                if (!cqs_active) {
                    #pragma unroll
                    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                        const int col_idx_base = col_idx_offset + nj * 8;
                        #pragma unroll
                        for (int j = 0; j < size<1, 0>(tensor); ++j) {
                            const int col_idx = col_idx_base + j;
                            #pragma unroll
                            for (int mi = 0; mi < size<0>(tensor); ++mi) {
                                // No causal, no local
                                if constexpr (Has_alibi) {
                                    tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                                }
                                if constexpr (!Is_even_MN) {
                                    if (col_idx >= max_seqlen_k) { tensor(mi, make_coord(j, nj)) = -INFINITY; }
                                }
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                        #pragma unroll
                        for (int i = 0; i < size<0, 0>(tensor); ++i) {
                            const int row_idx = row_idx_base + i * 8;
                            const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
                            const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
                            // See the sibling branch: one group-bits load per row
                            // instead of one per score element.
                            const bool cqs_bitmask_mode = cqs_active && cqs_group_bits != nullptr;
                            const int64_t cqs_row_bits = cqs_bitmask_mode ? cqs_row_bits_of(row_idx) : 0;
                            const bool cqs_row_active = cqs_active && (!cqs_bitmask_mode || cqs_row_bits != 0);
                            #pragma unroll
                            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                                const int col_idx_base = col_idx_offset + nj * 8;
                                #pragma unroll
                                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                                    const int col_idx = col_idx_base + j;
                                    if constexpr (Has_alibi) {
                                        if constexpr (Is_causal) {
                                            tensor(make_coord(i, mi), make_coord(j, nj)) += alibi_slope * col_idx;
                                        } else {
                                            tensor(make_coord(i, mi), make_coord(j, nj)) -= alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);
                                        }
                                    }
                                    if constexpr (Causal_mask) {
                                        if (col_idx >= col_idx_limit_right) {
                                            tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                        }
                                    }
                                    if constexpr (Is_local) {
                                        if (col_idx >= col_idx_limit_right || col_idx < col_idx_limit_left) {
                                            tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                        }
                                    }
                                    if constexpr (!Causal_mask && !Is_local && !Is_even_MN) {
                                        if (col_idx >= max_seqlen_k) {
                                            tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                        }
                                    }
                                    if (cqs_row_active) {
                                        const bool cqs_masked = cqs_bitmask_mode
                                            ? (cqs_smem_bits != nullptr
                                               ? (col_idx >= 0 && col_idx < max_seqlen_k
                                                  && (cqs_row_bits & cqs_smem_bits[col_idx - cqs_smem_col_base]) != 0)
                                               : cqs_should_mask_row_bits(cqs_row_bits, col_idx))
                                            : cqs_should_mask(row_idx, col_idx);
                                        if (cqs_masked) {
                                            tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                    const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                    #pragma unroll
                    for (int i = 0; i < size<0, 0>(tensor); ++i) {
                        const int row_idx = row_idx_base + i * 8;
                        const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
                        const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
                        // Hoisted out of the column loops: one group-bits load
                        // per row instead of one per score element, and rows
                        // that own their diagonal block (bits == 0) skip the
                        // per-element check entirely.
                        const bool cqs_bitmask_mode = cqs_active && cqs_group_bits != nullptr;
                        const int64_t cqs_row_bits = cqs_bitmask_mode ? cqs_row_bits_of(row_idx) : 0;
                        const bool cqs_row_active = cqs_active && (!cqs_bitmask_mode || cqs_row_bits != 0);
                        #pragma unroll
                        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                            const int col_idx_base = col_idx_offset + nj * 8;
                            #pragma unroll
                            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                                const int col_idx = col_idx_base + j;
                                if constexpr (Has_alibi) {
                                    if constexpr (Is_causal) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) += alibi_slope * col_idx;
                                    } else {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) -= alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);

                                    }
                                }
                                if constexpr (Causal_mask) {
                                    if (col_idx >= col_idx_limit_right) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                }
                                if constexpr (Is_local) {
                                    if (col_idx >= col_idx_limit_right || col_idx < col_idx_limit_left) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                }
                                if constexpr (!Causal_mask && !Is_local && !Is_even_MN) {
                                    // Causal and Local already handles MN masking
                                    if (col_idx >= max_seqlen_k) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                }
                                if (cqs_row_active) {
                                    const bool cqs_masked = cqs_bitmask_mode
                                        ? (cqs_smem_bits != nullptr
                                           ? (col_idx >= 0 && col_idx < max_seqlen_k
                                              && (cqs_row_bits & cqs_smem_bits[col_idx - cqs_smem_col_base]) != 0)
                                           : cqs_should_mask_row_bits(cqs_row_bits, col_idx))
                                        : cqs_should_mask(row_idx, col_idx);
                                    if (cqs_masked) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

};

} // namespace FLASH_NAMESPACE
