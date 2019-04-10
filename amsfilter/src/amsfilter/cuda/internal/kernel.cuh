#pragma once

#include <dtl/dtl.hpp>
#include "block_gather.cuh"
#include "cuda_helper.cuh"

namespace amsfilter {
namespace cuda {
namespace internal {
//===----------------------------------------------------------------------===//
/// A straight-forward kernel to probe a Bloom filter.
template<typename filter_t>
__global__
void
contains_kernel(
    const filter_t filter_logic,
    const typename filter_t::word_t* __restrict__ word_array,
    u32* __restrict__ keys, u32 key_cnt, $u32* __restrict__ result_bitmap) {

  // Who am I?
  u32 wid = global_warp_id();
  u32 lid = warp_local_thread_id();

  // The output this executing thread will write later on (until then, the
  // result is kept in a register).
  $u32 thread_local_bitmap = 0u;

  constexpr u32 elements_per_thread = warp_size; // ... processed sequentially
  constexpr u32 elements_per_warp = elements_per_thread * warp_size;

  // Where to start reading the input?
  $u32 read_pos = wid * elements_per_warp + lid;

  // Each thread processes multiple elements sequentially.
  for ($u32 i = 0; i != elements_per_thread; ++i) {
    auto is_contained = (read_pos < key_cnt)
        ? filter_logic.contains(word_array, keys[read_pos])
        : false;
    u32 bitmap = __ballot_sync(0xffffffff, is_contained);
    thread_local_bitmap = (lid == i) ? bitmap : thread_local_bitmap;
    read_pos += warp_size;
  }
  __syncwarp();
  // Every thread writes a single word of the output bitmap.
  u32 write_pos = global_thread_id();
  if (write_pos < ((key_cnt + 31) / 32)) {
    result_bitmap[write_pos] = thread_local_bitmap;
  }
}
//===----------------------------------------------------------------------===//
/// Similar to the straight-forward kernel above, but the blocks are explicitly
/// copied to shared memory before they are probed.
///
/// Note that this kernel only supports a fixed block size, which is 32.
// TODO make the kernel support arbitrary block sizes
template<typename filter_t>
__global__
void
contains_kernel_with_block_prefetch(
    const filter_t filter_logic,
    const typename filter_t::word_t* __restrict__ word_array,
    u32* __restrict__ keys, u32 key_cnt, $u32* __restrict__ result_bitmap) {

  using key_t = typename filter_t::key_t;
  using word_t = typename filter_t::word_t;
  static constexpr u32 word_cnt_per_block = filter_t::word_cnt_per_block;

  __shared__ word_t block_cache[word_cnt_per_block * warp_size];

  // Who am I?
  u32 wid = global_warp_id();
  u32 lid = warp_local_thread_id();

  word_t* block_cache_ptr = &block_cache[word_cnt_per_block * lid];

  // The output this executing thread will write later on (until then, the
  // result is kept in a register).
  $u32 thread_local_bitmap = 0u;

  constexpr u32 elements_per_thread = warp_size; // ... processed sequentially
  constexpr u32 elements_per_warp = elements_per_thread * warp_size;

  // Where to start reading the input?
  $u32 read_pos = wid * elements_per_warp + lid;

  // Each thread processes multiple elements sequentially.
  for ($u32 i = 0; i != elements_per_thread; ++i) {
    auto is_contained = false;
    key_t key = 0;
    const word_t* block_ptr = word_array;
    if (read_pos < key_cnt) {
      key = keys[read_pos];
      const auto block_idx = filter_logic.get_block_idx(key);
      block_ptr = word_array + (word_cnt_per_block * block_idx);
    }
    block_gather<word_t, word_cnt_per_block>::load(block_ptr, block_cache);
    if (read_pos < key_cnt) {
      is_contained = filter_t::block_t::contains(block_cache_ptr, key);
    }
    u32 bitmap = __ballot_sync(0xffffffff, is_contained);
    thread_local_bitmap = (lid == i) ? bitmap : thread_local_bitmap;
    read_pos += warp_size;
  }
  __syncwarp();
  // Every thread writes a single word of the output bitmap.
  u32 write_pos = global_thread_id();
  result_bitmap[write_pos] = thread_local_bitmap;
}
//===----------------------------------------------------------------------===//
} // namespace internal
} // namespace cuda
} // namespace amsfilter
