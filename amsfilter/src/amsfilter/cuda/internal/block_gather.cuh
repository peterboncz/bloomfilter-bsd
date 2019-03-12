#pragma once

#include <dtl/dtl.hpp>
#include <cuda_runtime.h>
#include <cub/thread/thread_load.cuh>
#include "cuda_helper.cuh"

namespace amsfilter {
namespace cuda {
namespace internal {
//===----------------------------------------------------------------------===//
// Helper to determine the largest native type suitable for data movement.
template<u32 size_in_bytes> struct largest_native_type_helper {
  using type = typename largest_native_type_helper<size_in_bytes / 2>::type;
};
template<> struct largest_native_type_helper<16> { using type = int4; };
template<> struct largest_native_type_helper< 8> { using type = $u64; };
template<> struct largest_native_type_helper< 4> { using type = $u32; };
template<> struct largest_native_type_helper< 2> { using type = $u16; };
template<> struct largest_native_type_helper< 1> { using type =  $u8; };
/// Determine the largest native type suitable for data movement.
template<typename T, u32 cnt> struct largest_native_type {
  using type = typename largest_native_type_helper<sizeof(T) * cnt>::type;
};
//===----------------------------------------------------------------------===//
// Gathers consecutive items from memory.
template<
    typename T,
    u32 items_per_thread,
    cub::CacheLoadModifier load_modifier = cub::LOAD_CA
>
struct block_gather_internal {
  __device__ __forceinline__
  static void
  load(const T* block_ptr, T* shared_mem_out) {
    __shared__ const T* block_ptrs[warp_size];
    u32 lid = warp_local_thread_id();
    // Write the block ptrs of all threads to shared memory.
    block_ptrs[lid] = block_ptr;
    __syncwarp();
    // Copy the block to shared memory.
    #pragma unroll
    for ($u32 l = 0; l < items_per_thread; ++l) {
      u32 j = (warp_size * l) + lid;
      u32 p = j / items_per_thread;
      u32 o = j % items_per_thread;
      const T* ptr = block_ptrs[p];
      shared_mem_out[j] = cub::ThreadLoad<cub::LOAD_CA>(&ptr[o]);
    }
    __syncwarp();
  }
};

/// Gathers consecutive items from memory.
template<
    typename T,
    u32 items_per_thread,
    cub::CacheLoadModifier load_modifier = cub::LOAD_DEFAULT
>
struct block_gather {
  __device__ __forceinline__
  static void
  load(const T* block_ptr, T* shared_mem_out) {
    // Determine the data movement type.
    using Tm = typename largest_native_type<T, items_per_thread>::type;
    static constexpr u32 move_cnt = items_per_thread / (sizeof(Tm) / sizeof(T));
    block_gather_internal<Tm, move_cnt, load_modifier>::load(
        reinterpret_cast<const Tm*>(block_ptr),
        reinterpret_cast<Tm*>(shared_mem_out));
  }
};
//===----------------------------------------------------------------------===//
} // namespace internal
} // namespace cuda
} // namespace amsfilter
