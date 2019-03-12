#pragma once

#include <dtl/dtl.hpp>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "cuda_api_helper.hpp"

static constexpr u32 warp_size = 32;

//===----------------------------------------------------------------------===//
// NOTE: The utility functions below assume a 1-dimensional grid.
//===----------------------------------------------------------------------===//
/// Returns the global (linear) ID of the executing thread.
__device__ __forceinline__
u32
global_thread_id() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

/// Returns the global size.
__device__ __forceinline__
u32
global_size() {
  return gridDim.x * blockDim.x;
}

/// Returns the total number of warps.
__device__ __forceinline__
u32
warp_cnt() {
  return (global_size() + (warp_size - 1)) / warp_size;
}

/// Returns the block size.
__device__ __forceinline__
u32
block_size() {
  return blockDim.x;
}

/// Returns the block ID in [0, grid_size).
__device__ __forceinline__
u32
block_id() {
  return blockIdx.x;
}

/// Returns the thread ID within the current block: [0, block_size)
__device__ __forceinline__
u32
block_local_thread_id() {
  return threadIdx.x;
}

/// Returns the warp ID within the current block the ID is in [0, u), where
/// u = block_size / warp_size
__device__ __forceinline__
u32
block_local_warp_id() {
  return block_local_thread_id() / warp_size;
}

/// Returns the warp ID (within the entire grid).
__device__ __forceinline__
u32
global_warp_id() {
  return global_thread_id() / warp_size;
}

/// Returns the thread id [0,32) within the current warp.
__device__ __forceinline__
u32
warp_local_thread_id() {
  return block_local_thread_id() % warp_size;
}
