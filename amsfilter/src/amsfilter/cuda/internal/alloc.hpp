#pragma once

#include <cstddef>
#include <exception>
#include <limits>

#include <dtl/dtl.hpp>
#include <cuda_runtime.h>

namespace amsfilter {
namespace cuda {
namespace internal {
//===----------------------------------------------------------------------===//
/// Allocator for host-side pinned memory. All allocations are "portable" which
/// means that they can be used from any CUDA context.
template<typename T>
struct cuda_host_allocator {

  using value_type = T;
  using pointer_type = value_type*;
  using size_type = std::size_t;

  cuda_host_allocator() = default;
  cuda_host_allocator(const cuda_host_allocator& other) = default;
  cuda_host_allocator(cuda_host_allocator&& other) noexcept = default;
  cuda_host_allocator& operator=(const cuda_host_allocator& other) = default;
  cuda_host_allocator& operator=(cuda_host_allocator&& other) noexcept = default;
  ~cuda_host_allocator() = default;

  pointer_type
  allocate(size_type n, const void* /* hint */ = nullptr) throw() {
    void* ptr = nullptr;
    size_type size = n * sizeof(T);

    if (n > std::numeric_limits<size_type>::max() / sizeof(value_type)) {
      throw std::bad_alloc();
    }

    cudaError cu_code = cudaMallocHost(&ptr, size, cudaHostAllocPortable);

    if (!ptr || cu_code != cudaSuccess) {
      throw std::bad_alloc();
    }
    return static_cast<pointer_type>(ptr);
  }

  void
  deallocate(pointer_type ptr, size_type /*n*/) throw() {
    cudaError cu_code = cudaFreeHost(ptr);
    if (cu_code != cudaSuccess) {
      const auto* err_msg = cudaGetErrorString(cu_code);
      throw std::runtime_error("Failed to free (pinned) memory. CUDA error: "
          + std::to_string(*err_msg));
    }
  }
};
//===----------------------------------------------------------------------===//
/// Allocator for device memory. Allocations are performed on the device with
/// the given number (default = 0).
template<typename T>
struct cuda_device_allocator {

  using value_type = T;
  using pointer_type = value_type*;
  using size_type = std::size_t;

  $u32 device_no_;

  explicit
  cuda_device_allocator(u32 device_no = 0) : device_no_(device_no) {};
  cuda_device_allocator(const cuda_device_allocator& other) = default;
  cuda_device_allocator(cuda_device_allocator&& other) noexcept = default;
  cuda_device_allocator& operator=(const cuda_device_allocator& other) = default;
  cuda_device_allocator& operator=(cuda_device_allocator&& other) noexcept = default;
  ~cuda_device_allocator() = default;

  pointer_type
  allocate(size_type n, const void* /* hint */ = nullptr) throw() {
    void* ptr = nullptr;
    size_type size = n * sizeof(T);

    if (n > std::numeric_limits<size_type>::max() / sizeof(value_type)) {
      throw std::bad_alloc();
    }

    int current_device;
    cudaGetDevice(&current_device);

    cudaError cu_code = cudaSetDevice(device_no_);
    if (cu_code != cudaSuccess) {
      cudaSetDevice(current_device);
      throw std::bad_alloc();
    }

    cu_code = cudaMalloc(&ptr, size);
    if (!ptr || cu_code != cudaSuccess) {
      cudaSetDevice(current_device);
      throw std::bad_alloc();
    }
    cudaSetDevice(current_device);
    return static_cast<pointer_type>(ptr);
  }

  void
  deallocate(pointer_type ptr, size_type /*n*/) throw() {
    int current_device;
    cudaGetDevice(&current_device);

    cudaError cu_code = cudaSetDevice(device_no_);
    if (cu_code != cudaSuccess) {
      cudaSetDevice(current_device);
      const auto* err_msg = cudaGetErrorString(cu_code);
      throw std::runtime_error("Failed to free (pinned) memory. CUDA error: "
          + std::to_string(*err_msg));
    }
    cu_code = cudaFree(ptr);
    if (cu_code != cudaSuccess) {
      cudaSetDevice(current_device);
      const auto* err_msg = cudaGetErrorString(cu_code);
      throw std::runtime_error("Failed to free (pinned) memory. CUDA error: "
          + std::to_string(*err_msg));
    }
  }
};
//===----------------------------------------------------------------------===//
} // namespace internal
} // namespace cuda
} // namespace amsfilter
