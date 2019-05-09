#pragma once

#include <mutex>
#include <dtl/dtl.hpp>
#include <thrust/device_vector.h>
#include <amsfilter/amsfilter.hpp>

namespace amsfilter {
namespace cuda {
namespace internal {
//===----------------------------------------------------------------------===//
/// Creates and manages data replicas in device memory.
struct replicas {
  using word_t = amsfilter::AmsFilter::word_t;
  using device_vector_t = thrust::device_vector<word_t>;
  using shared_device_vector_t = std::shared_ptr<device_vector_t>;

  $u32 cuda_device_cnt_;
  std::unique_ptr<std::mutex> mutex_;
  std::vector<$i32> replica_device_map_;
  std::vector<shared_device_vector_t> data_replicas_;

  $u32
  get_cuda_device_cnt() {
    $i32 count;
    cudaGetDeviceCount(&count);
    return static_cast<u32>(count);
  }

  replicas()
      : cuda_device_cnt_(get_cuda_device_cnt()),
        mutex_(std::make_unique<std::mutex>()),
        replica_device_map_(cuda_device_cnt_, -1) {}

  shared_device_vector_t
  get_replica(const word_t* input_begin, const word_t* input_end,
      u32 cuda_device_no) {
    if (cuda_device_no >= cuda_device_cnt_) {
      throw std::invalid_argument("Unknown CUDA device.");
    }
    std::size_t replica_idx = 0;
    if (replica_device_map_[cuda_device_no] == -1) { // FIXME mem leak?
      std::lock_guard<std::mutex> lock(*mutex_);
      if (replica_device_map_[cuda_device_no] == -1) { // race condition?
        replica_idx = data_replicas_.size();
        // Create a replica.
        cudaSetDevice(cuda_device_no);
        data_replicas_.emplace_back(
            std::make_shared<device_vector_t>(input_begin, input_end));
        replica_device_map_[cuda_device_no] = data_replicas_.size() - 1;
      }
      else {
        replica_idx =
            static_cast<std::size_t>(replica_device_map_[cuda_device_no]);
      }
    }
    return data_replicas_[replica_idx];
  }
};
//===----------------------------------------------------------------------===//
} // namespace internal
} // namespace cuda
} // namespace amsfilter
