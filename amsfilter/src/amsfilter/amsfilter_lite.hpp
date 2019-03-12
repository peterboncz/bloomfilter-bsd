#pragma once

#include <memory>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp>

#include <amsfilter/internal/buffer.hpp>
#include <amsfilter/amsfilter.hpp>
#include <amsfilter/common.hpp>
#include <amsfilter/probe_lite.hpp>

#ifdef HAVE_CUDA
#include <amsfilter/cuda/internal/replicas.hpp>
#include <amsfilter/cuda/probe_lite.hpp>
#endif

namespace amsfilter {
//===----------------------------------------------------------------------===//
/// The AMS-Filter with a lightweight interface.
class AmsFilterLite {

private:
  AmsFilter amsfilter_;
  shared_filter_data_t filter_data_;

public:
  explicit
  AmsFilterLite(const Config& config, const std::size_t desired_length)
      : amsfilter_(config, desired_length),
        filter_data_(std::make_shared<filter_data_t>(amsfilter_.size())) {}
  AmsFilterLite(const AmsFilterLite& other) = delete;
  AmsFilterLite(AmsFilterLite&& other) noexcept = default;
  AmsFilterLite& operator=(const AmsFilterLite& other) = delete;
  AmsFilterLite& operator=(AmsFilterLite&& other) = default;
  ~AmsFilterLite() = default;

  /// Inserts the given key.
  $u1
  insert(key_t key) {
    amsfilter_.insert(filter_data_->data(), key);
    return true; // inserts never fail
  }

  // TODO implement batch_insert

  /// Probes the filter for the given key.
  $u1
  contains(key_t key) const {
    return amsfilter_.contains(filter_data_->data(), key);
  }

  /// Returns the actual filter size in number of words.
  std::size_t
  size() const {
    return amsfilter_.size();
  }

  /// Returns a copy of the filter parameters.
  Config
  get_config() const {
    return amsfilter_.get_config();
  }

  /// Returns the desired length of the filter (in bits).
  std::size_t
  get_desired_length() const {
    return amsfilter_.get_desired_length();
  }

  ProbeLite
  batch_probe(std::size_t max_batch_size) {
    ProbeLite probe_instance(amsfilter_, filter_data_, max_batch_size);
    return std::move(probe_instance);
  }

#ifdef HAVE_CUDA
  /// Create and manage copies of the filter data in device memory.
  cuda::internal::replicas replicas_;

  cuda::ProbeLite
  batch_probe_cuda(std::size_t max_batch_size, $u32 cuda_device_no) {

    auto device_filter_data = replicas_.get_replica(filter_data_->begin(),
        filter_data_->end(), cuda_device_no);
    cuda::ProbeLite probe_instance(amsfilter_, device_filter_data,
        max_batch_size, cuda_device_no);
    return std::move(probe_instance);
  };
#endif

};
//===----------------------------------------------------------------------===//
} // namespace amsfilter
