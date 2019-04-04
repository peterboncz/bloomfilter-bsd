#pragma once

#include <memory>

#include <amsfilter/internal/buffer.hpp>
#include <amsfilter/internal/buffer.hpp>
#include <amsfilter/amsfilter.hpp>
#include <amsfilter/bitmap_view.hpp>
#include <amsfilter/common.hpp>
#include <amsfilter/probe.hpp>

namespace amsfilter {
//===----------------------------------------------------------------------===//
/// The AMS-Filter probe implementation with a lightweight interface.
///
/// In contrast to the Probe class, ProbeLite takes care of all the memory
/// management. Each instance allocates memory on the host-side as well as on
/// the device side. These memory resources are supposed to be re-used among
/// multiple batch-probes.
// TODO replicate filter data to NUMA nodes.
class ProbeLite {

  // Typedefs
  using bitmap_storage_t = amsfilter::AmsFilter::word_t;
  using bitmap_buffer_t = amsfilter::internal::buffer<bitmap_storage_t>;

  using key_t = amsfilter::key_t;

  /// The probe instance (with the low-level API).
  Probe probe_instance_;
  /// Shared pointer to the filter data.
  shared_filter_data_t filter_data_;
  /// The maximum number of key that can be tested in one go.
  std::size_t max_batch_size_;
  /// The CUDA device number, this instance is associated with.
  bitmap_buffer_t result_bitmap_;

  /// Returns the number of words required to store the resulting bitmap.
  static std::size_t
  bitmap_word_cnt(std::size_t max_batch_size) {
    return (max_batch_size + (bitwidth<bitmap_storage_t> - 1))
        / bitwidth<bitmap_storage_t>;
  }

public:

  using result_type = bitmap_view<bitmap_storage_t>;

  ProbeLite(const AmsFilter& filter,
      shared_filter_data_t& filter_data,
      std::size_t max_batch_size)
      : probe_instance_(filter),
        filter_data_(filter_data),
        max_batch_size_(max_batch_size),
        result_bitmap_(bitmap_word_cnt(max_batch_size_)) {};
  ProbeLite(const AmsFilter& filter,
      shared_filter_data_t& filter_data,
      std::size_t max_batch_size,
      const TuningParams& tuning_params)
      : probe_instance_(filter, tuning_params),
        filter_data_(filter_data),
        max_batch_size_(max_batch_size),
        result_bitmap_(bitmap_word_cnt(max_batch_size_)) {};
  ProbeLite(const ProbeLite& other) = delete;
  ProbeLite(ProbeLite&& other) noexcept = default;
  ProbeLite& operator=(const ProbeLite& other) = delete;
  ProbeLite& operator=(ProbeLite&& other) = default;
  ~ProbeLite() = default;

  /// Batch-probe the filter.
  void __attribute__((noinline))
  operator()(const key_t* keys, const std::size_t key_cnt) {
    if (key_cnt == 0) return;
    if (key_cnt > max_batch_size_) {
      throw std::invalid_argument("The 'key_cnt' argument must not exceed"
          " the max. batch size.");
    }
    // Execute the kernel (using the low-level API).
    probe_instance_(filter_data_->data(), keys, key_cnt, result_bitmap_.begin());
  }

  /// Returns true if the asynchronously executed probe is done, false otherwise.
  u1 __forceinline__
  is_done() {
    return true;
  }

  /// Blocks until a asynchronously executed probe is finished.
  void __forceinline__
  wait() {
    // Do nothing. Host-side execution is synchronous.
  }

  /// Gives access to the result bitmap.
  result_type
  get_results() {
    return bitmap_view<bitmap_storage_t> {
        result_bitmap_.begin(),
        result_bitmap_.end()
    };
  }

};
//===----------------------------------------------------------------------===//
} // namespace amsfilter
