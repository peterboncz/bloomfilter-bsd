#pragma once

#include <memory>
#include <amsfilter/common.hpp>
#include <amsfilter/tuning_params.hpp>
#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp>

namespace amsfilter {
//===----------------------------------------------------------------------===//
using Config = dtl::blocked_bloomfilter_config;
using TuningParams = amsfilter::tuning_params;
//===----------------------------------------------------------------------===//
class AmsFilter {
  class impl;

  /// The filter parameters.
  Config config_;
  /// The desired length of the Bloom filter (in bits).
  std::size_t desired_length_;
  /// Pointer to the implementation.
  std::unique_ptr<impl> pimpl_;

public:

  using key_t = amsfilter::key_t;
  using word_t = amsfilter::word_t;

  explicit
  AmsFilter(const Config& config, const std::size_t desired_length);
  AmsFilter(const AmsFilter& other) = delete;
  AmsFilter(AmsFilter&& other) noexcept = default;
  AmsFilter& operator=(const AmsFilter& other) = delete;
  AmsFilter& operator=(AmsFilter&& other) = default;
  ~AmsFilter();

  /// Inserts the given key.
  $u1
  insert(word_t* __restrict filter_data, key_t key);

  /// Probes the filter for the given key.
  $u1
  contains(const word_t* __restrict filter_data, key_t key) const;

  /// Returns the actual filter size in number of words.
  std::size_t
  size() const;

  /// Returns a copy of the filter parameters.
  Config
  get_config() const {
    return config_;
  }

  /// Returns the desired length of the filter (in bits).
  std::size_t
  get_desired_length() const {
    return desired_length_;
  }

};
//===----------------------------------------------------------------------===//
} // namespace amsfilter
