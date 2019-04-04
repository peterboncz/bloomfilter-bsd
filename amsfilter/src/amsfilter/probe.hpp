#pragma once

#include <memory>

#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp>
#include <amsfilter/amsfilter.hpp>
#include <amsfilter/common.hpp>

namespace amsfilter {
//===----------------------------------------------------------------------===//
// TODO
class Probe {

  class impl;
  std::unique_ptr<impl> pimpl_;

public:

  using key_t = amsfilter::key_t;
  using word_t = amsfilter::word_t;

  /// Constructs a new probe instance that is associated with the given filter.
  /// The max_batch_size refers to the maximum number of keys that can be
  /// tested in one go. The argument is required to allocate memory and to reuse
  /// that memory.
  explicit
  Probe(const AmsFilter& filter);
  Probe(const AmsFilter& filter, const TuningParams& tuning_params);
  Probe(const Probe& other) = delete;
  Probe(Probe&& other) noexcept;
  Probe& operator=(const Probe& other) = delete;
  Probe& operator=(Probe&& other) = delete;
  ~Probe();

  /// Probes the filter.
  void
  operator()(const word_t* __restrict filter_data,
      const key_t* __restrict keys, u32 key_cnt,
      word_t* __restrict result_bitmap);
};
//===----------------------------------------------------------------------===//
} // namespace amsfilter
