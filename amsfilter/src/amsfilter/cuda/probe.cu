#include "probe.hpp"

#include <dtl/dtl.hpp>

#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_logic.hpp>

#include <amsfilter/internal/blocked_bloomfilter_template.hpp>
#include <amsfilter/internal/blocked_bloomfilter_resolve.hpp>

#include "internal/probe_impl.cuh"

namespace amsfilter {
namespace cuda {
//===----------------------------------------------------------------------===//
struct Probe::impl {

  /// The actual probe instance.
  internal::probe_impl instance;

  impl(const Config& config, const std::size_t desired_length)
      : instance(config, desired_length) {}
  ~impl() = default;
  impl(impl&&) = default;
  impl(const impl&) = delete;
  impl& operator=(impl&&) = delete;
  impl& operator=(const impl&) = delete;

};
//===----------------------------------------------------------------------===//
Probe::Probe(const AmsFilter& filter)
    : pimpl_{std::make_unique<impl>(filter.get_config(),
          filter.get_desired_length())} {}
Probe::Probe(Probe&&) noexcept = default;
Probe& Probe::operator=(Probe&& other) noexcept = default;
Probe::~Probe() = default;

void
Probe::operator()(
    const amsfilter::internal::word_t* __restrict filter_data,
    const amsfilter::internal::key_t* __restrict keys, u32 key_cnt,
    amsfilter::internal::word_t* __restrict result_bitmap,
    const cudaStream_t& cuda_stream) {
  pimpl_->instance.contains(filter_data, keys, key_cnt, result_bitmap,
      cuda_stream);
}
//===----------------------------------------------------------------------===//
} // namespace cuda
} // namespace amsfilter
