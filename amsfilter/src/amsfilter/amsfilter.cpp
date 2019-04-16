#include <memory>

#include <dtl/dtl.hpp>
#include <dtl/simd.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_logic.hpp>

#include <amsfilter/internal/blocked_bloomfilter_template.hpp>
#include <amsfilter/internal/cuckoofilter_template.hpp>
#include <amsfilter/internal/filter_resolve.hpp>

#include "amsfilter.hpp"

#ifndef __AVX2__
#error CPU code compiled for a pre-AVX2 architecture!
#endif

namespace amsfilter {
using namespace amsfilter::internal;
//===----------------------------------------------------------------------===//
struct AmsFilter::impl {

  /// Pointer to the filter logic instance.
  std::unique_ptr<filter_base_t> instance;
  /// The blocked Bloom filter parameters.
  Config config;
  /// The desired length of the Bloom filter (in bits).
  std::size_t desired_length;

  impl(const Config& config, const std::size_t desired_length)
      : config(config),
        desired_length(desired_length) {
    init();
  };

  ~impl() = default;
  impl(impl&&) = default;
  impl(const impl&) = delete;
  impl& operator=(impl&&) = default;
  impl& operator=(const impl&) = delete;

  /// Resolves the (specialized) filter type and constructs an instance (on
  /// heap memory).
  void
  init() {
    get_instance(config, *this);
  }

  // This function is called by 'get_instance'.
  template<u32 w, u32 s, u32 z, u32 k, dtl::block_addressing a>
  void
  operator()(const dtl::blocked_bloomfilter_config& conf) {
    using resolved_type = typename bbf_type<w, s, z, k, a>::type;
    if (resolved_type::word_cnt_per_block != conf.word_cnt_per_block
        || resolved_type::sector_cnt != conf.sector_cnt
        || resolved_type::zone_cnt != conf.zone_cnt
        || resolved_type::k != conf.k
        || resolved_type::addr_mode != conf.addr_mode) {
      throw std::invalid_argument(
          "Failed to construct a blocked Bloom filter with the parameters w="
              + std::to_string(conf.word_cnt_per_block)
              + ", s=" + std::to_string(conf.sector_cnt)
              + ", z=" + std::to_string(conf.zone_cnt)
              + ", k=" + std::to_string(conf.k)
              + ", and, a=" + std::to_string(static_cast<i32>(conf.addr_mode))
              + ".");
    }
    // Instantiate the resolved type.
    instance = std::make_unique<resolved_type>(desired_length);
  }
};
//===----------------------------------------------------------------------===//
AmsFilter::AmsFilter(const Config& config, const std::size_t desired_length)
    : config_(config),
      desired_length_(desired_length),
      pimpl_{std::make_unique<impl>(config_, desired_length_)} {}

AmsFilter::~AmsFilter() = default;

$u1
AmsFilter::insert(AmsFilter::word_t* __restrict filter_data,
    const AmsFilter::key_t key) {
  pimpl_->instance->insert(filter_data, key);
  return true; // inserts never fail
}

$u1
AmsFilter::contains(const AmsFilter::word_t* __restrict filter_data,
    const AmsFilter::key_t key) const {
  return pimpl_->instance->contains(filter_data, key);
}

std::size_t
AmsFilter::size() const {
  return pimpl_->instance->word_cnt();
}
//===----------------------------------------------------------------------===//
} // namespace amsfilter