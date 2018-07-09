#pragma once

#include <chrono>
#include <map>
#include <random>

#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/block_addressing_logic.hpp>

#include "immintrin.h"
#include "cuckoofilter_config.hpp"

namespace dtl {
namespace cuckoofilter {


//===----------------------------------------------------------------------===//
/// Provides tuning parameters to the filter instance.
struct cuckoofilter_tune {

  /// Sets the SIMD unrolling factor for the given blocked filter config.
  /// Note: unrolling by 0 means -> scalar code (no SIMD)
  virtual void
  set_unroll_factor(u32 bits_per_tag,
                    u32 tags_per_bucket,
                    dtl::block_addressing addr_mode,
                    u32 unroll_factor) {
    throw std::runtime_error("Not supported");
  }


  /// Returns the SIMD unrolling factor for the given filter config.
  /// Note: unrolling by 0 means -> scalar code (no SIMD)
  virtual $u32
  get_unroll_factor(u32 bits_per_tag,
                    u32 tags_per_bucket,
                    dtl::block_addressing addr_mode) const {
    return 1; // default
  }


  /// Determines the best performing SIMD unrolling factor for the given
  /// filter config.
  virtual $u32
  tune_unroll_factor(u32 bits_per_tag,
                     u32 tags_per_bucket,
                     dtl::block_addressing addr_mode) {
    throw std::runtime_error("Not supported");
  }


  /// Determines the best performing SIMD unrolling factor for all valid
  /// filter configs.
  virtual void
  tune_unroll_factor() {
    throw std::runtime_error("Not supported");
  }

};
//===----------------------------------------------------------------------===//

} // namespace cuckoofilter
} // namespace dtl