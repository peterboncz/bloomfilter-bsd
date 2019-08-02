#pragma once

#include <chrono>
#include <map>
#include <random>

#include <dtl/dtl.hpp>

#include "block_addressing_logic.hpp"
#include "blocked_bloomfilter_config.hpp"

#include "immintrin.h"

#include "../model/tuning_params.hpp"

namespace dtl {


//===----------------------------------------------------------------------===//
/// Provides tuning parameters to the Bloom filter instance.
struct blocked_bloomfilter_tune {

  /// Sets the SIMD unrolling factor for the given blocked Bloom filter config.
  /// Note: unrolling by 0 means -> scalar code (no SIMD)
  virtual void
  set_unroll_factor(const blocked_bloomfilter_config& config,
                    u32 unroll_factor) {
    throw std::runtime_error("Not supported");
  }


  /// Returns the SIMD unrolling factor for the given blocked Bloom filter config.
  /// Note: unrolling by 0 means -> scalar code (no SIMD)
  virtual $u32
  get_unroll_factor(const blocked_bloomfilter_config& config) const {
    return 1; // default
  }


  /// Determines the best performing SIMD unrolling factor for the given
  /// blocked Bloom filter config.
  virtual $u32
  tune_unroll_factor(const blocked_bloomfilter_config& config,
      u64 filter_size_bits) {
    throw std::runtime_error("Not supported");
  }


  /// Determines the best performing SIMD unrolling factor for all valid
  /// blocked Bloom filter configs.
  virtual void
  tune_unroll_factor(u64 filter_size_bits) {
    throw std::runtime_error("Not supported");
  }

};
//===----------------------------------------------------------------------===//

} // namespace dtl