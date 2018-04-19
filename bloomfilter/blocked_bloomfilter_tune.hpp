#pragma once

#include <chrono>
#include <map>
#include <random>

#include <dtl/dtl.hpp>
#include <dtl/bloomfilter/block_addressing_logic.hpp>

#include "immintrin.h"

namespace dtl {


//===----------------------------------------------------------------------===//
struct blocked_bloomfilter_config {
  $u32 k = 8;
  $u32 word_size = 4; // [byte]
  $u32 word_cnt_per_block = 1;
  $u32 sector_cnt = 1;
  dtl::block_addressing addr_mode = dtl::block_addressing::POWER_OF_TWO;
  $u32 zone_cnt = 1;

  bool
  operator<(const blocked_bloomfilter_config &o) const {
    return k < o.k
        || (k == o.k && word_size  < o.word_size)
        || (k == o.k && word_size == o.word_size && word_cnt_per_block  < o.word_cnt_per_block)
        || (k == o.k && word_size == o.word_size && word_cnt_per_block == o.word_cnt_per_block && sector_cnt < o.sector_cnt)
        || (k == o.k && word_size == o.word_size && word_cnt_per_block == o.word_cnt_per_block && sector_cnt == o.sector_cnt && addr_mode < o.addr_mode)
        || (k == o.k && word_size == o.word_size && word_cnt_per_block == o.word_cnt_per_block && sector_cnt == o.sector_cnt && addr_mode == o.addr_mode && zone_cnt < o.zone_cnt);
  }

};
//===----------------------------------------------------------------------===//


namespace {

//===----------------------------------------------------------------------===//
struct tuning_params {
  $u32 unroll_factor = 1;

  tuning_params() = default;
  ~tuning_params() = default;
  tuning_params(const tuning_params& other) = default;
  tuning_params(tuning_params&& other) = default;

  tuning_params& operator=(const tuning_params& rhs) = default;
  tuning_params& operator=(tuning_params&& rhs) = default;
};
//===----------------------------------------------------------------------===//

} // anonymous namespace


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
  tune_unroll_factor(const blocked_bloomfilter_config& config) {
    throw std::runtime_error("Not supported");
  }


  /// Determines the best performing SIMD unrolling factor for all valid
  /// blocked Bloom filter configs.
  virtual void
  tune_unroll_factor() {
    throw std::runtime_error("Not supported");
  }

};
//===----------------------------------------------------------------------===//

} // namespace dtl