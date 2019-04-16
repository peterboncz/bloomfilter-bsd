#pragma once

#include <dtl/dtl.hpp>
#include <dtl/filter/cuckoofilter/cuckoofilter_config.hpp>
#include <dtl/filter/cuckoofilter/cuckoofilter_logic.hpp>
#include <dtl/filter/cuckoofilter/cuckoofilter_table.hpp>
#include <dtl/filter/blocked_bloomfilter/block_addressing_logic.hpp>

#include "filter_template.hpp"

namespace amsfilter {
namespace internal {

//===----------------------------------------------------------------------===//
template <std::size_t bits_per_tag, std::size_t tags_per_bucket>
using cf_table_t =
    dtl::cuckoofilter::cuckoofilter_table<bits_per_tag, tags_per_bucket>;
//===----------------------------------------------------------------------===//
// Disable the victim cache to improve performance.
static constexpr u1 cf_has_victim_cache = false;
//===----------------------------------------------------------------------===//
/// The template for Cuckoo filters.
template<
    std::size_t bits_per_tag,
    std::size_t tags_per_bucket,
    dtl::block_addressing addr = dtl::block_addressing::POWER_OF_TWO
>
using cf_t = dtl::cuckoofilter::cuckoofilter_logic<
    bits_per_tag, tags_per_bucket, cf_table_t, addr, cf_has_victim_cache>;
//===----------------------------------------------------------------------===//
/// The default instance.
using cf_default_t = cf_t<16, 4, dtl::block_addressing::POWER_OF_TWO>;
//===----------------------------------------------------------------------===//
/// Determines whether a given blocked Bloom filter configuration is valid.
template<
    std::size_t bits_per_tag,
    std::size_t tags_per_bucket,
    dtl::block_addressing addr = dtl::block_addressing::POWER_OF_TWO
>
struct cf_config_is_valid {
  static constexpr u1 value = (bits_per_tag == 8 || bits_per_tag== 16)
      && (tags_per_bucket == 1 || tags_per_bucket == 2 || tags_per_bucket == 4)
      && addr != dtl::block_addressing::DYNAMIC; // either MAGIC or POW2
};
//===----------------------------------------------------------------------===//
struct cf_invalid_t {};
//===----------------------------------------------------------------------===//
/// Resolves the Cuckoo filter type for the given parameters.  It refers
/// to 'cf_default_t' if the parameters are invalid or not supported.
template<
    std::size_t bits_per_tag,
    std::size_t tags_per_bucket,
    dtl::block_addressing addr = dtl::block_addressing::POWER_OF_TWO
>
struct cf_type {
  // Are the given parameters valid?
  static constexpr u1 config_is_valid =
      cf_config_is_valid<bits_per_tag, tags_per_bucket, addr>::value;
  // The fallback-type is used in case of invalid parameters.
//  using fallback_type = bbf_default_t;
  using fallback_type = cf_invalid_t;
  // The actual blocked Bloom filter type.
  using valid_type = cf_t<bits_per_tag, tags_per_bucket, addr>;

  /// The resolved Cuckoo filter type.
  using type = typename
    std::conditional<config_is_valid, valid_type, fallback_type>::type;
};
//===----------------------------------------------------------------------===//
} // namespace internal
} // namespace amsfilter
