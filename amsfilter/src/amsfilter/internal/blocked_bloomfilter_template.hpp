#pragma once

#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/block_addressing_logic.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_batch_probe_base.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_logic.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_block_logic_sgew.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_block_logic_sltw.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_block_logic_zoned.hpp>
#include <dtl/filter/blocked_bloomfilter/hash_family.hpp>

namespace amsfilter {
namespace internal {
//===----------------------------------------------------------------------===//
// Typedefs. - Currently, only 32-bit keys are supported.
using key_t = $u32;
using hash_value_t = $u32;
using word_t = $u32;
//===----------------------------------------------------------------------===//
/// Type switch for the different block types. - The proper block type is chosen
/// based on the parameters 'word_cnt', 'sector_cnt', and 'zone_cnt'.
template<u32 word_cnt, u32 sector_cnt, u32 zone_cnt, u32 k>
struct bbf_block_switch {

  // The first hash function (index) to use inside the block. Note: 0 is used
  // for block addressing.
  static constexpr u32 block_hash_fn_idx = 1;

  // DEPRECATED
  static constexpr u1 early_out = false;

  // Is zoned (aka cache-sectorized).
  static constexpr u1 is_zoned = zone_cnt > 1 && zone_cnt < sector_cnt
      && sector_cnt == word_cnt && sector_cnt > 1;

  /// Classic blocked Bloom filter, where the block size exceeds the size of a
  /// word.
  using blocked = dtl::multisector_block<key_t, word_t, word_cnt, 1, k,
      dtl::hasher, hash_value_t, block_hash_fn_idx, 0, sector_cnt, early_out>;

  /// Sectorized blocked Bloom filter, where the number of sectors is less than
  /// the number of words per block (causing a random block access pattern).
  ///
  /// This block type is know to be dominated by classic-blocked wrt. precision
  /// and by register-blocked and cache-sectorized wrt. performance on CPUs.
  using sectorized_rnd = dtl::multisector_block<key_t, word_t, word_cnt,
      sector_cnt, k, dtl::hasher, hash_value_t, block_hash_fn_idx, 0, sector_cnt,
      early_out>;

  /// Sectorized blocked Bloom filter, where the number of sectors is equal to
  /// the number of words per block.
  using sectorized_seq = dtl::multiword_block<key_t, word_t, word_cnt,
      sector_cnt, k, dtl::hasher, hash_value_t, block_hash_fn_idx, 0, word_cnt,
      early_out>;

  static_assert(!is_zoned || word_cnt == sector_cnt,
      "The number of words must be equal to the number of sectors.");

  /// Cache-Sectorized (aka zoned) blocked Bloom filter, where the number of
  /// sectors is equal to the number of words per block.
  using zoned = dtl::multizone_block<key_t, word_t, word_cnt, zone_cnt,
      k, dtl::hasher, hash_value_t, block_hash_fn_idx, 0, zone_cnt, early_out>;

  /// Refers to the implementation.
  using type =
  typename std::conditional<is_zoned,
      zoned,
      typename std::conditional<(sector_cnt >= word_cnt),
          sectorized_seq,
          typename std::conditional<(sector_cnt > 1),
              sectorized_rnd,
              blocked
          >::type
      >::type
  >::type;

  /// The number of accessed words per lookup. Actually, the number of LOAD-CMP
  /// sequences.
  static constexpr std::size_t word_access_cnt =
      (is_zoned)
          ? zone_cnt
          : (sector_cnt >= word_cnt)
              ? word_cnt
              : k;
};
//===----------------------------------------------------------------------===//
/// The template for (almost all kinds of) blocked Bloom filters.
template<
    u32 word_cnt,
    u32 sector_cnt,
    u32 zone_cnt,
    u32 k,
    dtl::block_addressing addr = dtl::block_addressing::POWER_OF_TWO,
    u1 early_out = false // DEPRECATED
>
using bbf_t = dtl::blocked_bloomfilter_logic<key_t, dtl::hasher,
    typename bbf_block_switch<word_cnt, sector_cnt, zone_cnt, k>::type, addr>;
//===----------------------------------------------------------------------===//
/// The base class for all blocked Bloom filters.
using bbf_base_t = dtl::blocked_bloomfilter_logic_base;
using bbf_batch_probe_base_t = dtl::blocked_bloomfilter_batch_probe_base;
//===----------------------------------------------------------------------===//
/// The default instance.
using bbf_default_t = bbf_t<1, 1, 1, 4, dtl::block_addressing::POWER_OF_TWO>;
//===----------------------------------------------------------------------===//
/// Returns the number of accessed words per lookup. Actually, the number of
/// LOAD-CMP sequences.
static constexpr
std::size_t
get_word_access_cnt(u32 word_cnt, u32 sector_cnt, u32 zone_cnt, u32 k) {
  u1 is_zoned = zone_cnt > 0 && zone_cnt < sector_cnt && sector_cnt == word_cnt
      && sector_cnt > 1;
  return (is_zoned)
            ? zone_cnt
            : (sector_cnt >= word_cnt)
                ? word_cnt
                : k;
};
static constexpr
std::size_t
get_word_access_cnt(const dtl::blocked_bloomfilter_config& c) {
  return get_word_access_cnt(c.word_cnt_per_block, c.sector_cnt, c.zone_cnt, c.k);
};
//===----------------------------------------------------------------------===//
/// Determines whether a given blocked Bloom filter configuration is valid.
template<
    u32 word_cnt,
    u32 sector_cnt,
    u32 zone_cnt,
    u32 k,
    dtl::block_addressing addr
>
struct bbf_config_is_valid {

  // No zoning when z = s (different to the definition given in the paper).
  static constexpr u1 is_zoned = zone_cnt < sector_cnt;

  static constexpr u1 value = word_cnt > 0
      && sector_cnt > 0
      && sector_cnt <= word_cnt
      && zone_cnt > 0
      && zone_cnt <= sector_cnt
      && (is_zoned ? sector_cnt == word_cnt : true)
      && (k % zone_cnt) == 0
      && k > 0 && k <= 16
      && addr != dtl::block_addressing::DYNAMIC // either MAGIC or POW2
      ;
};
//===----------------------------------------------------------------------===//
struct bbf_invalid_t {
  static constexpr u32 word_cnt_per_block = 0;
  static constexpr u32 k = 0;
  static constexpr u32 sector_cnt = 0;
  static constexpr u32 zone_cnt = 0;
  static constexpr dtl::block_addressing addr_mode =
      dtl::block_addressing::POWER_OF_TWO;
};
//===----------------------------------------------------------------------===//
/// Resolves the blocked Bloom filter type for the given parameters.  It refers
/// to 'bbf_default_t' if the parameters are invalid or not supported.
template<u32 w, u32 s, u32 z, u32 k, dtl::block_addressing a>
struct bbf_type {
  // Are the given parameters valid?
  static constexpr u1 config_is_valid = bbf_config_is_valid<w,s,z,k,a>::value;
  // The fallback-type is used in case of invalid parameters.
//  using fallback_type = bbf_default_t;
  using fallback_type = bbf_invalid_t;
  // The actual blocked Bloom filter type.
  using valid_type = bbf_t<w, s, z, k, a>;

  /// The resolved blocked Bloom filter type.
  using type = typename
  std::conditional<config_is_valid, valid_type, fallback_type>::type;
};
//===----------------------------------------------------------------------===//
} // namespace internal
} // namespace amsfilter
