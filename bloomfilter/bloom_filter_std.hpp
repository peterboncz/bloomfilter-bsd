#pragma once

#include <bitset>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include <dtl/dtl.hpp>
#include <dtl/div.hpp>
#include <dtl/math.hpp>

#include "bloomfilter_addressing_logic.hpp"

namespace dtl {
namespace bloom_filter {

template<
    typename Tk,           // the key type
    typename HashFn,       // the hash function (family) to use
    block_addressing AddrMode = block_addressing::POWER_OF_TWO  // the addressing scheme
>
struct bloom_filter_std {

  using key_t = Tk;
  using word_t = uint32_t;

  //===----------------------------------------------------------------------===//
  // Inspect the given hash functions
  //===----------------------------------------------------------------------===//

  using hash_value_t = $u32; //decltype(HashFn<key_t>::hash(0)); // TODO find out why NVCC complains
  static_assert(std::is_integral<hash_value_t>::value, "Hash function must return an integral type.");
  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;

  //===----------------------------------------------------------------------===//

  // A fake block; required by addressing logic
  struct block_t {
    static constexpr uint32_t block_bitlength = sizeof(word_t) * 8;
    static constexpr uint32_t word_cnt = 1;
  };

  using addr_t = bloomfilter_addressing_logic<AddrMode, hash_value_t, block_t>;
  using size_t = $u32;


  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  /// The addressing scheme.
  const addr_t addr;
  const uint32_t k;
  //===----------------------------------------------------------------------===//


 public:

  explicit
  bloom_filter_std(const std::size_t length, const uint32_t k) noexcept
      : addr(length), k(k) { }

  bloom_filter_std(const bloom_filter_std&) noexcept = default;

  bloom_filter_std(bloom_filter_std&&) noexcept = default;


  /// Returns the size of the Bloom filter (in number of bits).
  __forceinline__ __host__ __device__
  std::size_t
  length() const noexcept {
    return static_cast<std::size_t>(addr.block_cnt) * sizeof(word_t) * 8;
  }


//  /// Returns the number of blocks the Bloom filter consists of.
//  __forceinline__ __host__ __device__
//  std::size_t
//  block_cnt() const noexcept {
//    return addr.block_cnt;
//  }


  /// Returns the number of words the Bloom filter consists of.
  __forceinline__ __host__ __device__
  std::size_t
  word_cnt() const noexcept {
    return addr.block_cnt;
  }


  __forceinline__ __host__
  void
  insert(const key_t& key, word_t* __restrict filter) const noexcept {
    constexpr uint32_t word_bitlength = sizeof(word_t) * 8;
    constexpr uint32_t word_bitlength_log2 = dtl::ct::log_2<word_bitlength>::value;
    constexpr uint32_t word_mask = (word_t(1u) << word_bitlength_log2) - 1;
    const auto addressing_bits = addr.get_required_addressing_bits();

    // Set one bit per word at a time.
    for (uint32_t current_k = 0; current_k < k; current_k++) {
      const hash_value_t hash_val = HashFn::hash(key, current_k);
      const size_t word_idx = addr.get_word_idx(hash_val);
      const size_t bit_idx = (hash_val >> (word_bitlength - addressing_bits)) & word_mask;
      filter[word_idx] |= word_t(1u) << bit_idx;
    }
  }


  __forceinline__
  uint64_t
  batch_insert(const key_t* __restrict keys, const uint32_t key_cnt,
                 const word_t* __restrict filter,
                 uint32_t* __restrict match_positions, const uint32_t match_offset) const {
    $u32* match_writer = match_positions;
    for (uint32_t j = 0; j < key_cnt; j++) {
      const auto is_contained = contains(keys[j], filter);
      *match_writer = j + match_offset;
      match_writer += is_contained;
    }
    return match_writer - match_positions;
  };


  __forceinline__ __host__ __device__
  u1
  contains(const key_t& key, const word_t* __restrict filter) const noexcept {
    constexpr uint32_t word_bitlength = sizeof(word_t) * 8;
    constexpr uint32_t word_bitlength_log2 = dtl::ct::log_2<word_bitlength>::value;
    constexpr uint32_t word_mask = (word_t(1u) << word_bitlength_log2) - 1;
    const auto addressing_bits = addr.get_required_addressing_bits();

    // Test one bit per word at a time.
    for (uint32_t current_k = 0; current_k < k; current_k++) {
      const hash_value_t hash_val = HashFn::hash(key, current_k);
      const size_t word_idx = addr.get_word_idx(hash_val);
      const size_t bit_idx = (hash_val >> (word_bitlength - addressing_bits)) & word_mask;
      const bool hit = filter[word_idx] & (word_t(1u) << bit_idx);
      if (!hit) return false;
    }
    return true;
  }


  __forceinline__
  uint64_t
  batch_contains(const key_t* __restrict keys, const uint32_t key_cnt,
                 const word_t* __restrict filter,
                 uint32_t* __restrict match_positions, const uint32_t match_offset) const {
    $u32* match_writer = match_positions;
    for (uint32_t j = 0; j < key_cnt; j++) {
      const auto is_contained = contains(keys[j], filter);
      *match_writer = j + match_offset;
      match_writer += is_contained;
    }
    return match_writer - match_positions;
  };


};

} // namespace bloom_filter
} // namespace dtl
