#pragma once

#include <bitset>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include <dtl/dtl.hpp>
#include <dtl/div.hpp>
#include <dtl/math.hpp>
#include <dtl/mem.hpp>
#include <dtl/simd.hpp>

#include "block_addressing_logic.hpp"
#include "hash_family.hpp"

namespace dtl {

// TODO rename to bloomfilter_logic
template<
    typename Tk,           // the key type
//    typename HashFn,       // the hash function (family) to use
    block_addressing AddrMode = block_addressing::POWER_OF_TWO  // the addressing scheme
>
struct std_bloomfilter {

  using key_t = Tk;
  using word_t = uint32_t;

  using HashFn = dtl::hash::dyn::mul32;

  //===----------------------------------------------------------------------===//
  // Inspect the given hash functions
  //===----------------------------------------------------------------------===//
  using hash_value_t = $u32; //decltype(HashFn<key_t>::hash(0)); // TODO find out why NVCC complains
  static_assert(std::is_integral<hash_value_t>::value, "Hash function must return an integral type.");
  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;
  //===----------------------------------------------------------------------===//


  static constexpr uint32_t block_bitlength = sizeof(word_t) * 8;

  using addr_t = block_addressing_logic<AddrMode>;
  using size_t = $u64;


  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  /// The addressing scheme.
  const addr_t addr;
  /// The number of bits to set/test per element.
  const uint32_t k;
  //===----------------------------------------------------------------------===//


 public:

  explicit
  std_bloomfilter(const std::size_t length, const uint32_t k) noexcept
      : addr(length, block_bitlength), k(k) { }

  std_bloomfilter(const std_bloomfilter&) noexcept = default;

  std_bloomfilter(std_bloomfilter&&) noexcept = default;


  //===----------------------------------------------------------------------===//
  /// Returns the size of the Bloom filter (in number of bits).
  __forceinline__ __host__ __device__
  std::size_t
  length() const noexcept {
    return static_cast<std::size_t>(addr.block_cnt) * sizeof(word_t) * 8;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the number of words the Bloom filter consists of.
  __forceinline__ __host__ __device__
  std::size_t
  word_cnt() const noexcept {
    return addr.block_cnt; // word == block
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Insert the given key/element into the filter.
  __forceinline__ __host__
  void
  insert(const key_t& key, word_t* __restrict filter) const noexcept {
    constexpr uint32_t word_bitlength = sizeof(word_t) * 8;
    constexpr uint32_t word_bitlength_log2 = dtl::ct::log_2<word_bitlength>::value;
    constexpr word_t word_mask = (word_t(1u) << word_bitlength_log2) - 1;
    const auto addressing_bits = addr.get_required_addressing_bits();

    // Set one bit per word at a time.
    for (uint32_t current_k = 0; current_k < k; current_k++) {
      const hash_value_t hash_val = HashFn::hash(key, current_k);
      const hash_value_t word_idx = addr.get_block_idx(hash_val); // word == block
      const hash_value_t bit_idx = (hash_val >> (word_bitlength - word_bitlength_log2 - addressing_bits)) & word_mask;
      filter[word_idx] |= word_t(1u) << bit_idx;
    }
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Inserts multiple keys/elements into the filter.
  __forceinline__
  void
  batch_insert(const key_t* __restrict keys, const uint32_t key_cnt,
               word_t* __restrict filter) const {
    for (uint32_t j = 0; j < key_cnt; j++) {
      insert(keys[j], filter);
    }
  };
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  __forceinline__ __host__ __device__
  u1
  contains(const key_t& key, const word_t* __restrict filter) const noexcept {
    constexpr uint32_t word_bitlength = sizeof(word_t) * 8;
    constexpr uint32_t word_bitlength_log2 = dtl::ct::log_2<word_bitlength>::value;
    constexpr word_t word_mask = (word_t(1u) << word_bitlength_log2) - 1;
    const auto addressing_bits = addr.get_required_addressing_bits();

    // Test one bit per word at a time.
    for (uint32_t current_k = 0; current_k < k; current_k++) {
      const hash_value_t hash_val = HashFn::hash(key, current_k);
      const hash_value_t word_idx = addr.get_block_idx(hash_val);
      const hash_value_t bit_idx = (hash_val >> (word_bitlength - word_bitlength_log2 - addressing_bits)) & word_mask;
      const bool hit = filter[word_idx] & (word_t(1u) << bit_idx);
      if (!hit) return false;
    }
    return true;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // TODO should be private
  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  typename dtl::vec<key_t, dtl::vector_length<Tv>::value>::mask_t
  simd_contains(const Tv& keys,
                const word_t* __restrict filter) const noexcept {
    using vec_t = dtl::vec<key_t, dtl::vector_length<Tv>::value>;
    using hash_value_vt = dtl::vec<hash_value_t, dtl::vector_length<Tv>::value>;
    using mask_t = typename vec_t::mask_t;

    constexpr uint32_t word_bitlength = sizeof(word_t) * 8;
    constexpr uint32_t word_bitlength_log2 = dtl::ct::log_2<word_bitlength>::value;
    constexpr uint32_t word_mask = (word_t(1u) << word_bitlength_log2) - 1;
    const auto addressing_bits = addr.get_required_addressing_bits();

    // Test one bit per word at a time.
    const hash_value_vt hash_vals = HashFn::hash(keys, 0);
    const auto word_idxs = addr.get_block_idx(hash_vals);
    const auto bit_idxs = (hash_vals >> (word_bitlength - word_bitlength_log2 - addressing_bits)) & word_mask;
    const auto lsb_set = vec_t::make(word_t(1u));
    mask_t exec_mask = (dtl::gather(filter, word_idxs) & (lsb_set << bit_idxs)) != 0;

    if (exec_mask.any()) {
      for (uint32_t current_k = 1; current_k < k; current_k++) {
        const vec_t hash_vals = HashFn::hash(keys, current_k);
        const vec_t word_idxs = addr.get_block_idx(hash_vals).zero_mask(exec_mask);
        const auto bit_idxs = (hash_vals >> (word_bitlength - word_bitlength_log2 - addressing_bits)) & word_mask;
        exec_mask &= (dtl::gather(filter, word_idxs) & (lsb_set << bit_idxs)) != 0;
        if (exec_mask.none()) return !exec_mask;
      }

    }
    return !exec_mask;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  __forceinline__
  uint64_t
  batch_contains(const word_t* __restrict filter,
                 const key_t* __restrict keys, const uint32_t key_cnt,
                 uint32_t* __restrict match_positions, const uint32_t match_offset) const {
    constexpr u32 mini_batch_size = 16;
    const u32 mini_batch_cnt = key_cnt / mini_batch_size;

    $u32* match_writer = match_positions;
    for ($u32 mb = 0; mb < mini_batch_cnt; mb++) {
      for (uint32_t j = mb * mini_batch_size; j < ((mb + 1) * mini_batch_size); j++) {
        const auto is_contained = contains(keys[j], filter);
        *match_writer = j + match_offset;
        match_writer += is_contained;
      }
    }
    for (uint32_t j = (mini_batch_cnt * mini_batch_size); j < key_cnt; j++) {
      const auto is_contained = contains(keys[j], filter);
      *match_writer = j + match_offset;
      match_writer += is_contained;
    }
    return match_writer - match_positions;
  };
  //===----------------------------------------------------------------------===//

};

} // namespace dtl
