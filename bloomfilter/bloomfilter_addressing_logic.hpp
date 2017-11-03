#pragma once

#include <bitset>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include <dtl/dtl.hpp>
#include <dtl/div.hpp>
#include <dtl/math.hpp>
#include <dtl/simd.hpp>


namespace dtl {

enum class block_addressing {
  POWER_OF_TWO,
  MAGIC
};

template<
    block_addressing,          // the block addressing mode
    typename _hash_value_t,    // the hash value type
    typename _block_t          // the block type
>
struct bloomfilter_addressing_logic {};


template<
    typename _hash_value_t,    // the hash value type
    typename _block_t          // the block type
>
struct bloomfilter_addressing_logic<block_addressing::MAGIC, _hash_value_t, _block_t> {

  using block_t = _block_t;
  using size_t = $u64;
  using hash_value_t = _hash_value_t;


  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  const size_t block_cnt; // the number of blocks
  const size_t block_cnt_log2; // the number of bits required to address the individual blocks
  const dtl::fast_divisor_u32_t fast_divisor;
  //===----------------------------------------------------------------------===//


  /// Determines the actual block count based on the given bitlength of the Bloom filter.
  /// Note: The actual size of the Bloom filter might be larger than the given 'bitlength'.
  static constexpr
  size_t
  determine_block_cnt(const std::size_t bitlength) {
    u32 desired_block_cnt = (bitlength + (block_t::block_bitlength - 1)) / block_t::block_bitlength;
    u32 actual_block_cnt = dtl::next_cheap_magic(desired_block_cnt).divisor;
    return actual_block_cnt;
  }


 public:

  explicit
  bloomfilter_addressing_logic(const std::size_t length) noexcept
      : block_cnt(determine_block_cnt(length)),
        block_cnt_log2(dtl::log_2(dtl::next_power_of_two(block_cnt))),
        fast_divisor(dtl::next_cheap_magic(block_cnt)) { }

  bloomfilter_addressing_logic(const bloomfilter_addressing_logic&) noexcept = default;

  bloomfilter_addressing_logic(bloomfilter_addressing_logic&&) noexcept = default;

  ~bloomfilter_addressing_logic() noexcept = default;


  __forceinline__ __host__ __device__
  std::size_t
  length() const noexcept {
    return block_cnt * block_t::block_bitlength;
  }


  __forceinline__ __host__ __device__
  std::size_t
  word_cnt() const noexcept {
    return block_cnt * block_t::word_cnt;
  }


  /// Returns the index of the block the hash value maps to.
  __forceinline__ __host__ __device__
  hash_value_t
  get_block_idx(const hash_value_t hash_value) const noexcept {
    const auto block_idx = dtl::fast_mod_u32(hash_value, fast_divisor);
    return block_idx;
  }


  /// Returns the index of the block the hash value maps to.
  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  dtl::vec<hash_value_t, dtl::vector_length<Tv>::value>
  get_block_idxs(const Tv& hash_value) const noexcept {
    const auto block_idx = dtl::fast_mod_u32(hash_value, fast_divisor);
    return block_idx;
  }


  /// Returns the index of the first word of the block the hash value maps to.
  __forceinline__ __host__ __device__
  hash_value_t
  get_word_idx(const hash_value_t hash_value) const noexcept {
    const hash_value_t block_idx = get_block_idx(hash_value);
    const hash_value_t word_idx = block_idx * block_t::word_cnt;
    return word_idx;
  }


  /// Returns the number of bits required to address the individual blocks.
  __forceinline__ __host__ __device__
  uint32_t
  get_required_addressing_bits() const noexcept {
    return block_cnt_log2;
  }


};


//===----------------------------------------------------------------------===//


template<
    typename _hash_value_t,    // the hash value type
    typename _block_t          // the block type
>
struct bloomfilter_addressing_logic<block_addressing::POWER_OF_TWO, _hash_value_t, _block_t> {

  using block_t = _block_t;
  using size_t = $u32;
  using hash_value_t = _hash_value_t;
  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;

  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  const size_t block_cnt;      // the number of blocks
  const size_t block_cnt_log2; // the number of bits required to address the individual blocks
  const size_t block_cnt_mask;
  //===----------------------------------------------------------------------===//


  /// Determines the actual block count based on the given bitlength of the Bloom filter.
  /// Note: The actual size of the Bloom filter might be larger than the given 'bitlength'.
  static constexpr
  size_t
  determine_block_cnt(const std::size_t bitlength) {
    u32 desired_block_cnt = (bitlength + (block_t::block_bitlength - 1)) / block_t::block_bitlength;
    u32 actual_block_cnt = dtl::next_power_of_two(desired_block_cnt);
    return actual_block_cnt;
  }


 public:

  explicit
  bloomfilter_addressing_logic(const std::size_t length) noexcept
      : block_cnt(determine_block_cnt(length)),
        block_cnt_log2(dtl::log_2(block_cnt)),
        block_cnt_mask(block_cnt - 1) { }

  bloomfilter_addressing_logic(const bloomfilter_addressing_logic&) noexcept = default;

  bloomfilter_addressing_logic(bloomfilter_addressing_logic&&) noexcept = default;

  ~bloomfilter_addressing_logic() noexcept = default;


  __forceinline__ __host__ __device__
  std::size_t
  length() const noexcept {
    return block_cnt * block_t::block_bitlength;
  }


  __forceinline__ __host__ __device__
  std::size_t
  word_cnt() const noexcept {
    return block_cnt * block_t::word_cnt;
  }


  /// Returns the index of the block the hash value maps to.
  __forceinline__ __host__ __device__
  hash_value_t
  get_block_idx(const hash_value_t hash_value) const noexcept {
    const auto block_idx = (hash_value >> (hash_value_bitlength - block_cnt_log2)) & block_cnt_mask;
    return block_idx;
  }


  /// Returns the index of the block the hash value maps to.
  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  dtl::vec<hash_value_t, dtl::vector_length<Tv>::value>
  get_block_idxs(const Tv& hash_value) const noexcept {
    const auto block_idx = (hash_value >> (hash_value_bitlength - block_cnt_log2)) & block_cnt_mask;
    return block_idx;
  }


  /// Returns the index of the first word of the block the hash value maps to.
  __forceinline__ __host__ __device__
  size_t
  get_word_idx(const hash_value_t hash_value) const noexcept {
    const size_t block_idx = get_block_idx(hash_value);
    const size_t word_idx = block_idx * block_t::word_cnt;
    return word_idx;
  }


  /// Returns the number of bits required to address the individual blocks.
  __forceinline__ __host__ __device__
  uint32_t
  get_required_addressing_bits() const noexcept {
    return block_cnt_log2;
  }


};

} // namespace dtl
