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


//===----------------------------------------------------------------------===//
/// The block addressing modes.
enum class block_addressing : u32 {
  /// The numbers of blocks is a power of two.
  POWER_OF_TWO,
  /// The numbers of blocks is restricted to 'cheap' magic numbers.
  MAGIC
};
//===----------------------------------------------------------------------===//


template<
    block_addressing          // the block addressing mode
>
struct block_addressing_logic {};


template<>
struct block_addressing_logic<block_addressing::MAGIC> {

  using size_t = $u32;
  using hash_value_t = $u32;
  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;


  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  const size_t block_cnt; // the number of blocks
  const size_t block_cnt_log2; // the number of bits required to address the individual blocks
  const size_t block_cnt_mask;
  const dtl::fast_divisor_u32_t fast_divisor;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Determines the actual block count based on the given bitlength of the Bloom filter.
  /// Note: The actual size of the Bloom filter might be larger than the given 'bitlength'.
  static size_t
  determine_block_cnt(const std::size_t data_bitlength, const std::size_t block_bitlength) {
    auto desired_block_cnt = (data_bitlength + (block_bitlength - 1)) / block_bitlength;
    auto actual_block_cnt = dtl::next_cheap_magic(desired_block_cnt).divisor;
    return actual_block_cnt;
  }
  //===----------------------------------------------------------------------===//


 public:

  explicit
  block_addressing_logic(const std::size_t data_bitlength, const std::size_t block_bitlength) noexcept
      : block_cnt(determine_block_cnt(data_bitlength, block_bitlength)),
        block_cnt_log2(dtl::log_2(dtl::next_power_of_two(block_cnt))),
        block_cnt_mask(dtl::next_power_of_two(block_cnt) - 1),
        fast_divisor(dtl::next_cheap_magic(block_cnt)) { }

  block_addressing_logic(const block_addressing_logic&) noexcept = default;

  block_addressing_logic(block_addressing_logic&&) noexcept = default;

  ~block_addressing_logic() noexcept = default;


  //===----------------------------------------------------------------------===//
  /// Returns the number of blocks.
  __forceinline__ __host__ __device__
  size_t
  get_block_cnt() const noexcept {
    return block_cnt;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the index of the block the hash value maps to.
  __forceinline__ __host__ __device__
  hash_value_t
  get_block_idx(const hash_value_t hash_value) const noexcept {
    const auto h = hash_value >> (hash_value_bitlength - get_required_addressing_bits());
    const auto block_idx = dtl::fast_mod_u32(h, fast_divisor);
    return block_idx;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the index of the block the hash value maps to.
  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  dtl::vec<hash_value_t, dtl::vector_length<Tv>::value>
  get_block_idxs(const Tv& hash_value) const noexcept {
    const auto h = hash_value >> (hash_value_bitlength - get_required_addressing_bits());
    const auto block_idx = dtl::fast_mod_u32(h, fast_divisor);
    return block_idx;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the number of bits required to address the individual blocks.
  __forceinline__ __host__ __device__
  uint32_t
  get_required_addressing_bits() const noexcept {
    return block_cnt_log2;
  }
  //===----------------------------------------------------------------------===//


};
//===----------------------------------------------------------------------===//


template<>
struct block_addressing_logic<block_addressing::POWER_OF_TWO> {

  using size_t = $u32;
  using hash_value_t = $u32;
  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;

  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  const size_t block_cnt;      // the number of blocks
  const size_t block_cnt_log2; // the number of bits required to address the individual blocks
  const size_t block_cnt_mask;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Determines the actual block count based on the given bitlength of the Bloom filter.
  /// Note: The actual size of the Bloom filter might be larger than the given 'bitlength'.
  static size_t
  determine_block_cnt(const std::size_t data_bitlength, const std::size_t block_bitlength) {
    auto desired_block_cnt = (data_bitlength + (block_bitlength - 1)) / block_bitlength;
    auto actual_block_cnt = dtl::next_power_of_two(desired_block_cnt);
    return actual_block_cnt;
  }
  //===----------------------------------------------------------------------===//


 public:

  explicit
  block_addressing_logic(const std::size_t data_bitlength, const std::size_t block_bitlength) noexcept
      : block_cnt(determine_block_cnt(data_bitlength, block_bitlength)),
        block_cnt_log2(dtl::log_2(block_cnt)),
        block_cnt_mask(block_cnt - 1) { }

  block_addressing_logic(const block_addressing_logic&) noexcept = default;

  block_addressing_logic(block_addressing_logic&&) noexcept = default;

  ~block_addressing_logic() noexcept = default;


  //===----------------------------------------------------------------------===//
  /// Returns the number of blocks.
  __forceinline__ __host__ __device__
  size_t
  get_block_cnt() const noexcept {
    return block_cnt;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the index of the block the hash value maps to.
  __forceinline__ __host__ __device__
  hash_value_t
  get_block_idx(const hash_value_t hash_value) const noexcept {
    const auto block_idx = (hash_value >> (hash_value_bitlength - get_required_addressing_bits()));
    return block_idx;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the index of the block the hash value maps to.
  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  dtl::vec<hash_value_t, dtl::vector_length<Tv>::value>
  get_block_idxs(const Tv& hash_value) const noexcept {
    const auto block_idx = (hash_value >> (hash_value_bitlength - get_required_addressing_bits()));
    return block_idx;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the number of bits required to address the individual blocks.
  __forceinline__ __host__ __device__
  uint32_t
  get_required_addressing_bits() const noexcept {
    return block_cnt_log2;
  }
  //===----------------------------------------------------------------------===//


};

} // namespace dtl
