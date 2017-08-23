#pragma once

#include <bitset>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include <dtl/dtl.hpp>
#include <dtl/div.hpp>
#include <dtl/math.hpp>


namespace dtl {

/// A multi-word block. The k bits are distributed among all words of the block (optionally in a sectorized manner).
template<typename Tk,      // the key type
    class Tblock,       // the block type
    template<typename Ty> class HashFn1,    // the first hash function to use
    template<typename Ty> class HashFn2    // the second hash function to use
>
struct bloomfilter_logic {

  using key_t = Tk;
  using block_t = Tblock;
  using size_t = $u32;

  // Inspect the given hash function
  static_assert(
      std::is_same<decltype(HashFn1<key_t>::hash(0)), decltype(HashFn2<key_t>::hash(0))>::value,
      "The two hash functions must return the same type.");
  using hash_value_t = $u32; //decltype(HashFn<key_t>::hash(0)); // TODO find out why NVCC complains
  static_assert(std::is_integral<hash_value_t>::value, "Hash function must return an integral type.");
  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;
  static constexpr u32 hash_fn_cnt = 2;

  static_assert(hash_value_bitlength == block_t::hash_value_bitlength, "Hash value bitlength mismatch.");


  // ---- Members ----
  const size_t block_cnt; // the number of blocks
//  const size_t block_cnt_log2; // the number of bits required to address the individual blocks
  const dtl::fast_divisor_u32_t fast_divisor;
  // ----


  static constexpr
  size_t
  determine_block_cnt(const std::size_t length) {
    u32 desired_block_cnt = (length + (block_t::block_bitlength - 1)) / block_t::block_bitlength;
    u32 actual_block_cnt = dtl::next_cheap_magic(desired_block_cnt).divisor;
    return actual_block_cnt;
//    u32 min_word_cnt = static_cast<size_t>(min_m / word_bitlength);
//    return std::max(actual_block_cnt, min_word_cnt);
  }



 public:

  explicit
  bloomfilter_logic(const std::size_t length)
      : block_cnt(determine_block_cnt(length)),
        fast_divisor(dtl::next_cheap_magic(block_cnt)) { }

  bloomfilter_logic(const bloomfilter_logic&) = default;

  bloomfilter_logic(bloomfilter_logic&&) = default;


  __forceinline__ __unroll_loops__ __host__ __device__
  std::size_t
  length() const noexcept {
    return block_cnt * block_t::block_bitlength;
  }

  __forceinline__ __unroll_loops__ __host__ __device__
  std::size_t
  word_cnt() const noexcept {
    return block_cnt * block_t::word_cnt;
  }


  __forceinline__ __unroll_loops__ __host__ //__device__
  void
  insert(const key_t& key, typename block_t::word_t* __restrict word_array) const noexcept {
    const hash_value_t hash_val_1 = HashFn1<key_t>::hash(key);
    const size_t block_idx = dtl::fast_mod_u32(hash_val_1, fast_divisor);
    const size_t word_idx = block_idx * block_t::word_cnt;
    const hash_value_t hash_val_2 = HashFn2<key_t>::hash(key);
    block_t::insert(hash_val_2, &word_array[word_idx]);
  }
  //===----------------------------------------------------------------------===//


  __forceinline__ __unroll_loops__ __host__ __device__
  u1
  contains(const key_t& key, const typename block_t::word_t* __restrict word_array) const noexcept {
    const hash_value_t hash_val_1 = HashFn1<key_t>::hash(key);
    const size_t block_idx = dtl::fast_mod_u32(hash_val_1, fast_divisor);
    const size_t word_idx = block_idx * block_t::word_cnt;
    const hash_value_t hash_val_2 = HashFn2<key_t>::hash(key);
    return block_t::contains(hash_val_2, &word_array[word_idx]);
  }
  //===----------------------------------------------------------------------===//


};

} // namespace dtl
