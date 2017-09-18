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

template<typename Tk,      // the key type
    typename Taddr,        // the addressing logic
    typename Tblock,       // the block type
    template<typename Ty> class HashFn1,    // the first hash function to use
    template<typename Ty> class HashFn2     // the second hash function to use
>
struct bloomfilter_logic {

  using key_t = Tk;
  using addr_t = Taddr;
  using block_t = Tblock;
  using size_t = $u32;

  //===----------------------------------------------------------------------===//
  // Inspect the given hash functions
  //===----------------------------------------------------------------------===//
  static_assert(
      std::is_same<decltype(HashFn1<key_t>::hash(0)), decltype(HashFn2<key_t>::hash(0))>::value,
      "The two hash functions must return the same type.");

  using hash_value_t = $u32; //decltype(HashFn<key_t>::hash(0)); // TODO find out why NVCC complains

  static_assert(std::is_integral<hash_value_t>::value, "Hash function must return an integral type.");

  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;

  static_assert(hash_value_bitlength == block_t::hash_value_bitlength, "Hash value bitlength mismatch.");

  static constexpr u32 hash_fn_cnt = 2;


  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  const addr_t addr;
  //===----------------------------------------------------------------------===//


 public:

  explicit
  bloomfilter_logic(const std::size_t length) noexcept
      : addr(length) { }

  bloomfilter_logic(const bloomfilter_logic&) noexcept = default;

  bloomfilter_logic(bloomfilter_logic&&) noexcept = default;


  /// Returns the size of the Bloom filter (in number of bits).
  __forceinline__ __host__ __device__
  std::size_t
  length() const noexcept {
    return static_cast<std::size_t>(addr.block_cnt) * block_t::block_bitlength;
  }


  /// Returns the number of blocks the bitvector consists of.
  __forceinline__ __host__ __device__
  std::size_t
  block_cnt() const noexcept {
    return addr.block_cnt;
  }


  /// Returns the number of words the bitvector consists of.
  __forceinline__ __host__ __device__
  std::size_t
  word_cnt() const noexcept {
    return addr.block_cnt * block_t::word_cnt;
  }


  __forceinline__ __host__ __device__
  hash_value_t
  hash1(const key_t key) const noexcept {
    return HashFn1<key_t>::hash(key);
  }


  __forceinline__ __host__ __device__
  hash_value_t
  hash2(const key_t key) const noexcept {
    return HashFn2<key_t>::hash(key);
  }


  __forceinline__ __host__
  void
  insert(const key_t& key, typename block_t::word_t* __restrict__ word_array) const noexcept {
    const hash_value_t hash_val_1 = hash1(key);
    const size_t word_idx = addr.get_word_idx(hash_val_1);
    const hash_value_t hash_val_2 = hash2(key);
    block_t::insert(hash_val_2, &word_array[word_idx]);
  }
  //===----------------------------------------------------------------------===//


  __forceinline__ __host__ __device__
  u1
  contains(const key_t& key, const typename block_t::word_t* __restrict__ word_array) const noexcept {
    const hash_value_t hash_val_1 = hash1(key);
    const size_t word_idx = addr.get_word_idx(hash_val_1);
    const hash_value_t hash_val_2 = hash2(key);
    return block_t::contains(hash_val_2, &word_array[word_idx]);
  }
  //===----------------------------------------------------------------------===//


};

} // namespace dtl
