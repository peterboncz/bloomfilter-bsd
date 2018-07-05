#pragma once

#include <bitset>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>
#include <dtl/math.hpp>
#include <dtl/simd.hpp>
#include <dtl/filter/block_addressing_logic.hpp>
#include <dtl/filter/blocked_bloomfilter_batch_dispatch.hpp>
#include <dtl/filter/blocked_bloomfilter_block_logic.hpp>

#include <boost/integer/static_min_max.hpp>

#include "immintrin.h"


namespace dtl {

//===----------------------------------------------------------------------===//
// A high-performance blocked Bloom filter template.
//===----------------------------------------------------------------------===//
template<
    typename Tk,                  // the key type
    template<typename Ty, u32 i> class Hasher,      // the hash function family to use
    typename Tb,                  // the block type
    dtl::block_addressing block_addressing = dtl::block_addressing::POWER_OF_TWO,
    u1 early_out = false          // branch out early, if possible
>
struct blocked_bloomfilter_logic {

  //===----------------------------------------------------------------------===//
  // The static part.
  //===----------------------------------------------------------------------===//
  using key_t = typename std::remove_cv<Tk>::type;
  using word_t = typename std::remove_cv<typename Tb::word_t>::type;
  using block_t = typename std::remove_cv<Tb>::type;
  static_assert(std::is_integral<key_t>::value, "The key type must be an integral type.");
  static_assert(std::is_integral<word_t>::value, "The word type must be an integral type.");

  using size_t = $u64;

  static constexpr u32 word_cnt_per_block = block_t::word_cnt;
  static constexpr u32 word_cnt_per_block_log2 = dtl::ct::log_2<word_cnt_per_block>::value;
  static_assert(dtl::is_power_of_two(word_cnt_per_block), "The number of words per block must be a power of two.");

  static constexpr u32 word_bitlength = sizeof(word_t) * 8;
  static constexpr u32 word_bitlength_log2 = dtl::ct::log_2_u32<word_bitlength>::value;

  static constexpr u32 block_bitlength = word_bitlength * word_cnt_per_block;

  // Inspect the given hash function.
  using hash_value_t = decltype(Hasher<key_t, 0>::hash(42)); // TODO find out why NVCC complains
  static_assert(std::is_integral<hash_value_t>::value, "Hash function must return an integral type.");
  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;

  static constexpr u32 k = block_t::k;
  static constexpr u32 sector_cnt = block_t::sector_cnt;

  // The block addressing logic (either MAGIC or POWER_OF_TWO).
  using addr_t = block_addressing_logic<block_addressing>;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  /// Addressing logic instance.
  const addr_t addr;
  //===----------------------------------------------------------------------===//


  /// C'tor.
  /// Note, that the actual length might be (slightly) different to the
  /// desired length. The function get_length() returns the actual length.
  explicit
  blocked_bloomfilter_logic(const std::size_t desired_length)
      : addr((desired_length + (block_bitlength - 1)) / block_bitlength) { }

  /// Copy c'tor
  blocked_bloomfilter_logic(const blocked_bloomfilter_logic&) = default;

  ~blocked_bloomfilter_logic() = default;


  //===----------------------------------------------------------------------===//
  // Insert
  //===----------------------------------------------------------------------===//
//  __forceinline__ __host__
  void
  insert(word_t* __restrict filter_data,
         const key_t key) noexcept {
    const hash_value_t block_addressing_hash_val = Hasher<const key_t, 0>::hash(key);
    const hash_value_t block_idx = addr.get_block_idx(block_addressing_hash_val);
    const hash_value_t bitvector_word_idx = block_idx << word_cnt_per_block_log2;

    auto block_ptr = &filter_data[bitvector_word_idx];

    block_t::insert(block_ptr, key);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Batch Insert
  //===----------------------------------------------------------------------===//
  void
  batch_insert(word_t* __restrict filter_data,
               const key_t* keys, u32 key_cnt) noexcept {
    for (std::size_t i = 0; i < key_cnt; i++) {
      insert(filter_data, keys[i]);
    }
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains
  //===----------------------------------------------------------------------===//
  u1
  contains(const word_t* __restrict filter_data,
           const key_t key) const noexcept {
    const hash_value_t block_addressing_hash_val = Hasher<const key_t, 0>::hash(key);
    const hash_value_t block_idx = addr.get_block_idx(block_addressing_hash_val);
    const hash_value_t bitvector_word_idx = block_idx << word_cnt_per_block_log2;

    const auto block_ptr = &filter_data[bitvector_word_idx];

    u1 found = block_t::contains(block_ptr, key);
    return found;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains (SIMD) - called by batch dispatch
  //===----------------------------------------------------------------------===//
  template<u64 n> // the vector length
  __forceinline__ __host__
  typename vec<word_t, n>::mask
  contains_vec(const word_t* __restrict filter_data,
               const vec<key_t, n>& keys) const noexcept {
    // Typedef the vector types.
    using key_vt = vec<key_t, n>;
    using hash_value_vt = vec<hash_value_t, n>;

    const hash_value_vt block_addressing_hash_vals = Hasher<key_vt, 0>::hash(keys);
    const hash_value_vt block_idxs = addr.get_block_idxs(block_addressing_hash_vals);
    const hash_value_vt bitvector_word_idx = block_idxs << word_cnt_per_block_log2;

    auto found = block_t::contains(keys, filter_data, bitvector_word_idx);
    return found;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Batch-wise Contains
  //===----------------------------------------------------------------------===//
  template<u64 vector_len>
  $u64
  batch_contains(const word_t* __restrict filter_data,
                 const key_t* __restrict keys, u32 key_cnt,
                 $u32* __restrict match_positions, u32 match_offset) const {
    return internal::dispatch<blocked_bloomfilter_logic, vector_len>
             ::batch_contains(*this, filter_data,
                              keys, key_cnt,
                              match_positions, match_offset);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the (actual) length in bits.
  std::size_t
  get_length() const noexcept {
    return u64(addr.get_block_cnt()) * block_bitlength;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the (actual) length in number of words.
  std::size_t
  word_cnt() const noexcept {
    return u64(addr.get_block_cnt()) * word_cnt_per_block;
  }
  //===----------------------------------------------------------------------===//

};

} // namespace dtl
