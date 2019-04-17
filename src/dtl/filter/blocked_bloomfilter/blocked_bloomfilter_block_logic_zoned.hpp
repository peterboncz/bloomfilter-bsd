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

#include "immintrin.h"

#include <boost/integer/static_min_max.hpp>

#include "blocked_bloomfilter_block_logic_sgew.hpp"
#include "vector_helper.hpp"

namespace dtl {


//===----------------------------------------------------------------------===//
// Recursive template to work with blocks consisting of multiple sectors,
// whereas a sector corresponds to a native processor word.
//
// In contrast to a sectorized block, the bits are not set/tested in ALL
// sectors. Instead, multiple sectors are grouped to z zones, and k/z bits are
// set in each zone. Within each zone, only a single word (=sector) is accessed.
// This scheme allows to spread out the k bits over larger blocks
// (preferably 512 bits) rather than s*S bits, while the sequential access
// pattern is retained.
//===----------------------------------------------------------------------===//
template<
    typename _key_t,              // the key type
    typename _word_t,             // the word type
    u32 _word_cnt,                // the number of words per block
    u32 z,                        // the numbers of zones (must be a power of two)
    u32 _k,                       // the number of bits to set/test
    template<typename Ty, u32 i> class hasher,      // the hash function family to use
    typename hash_value_t,        // the hash value type to use

    u32 hash_fn_idx,              // current hash function index (used for recursion)
    u32 remaining_hash_bit_cnt,   // the number of remaining hash bits (used for recursion)
    u32 remaining_zone_cnt,       // the remaining number of zones (used for recursion)

    u1 early_out = false          // allows for branching out during lookups (before the next sector is tested)
>
struct multizone_block {

  static_assert(early_out == false, "The early-out feature is not supported.");
  static_assert(z < _word_cnt, "The number of zones must be less than the number of words per block.");

  //===----------------------------------------------------------------------===//
  // Static part
  //===----------------------------------------------------------------------===//
  using key_t = _key_t;
  using word_t = _word_t;

  static constexpr u32 word_cnt = _word_cnt;
  static_assert(dtl::is_power_of_two(word_cnt), "Parameter 'word_cnt' must be a power of two.");
  static constexpr u32 sector_cnt = word_cnt;
  static constexpr u32 sector_cnt_log2 = dtl::ct::log_2<sector_cnt>::value;

  static constexpr u32 zone_cnt = z;
  static_assert(dtl::is_power_of_two(zone_cnt), "Parameter 'zone_cnt' must be a power of two.");
  static constexpr u32 current_zone_idx = zone_cnt - remaining_zone_cnt;

  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;

  static constexpr u32 k = _k;
  static constexpr u32 k_cnt_per_zone = k / zone_cnt;
  static_assert(k % zone_cnt == 0, "Parameter 'k' must be dividable by 'z'.");

  static constexpr u32 word_cnt_per_zone = word_cnt / zone_cnt;
  static_assert(word_cnt % zone_cnt == 0, "Parameter 'word_cnt' must be dividable by 'z'.");
  __forceinline__ __host__ __device__
  static constexpr word_t zone_mask() { return static_cast<word_t>(word_cnt_per_zone) - 1; }

  // The number of hash bits required to determine the sector within a zone.
  static constexpr u32 hash_bit_cnt_per_zone = dtl::ct::log_2_u32<word_cnt_per_zone>::value;

  // Rehash if necessary
  static constexpr u1 rehash = remaining_hash_bit_cnt < hash_bit_cnt_per_zone;
  static constexpr u32 hash_fn_idx_after_rehash = rehash ? hash_fn_idx + 1 : hash_fn_idx;
  static constexpr u32 remaining_hash_bit_cnt_after_rehash = rehash ? hash_value_bitlength : remaining_hash_bit_cnt;

  //===----------------------------------------------------------------------===//
  // Insert
  //===----------------------------------------------------------------------===//
  __forceinline__
  static void
  insert(word_t* __restrict block_ptr, const key_t key) noexcept {

    hash_value_t hash_val = 0;

    // Call the recursive function
    static constexpr u32 remaining_hash_bits = 0;
    multizone_block<key_t, word_t, word_cnt, zone_cnt, k,
                    hasher, hash_value_t, hash_fn_idx, remaining_hash_bits,
                    remaining_zone_cnt, early_out>
      ::insert(block_ptr, key, hash_val);
  }

  __forceinline__
  static void
  insert_atomic(word_t* __restrict block_ptr, const key_t key) noexcept {

    hash_value_t hash_val = 0;

    // Call the recursive function
    static constexpr u32 remaining_hash_bits = 0;
    multizone_block<key_t, word_t, word_cnt, zone_cnt, k,
                    hasher, hash_value_t, hash_fn_idx, remaining_hash_bits,
                    remaining_zone_cnt, early_out>
      ::insert_atomic(block_ptr, key, hash_val);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Insert (Recursive)
  //===----------------------------------------------------------------------===//
  __forceinline__
  static void
  insert(word_t* __restrict block_ptr, const key_t key, hash_value_t& hash_val) noexcept {

    // Rehash if necessary
    hash_val = rehash ? hasher<key_t, hash_fn_idx>::hash(key) : hash_val;

    // Word index within zone
    constexpr u32 shift = remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_zone;
    u32 word_idx_within_zone = (hash_val >> shift) & zone_mask(); // TODO consider using bit-extract

    // Word index within block
    u32 word_idx = (word_cnt_per_zone * current_zone_idx) + word_idx_within_zone;

    // Load the word of interest
    word_t word = block_ptr[word_idx];

    word_t bit_mask = 0;

    using word_block_t =
      word_block<key_t, word_t, 1 /*sector_cnt_per_word*/, k_cnt_per_zone,
                 hasher, hash_value_t, hash_fn_idx_after_rehash, remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_zone,
                 k_cnt_per_zone>;
    word_block_t::which_bits(key, hash_val, bit_mask);

    // Update the bit vector
    word |= bit_mask;
    block_ptr[word_idx] = word;


    // Process remaining zones recursively, if any
    using block_t =
      multizone_block<key_t, word_t, word_cnt, zone_cnt, k,
                      hasher, hash_value_t,
                      word_block_t::hash_fn_idx_end, word_block_t::remaining_hash_bits,
                      remaining_zone_cnt - 1>;
    block_t::insert(block_ptr, key, hash_val);
  }

  __forceinline__
  static void
  insert_atomic(word_t* __restrict block_ptr, const key_t key, hash_value_t& hash_val) noexcept {

    // Rehash if necessary
    hash_val = rehash ? hasher<key_t, hash_fn_idx>::hash(key) : hash_val;

    // Word index within zone
    constexpr u32 shift = remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_zone;
    u32 word_idx_within_zone = (hash_val >> shift) & zone_mask(); // TODO consider using bit-extract

    // Word index within block
    u32 word_idx = (word_cnt_per_zone * current_zone_idx) + word_idx_within_zone;

    word_t bit_mask = 0;

    using word_block_t =
      word_block<key_t, word_t, 1 /*sector_cnt_per_word*/, k_cnt_per_zone,
                 hasher, hash_value_t, hash_fn_idx_after_rehash, remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_zone,
                 k_cnt_per_zone>;
    word_block_t::which_bits(key, hash_val, bit_mask);

    // Atomically update the bit vector
    word_t* word_ptr = &block_ptr[word_idx];
    std::atomic<word_t>* atomic_word_ptr = reinterpret_cast<std::atomic<word_t>*>(word_ptr);
    $u1 success = false;
    do {
      // Load the word of interest
      word_t word = atomic_word_ptr->load();
      // Update the bit vector
      word_t updated_word = word | bit_mask;
      success = atomic_word_ptr->compare_exchange_weak(word, updated_word);
    } while (!success);

    // Process remaining zones recursively, if any
    using block_t =
      multizone_block<key_t, word_t, word_cnt, zone_cnt, k,
                      hasher, hash_value_t,
                      word_block_t::hash_fn_idx_end, word_block_t::remaining_hash_bits,
                      remaining_zone_cnt - 1>;
    block_t::insert_atomic(block_ptr, key, hash_val);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains
  //===----------------------------------------------------------------------===//
  __forceinline__ __unroll_loops__ __host__ __device__
  static u1
  contains(const word_t* __restrict block_ptr, const key_t key) noexcept {

    hash_value_t hash_val = 0;

    // Call the recursive function
    static constexpr u32 remaining_hash_bits = 0;
    return multizone_block<key_t, word_t, word_cnt, zone_cnt, k,
                           hasher, hash_value_t, hash_fn_idx, remaining_hash_bits,
                           remaining_zone_cnt, early_out>
      ::contains(block_ptr, key, hash_val, true);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains (Recursive)
  //===----------------------------------------------------------------------===//
  __forceinline__ __unroll_loops__ __host__ __device__
  static u1
  contains(const word_t* __restrict block_ptr, const key_t key, hash_value_t& hash_val, u1 is_contained_in_block) noexcept {

    // Rehash if necessary
    hash_val = rehash ? hasher<key_t, hash_fn_idx>::hash(key) : hash_val;

    // Word index within zone
    constexpr u32 shift = remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_zone;
    u32 word_idx_within_zone = (hash_val >> shift) & zone_mask(); // TODO consider using bit-extract

    // Word index within block
    u32 word_idx = (word_cnt_per_zone * current_zone_idx) + word_idx_within_zone;

    // Load the word of interest
    word_t word = block_ptr[word_idx];

    word_t bit_mask = 0;

    using word_block_t =
      word_block<key_t, word_t, 1 /*sector_cnt_per_word*/, k_cnt_per_zone,
                 hasher, hash_value_t, hash_fn_idx_after_rehash, remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_zone,
                 k_cnt_per_zone>;
    word_block_t::which_bits(key, hash_val, bit_mask);

    // Bit testing
    u1 found_in_zone = (word & bit_mask) == bit_mask;


    // Process remaining zones recursively, if any
    using block_t =
      multizone_block<key_t, word_t, word_cnt, zone_cnt, k,
                      hasher, hash_value_t,
                      word_block_t::hash_fn_idx_end, word_block_t::remaining_hash_bits,
                      remaining_zone_cnt - 1>;
    return block_t::contains(block_ptr, key, hash_val, found_in_zone & is_contained_in_block);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains (SIMD)
  //===----------------------------------------------------------------------===//
  template<u64 n>
  __forceinline__ __unroll_loops__
  static auto
  contains(const vec<key_t,n>& keys,
           const word_t* __restrict bitvector_base_address,
           const vec<hash_value_t,n>& block_start_word_idxs) noexcept {

    vec<hash_value_t, n> hash_vals(0);
    const auto is_contained_in_block_mask = vec<word_t,n>::mask::make_all_mask(); // true

    // Call recursive function
    static constexpr u32 remaining_hash_bits = 0;
    return multizone_block<key_t, word_t, word_cnt, z, k,
                           hasher, hash_value_t, hash_fn_idx, remaining_hash_bits,
                           remaining_zone_cnt, early_out>
      ::contains(keys, hash_vals, bitvector_base_address, block_start_word_idxs, is_contained_in_block_mask);

  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains (SIMD, Recursive)
  //===----------------------------------------------------------------------===//
  template<u64 n>
  __forceinline__ __unroll_loops__
  static auto
  contains(const vec<key_t,n>& keys,
           vec<hash_value_t,n>& hash_vals,
           const word_t* __restrict bitvector_base_address,
           const vec<hash_value_t,n>& block_start_word_idxs,
           const typename vec<word_t,n>::mask is_contained_in_block_mask) noexcept {

    // Rehash if necessary
    hash_vals = rehash ? hasher<vec<key_t, n>, hash_fn_idx>::hash(keys) : hash_vals;

    // Word index within zone
    constexpr u32 shift = remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_zone;
    auto word_idx_within_zone = (hash_vals >> shift) & zone_mask();

    // Word index (abs)
    auto word_idxs = block_start_word_idxs + word_idx_within_zone + (word_cnt_per_zone * current_zone_idx);

    // Load the words of interest
    const auto words = internal::vector_gather<word_t, hash_value_t, n>::gather(bitvector_base_address, word_idxs);

    vec<word_t,n> bit_masks = 0;

    using word_block_t =
      word_block<key_t, word_t, 1 /*sector_cnt_per_word*/, k_cnt_per_zone,
                 hasher, hash_value_t, hash_fn_idx_after_rehash, remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_zone,
                 k_cnt_per_zone>;
    word_block_t::which_bits(keys, hash_vals, bit_masks);

    // Bit testing
    typename vec<word_t,n>::mask found_in_zone = (words & bit_masks) == bit_masks;


    // Process remaining zones recursively, if any
    using block_t =
      multizone_block<key_t, word_t, word_cnt, zone_cnt, k,
                      hasher, hash_value_t,
                      word_block_t::hash_fn_idx_end, word_block_t::remaining_hash_bits,
                      remaining_zone_cnt - 1>;
    return block_t::contains(keys, hash_vals, bitvector_base_address, block_start_word_idxs, found_in_zone & is_contained_in_block_mask);
  }
  //===----------------------------------------------------------------------===//

};


//===----------------------------------------------------------------------===//
template<
    typename key_t,               // the key type
    typename word_t,              // the word type
    u32 word_cnt,                 // the number of words per block
    u32 z,                        // the numbers of zones (must be a power of two)
    u32 k,                        // the number of bits to set/test
    template<typename Ty, u32 i> class hasher,      // the hash function family to use
    typename hash_value_t,        // the hash value type to use

    u32 hash_fn_idx,              // current hash function index (used for recursion)
    u32 remaining_hash_bit_cnt,   // the number of remaining hash bits (used for recursion)

    u1 early_out                  // allows for branching out during lookups (before the next sector is tested)
>
struct multizone_block<key_t, word_t, word_cnt, z, k, hasher, hash_value_t, hash_fn_idx, remaining_hash_bit_cnt,
                       0 /* no more remaining zones */, early_out>  {

  //===----------------------------------------------------------------------===//
  // Insert
  //===----------------------------------------------------------------------===//
  __forceinline__
  static void
  insert(word_t* __restrict block_ptr, const key_t key, hash_value_t& hash_val) noexcept {
    // End of recursion.
  }
  __forceinline__
  static void
  insert_atomic(word_t* __restrict block_ptr, const key_t key, hash_value_t& hash_val) noexcept {
    // End of recursion.
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains
  //===----------------------------------------------------------------------===//
  __forceinline__ __host__ __device__
  static u1
  contains(const word_t* __restrict block_ptr, const key_t key, hash_value_t& hash_val, u1 is_contained_in_block) noexcept {
    // End of recursion.
    return is_contained_in_block;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains (SIMD)
  //===----------------------------------------------------------------------===//
  template<u64 n>
  __forceinline__ __unroll_loops__
  static auto
  contains(const vec<key_t,n>& keys,
           vec<hash_value_t,n>& hash_vals,
           const word_t* __restrict bitvector_base_address,
           const vec<key_t,n>& block_start_word_idxs,
           const typename vec<word_t,n>::mask is_contained_in_block_mask) noexcept {
    // End of recursion.
    return is_contained_in_block_mask;
  }
  //===----------------------------------------------------------------------===//

};

} // namespace dtl
