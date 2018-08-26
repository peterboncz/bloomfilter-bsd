#pragma once

#include <atomic>
#include <bitset>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>
#include <dtl/math.hpp>
#include <dtl/simd.hpp>
#include <dtl/filter/blocked_bloomfilter/vector_helper.hpp>

#include "immintrin.h"

#include <boost/integer/static_min_max.hpp>


namespace dtl {

// Specialization for k = 8 with manual unrolling to improve IPC.
template<
    typename key_t,               // the key type
    typename word_t,              // the word type
    u32 s,                        // the numbers of sectors (must be a power of two)

    template<typename Ty, u32 i> class hasher,      // the hash function family to use
    typename hash_value_t,        // the hash value type

    u32 hash_fn_idx,              // current hash function index (used for recursion)
    u32 remaining_hash_bit_cnt    // the number of remaining hash bits (used for recursion)
>
struct word_block<key_t, word_t, s, 8, hasher, hash_value_t, hash_fn_idx, remaining_hash_bit_cnt, 8> {

  //===----------------------------------------------------------------------===//
  // Static part
  //===----------------------------------------------------------------------===//
  static constexpr u32 k = 8; // Specialization for k = 8
  static constexpr u32 remaining_k_cnt = k;

  static_assert(dtl::is_power_of_two(s), "Parameter 's' must be a power of two.");

  static constexpr u32 word_bitlength = sizeof(word_t) * 8;
  static constexpr u32 word_bitlength_mask = word_bitlength - 1;

  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;

  static constexpr u32 sector_bitlength = word_bitlength / s;
  static constexpr u32 sector_bitlength_log2 = dtl::ct::log_2_u32<sector_bitlength>::value;
  static constexpr word_t sector_mask() { return static_cast<word_t>(sector_bitlength) - 1; }

  static_assert(sector_bitlength >= 8, "A sector must be at least one byte in size.");

  static constexpr u32 hash_bit_cnt_per_k = sector_bitlength_log2;
  static constexpr u32 k_cnt_per_hash_value = ((sizeof(hash_value_t) * 8) / hash_bit_cnt_per_k) ; // consider -1 to respect hash fn weakness in the low order bits
  static constexpr u32 k_cnt_per_sector = k / s;

  static constexpr u32 current_k = k - remaining_k_cnt;

  static constexpr u1 rehash = remaining_hash_bit_cnt < hash_bit_cnt_per_k;
  static constexpr u32 remaining_hash_bit_cnt_after_rehash = rehash ? hash_value_bitlength : remaining_hash_bit_cnt;
  //===----------------------------------------------------------------------===//


  __forceinline__ __unroll_loops__ __host__ __device__
  static void
  which_bits(const key_t key, hash_value_t& hash_val, word_t& word) noexcept {

    // First k. - Set one bit in the given word; rehash if necessary
    static constexpr u32 hash_fn_idx_0 = hash_fn_idx;
    static constexpr u1 rehash_0 = remaining_hash_bit_cnt < hash_bit_cnt_per_k;
    static constexpr u32 remaining_hash_bit_cnt_after_rehash_0 = rehash_0 ? hash_value_bitlength : remaining_hash_bit_cnt;
    const hash_value_t hash_val_0 = rehash_0 ? hasher<key_t, hash_fn_idx_0>::hash(key) : hash_val;
    constexpr u32 sector_idx_0 = /*current_k = */ 0  / k_cnt_per_sector;
    constexpr u32 shift_0 = remaining_hash_bit_cnt_after_rehash_0 - hash_bit_cnt_per_k;
    static constexpr u32 remaining_hash_bit_cnt_0 = shift_0;

    // Second k. - Set one bit in the given word; rehash if necessary
    static constexpr u32 hash_fn_idx_1 = rehash_0 ? hash_fn_idx_0 + 1 : hash_fn_idx_0;
    static constexpr u1 rehash_1 = remaining_hash_bit_cnt_0 < hash_bit_cnt_per_k;
    static constexpr u32 remaining_hash_bit_cnt_after_rehash_1 = rehash_1 ? hash_value_bitlength : remaining_hash_bit_cnt_0;
    const hash_value_t hash_val_1 = rehash_1 ? hasher<key_t, hash_fn_idx_1>::hash(key) : hash_val_0;
    constexpr u32 sector_idx_1 = /*current_k = */ 1  / k_cnt_per_sector;
    constexpr u32 shift_1 = remaining_hash_bit_cnt_after_rehash_1 - hash_bit_cnt_per_k;
    static constexpr u32 remaining_hash_bit_cnt_1 = shift_1;

    // Third k. - Set one bit in the given word; rehash if necessary
    static constexpr u32 hash_fn_idx_2 = rehash_1 ? hash_fn_idx_1 + 1 : hash_fn_idx_1;
    static constexpr u1 rehash_2 = remaining_hash_bit_cnt_1 < hash_bit_cnt_per_k;
    static constexpr u32 remaining_hash_bit_cnt_after_rehash_2 = rehash_2 ? hash_value_bitlength : remaining_hash_bit_cnt_1;
    const hash_value_t hash_val_2 = rehash_2 ? hasher<key_t, hash_fn_idx_2>::hash(key) : hash_val_1;
    constexpr u32 sector_idx_2 = /*current_k = */ 2  / k_cnt_per_sector;
    constexpr u32 shift_2 = remaining_hash_bit_cnt_after_rehash_2 - hash_bit_cnt_per_k;
    static constexpr u32 remaining_hash_bit_cnt_2 = shift_2;

    // Forth k. - Set one bit in the given word; rehash if necessary
    static constexpr u32 hash_fn_idx_3 = rehash_2 ? hash_fn_idx_2 + 1 : hash_fn_idx_2;
    static constexpr u1 rehash_3 = remaining_hash_bit_cnt_2 < hash_bit_cnt_per_k;
    static constexpr u32 remaining_hash_bit_cnt_after_rehash_3 = rehash_3 ? hash_value_bitlength : remaining_hash_bit_cnt_2;
    const hash_value_t hash_val_3 = rehash_3 ? hasher<key_t, hash_fn_idx_3>::hash(key) : hash_val_2;
    constexpr u32 sector_idx_3 = /*current_k = */ 3  / k_cnt_per_sector;
    constexpr u32 shift_3 = remaining_hash_bit_cnt_after_rehash_3 - hash_bit_cnt_per_k;
    static constexpr u32 remaining_hash_bit_cnt_3 = shift_3;

    // Fifth k. - Set one bit in the given word; rehash if necessary
    static constexpr u32 hash_fn_idx_4 = rehash_3 ? hash_fn_idx_3 + 1 : hash_fn_idx_3;
    static constexpr u1 rehash_4 = remaining_hash_bit_cnt_3 < hash_bit_cnt_per_k;
    static constexpr u32 remaining_hash_bit_cnt_after_rehash_4 = rehash_4 ? hash_value_bitlength : remaining_hash_bit_cnt_3;
    const hash_value_t hash_val_4 = rehash_4 ? hasher<key_t, hash_fn_idx_4>::hash(key) : hash_val_3;
    constexpr u32 sector_idx_4 = /*current_k = */ 4  / k_cnt_per_sector;
    constexpr u32 shift_4 = remaining_hash_bit_cnt_after_rehash_4 - hash_bit_cnt_per_k;
    static constexpr u32 remaining_hash_bit_cnt_4 = shift_4;

    // Sixth k. - Set one bit in the given word; rehash if necessary
    static constexpr u32 hash_fn_idx_5 = rehash_4 ? hash_fn_idx_4 + 1 : hash_fn_idx_4;
    static constexpr u1 rehash_5 = remaining_hash_bit_cnt_4 < hash_bit_cnt_per_k;
    static constexpr u32 remaining_hash_bit_cnt_after_rehash_5 = rehash_5 ? hash_value_bitlength : remaining_hash_bit_cnt_4;
    const hash_value_t hash_val_5 = rehash_5 ? hasher<key_t, hash_fn_idx_5>::hash(key) : hash_val_4;
    constexpr u32 sector_idx_5 = /*current_k = */ 5  / k_cnt_per_sector;
    constexpr u32 shift_5 = remaining_hash_bit_cnt_after_rehash_5 - hash_bit_cnt_per_k;
    static constexpr u32 remaining_hash_bit_cnt_5 = shift_5;

    // Seventh k. - Set one bit in the given word; rehash if necessary
    static constexpr u32 hash_fn_idx_6 = rehash_5 ? hash_fn_idx_5 + 1 : hash_fn_idx_5;
    static constexpr u1 rehash_6 = remaining_hash_bit_cnt_5 < hash_bit_cnt_per_k;
    static constexpr u32 remaining_hash_bit_cnt_after_rehash_6 = rehash_6 ? hash_value_bitlength : remaining_hash_bit_cnt_5;
    const hash_value_t hash_val_6 = rehash_6 ? hasher<key_t, hash_fn_idx_6>::hash(key) : hash_val_5;
    constexpr u32 sector_idx_6 = /*current_k = */ 6  / k_cnt_per_sector;
    constexpr u32 shift_6 = remaining_hash_bit_cnt_after_rehash_6 - hash_bit_cnt_per_k;
    static constexpr u32 remaining_hash_bit_cnt_6 = shift_6;

    // Eighth k. - Set one bit in the given word; rehash if necessary
    static constexpr u32 hash_fn_idx_7 = rehash_6 ? hash_fn_idx_6 + 1 : hash_fn_idx_6;
    static constexpr u1 rehash_7 = remaining_hash_bit_cnt_6 < hash_bit_cnt_per_k;
    static constexpr u32 remaining_hash_bit_cnt_after_rehash_7 = rehash_7 ? hash_value_bitlength : remaining_hash_bit_cnt_6;
    const hash_value_t hash_val_7 = rehash_7 ? hasher<key_t, hash_fn_idx_7>::hash(key) : hash_val_6;
    constexpr u32 sector_idx_7 = /*current_k = */ 7  / k_cnt_per_sector;
    constexpr u32 shift_7 = remaining_hash_bit_cnt_after_rehash_7 - hash_bit_cnt_per_k;

    u32 bit_idx_0 = ((hash_val_0 >> shift_0) & sector_mask()) + (sector_idx_0 * sector_bitlength);
    const word_t word_0 = word_t(1) << bit_idx_0;
    u32 bit_idx_1 = ((hash_val_1 >> shift_1) & sector_mask()) + (sector_idx_1 * sector_bitlength);
    const word_t word_1 = word_t(1) << bit_idx_1;
    u32 bit_idx_2 = ((hash_val_2 >> shift_2) & sector_mask()) + (sector_idx_2 * sector_bitlength);
    const word_t word_2 = word_t(1) << bit_idx_2;
    u32 bit_idx_3 = ((hash_val_3 >> shift_3) & sector_mask()) + (sector_idx_3 * sector_bitlength);
    const word_t word_3 = word_t(1) << bit_idx_3;
    u32 bit_idx_4 = ((hash_val_4 >> shift_4) & sector_mask()) + (sector_idx_4 * sector_bitlength);
    const word_t word_4 = word_t(1) << bit_idx_4;
    u32 bit_idx_5 = ((hash_val_5 >> shift_5) & sector_mask()) + (sector_idx_5 * sector_bitlength);
    const word_t word_5 = word_t(1) << bit_idx_5;
    u32 bit_idx_6 = ((hash_val_6 >> shift_6) & sector_mask()) + (sector_idx_6 * sector_bitlength);
    const word_t word_6 = word_t(1) << bit_idx_6;
    u32 bit_idx_7 = ((hash_val_7 >> shift_7) & sector_mask()) + (sector_idx_7 * sector_bitlength);
    const word_t word_7 = word_t(1) << bit_idx_7;

    // Reduce
    const word_t word_01 = word_0 | word_1;
    const word_t word_23 = word_2 | word_3;
    const word_t word_45 = word_4 | word_5;
    const word_t word_67 = word_6 | word_7;
    const word_t word_0123 = word_01 | word_23;
    const word_t word_4567 = word_45 | word_67;
    word |= word_0123 | word_4567;
  }
  //===----------------------------------------------------------------------===//


  template<u64 n>
  __forceinline__ __unroll_loops__ __host__ __device__
  static void
  which_bits(const vec<key_t, n>& keys,
             vec<hash_value_t, n>& hash_vals,
             vec<word_t, n>& words) noexcept {

    // Typedef vector types
    using key_vt = vec<key_t, n>;
    using hash_value_vt = vec<hash_value_t, n>;
    using word_vt = vec<word_t, n>;

    hash_vals = rehash ? hasher<key_vt, hash_fn_idx>::hash(keys) : hash_vals;
    const hash_value_t remaining_hash_bit_cnt_after_rehash = rehash ? hash_value_bitlength : remaining_hash_bit_cnt;

    // Set one bit in the given word; rehash if necessary
    constexpr u32 sector_idx = current_k / k_cnt_per_sector;
    constexpr u32 shift = remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_k;
    hash_value_vt bit_idxs = sector_idx == 0 ? ((hash_vals >> shift) & sector_mask())
                                             : ((hash_vals >> shift) & sector_mask()) + (sector_idx * sector_bitlength);
    words |= word_vt(1) << internal::vector_convert<hash_value_t, word_t, n>::convert(bit_idxs);

    // Recurse
    word_block<key_t, word_t, s, k,
        hasher, hash_value_t,
        (rehash ? hash_fn_idx + 1 : hash_fn_idx), // increment the hash function index
        remaining_hash_bit_cnt_after_rehash >= hash_bit_cnt_per_k ? remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_k : 0, // the number of remaining hash bits
        remaining_k_cnt - 1> // decrement the remaining k counter
    ::which_bits(keys, hash_vals, words);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // The number of required hash functions.
  //===----------------------------------------------------------------------===//
  static constexpr u32 hash_fn_idx_end =
      word_block<key_t, word_t, s, k,
          hasher, hash_value_t,
          (rehash ? hash_fn_idx + 1 : hash_fn_idx), // increment the hash function index
          remaining_hash_bit_cnt_after_rehash >= hash_bit_cnt_per_k ? remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_k : 0, // the number of remaining hash bits
          remaining_k_cnt - 1> // decrement the remaining k counter
      ::hash_fn_idx_end;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // The number of required hash bits.
  //===----------------------------------------------------------------------===//
  static constexpr u32 remaining_hash_bits =
      word_block<key_t, word_t, s, k,
          hasher, hash_value_t,
          (rehash ? hash_fn_idx + 1 : hash_fn_idx), // increment the hash function index
          remaining_hash_bit_cnt_after_rehash >= hash_bit_cnt_per_k ? remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_k : 0, // the number of remaining hash bits
          remaining_k_cnt - 1> // decrement the remaining k counter
      ::remaining_hash_bits;
  //===----------------------------------------------------------------------===//


};

} // namespace dtl
