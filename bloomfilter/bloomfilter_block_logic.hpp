#pragma once

#include <bitset>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>
#include <dtl/math.hpp>

#include <cub/cub.cuh>

#include "immintrin.h"

namespace dtl {

/// A multi-word block. The k bits are distributed among all words of the block (optionally in a sectorized manner).
template<typename Tk,      // the key type
    typename Tw = u32,     // the word type to use for the bitset
    typename Th = u32,     // the hash value type7
    u32 K = 3,             // the number of bits to set
    u32 B = 2,             // the number of words per block (block size)
    u1 Sectorized = false
>
struct bloomfilter_block_logic {

  using key_t = typename std::remove_cv<Tk>::type;
  using word_t = typename std::remove_cv<Tw>::type;
  using size_t = $u32;

  static_assert(std::is_integral<key_t>::value, "The key type must be an integral type.");
  static_assert(std::is_integral<word_t>::value, "The word type must be an integral type.");


  static constexpr u32 word_bitlength = sizeof(word_t) * 8;
  static constexpr u32 word_bitlength_log2 = dtl::ct::log_2_u32<word_bitlength>::value;
  static constexpr u32 word_bitlength_log2_mask = (1u << word_bitlength_log2) - 1;
  static constexpr u32 word_bitlength_mask = word_bitlength - 1;

  // The number of words per block.
  static constexpr u32 word_cnt = B;
  static_assert(is_power_of_two(B), "The number of words per block must be a power of two.");
  // The number of bits needed to address the individual word within a block.
  static constexpr u32 word_cnt_log2 = dtl::ct::log_2_u32<word_cnt>::value;
  static constexpr u32 word_cnt_mask = word_cnt - 1;

  static constexpr u32 block_bitlength = word_bitlength * word_cnt;
  static constexpr u32 block_bitlength_log2 = dtl::ct::log_2_u32<block_bitlength>::value;
  static constexpr u32 block_bitlength_mask = word_cnt - 1;


  // The number of bits to set per element.
  static constexpr u32 k = K;
  static_assert(k > 0, "Parameter 'k' must be at least '1'.");


  // Split the block into multiple sectors (or sub-blocks) with a length of a power of two.
  // Note that sectorization is a specialization. Having only one sector = no sectorization.
  static constexpr u1 sectorized = Sectorized;
  static_assert(!sectorized || ((block_bitlength / dtl::next_power_of_two(k)) != 0),
                "The number of sectors must be greater than zero. Probably the given k is set to high.");

  static constexpr u32 sector_cnt = (!sectorized) ? 1
                                                  : block_bitlength / (block_bitlength / static_cast<u32>(dtl::next_power_of_two(k)));
  static constexpr u32 sector_cnt_mask = sector_cnt - 1;
  static constexpr u32 sector_bitlength = block_bitlength / sector_cnt;
  // The number of bits needed to address the individual bits within a sector.
  static constexpr u32 sector_bitlength_log2 = dtl::ct::log_2_u32<sector_bitlength>::value;
//  static constexpr u32 sector_bitlength_log2_mask = (1u << sector_bitlength_log2) - 1;
  static constexpr word_t sector_mask = sector_bitlength - 1;

  // The (static) length of the Bloom filter block.
  static constexpr size_t m = word_cnt * word_bitlength;

  using hash_value_t = Th;
  static constexpr size_t hash_value_bitlength = sizeof(hash_value_t) * 8;

  // The number of hash bits required per k.
  static constexpr size_t required_hash_bits_per_k = sector_bitlength_log2;

  // The number of hash bits required per element.
  static constexpr size_t required_hash_bits_per_element = k * required_hash_bits_per_k;

  static constexpr size_t max_k = hash_value_bitlength / required_hash_bits_per_k;

  static_assert(required_hash_bits_per_element <= hash_value_bitlength,
                "The required hash bits exceed the number of bits provided by the hash function.");


  static constexpr u32 shift = sector_cnt >= word_cnt
                               ? dtl::ct::log_2_u32<sector_cnt / word_cnt>::value
                               : dtl::ct::log_2_u32<word_cnt / sector_cnt>::value;


  static constexpr cub::CacheLoadModifier cache_load_modifier = cub::LOAD_CG;

private:

  //===----------------------------------------------------------------------===//
  // Sectorization
  //===----------------------------------------------------------------------===//

  __forceinline__ __unroll_loops__ __host__ __device__
  static void
  insert_sectorized(const hash_value_t& hash_val, word_t* __restrict word_array) noexcept {
    if (sector_cnt >= word_cnt) {
      // a sector does not exceed a word
      auto h = hash_val;
      for (size_t i = 0; i < k; i++) {
        // consume the high order bits
        u32 sector_idx = i & sector_cnt_mask;
        u32 word_idx = sector_idx >> shift;
        u32 in_word_sector_idx = sector_idx & ((word_bitlength / sector_bitlength) - 1);
        word_t word = word_array[word_idx];
        u32 bit_idx = h >> (hash_value_bitlength - sector_bitlength_log2);
        word |= word_t(1) << bit_idx + (in_word_sector_idx * sector_bitlength);
        word_array[word_idx] = word;
        h <<= required_hash_bits_per_k;
      }
      return;
    }
    else {
      // a sector exceeds a word
      auto h = hash_val;
      for (size_t i = 0; i < k; i++) {
        // consume the high order bits
        u32 sector_idx = i & sector_cnt_mask;
        u32 bit_idx_in_sector = h >> (hash_value_bitlength - sector_bitlength_log2);
        u32 word_idx = (sector_idx << shift) + (bit_idx_in_sector >> word_bitlength_log2);
        u32 bit_idx_in_word = bit_idx_in_sector & word_bitlength_log2_mask;
        word_t word = word_array[word_idx];
        word |= word_t(1) << bit_idx_in_word;
        word_array[word_idx] = word;
        h <<= required_hash_bits_per_k;
      }
      return;
    }
  }
  //===----------------------------------------------------------------------===//

  __forceinline__ __unroll_loops__ __host__ __device__
  static u1
  contains_sectorized(const hash_value_t& hash_val, const word_t* __restrict word_array) noexcept {
    if (sector_cnt == word_cnt) {
      // sector size == word size => 1 bit to test per word
      auto h = hash_val;
      auto r = true;
      for (size_t word_idx = 0; word_idx < word_cnt; word_idx++) {
        // consume the high order bits
#if defined(__CUDA_ARCH__)
        const word_t word = cub::ThreadLoad<cache_load_modifier>(word_array + word_idx);
#else
        const word_t word = word_array[word_idx];
#endif // defined(__CUDA_ARCH__)
        u32 bit_idx = h >> (hash_value_bitlength - sector_bitlength_log2);
        r &= dtl::bits::bit_test(word, bit_idx);
        h <<= required_hash_bits_per_k;
      }
      return r;
    }
    else if (sector_cnt > word_cnt) {
      // a sector does not exceed a word
      // further the numbers of sectors per word is >= 2
      // => test sector_cnt/word_cnt bits at once
      static constexpr u32 sector_cnt_per_word = sector_cnt / word_cnt;
      auto h = hash_val;
      auto r = true;
      for (size_t word_idx = 0; word_idx < word_cnt; word_idx++) {
#if defined(__CUDA_ARCH__)
        const word_t word = cub::ThreadLoad<cache_load_modifier>(word_array + word_idx);
#else
        const word_t word = word_array[word_idx];
#endif // defined(__CUDA_ARCH__)
        word_t search_mask = 0;
        for (size_t in_word_sector_idx = 0; in_word_sector_idx < sector_cnt_per_word; in_word_sector_idx++) {
          // consume the high order bits
          u32 bit_idx = h >> (hash_value_bitlength - sector_bitlength_log2);
          search_mask |= word_t(1) << (bit_idx + in_word_sector_idx * sector_bitlength);
          h <<= required_hash_bits_per_k;
        }
        r &= (word & search_mask) == search_mask;
      }
      return r;
    }
    else {
      // a sector exceeds a word
      auto h = hash_val;
      auto r = true;
      for (size_t i = 0; i < k; i++) {
        // consume the high order bits
        u32 sector_idx = i & sector_cnt_mask;
        u32 bit_idx_in_sector = h >> (hash_value_bitlength - sector_bitlength_log2);
        u32 word_idx = (sector_idx << shift) + (bit_idx_in_sector >> word_bitlength_log2);
#if defined(__CUDA_ARCH__)
        const word_t word = cub::ThreadLoad<cache_load_modifier>(word_array + word_idx);
#else
        const word_t word = word_array[word_idx];
#endif // defined(__CUDA_ARCH__)
        u32 bit_idx_in_word = bit_idx_in_sector & word_bitlength_log2_mask;
        r &= dtl::bits::bit_test(word, bit_idx_in_word);
        h <<= required_hash_bits_per_k;
      }
      return r;
    }
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Default (non-sectorized)
  //===----------------------------------------------------------------------===//

  /// Set k bits within the block
  __forceinline__ __unroll_loops__ __host__ __device__
  static void
  insert_default(const hash_value_t& hash_val, word_t* __restrict word_array) noexcept {
    auto h = hash_val;
    for (size_t i = 0; i < k; i++) {
      // consume the high order bits
      u32 word_idx = word_cnt == 1 ? 0u : h >> (hash_value_bitlength - word_cnt_log2);
      word_t word = word_array[word_idx];
      u32 bit_idx = (h >> (hash_value_bitlength - word_cnt_log2 - sector_bitlength_log2)) & ((hash_value_t(1) << sector_bitlength_log2) - 1);
      word_t word1 = word_t(1) << bit_idx;
      word |= word1;
      word_array[word_idx] = word;
      h <<= required_hash_bits_per_k;
    }
  }
  //===----------------------------------------------------------------------===//


  __forceinline__ __unroll_loops__ __host__ __device__
  static u1
  contains_default(const hash_value_t& hash_val, const word_t* __restrict word_array) noexcept {
    if (word_cnt == 1) {
      if (k == 1) {
        u32 bit_idx = hash_val >> (hash_value_bitlength - sector_bitlength_log2);
#if defined(__CUDA_ARCH__)
        const word_t word = cub::ThreadLoad<cache_load_modifier>(word_array);
#else
        const word_t word = *word_array;
#endif // defined(__CUDA_ARCH__)
        return dtl::bits::bit_test(word, bit_idx);
      }
      else {
        // test all bits in one go
#if defined(__CUDA_ARCH__)
        const word_t word = cub::ThreadLoad<cache_load_modifier>(word_array);
#else
        const word_t word = word_array[0];
#endif // defined(__CUDA_ARCH__)
        word_t search_mask = 0;
        for (size_t i = 0; i < k; i++) {
          // consume the high order bits
          u32 bit_idx = hash_val >> (hash_value_bitlength - sector_bitlength_log2 * (i + 1)) & ((hash_value_t(1) << sector_bitlength_log2) - 1);
          search_mask |= word_t(1) << bit_idx;
        }
        return (word & search_mask) == search_mask;;
      }
    }
    else {
      auto h = hash_val;
      auto r = true;
      for (size_t i = 0; i < k; i++) {
        // consume the high order bits
        u32 word_idx = word_cnt == 1 ? 0u : h >> (hash_value_bitlength - word_cnt_log2);
#if defined(__CUDA_ARCH__)
        const word_t word = cub::ThreadLoad<cache_load_modifier>(word_array + word_idx);
#else
        const word_t word = word_array[word_idx];
#endif // defined(__CUDA_ARCH__)
        u32 bit_idx = (h >> (hash_value_bitlength - word_cnt_log2 - sector_bitlength_log2)) & ((hash_value_t(1) << sector_bitlength_log2) - 1);
        r &= dtl::bits::bit_test(word, bit_idx);
        h <<= required_hash_bits_per_k;
      }
      return r;
    }
  }
  //===----------------------------------------------------------------------===//


public:

  __forceinline__ __unroll_loops__ __host__ //__device__
  static void
  insert(const hash_value_t& hash_val, word_t* __restrict word_array) noexcept {
    if (sectorized) {
      insert_sectorized(hash_val, word_array);
    }
    else {
      insert_default(hash_val, word_array);
    }
  }
  //===----------------------------------------------------------------------===//


  __forceinline__ __unroll_loops__ __host__ __device__
  static u1
  contains(const hash_value_t& hash_val, const word_t* __restrict word_array) noexcept {
    if (sectorized) {
      return contains_sectorized(hash_val, word_array);
    }
    else {
      return contains_default(hash_val, word_array);
    }
  }
  //===----------------------------------------------------------------------===//


};

} // namespace dtl
