#pragma once

#include <functional>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/math.hpp>

#include "immintrin.h"

namespace dtl {

template<typename Tk, template<typename Ty> class hash_fn, typename Tw = u64>
struct bloomfilter {

  using key_t = typename std::remove_cv<Tk>::type;
  using word_t = typename std::remove_cv<Tw>::type;

  /// the length in bits. guaranteed to be a power of two
  u32 length;
  u32 length_mask;
  u32 word_bitlength = sizeof(word_t) * 8;
  u32 word_bitlength_log2 = dtl::log_2(word_bitlength);
  u32 word_bitlength_mask = word_bitlength - 1;
  std::vector<word_t> word_array;

  bloomfilter(u32 length)
      : length(next_power_of_two(length)), length_mask(next_power_of_two(length) - 1),
        word_array(next_power_of_two(length) / word_bitlength, 0) { }

  inline void
  insert(const key_t& key) {
    u32 hash_val = hash_fn<key_t>::hash(key);
    u32 bit_idx = hash_val & length_mask;
    u32 word_idx = bit_idx >> word_bitlength_log2;
    u32 in_word_idx = bit_idx & word_bitlength_mask;
    u32 second_in_word_idx = hash_val >> (32 - word_bitlength_log2);
    word_t word = word_array[word_idx];
    word |= word_t(1) << in_word_idx;
    word |= word_t(1) << second_in_word_idx;
    word_array[word_idx] = word;
  }


  inline u1
  contains(const key_t& key) const {
    u32 hash_val = hash_fn<key_t>::hash(key);
    u32 bit_idx = hash_val & length_mask;
    u32 word_idx = bit_idx >> word_bitlength_log2;
    u32 in_word_idx = bit_idx & word_bitlength_mask;
    u32 second_in_word_idx = hash_val >> (32 - word_bitlength_log2);
    word_t search_mask = (word_t(1) << in_word_idx) | (word_t(1) << second_in_word_idx);
    return (word_array[word_idx] & search_mask) == search_mask;
  }


  u64 popcnt() {
    $u64 pc = 0;
    for (auto& w : word_array) {
      pc += _mm_popcnt_u64(w);
    }
    return pc;
  }};

}
