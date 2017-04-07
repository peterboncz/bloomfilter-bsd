#pragma once

#include "adept.hpp"
#include <vector>
#include <functional>
#include "immintrin.h"
#include "math.hpp"
#include "simd.hpp"

namespace dtl {

template<typename Tk, template<typename Ty> class hash_fn>
struct bloomfilter {

  using word_t = $i64;
  u32 word_bitlength = sizeof(word_t) * 8;
  u32 word_bitlength_log2 = dtl::log_2(word_bitlength);
  u32 word_bitlength_mask = word_bitlength - 1;
  u32 length_mask;
  std::vector<word_t> word_array;

  bloomfilter(u32 length)
      : length_mask(next_power_of_two(length) - 1),
        word_array(next_power_of_two(length) / word_bitlength, 0) { }

  inline void
  insert(const Tk& key) {
    u32 hash_val = hash_fn<Tk>::hash(key);
    u32 bit_idx = hash_val & length_mask;
    u32 word_idx = bit_idx >> word_bitlength_log2;
    u32 in_word_idx = bit_idx & word_bitlength_mask;
    u32 second_in_word_idx = hash_val >> (word_bitlength - word_bitlength_log2);
    word_t word = word_array[word_idx];
    word |= 1 << in_word_idx;
    word |= 1 << second_in_word_idx;
    word_array[word_idx] = word;
  }


  inline u1
  contains(const Tk& key) const {
    u32 hash_val = hash_fn<Tk>::hash(key);
    u32 bit_idx = hash_val & length_mask;
    u32 word_idx = bit_idx >> word_bitlength_log2;
    u32 in_word_idx = bit_idx & word_bitlength_mask;
    u32 second_in_word_idx = hash_val >> (word_bitlength - word_bitlength_log2);
    word_t search_mask = (1 << in_word_idx) | (1 << second_in_word_idx);
    return (word_array[word_idx] & search_mask) == search_mask;
  }


  template<u64 vector_len>
  typename vec<Tk, vector_len>::mask_t
  contains(const vec<Tk, vector_len>& keys) const {
    using vec_t = vec<Tk, vector_len>;
    const vec_t bit_idxs = hash_fn<vec_t>::hash(keys) & length_mask;
    const vec_t word_idxs = bit_idxs >> word_bitlength_log2;
    const vec_t in_word_idxs = bit_idxs & word_bitlength_mask;
    const vec_t words = word_idxs.load(word_array.data());
    return (words & (Tk(1) << in_word_idxs)) != 0;
  }


  u64 popcnt() {
    $u64 pc = 0;
    for (auto& w : word_array) {
      pc += _mm_popcnt_u64(w);
    }
    return pc;
  }};

}