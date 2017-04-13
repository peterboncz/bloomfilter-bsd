#pragma once

#include <functional>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/math.hpp>
//#include <dtl/simd.hpp>

#include "immintrin.h"

namespace dtl {

template<typename Tk, template<typename Ty> class hash_fn>
struct bloomfilter {

  using key_t = typename std::remove_cv<Tk>::type;
  using word_t = $u64;

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


//  template<u64 vector_len>
//  typename vec<key_t, vector_len>::mask_t
//  contains(const vec<key_t, vector_len>& keys) const {
//    using key_vt = vec<key_t, vector_len>;
//    using word_vt = vec<word_t, vector_len>;
//    const key_vt hash_vals = hash_fn<key_vt>::hash(keys);
//    const key_vt bit_idxs = hash_vals & length_mask;
//    const key_vt word_idxs = bit_idxs >> word_bitlength_log2;
//    const key_vt in_word_idxs = bit_idxs & word_bitlength_mask;
//    const key_vt second_in_word_idxs = hash_vals >> (32 - word_bitlength_log2);
//    const word_vt search_masks = (word_vt::make(1) << in_word_idxs) | (word_vt::make(1) << second_in_word_idxs);
//    const word_vt words = word_idxs.load(word_array.data());
//    return (words & search_masks) == search_masks;
//  }




  u64 popcnt() {
    $u64 pc = 0;
    for (auto& w : word_array) {
      pc += _mm_popcnt_u64(w);
    }
    return pc;
  }};

}
