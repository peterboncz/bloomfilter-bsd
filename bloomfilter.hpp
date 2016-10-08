#pragma once

#include "adept.hpp"
#include <vector>
#include <functional>
#include "immintrin.h"
#include "math.hpp"

namespace dtl {

template<typename Tk, typename hash_fn>
struct bloomfilter {

  using word_t = $u32;
  const u64 word_bitlength = sizeof(word_t) * 8;
  const u64 word_bitlength_log2 = log_2(word_bitlength);
  const u64 word_bitlength_mask = word_bitlength - 1;
  u64 length_mask;
  std::vector<word_t> bitarray;

  bloomfilter(u64 length)
      : length_mask(next_power_of_two(length) - 1),
        bitarray(next_power_of_two(length) / word_bitlength, 0) { }

  void insert(const Tk& key) {
    u64 bit_idx = hash_fn::hash(key) & length_mask;
    u64 word_idx = bit_idx >> word_bitlength_log2;
    u64 in_word_idx = bit_idx & word_bitlength_mask;
    bitarray[word_idx] |= 1 << in_word_idx;
  }

  u1 contains(const Tk& key) {
    u64 bit_idx = hash_fn::hash(key) & length_mask;
    u64 word_idx = bit_idx >> word_bitlength_log2;
    u64 in_word_idx = bit_idx & word_bitlength_mask;
    return (bitarray[word_idx] & (1 << in_word_idx)) != 0;
  }

  u64 popcnt() {
    $u64 pc = 0;
    for (auto& w : bitarray) {
      pc += _mm_popcnt_u64(w);
    }
    return pc;
  }};

}