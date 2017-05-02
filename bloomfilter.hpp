#pragma once

#include <functional>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>
#include <dtl/math.hpp>

#include "immintrin.h"

namespace dtl {

template<typename Tk, // the key type
    template<typename Ty> class hash_fn, // the hash function to use
    typename Tw = u64, // the word type to use for the bitset
    typename Alloc = std::allocator<Tw>>
struct bloomfilter {

  using key_t = typename std::remove_cv<Tk>::type;
  using word_t = typename std::remove_cv<Tw>::type;
  using allocator_t = Alloc;

  /// the length in bits. guaranteed to be a power of two
  u32 length;
  u32 length_mask;
  u32 word_bitlength = sizeof(word_t) * 8;
  u32 word_bitlength_log2 = dtl::log_2(word_bitlength);
  u32 word_bitlength_mask = word_bitlength - 1;

  Alloc allocator;
  std::vector<word_t, Alloc> word_array;


  bloomfilter(u32 length, Alloc allocator = Alloc())
      : length(next_power_of_two(length)), length_mask(next_power_of_two(length) - 1),
        allocator(allocator),
        word_array(next_power_of_two(length) / word_bitlength, 0, this->allocator) { }

  /// creates a copy of the bloomfilter (allows to specify a different allocator)
  template<typename AllocOfCopy = Alloc>
  bloomfilter<Tk, hash_fn, Tw, AllocOfCopy>
  make_copy(AllocOfCopy alloc = AllocOfCopy()) {
    using return_t = bloomfilter<Tk, hash_fn, Tw, AllocOfCopy>;
    return_t bf_copy(this->length, alloc);
    bf_copy.word_array.clear();
    bf_copy.word_array.insert(bf_copy.word_array.begin(), word_array.begin(), word_array.end());
    return bf_copy;
  };

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
