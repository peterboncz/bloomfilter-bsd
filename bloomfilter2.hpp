#pragma once

#include <functional>
#include <stdexcept>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>
#include <dtl/math.hpp>

#include "immintrin.h"

namespace dtl {

template<typename Tk,      // the key type
    template<typename Ty> class HashFn,     // the first hash function to use
    template<typename Ty> class HashFn2,    // the second hash function to use
    typename Tw = u64,     // the word type to use for the bitset
    typename Alloc = std::allocator<Tw>,
    u32 K = 2,             // the number of hash functions to use
    u1 Sectorized = false
>
struct bloomfilter2 {

  using key_t = typename std::remove_cv<Tk>::type;
  using word_t = typename std::remove_cv<Tw>::type;
  using allocator_t = Alloc;
  using size_t = $u32;

  static_assert(std::is_integral<key_t>::value, "The key type must be an integral type.");
  static_assert(std::is_integral<word_t>::value, "The word type must be an integral type.");


  static constexpr u32 word_bitlength = sizeof(word_t) * 8;
  static constexpr u32 word_bitlength_log2 = dtl::ct::log_2<word_bitlength>::value;
  static constexpr u32 word_bitlength_mask = word_bitlength - 1;


  // inspect the given hash function
  static_assert(
      std::is_same<decltype(HashFn<key_t>::hash(0)), decltype(HashFn2<key_t>::hash(0))>::value,
      "The two hash functions must return the same type.");
  using hash_value_t = decltype(HashFn<key_t>::hash(0));
  static_assert(std::is_integral<hash_value_t>::value, "Hash function must return an integral type.");
  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;

  // the number of hash functions to use
  static constexpr u32 k = K;
  static_assert(k > 1, "Parameter 'k' must be at least '2'.");

  // split each word into multiple sectors (sub words, with a length of a power of two)
  // note that sectorization is a specialization. having only one sector = no sectorization
  static constexpr u1 sectorized = Sectorized;
  static constexpr u32 compute_sector_cnt() {
    if (!sectorized) return 1;
    u32 k_pow_2 = dtl::next_power_of_two(k);
    static_assert((word_bitlength / k_pow_2) != 0, "The number of sectors must be greater than zero. Probably the given number of hash functions is set to high.");
    return word_bitlength / (word_bitlength / k_pow_2);
  }
  static constexpr u32 sector_cnt = compute_sector_cnt();
  static constexpr u32 sector_bitlength = word_bitlength / sector_cnt;
  // the number of bits needed to address the individual bits within a sector
  static constexpr u32 sector_bitlength_log2 = dtl::ct::log_2<sector_bitlength>::value;
//  static constexpr word_t sector_mask() { return sector_bitlength_log2 - 1; } // a function, to work around a compiler bug
  static constexpr word_t sector_mask() { return sector_bitlength - 1; }

  // the number of remaining bits of the FIRST hash value (used to identify the word)
  static constexpr i32 remaining_hash_bit_cnt = static_cast<i32>(hash_value_bitlength) - (sectorized ? sector_bitlength_log2 : word_bitlength_log2);
  static constexpr u64 max_m = (1ull << remaining_hash_bit_cnt) * word_bitlength;

  // members
  size_t length_mask; // the length of the bitvector (length_mask + 1) is not stored explicitly
  size_t word_cnt_log2; // the number of bits to address the individual words of the bitvector
  Alloc allocator;
  std::vector<word_t, Alloc> word_array;


  bloomfilter2(const size_t length, const Alloc allocator = Alloc())
      : length_mask(next_power_of_two(length) - 1),
        word_cnt_log2(dtl::log_2(next_power_of_two(length) / word_bitlength)),
        allocator(allocator),
        word_array(next_power_of_two(length) / word_bitlength, 0, this->allocator) {
    if (length > max_m) throw std::invalid_argument("Length must not exceed 'max_m'.");
  }

  /// creates a copy of the bloomfilter (allows to specify a different allocator)
  template<typename AllocOfCopy = Alloc>
  bloomfilter2<Tk, HashFn, HashFn2, Tw, AllocOfCopy>
  make_copy(AllocOfCopy alloc = AllocOfCopy()) {
    using return_t = bloomfilter2<Tk, HashFn, HashFn2, Tw, AllocOfCopy>;
    return_t bf_copy(this->length_mask + 1, alloc);
    bf_copy.word_array.clear();
    bf_copy.word_array.insert(bf_copy.word_array.begin(), word_array.begin(), word_array.end());
    return bf_copy;
  };


  forceinline size_t
  which_word(const hash_value_t hash_val) const noexcept{
    const size_t word_idx = hash_val >> (hash_value_bitlength - word_cnt_log2);
    assert(word_idx < ((length_mask + 1) / word_bitlength));
    return word_idx;
  }


  forceinline unroll_loops
  word_t
  which_bits(const hash_value_t first_hash_val,
             const hash_value_t second_hash_val) const noexcept {
    // take the LSBs of first hash value
    word_t word = word_t(1) << (first_hash_val & sector_mask());
    for (size_t i = 0; i < k - 1; i++) {
      const u32 bit_idx = (second_hash_val >> (i * sector_bitlength_log2)) & sector_mask();
      word |= word_t(1) << (bit_idx + ((i + 1) * sector_bitlength));
    }
    return word;
  }


  forceinline void
  insert(const key_t& key) noexcept {
    const hash_value_t first_hash_val = HashFn<key_t>::hash(key);
    const hash_value_t second_hash_val = HashFn2<key_t>::hash(key);
    u32 word_idx = which_word(first_hash_val);
    word_t word = word_array[word_idx];
    word |= which_bits(first_hash_val, second_hash_val);
    word_array[word_idx] = word;
  }


  forceinline u1
  contains(const key_t& key) const noexcept {
    const hash_value_t first_hash_val = HashFn<key_t>::hash(key);
    const hash_value_t second_hash_val = HashFn2<key_t>::hash(key);
    u32 word_idx = which_word(first_hash_val);
    const word_t search_mask = which_bits(first_hash_val, second_hash_val);
    return (word_array[word_idx] & search_mask) == search_mask;
  }


  u64 popcnt() const noexcept {
    $u64 pc = 0;
    for (auto& w : word_array) {
      pc += _mm_popcnt_u64(w);
    }
    return pc;
  }


  f64 load_factor() const noexcept {
    f64 m = length_mask + 1;
    return popcnt() / m;
  }

};

} // namespace dtl
