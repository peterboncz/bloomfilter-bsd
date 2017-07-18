#pragma once

#include <bitset>
#include <functional>
#include <stdexcept>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>
#include <dtl/math.hpp>

#include "immintrin.h"
#include "dtl/bloomfilter/bloomfilter.hpp"

namespace dtl {

/// A high-performance Bloom filter.
template<typename Tk,      // the key type
    template<typename Ty> class HashFn,     // the first hash function to use
    template<typename Ty> class HashFn2,    // the second hash function to use
    typename Tw = u64,     // the word type to use for the bitset
    typename Alloc = std::allocator<Tw>,
    u32 K = 5,             // the number of hash functions to use
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
  static constexpr u32 word_bitlength_log2 = dtl::ct::log_2_u32<word_bitlength>::value;
  static constexpr u32 word_bitlength_mask = word_bitlength - 1;


  // Inspect the given hash function
  static_assert(
      std::is_same<decltype(HashFn<key_t>::hash(0)), decltype(HashFn2<key_t>::hash(0))>::value,
      "The two hash functions must return the same type.");
  using hash_value_t = decltype(HashFn<key_t>::hash(0));
  static_assert(std::is_integral<hash_value_t>::value, "Hash function must return an integral type.");
  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;
  static constexpr u32 hash_fn_cnt = 2;


  // The number of hash functions to use.
  static constexpr u32 k = K;
  static_assert(k > 1, "Parameter 'k' must be at least '2'.");

  // Split each word into multiple sectors (sub words, with a length of a power of two).
  // Note that sectorization is a specialization. Having only one sector = no sectorization.
  static constexpr u1 sectorized = Sectorized;

// incompatible with C++11
//  static constexpr u32 compute_sector_cnt() {
//    if (!sectorized) return 1;
//    u32 k_pow_2 = dtl::next_power_of_two(k);
//    static_assert((word_bitlength / k_pow_2) != 0, "The number of sectors must be greater than zero. Probably the given number of hash functions is set to high.");
//    return word_bitlength / (word_bitlength / k_pow_2);
//  }

  static constexpr u32 compute_sector_cnt() {
    static_assert(!sectorized || ((word_bitlength / dtl::next_power_of_two(k)) != 0), "The number of sectors must be greater than zero. Probably the given number of hash functions is set to high.");
    return (!sectorized) ? 1
                         : word_bitlength / (word_bitlength / dtl::next_power_of_two(k));
  }
  static constexpr u32 sector_cnt = compute_sector_cnt();
  static constexpr u32 sector_bitlength = word_bitlength / sector_cnt;
  // the number of bits needed to address the individual bits within a sector
  static constexpr u32 sector_bitlength_log2 = dtl::ct::log_2_u32<sector_bitlength>::value;
//  static constexpr word_t sector_mask() { return sector_bitlength_log2 - 1; } // a function, to work around a compiler bug
  static constexpr word_t sector_mask() { return sector_bitlength - 1; }

  // the number of remaining bits of the FIRST hash value (used to identify the word)
  static constexpr i32 remaining_hash_bit_cnt = static_cast<i32>(hash_value_bitlength) - (sectorized ? sector_bitlength_log2 : word_bitlength_log2);
  static constexpr u64 min_m = 2 * word_bitlength; // Using only one word would cause undefined behaviour in bit shifts later on.
  static constexpr u64 max_m = (1ull << remaining_hash_bit_cnt) * word_bitlength;

  // ---- Members ----
  size_t length_mask; // the length of the bitvector (length_mask + 1) is not stored explicitly
  size_t word_cnt_log2; // the number of bits to address the individual words of the bitvector
  allocator_t allocator;
  std::vector<word_t, allocator_t> word_array;
  // ----


// incompatible with C++11
//  static constexpr
//  size_t
//  determine_actual_length(const size_t length) {
//    // round up to the next power of two
//    const size_t m = static_cast<size_t>(next_power_of_two(length));
//    const size_t min = static_cast<size_t>(min_m);
//    return std::max(m, min);
//  }


  static constexpr
  size_t
  determine_actual_length(const size_t length) {
    // round up to the next power of two
    return std::max(
        static_cast<size_t>(next_power_of_two(length)),
        static_cast<size_t>(min_m)
    );
  }


  bloomfilter2(const size_t length,
               const allocator_t allocator = allocator_t())
      : length_mask(determine_actual_length(length) - 1),
        word_cnt_log2(dtl::log_2((length_mask + 1) / word_bitlength)),
        allocator(allocator),
        word_array((length_mask + 1) / word_bitlength, 0, this->allocator) {
    if (((length_mask + 1)) > max_m) throw std::invalid_argument("Length must not exceed 'max_m'.");
  }


  // FIXME
  ~bloomfilter2() {
    word_array.resize(8);
  }


  /// Creates a copy of the bloomfilter (allows to specify a different allocator)
  template<typename AllocOfCopy = Alloc>
  bloomfilter2<Tk, HashFn, HashFn2, Tw, AllocOfCopy, K, Sectorized>
  make_copy(AllocOfCopy alloc = AllocOfCopy()) const {
    using return_t = bloomfilter2<Tk, HashFn, HashFn2, Tw, AllocOfCopy, K, Sectorized>;
    return_t bf_copy(this->length_mask + 1, alloc);
    bf_copy.word_array.clear();
    bf_copy.word_array.insert(bf_copy.word_array.begin(), word_array.begin(), word_array.end());
    return bf_copy;
  }


  /// Creates a copy of the bloomfilter (allows to specify a different allocator)
  template<typename AllocOfCopy = Alloc>
  bloomfilter2<Tk, HashFn, HashFn2, Tw, AllocOfCopy, K, Sectorized>*
  make_heap_copy(AllocOfCopy alloc = AllocOfCopy()) const {
    using bf_t = bloomfilter2<Tk, HashFn, HashFn2, Tw, AllocOfCopy, K, Sectorized>;
    bf_t* bf_copy = new bf_t(this->length_mask + 1, alloc);
    bf_copy->word_array.clear();
    bf_copy->word_array.insert(bf_copy->word_array.begin(), word_array.begin(), word_array.end());
    return bf_copy;
  }


  __forceinline__ __host__ __device__
  static size_t
  which_word(const /*hash_value_t*/ uint32_t hash_val,
             const size_t length_mask,
             const size_t word_cnt_log2) noexcept {
    const size_t word_idx = hash_val >> (hash_value_bitlength - word_cnt_log2);
    assert(word_idx < ((length_mask + 1) / word_bitlength));
    return word_idx;
  }


  __forceinline__ __unroll_loops__ __host__ __device__
  static word_t
  which_bits(const hash_value_t first_hash_val,
             const hash_value_t second_hash_val,
             const size_t word_cnt_log2) noexcept {
    u32 first_bit_idx = ((first_hash_val >> (hash_value_bitlength - word_cnt_log2 - sector_bitlength_log2)) & sector_mask());
    word_t word = word_t(1) << first_bit_idx;
    for (size_t i = 0; i < k - 1; i++) {
      u32 shift = hash_value_bitlength - (i * sector_bitlength_log2 + 2); // FIXME for 64-bit words
      u32 bit_idx = (second_hash_val >> shift) & sector_mask();
      u32 sector_offset = ((i + 1) * sector_bitlength) & word_bitlength_mask;
      word |= word_t(1) << (bit_idx + sector_offset);
    }
    return word;
  }


  __forceinline__
  void
  insert(const key_t& key) noexcept {
    const hash_value_t first_hash_val = HashFn<key_t>::hash(key);
    const hash_value_t second_hash_val = HashFn2<key_t>::hash(key);
    u32 word_idx = which_word(first_hash_val, length_mask, word_cnt_log2);
    word_t word = word_array[word_idx];
    word |= which_bits(first_hash_val, second_hash_val, word_cnt_log2);
    word_array[word_idx] = word;
  }


  __forceinline__
  u1
  contains(const key_t& key) const noexcept {
    const hash_value_t first_hash_val = HashFn<key_t>::hash(key);
    const hash_value_t second_hash_val = HashFn2<key_t>::hash(key);
    u32 word_idx = which_word(first_hash_val, length_mask, word_cnt_log2);
    const word_t search_mask = which_bits(first_hash_val, second_hash_val, word_cnt_log2);
    return (word_array[word_idx] & search_mask) == search_mask;
  }


  u64
  popcnt() const noexcept {
    return std::accumulate(word_array.begin(), word_array.end(), 0ull,
                           [](u64 cntr, word_t word) { return cntr + dtl::bits::pop_count(word); });
  }


  f64
  load_factor() const noexcept {
    f64 m = length_mask + 1;
    return popcnt() / m;
  }


  void
  print_info() const noexcept {
    std::cout << "-- bloomfilter parameters --" << std::endl;
    std::cout << "static" << std::endl;
    std::cout << "  h:                    " << hash_fn_cnt << std::endl;
    std::cout << "  k:                    " << k << std::endl;
    std::cout << "  word bitlength:       " << word_bitlength << std::endl;
    std::cout << "  hash value bitlength: " << hash_value_bitlength << std::endl;
    std::cout << "  sectorized:           " << (sectorized ? "true" : "false") << std::endl;
    std::cout << "  sector count:         " << sector_cnt << std::endl;
    std::cout << "  sector bitlength:     " << sector_bitlength << std::endl;
    std::cout << "  hash bits per sector: " << sector_bitlength_log2 << std::endl;
    std::cout << "  hash bits per word:   " << (k * sector_bitlength_log2) << std::endl;
    std::cout << "  hash bits wasted:     " << (sectorized ? (word_bitlength - (sector_bitlength * k)) : 0) << std::endl;
    std::cout << "  remaining hash bits:  " << remaining_hash_bit_cnt << std::endl;
    std::cout << "  max m:                " << max_m << std::endl;
    std::cout << "  max size [MiB]:       " << (max_m / 8.0 / 1024.0 / 1024.0 ) << std::endl;
    std::cout << "dynamic" << std::endl;
    std::cout << "  m:                    " << (length_mask + 1) << std::endl;
    f64 size_MiB = (length_mask + 1) / 8.0 / 1024.0 / 1024.0;
    if (size_MiB < 1) {
      std::cout << "  size [KiB]:           " << (size_MiB * 1024) << std::endl;
    }
    else {
      std::cout << "  size [MiB]:           " << size_MiB << std::endl;
    }
    std::cout << "  population count:     " << popcnt() << std::endl;
    std::cout << "  load factor:          " << load_factor() << std::endl;
  }


  void
  print() const noexcept {
    std::cout << "-- bloomfilter dump --" << std::endl;
    $u64 i = 0;
    for (const word_t word : word_array) {
      std::cout << std::bitset<word_bitlength>(word);
      i++;
      if (i % (128 / word_bitlength) == 0) {
        std::cout << std::endl;
      }
      else {
        std::cout << " ";
      }
    }
    std::cout << std::endl;
  }


};

} // namespace dtl
