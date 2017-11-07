#pragma once

#include <bitset>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>
#include <dtl/math.hpp>
#include <dtl/bloomfilter/blocked_bloomfilter_block_logic.hpp>

#include "immintrin.h"
#include "dtl/bloomfilter/bloomfilter_h1.hpp"

#include <boost/integer/static_min_max.hpp>


namespace dtl {


//===----------------------------------------------------------------------===//
// A high-performance blocked Bloom filter template.
//===----------------------------------------------------------------------===//
namespace internal {

//===----------------------------------------------------------------------===//
// Batch-wise Contains
//===----------------------------------------------------------------------===//
template<
    typename filter_t,
    u64 vector_len
>
struct dispatch {

  __forceinline__
  static $u64
  batch_contains(const filter_t& filter,
                 const typename filter_t::word_t* __restrict filter_data,
                 const typename filter_t::key_t* __restrict keys, u32 key_cnt,
                 $u32* __restrict match_positions, u32 match_offset) {
    // Typedefs
    using key_t = typename filter_t::key_t;
    using word_t_t = typename filter_t::word_t;
    using vec_t = vec<key_t, vector_len>;
    using mask_t = typename vec<key_t, vector_len>::mask;

    const key_t* reader = keys;
    $u32* match_writer = match_positions;

    // Determine the number of keys that need to be probed sequentially, due to alignment
    u64 required_alignment_bytes = vec_t::byte_alignment;
    u64 t = dtl::mem::is_aligned(reader)  // should always be true
            ? (required_alignment_bytes - (reinterpret_cast<uintptr_t>(reader) % required_alignment_bytes)) / sizeof(key_t) // FIXME first elements are processed sequentially even if aligned
            : key_cnt;
    u64 unaligned_key_cnt = std::min(static_cast<$u64>(key_cnt), t);
    // process the unaligned keys sequentially
    $u64 read_pos = 0;
    for (; read_pos < unaligned_key_cnt; read_pos++) {
      u1 is_match = filter.contains(filter_data, *reader);
      *match_writer = static_cast<$u32>(read_pos) + match_offset;
      match_writer += is_match;
      reader++;
    }
    // Process the aligned keys vectorized
    u64 aligned_key_cnt = ((key_cnt - unaligned_key_cnt) / vector_len) * vector_len;
    for (; read_pos < (unaligned_key_cnt + aligned_key_cnt); read_pos += vector_len) {
      const auto mask = filter.template contains_vec<vector_len>(filter_data, *reinterpret_cast<const vec_t*>(reader));
      u64 match_cnt = mask.to_positions(match_writer, read_pos + match_offset);
      match_writer += match_cnt;
      reader += vector_len;
    }
    // process remaining keys sequentially
    for (; read_pos < key_cnt; read_pos++) {
      u1 is_match = filter.contains(filter_data, *reader);
      *match_writer = static_cast<$u32>(read_pos) + match_offset;
      match_writer += is_match;
      reader++;
    }
    return match_writer - match_positions;
  }

};


//===----------------------------------------------------------------------===//
// Batch-wise Contains (no SIMD)
//===----------------------------------------------------------------------===//
template<
    typename filter_t
>
struct dispatch<filter_t, 0> {

  __forceinline__
  static $u64
  batch_contains(const filter_t& filter,
                 const typename filter_t::word_t* __restrict filter_data,
                 const typename filter_t::key_t* __restrict keys, u32 key_cnt,
                 $u32* __restrict match_positions, u32 match_offset) {
    $u32* match_writer = match_positions;
    $u32 i = 0;
    for (; i < key_cnt; i += 4) {
      u1 is_match_0 = filter.contains(filter_data, keys[i]);
      u1 is_match_1 = filter.contains(filter_data, keys[i + 1]);
      u1 is_match_2 = filter.contains(filter_data, keys[i + 2]);
      u1 is_match_3 = filter.contains(filter_data, keys[i + 3]);
      *match_writer = i + match_offset;
      match_writer += is_match_0;
      *match_writer = (i + 1) + match_offset;
      match_writer += is_match_1;
      *match_writer = (i + 2) + match_offset;
      match_writer += is_match_2;
      *match_writer = (i + 3) + match_offset;
      match_writer += is_match_3;
    }
    for (; i < key_cnt; i++) {
      u1 is_match = filter.contains(filter_data, keys[i]);
      *match_writer = i + match_offset;
      match_writer += is_match;
    }
    return match_writer - match_positions;
  }

};


} // namespace internal


template<
    typename Tk,                  // the key type
    template<typename Ty, u32 i> class Hasher,      // the hash function family to use
    typename Tw,                  // the word type to use for the bitset
    u32 Wc = 2,                   // the number of words per block
    u32 s = Wc,                   // the word type to use for the bitset
    u32 K = 8,                    // the number of hash functions to use
    dtl::block_addressing block_addressing = dtl::block_addressing::POWER_OF_TWO,
    typename Alloc = std::allocator<Tw>
>
struct blocked_bloomfilter_logic {

  using key_t = typename std::remove_cv<Tk>::type;
  using word_t = typename std::remove_cv<Tw>::type;
  static_assert(std::is_integral<key_t>::value, "The key type must be an integral type.");
  static_assert(std::is_integral<word_t>::value, "The word type must be an integral type.");

  using allocator_t = Alloc;
  using size_t = $u64;

  static constexpr u32 word_cnt_per_block = Wc;
  static constexpr u32 word_cnt_per_block_log2 = dtl::ct::log_2<Wc>::value;
  static_assert(dtl::is_power_of_two(Wc), "Parameter 'Wc' must be a power of two.");

  static constexpr u32 sector_cnt = s;

  static constexpr u32 word_bitlength = sizeof(word_t) * 8;
  static constexpr u32 word_bitlength_log2 = dtl::ct::log_2_u32<word_bitlength>::value;
  static constexpr u32 word_bitlength_mask = word_bitlength - 1;

  static constexpr u32 block_bitlength = sizeof(word_t) * 8 * word_cnt_per_block;
  static constexpr u32 block_bitlength_log2 = dtl::ct::log_2_u32<block_bitlength>::value;
  static constexpr u32 block_bitlength_mask = block_bitlength - 1;


  // Inspect the given hash function
  using hash_value_t = decltype(Hasher<key_t, 0>::hash(42)); // TODO find out why NVCC complains
  static_assert(std::is_integral<hash_value_t>::value, "Hash function must return an integral type.");
  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;


  // The number of hash functions to use.
  static constexpr u32 k = K;

  static constexpr u64 max_m = 256ull * 1024 * 1024 * 8; // FIXME

  static constexpr u32 block_hash_fn_idx = 1; // 0 is used for block addressing
  using block_t = typename blocked_bloomfilter_block_logic<key_t, word_t, word_cnt_per_block, sector_cnt, k,
                                                           Hasher, hash_value_t, block_hash_fn_idx>::type;

  using addr_t = block_addressing_logic<block_addressing>;

  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  const addr_t addr;
  //===----------------------------------------------------------------------===//


  /// C'tor
  explicit
  blocked_bloomfilter_logic(const size_t length)
      : addr(length, block_bitlength) {
    if (addr.get_block_cnt() * block_bitlength > max_m) throw std::invalid_argument("Length must not exceed 'max_m'.");
  }

  /// Copy c'tor
  blocked_bloomfilter_logic(const blocked_bloomfilter_logic&) = default;


  ~blocked_bloomfilter_logic() = default;


  //===----------------------------------------------------------------------===//
  // Insert
  //===----------------------------------------------------------------------===//
  __forceinline__ __host__
  void
  insert(word_t* __restrict filter_data,
         const key_t key) noexcept {
    const hash_value_t block_addressing_hash_val = Hasher<const key_t, 0>::hash(key);
    const hash_value_t block_idx = addr.get_block_idx(block_addressing_hash_val);
    const hash_value_t bitvector_word_idx = block_idx << word_cnt_per_block_log2;

    auto block_ptr = &filter_data[bitvector_word_idx];

    block_t::insert(block_ptr, key);
  }


  //===----------------------------------------------------------------------===//
  // Contains
  //===----------------------------------------------------------------------===//
  __forceinline__ __host__ __device__
  u1
  contains(const word_t* __restrict filter_data,
           const key_t key) const noexcept {
    const hash_value_t block_addressing_hash_val = Hasher<const key_t, 0>::hash(key);
    const hash_value_t block_idx = addr.get_block_idx(block_addressing_hash_val);
    const hash_value_t bitvector_word_idx = block_idx << word_cnt_per_block_log2;

    const auto block_ptr = &filter_data[bitvector_word_idx];

    u1 found = block_t::contains(block_ptr, key);
    return found;
  }


  //===----------------------------------------------------------------------===//
  // Contains (SIMD)
  //===----------------------------------------------------------------------===//
  template<u64 n> // the vector length
  __forceinline__ __host__
  typename vec<word_t, n>::mask
  contains_vec(const word_t* __restrict filter_data,
               const vec<key_t, n>& keys) const noexcept {
    using key_vt = vec<key_t, n>;
    using hash_value_vt = vec<hash_value_t, n>;
    using word_vt = vec<word_t, n>;

    const hash_value_vt block_addressing_hash_vals = Hasher<key_vt, 0>::hash(keys);
    const hash_value_vt block_idxs = addr.get_block_idxs(block_addressing_hash_vals);
    const hash_value_vt bitvector_word_idx = block_idxs << word_cnt_per_block_log2;

    auto found = block_t::contains(keys, filter_data, bitvector_word_idx);
    return found;
  }


  //===----------------------------------------------------------------------===//
  // Batch-wise Contains
  //===----------------------------------------------------------------------===//
  template<u64 vector_len = dtl::simd::lane_count<key_t>>
  __forceinline__
  $u64
  batch_contains(const word_t* __restrict filter_data,
                 const key_t* __restrict keys, u32 key_cnt,
                 $u32* __restrict match_positions, u32 match_offset) const {
    return internal::dispatch<blocked_bloomfilter_logic, vector_len>
             ::batch_contains(*this, filter_data,
                              keys, key_cnt,
                              match_positions, match_offset);
  }


  /// Returns length in bits.
  __forceinline__
  size_t
  length() const noexcept {
    return addr.get_block_cnt() * block_bitlength;
  }


  void
  print_info() const noexcept {
    std::cout << "-- bloomfilter parameters --" << std::endl;
    std::cout << "  k:                    " << k << std::endl;
    std::cout << "  word bitlength:       " << word_bitlength << std::endl;
    std::cout << "  hash value bitlength: " << hash_value_bitlength << std::endl;
    std::cout << "  word count:           " << word_cnt_per_block << std::endl;
    std::cout << "  sector count:         " << s << std::endl;
    std::cout << "  max m:                " << max_m << std::endl;
    std::cout << "  max size [MiB]:       " << (max_m / 8.0 / 1024.0 / 1024.0 ) << std::endl;
    std::cout << "dynamic" << std::endl;
    std::cout << "  m:                    " << (addr.get_block_cnt() * block_bitlength) << std::endl;
    f64 size_MiB = (addr.get_block_cnt() * block_bitlength) / 8.0 / 1024.0 / 1024.0;
    if (size_MiB < 1) {
      std::cout << "  size [KiB]:           " << (size_MiB * 1024) << std::endl;
    }
    else {
      std::cout << "  size [MiB]:           " << size_MiB << std::endl;
    }
  }


//  void
//  print() const noexcept {
//    std::cout << "-- Bloom filter dump --" << std::endl;
//    $u64 i = 0;
//    for (const word_t word : word_array) {
//      std::cout << std::bitset<word_bitlength>(word);
//      i++;
//      if (i % (128 / word_bitlength) == 0) {
//        std::cout << std::endl;
//      }
//      else {
//        std::cout << " ";
//      }
//    }
//    std::cout << std::endl;
//  }


};

} // namespace dtl
