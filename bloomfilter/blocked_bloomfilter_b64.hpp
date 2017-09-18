#pragma once

#include <bitset>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <typeinfo>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>
#include <dtl/math.hpp>

#include "blocked_bloomfilter.hpp"
#include "bloomfilter_addressing_logic.hpp"

#include <cxxabi.h>

namespace dtl {

template<
    typename key_t,        // the key type
    typename word_t,       // the word type to use for the bitset
    typename hash_value_t, // the hash value type
    u32 K,                 // the number of bits to set per sector
    u32 B,                 // the block size in bytes
    u32 S                  // the sector size in bytes
>
struct bloomfilter_block_b64 {};

template<>
struct bloomfilter_block_b64<$u32, $u32, $u32, 1, 64, 64> {

  using bbf = blocked_bloomfilter_params<$u32, $u32, $u32, 1, 64, 64>;

  __forceinline__ __unroll_loops__ __host__ __device__
  static u1
  contains_hash(const typename bbf::hash_value_t& hash_val, const typename bbf::word_t* __restrict block) noexcept {
    // consume the high order bits
    u32 word_idx = hash_val >> (bbf::hash_value_bitlength - bbf::word_cnt_log2);
    const typename bbf::word_t word = bbf::util::load(block + word_idx);
    u32 bit_idx = (hash_val >> (bbf::hash_value_bitlength - bbf::word_cnt_log2 - bbf::sector_bitlength_log2))
        & ((bbf::hash_value_t(1) << bbf::sector_bitlength_log2) - 1);
    return dtl::bits::bit_test(word, bit_idx);
  }

  __forceinline__ __unroll_loops__ __host__ __device__
  static u1
  contains_key(const typename bbf::key_t& key, const typename bbf::word_t* __restrict block) noexcept {
    const typename bbf::hash_value_t hash_val = hash<1>(key);
    return contains_hash(hash_val, block);
  }

};

template<u32 K>
struct bloomfilter_block_b64_k2_k3 {

  static_assert(K >= 2 && K <= 3, "The parameter 'K' must be in (1,3].");
  using bbf = blocked_bloomfilter_params<$u32, $u32, $u32, K, 64, 64>;

  __forceinline__ __unroll_loops__ __host__ __device__
  static u1
  contains_hash(const typename bbf::hash_value_t& hash_val, const typename bbf::word_t* __restrict block) noexcept {
    auto h = hash_val;
    auto r = true;
    for (size_t i = 0; i < bbf::k; i++) {
      // consume the high order bits
      u32 word_idx = h >> (bbf::hash_value_bitlength - bbf::word_cnt_log2);
      const typename bbf::word_t word = bbf::util::load(block + word_idx);
      u32 bit_idx = (h >> (bbf::hash_value_bitlength - bbf::word_cnt_log2 - bbf::sector_bitlength_log2))
          & ((bbf::hash_value_t(1) << bbf::sector_bitlength_log2) - 1);
      r &= dtl::bits::bit_test(word, bit_idx);
      h <<= bbf::required_hash_bits_per_k;
    }
    return r;
  }

  __forceinline__ __unroll_loops__ __host__ __device__
  static u1
  contains_key(const typename bbf::key_t& key, const typename bbf::word_t* __restrict block) noexcept {
    const typename bbf::hash_value_t hash_val = hash<1>(key);
    return contains_hash(hash_val, block);
  }

};

template<>
struct bloomfilter_block_b64<$u32, $u32, $u32, 2, 64, 64> : bloomfilter_block_b64_k2_k3<2> {};

template<>
struct bloomfilter_block_b64<$u32, $u32, $u32, 3, 64, 64> : bloomfilter_block_b64_k2_k3<3> {};



template<u32 K>
struct bloomfilter_block_b64_k4_k5_k6 {

  static_assert(K >= 4 && K <=6, "The parameter 'K' must be in [4,6].");
  using bbf = blocked_bloomfilter_params<$u32, $u32, $u32, K, 64, 64>;

  __forceinline__ __unroll_loops__ __host__ __device__
  static u1
  contains_hash(const typename bbf::hash_value_t& hash_val, const typename bbf::word_t* __restrict block) noexcept {
    assert(false); // no fast path
    static_assert(false, "No fast path.");
  }

  __forceinline__ __unroll_loops__ __host__ __device__
  static u1
  contains_key(const typename bbf::key_t& key, const typename bbf::word_t* __restrict block) noexcept {
    auto r = true;
    {
      auto h = hash<1>(key);
      for (size_t i = 0; i < 3; i++) {
        // consume the high order bits
        u32 word_idx = h >> (bbf::hash_value_bitlength - bbf::word_cnt_log2);
        const typename bbf::word_t word = bbf::util::load(block + word_idx);
        u32 bit_idx = (h >> (bbf::hash_value_bitlength - bbf::word_cnt_log2 - bbf::sector_bitlength_log2))
            & ((bbf::hash_value_t(1) << bbf::sector_bitlength_log2) - 1);
        r &= dtl::bits::bit_test(word, bit_idx);
        h <<= bbf::required_hash_bits_per_k;
      }
    }
    {
      auto h = hash<2>(key);
      for (size_t i = 3; i < bbf::k; i++) {
        // consume the high order  bits
        u32 word_idx = h >> (bbf::hash_value_bitlength - bbf::word_cnt_log2);
        const typename bbf::word_t word = bbf::util::load(block + word_idx);
        u32 bit_idx = (h >> (bbf::hash_value_bitlength - bbf::word_cnt_log2 - bbf::sector_bitlength_log2))
            & ((bbf::hash_value_t(1) << bbf::sector_bitlength_log2) - 1);
        r &= dtl::bits::bit_test(word, bit_idx);
        h <<= bbf::required_hash_bits_per_k;
      }
    }
    return r;
  }

};

template<>
struct bloomfilter_block_b64<$u32, $u32, $u32, 4, 64, 64> : bloomfilter_block_b64_k4_k5_k6<4> {};

template<>
struct bloomfilter_block_b64<$u32, $u32, $u32, 5, 64, 64> : bloomfilter_block_b64_k4_k5_k6<5> {};

template<>
struct bloomfilter_block_b64<$u32, $u32, $u32, 6, 64, 64> : bloomfilter_block_b64_k4_k5_k6<6> {};




//===----------------------------------------------------------------------===//

/// The primary template for a blocked Bloom filter with block size = 64 bytes
template<
    typename key_t,        // the key type
    typename word_t,       // the word type to use for the bitset
    typename hash_value_t, // the hash value type
    u32 K,                 // the number of bits to set per sector
    u32 B,                 // the block size in bytes
    u32 S,                 // the sector size in bytes
    block_addressing addr_mode // the block addressing mode
>
struct blocked_bloomfilter_b64 {
  using bbf = blocked_bloomfilter_params<key_t, word_t, hash_value_t, K, B, S>;
  using addr_t = bloomfilter_addressing_logic<addr_mode, hash_value_t, bbf>;

  addr_t addr;
  using block_t = bloomfilter_block_b64<$u32, $u32, $u32, K, 64, 64>;
  block_t t;

  __forceinline__ __unroll_loops__ __host__ __device__
  u1
  _fast_contains(const bbf::key_t key, const bbf::word_t* filter) {
    const hash_value_t hash_value = hash<0>(key);
    const u32 block_idx = addr.get_block_idx(hash_value);
    const bbf::word_t* block = filter + (block_idx * sizeof(bbf::word_t));
    return block_t::contains_hash(hash_value << addr.block_cnt_log2, block);
  }

  __forceinline__ __unroll_loops__ __host__ __device__
  _contains(const bbf::key_t key, const bbf::word_t* filter) {
    const hash_value_t hash_value = hash<0>(key);
    const u32 block_idx = addr.get_block_idx(hash_value);
    const bbf::word_t* block = filter + (block_idx * sizeof(bbf::word_t));
    return block_t::contains_key(key, block);
  }

  blocked_bloomfilter_b64() : addr(2048) {
    int status;
    const auto& ti = typeid(typename block_t::bbf);
    std::cout << "block type: " << abi::__cxa_demangle(ti.name(), 0, 0, &status) << std::endl;
  };

  u1
  contains_key(const bbf::key_t key, const bbf::word_t* filter) {

  }
};


} // namespace dtl