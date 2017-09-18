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

//#include <cub/cub.cuh>

#include "bloomfilter_addressing_logic.hpp"


namespace dtl {


template<
    typename Tk,           // the key type
    typename Tw = u32,     // the word type to use for the bitset
    typename Th = u32,     // the hash value type
    u32 K = 2,             // the number of bits to set per sector
    u32 B = 4,             // the block size in bytes
    u32 S = 4              // the sector size in bytes
>
struct blocked_bloomfilter_params {

  using key_t = typename std::remove_cv<Tk>::type;
  using word_t = typename std::remove_cv<Tw>::type;
  using size_t = $u32;

  static_assert(std::is_integral<key_t>::value, "The key type must be an integral type.");
  static_assert(std::is_integral<word_t>::value, "The word type must be an integral type.");

  static_assert(B >= sizeof(word_t), "The block size must be greater or equal to the word size.");
  static_assert(S <= B, "The sector size must not exceed the block size.");
  static_assert(is_power_of_two(S), "The sector size must be a power of two.");

  static constexpr u32 word_bitlength = sizeof(word_t) * 8;
  static constexpr u32 word_bitlength_log2 = dtl::ct::log_2_u32<word_bitlength>::value;
  static constexpr u32 word_bitlength_log2_mask = (1u << word_bitlength_log2) - 1;
  static constexpr u32 word_bitlength_mask = word_bitlength - 1;

  // The number of words per block.
  static constexpr u32 word_cnt = B / sizeof(word_t);
  static_assert(is_power_of_two(word_cnt), "The number of words per block must be a power of two.");
  // The number of bits needed to address the individual words within a block.
  static constexpr u32 word_cnt_log2 = dtl::ct::log_2_u32<word_cnt>::value;
  static constexpr u32 word_cnt_mask = word_cnt - 1;

  static constexpr u32 block_bitlength = word_bitlength * word_cnt;
  static constexpr u32 block_bitlength_log2 = dtl::ct::log_2_u32<block_bitlength>::value;
  static constexpr u32 block_bitlength_mask = word_cnt - 1;


  // The number of bits to set per element per sector.
  static constexpr u32 k = K;
  static_assert(k > 0, "Parameter 'k' must be at least '1'.");


  // Split the block into multiple sectors (or sub-blocks) with a length of a power of two.
  // Note that sectorization is a specialization. Having only one sector = no sectorization.
  static constexpr u1 sectorized = S < B;
  static constexpr u32 sector_cnt = B / S;
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
  static constexpr size_t required_hash_bits_per_element = k * required_hash_bits_per_k * sector_cnt;

  struct util {


    template<typename T>
    __forceinline__ __unroll_loops__ __host__ __device__
    static
    T load(T const* ptr) {
#if defined(__CUDA_ARCH__)
      static constexpr cub::CacheLoadModifier cache_load_modifier = cub::LOAD_CG;
      return cub::ThreadLoad<cache_load_modifier>(ptr);
#else
      return *ptr;
#endif // defined(__CUDA_ARCH__)
    }

//    template<typename T, u32 p = 32>
//    struct knuth_32 {
//      using Ty = typename std::remove_cv<T>::type;
//
//      __host__ __device__
//      static inline Ty
//      hash(const Ty& key) {
////    Ty knuth = 2654435769u; // 0b10011110001101110111100110111001
//        Ty knuth = 596572387u; // Peter 1
//        return (key * knuth) >> (32 - p);
//      }
//    };
//
//
//    template<typename T, u32 p = 32>
//    struct knuth_32_alt {
//      using Ty = typename std::remove_cv<T>::type;
//
//      __host__ __device__
//      static inline Ty
//      hash(const Ty& key) {
////    Ty knuth = 1799596469u; // 0b01101011010000111010100110110101
//        Ty knuth = 370248451u; // Peter 2
//        return (key * knuth) >> (32 - p);
//      }
//    };
  };

};

} // namespace dtl