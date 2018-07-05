#pragma once

#include <cstddef>

#include <dtl/dtl.hpp>

namespace dtl {
namespace cuckoofilter {
namespace internal {

//===----------------------------------------------------------------------===//
// CUDA implementations to find tags in buckets.
//
// Note: Not all cuckoo filter configurations need to be specialized for
//       for CUDA. If no specialization is found (using SFINAE), the
//       the default scalar implementation is used.
//===----------------------------------------------------------------------===//


template<>
struct find_tag_in_buckets_scalar<8,2> {

  template<typename table_t> // inferred
  __forceinline__ __host__ __device__
  static bool
  dispatch(const table_t& table,
           const uint32_t* __restrict filter_data,
           const std::size_t bucket_idx1,
           const std::size_t bucket_idx2,
           const uint32_t tag) {

    constexpr std::size_t bits_per_tag = 8;
    constexpr auto tag_mask = static_cast<uint32_t>((1ull << bits_per_tag) - 1);

    bool found = false;

    const auto word_idx1 = bucket_idx1 >> 1; // two buckets per word
    const auto word1 = filter_data[word_idx1];
    const auto in_word_idx1 = (bucket_idx1 & (2 - 1)) << 4; // offset is either 0 or 16 bit
    found |= ((word1 >> in_word_idx1) & tag_mask) == tag;
    found |= ((word1 >> (in_word_idx1 + bits_per_tag)) & tag_mask) == tag;

    const auto word_idx2 = bucket_idx2 >> 1; // two buckets per word
    const auto word2 = filter_data[word_idx2];
    const auto in_word_idx2 = (bucket_idx2 & (2 - 1)) << 4; // offset is either 0 or 16 bit
    found |= ((word2 >> in_word_idx2) & tag_mask) == tag;
    found |= ((word2 >> (in_word_idx2 + bits_per_tag)) & tag_mask) == tag;

    return found;
  }

};


template<>
struct find_tag_in_buckets_scalar<8,4> {

  template<typename table_t> // inferred
  __forceinline__ __host__ __device__
  static bool
  dispatch(const table_t& table,
           const uint32_t* __restrict filter_data,
           const std::size_t bucket_idx1,
           const std::size_t bucket_idx2,
           const uint32_t tag) {

    const auto* d = reinterpret_cast<const uchar4*>(filter_data);

    bool found = false;

    const auto load_idx1 = bucket_idx1;
    const auto bucket1 = d[load_idx1];
    found |= bucket1.x == tag;
    found |= bucket1.y == tag;
    found |= bucket1.z == tag;
    found |= bucket1.w == tag;

    const auto load_idx2 = bucket_idx2;
    const auto bucket2 = d[load_idx2];
    found |= bucket2.x == tag;
    found |= bucket2.y == tag;
    found |= bucket2.z == tag;
    found |= bucket2.w == tag;

    return found;
  }

};


template<>
struct find_tag_in_buckets_scalar<16,2> {

  template<typename table_t> // inferred
  __forceinline__ __host__ __device__
  static bool
  dispatch(const table_t& table,
           const uint32_t* __restrict filter_data,
           const std::size_t bucket_idx1,
           const std::size_t bucket_idx2,
           const uint32_t tag) {

    constexpr std::size_t bits_per_tag = 16;
    constexpr auto tag_mask = static_cast<uint32_t>((1ull << bits_per_tag) - 1);

    bool found = false;

    const auto word_idx1 = bucket_idx1; // one bucket per word
    const auto word1 = filter_data[word_idx1];
    found |= (word1 & tag_mask) == tag;
    found |= ((word1 >> bits_per_tag) & tag_mask) == tag;

    const auto word_idx2 = bucket_idx2; // one bucket per word
    const auto word2 = filter_data[word_idx2];
    found |= (word2 & tag_mask) == tag;
    found |= ((word2 >> bits_per_tag) & tag_mask) == tag;

    return found;
  }

};


template<>
struct find_tag_in_buckets_scalar<16,4> {

  template<typename table_t> // inferred
  __forceinline__ __host__ __device__
  static bool
  dispatch(const table_t& table,
           const uint32_t* __restrict filter_data,
           const std::size_t bucket_idx1,
           const std::size_t bucket_idx2,
           const uint32_t tag) {

    // Load tags using 64-bit loads.
    const auto* d = reinterpret_cast<const ushort4*>(filter_data);

    bool found = false;

    const auto load_idx1 = bucket_idx1;
    const auto bucket1 = d[load_idx1];
    found |= bucket1.x == tag;
    found |= bucket1.y == tag;
    found |= bucket1.z == tag;
    found |= bucket1.w == tag;

    const auto load_idx2 = bucket_idx2;
    const auto bucket2 = d[load_idx2];
    found |= bucket2.x == tag;
    found |= bucket2.y == tag;
    found |= bucket2.z == tag;
    found |= bucket2.w == tag;

    return found;
  }

};


template<>
struct find_tag_in_buckets_scalar<32,2> {

  template<typename table_t> // inferred
  __forceinline__ __host__ __device__
  static bool
  dispatch(const table_t& table,
           const uint32_t* __restrict filter_data,
           const std::size_t bucket_idx1,
           const std::size_t bucket_idx2,
           const uint32_t tag) {

    const auto* d = reinterpret_cast<const uint2*>(filter_data);

    bool found = false;

    const auto load_idx1 = bucket_idx1;
    const auto bucket1 = d[load_idx1];
    found |= bucket1.x == tag;
    found |= bucket1.y == tag;

    const auto load_idx2 = bucket_idx2;
    const auto bucket2 = d[load_idx2];
    found |= bucket2.x == tag;
    found |= bucket2.y == tag;

    return found;
  }

};


template<>
struct find_tag_in_buckets_scalar<32,4> {

  template<typename table_t> // inferred
  __forceinline__ __host__ __device__
  static bool
  dispatch(const table_t& table,
           const uint32_t* __restrict filter_data,
           const std::size_t bucket_idx1,
           const std::size_t bucket_idx2,
           const uint32_t tag) {

    const auto* d = reinterpret_cast<const uint4*>(filter_data);

    bool found = false;

    const auto load_idx1 = bucket_idx1;
    const auto bucket1 = d[load_idx1];
    found |= bucket1.x == tag;
    found |= bucket1.y == tag;
    found |= bucket1.z == tag;
    found |= bucket1.w == tag;

    const auto load_idx2 = bucket_idx2;
    const auto bucket2 = d[load_idx2];
    found |= bucket2.x == tag;
    found |= bucket2.y == tag;
    found |= bucket2.z == tag;
    found |= bucket2.w == tag;

    return found;
  }

};


} // namespace internal
} // namespace cuckoofilter
} // namespace dtl
