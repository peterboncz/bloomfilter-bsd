#pragma once

//===----------------------------------------------------------------------===//
// This file is based on the original Cuckoo filter implementation,
// that can be found on GitHub: https://github.com/efficient/cuckoofilter
// Usable under the terms in the Apache License, Version 2.0.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2013, Carnegie Mellon University and Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

// Change log:
// 2017-2018 | H. Lang (TUM), P. Boncz (CWI):
//  - SIMDized implementation for the 'find_tag_in_bucket' function (supports 8,16, and 32 bit tags)

#include <sstream>
#include <xmmintrin.h>

#include <dtl/dtl.hpp>
#include <dtl/thread.hpp>
#include <dtl/simd.hpp>
#include <dtl/filter/blocked_bloomfilter/vector_helper.hpp>

// include the scalar "find tag in buckets" implementations
#include "cuckoofilter_table_scalar.hpp"

// include the vectorized "find tag in buckets" implementations
#include "cuckoofilter_table_simd.hpp"
#if defined(__AVX2__) && !defined(__AVX512F__)
#include "cuckoofilter_table_simd_avx2.hpp"
#endif
#if defined(__AVX512F__)
#include "cuckoofilter_table_simd_avx512f.hpp"
#endif

#include "bitsutil.hpp"


namespace dtl {
namespace cuckoofilter {


//===----------------------------------------------------------------------===//
// A cuckoo filter table implementation with SIMD accelerated lookup
// functions for the most common configurations.
//===----------------------------------------------------------------------===//

template<
    std::size_t bits_per_tag,
    std::size_t tags_per_bucket
>
class cuckoofilter_table {

public:

  // Determine whether a SIMD implementation is available.
  static constexpr u1 is_vectorized = internal::find_tag_in_buckets_simd<bits_per_tag, tags_per_bucket>::vectorized;

  // The basic storage type. We choose 32-bit integers to be at least 4 byte aligned.
  using word_t = uint32_t;

  static_assert(bits_per_tag == 4
                || bits_per_tag == 8
                || bits_per_tag == 12
                || bits_per_tag == 16
                || bits_per_tag == 32, "Tag size not supported.");

  static_assert(tags_per_bucket == 1
                || tags_per_bucket == 2
                || tags_per_bucket == 4
                || tags_per_bucket == 8
                || tags_per_bucket == 16, "Associativity not supported.");

  static constexpr std::size_t bytes_per_bucket = (bits_per_tag * tags_per_bucket + 7) / 8;

  static constexpr uint32_t tag_mask = static_cast<uint32_t>((1ull << bits_per_tag) - 1);

  static constexpr bool delete_supported = false;


  struct bucket_t {
    char bits_[bytes_per_bucket];
  } __attribute__((__packed__));

  const size_t num_buckets_;


 public:

  explicit cuckoofilter_table(const size_t num) : num_buckets_(num) { }
  ~cuckoofilter_table() = default;

  //===----------------------------------------------------------------------===//
  // Read tag from pos(i,j)
  //===----------------------------------------------------------------------===//
  __forceinline__ __host__ __device__
  uint32_t
  read_tag(const word_t* __restrict filter_data, const std::size_t i, const std::size_t j) const {
    const auto* buckets = reinterpret_cast<const bucket_t*>(filter_data);
    const char* p = buckets[i].bits_;
    uint32_t tag;
    /* following code only works for little-endian */
    if (bits_per_tag == 2) {
      tag = *((uint8_t *)p) >> (j * 2);
    } else if (bits_per_tag == 4) {
      p += (j >> 1);
      tag = *((uint8_t *)p) >> ((j & 1) << 2);
    } else if (bits_per_tag == 8) {
      p += j;
      tag = *((uint8_t *)p);
    } else if (bits_per_tag == 12) {
      p += j + (j >> 1);
      tag = *((uint16_t *)p) >> ((j & 1) << 2);
    } else if (bits_per_tag == 16) {
      p += (j << 1);
      tag = *((uint16_t *)p);
    } else if (bits_per_tag == 32) {
      tag = ((uint32_t *)p)[j];
    }
    return tag & tag_mask;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Write tag to pos(i,j)
  //===----------------------------------------------------------------------===//
  __forceinline__
  void
  write_tag(word_t* __restrict filter_data,
            const std::size_t i, const std::size_t j, const uint32_t t) const {
    auto* buckets = reinterpret_cast<bucket_t*>(filter_data);
    char *p = buckets[i].bits_;
    uint32_t tag = t & tag_mask;
    /* following code only works for little-endian */
    if (bits_per_tag == 2) {
      *((uint8_t *)p) |= tag << (2 * j);
    } else if (bits_per_tag == 4) {
      p += (j >> 1);
      if ((j & 1) == 0) {
        *((uint8_t *)p) &= 0xf0;
        *((uint8_t *)p) |= tag;
      } else {
        *((uint8_t *)p) &= 0x0f;
        *((uint8_t *)p) |= (tag << 4);
      }
    } else if (bits_per_tag == 8) {
      ((uint8_t *)p)[j] = tag;
    } else if (bits_per_tag == 12) {
      p += (j + (j >> 1));
      if ((j & 1) == 0) {
        ((uint16_t *)p)[0] &= 0xf000;
        ((uint16_t *)p)[0] |= tag;
      } else {
        ((uint16_t *)p)[0] &= 0x000f;
        ((uint16_t *)p)[0] |= (tag << 4);
      }
    } else if (bits_per_tag == 16) {
      ((uint16_t *)p)[j] = tag;
    } else if (bits_per_tag == 32) {
      ((uint32_t *)p)[j] = tag;
    }
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Insert
  //===----------------------------------------------------------------------===//
  __forceinline__
  bool
  insert_tag_to_bucket(word_t* __restrict filter_data,
                       const std::size_t i, const uint32_t tag,
                       const bool kickout, uint32_t& oldtag) const {
    for (size_t j = 0; j < tags_per_bucket; j++) {
      if (read_tag(filter_data, i, j) == 0) {
        write_tag(filter_data, i, j, tag);
        return true;
      }
    }
    if (kickout) {
      std::size_t r = dtl::this_thread::rand32() % tags_per_bucket;
      oldtag = read_tag(filter_data, i, r);
      write_tag(filter_data, i, r, tag);
    }
    return false;
  }
  //===----------------------------------------------------------------------===//



  //===----------------------------------------------------------------------===//
  // Find tags in buckets.
  //===----------------------------------------------------------------------===//
  // Scalar
  __forceinline__ __host__ __device__
  bool
  find_tag_in_buckets(const word_t* __restrict filter_data,
                      const std::size_t i1, const std::size_t i2,
                      const uint32_t tag) const {
    return internal::find_tag_in_buckets_scalar<bits_per_tag, tags_per_bucket>::dispatch(*this, filter_data, i1, i2, tag);
  }

  // vectorized
  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  typename dtl::vec<uint32_t, dtl::vector_length<Tv>::value>::mask
  find_tag_in_buckets(const word_t* __restrict filter_data,
                      const Tv& i1,
                      const Tv& i2,
                      const Tv& tag) const {
    return internal::find_tag_in_buckets_simd<bits_per_tag, tags_per_bucket>::dispatch(filter_data, i1, i2, tag);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Convenience functions.
  //===----------------------------------------------------------------------===//
  std::size_t
  num_buckets() const {
    return num_buckets_;
  }

  std::size_t
  size_in_bytes() const {
    return word_cnt() * sizeof(word_t);
  }

  __forceinline__ __host__ __device__
  std::size_t
  word_cnt() const {
    return ((bytes_per_bucket * num_buckets_ + (sizeof(word_t) - 1)) / sizeof(word_t));
  }

  std::size_t size_in_tags() const {
    return tags_per_bucket * num_buckets_;
  }

  std::size_t
  count_occupied_entries_in_bucket(const word_t* __restrict filter_data,
                                   const std::size_t bucket_idx) const {
    std::size_t num = 0;
    for (std::size_t tag_idx = 0; tag_idx < tags_per_bucket; tag_idx++) {
      if (read_tag(filter_data, bucket_idx, tag_idx) != 0) {
        num++;
      }
    }
    return num;
  }

  std::size_t
  count_occupied_entires(const word_t* __restrict filter_data) const {
    std::size_t num = 0;
    for (std::size_t bucket_idx = 0; bucket_idx < num_buckets_; bucket_idx++) {
      num += count_occupied_entries_in_bucket(filter_data, bucket_idx);
    }
    return num;
  }
  //===----------------------------------------------------------------------===//

};


} // namespace cuckoofilter
} // namespace dtl
