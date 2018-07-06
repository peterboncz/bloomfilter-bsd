#pragma once

//===----------------------------------------------------------------------===//
// This file is based on the original Cuckoo filter implementation,
// that can be found on GitHub: https://github.com/efficient/cuckoofilter
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
//  - Scalar code as a fall back (if AVX2 or AVX-512 is not supported)
//  - Experimental CUDA implementation

#include <cstddef>

#include <dtl/dtl.hpp>

#include "bitsutil.hpp"

namespace dtl {
namespace cuckoofilter {
namespace internal {

//===----------------------------------------------------------------------===//
// Find tag in buckets.
//===----------------------------------------------------------------------===//


// Generic scalar implementation.
template<std::size_t bits_per_tag, std::size_t tags_per_bucket>
struct find_tag_in_buckets_scalar {

  template<typename table_t>
  __forceinline__ __host__ __device__
  static bool
  dispatch(const table_t& table,
           const typename table_t::word_t* __restrict filter_data,
           const std::size_t i1, const std::size_t i2,
           const uint32_t tag) {
    const auto* buckets = reinterpret_cast<const typename table_t::bucket_t*>(filter_data);
    const char *p1 = buckets[i1].bits_;
    const char *p2 = buckets[i2].bits_;

    uint64_t v1 = *((uint64_t *)p1);
    uint64_t v2 = *((uint64_t *)p2);

    // caution: unaligned access & assuming little endian
    if (bits_per_tag == 4 && tags_per_bucket == 4) {
      return hasvalue4(v1, tag) | hasvalue4(v2, tag);
    } else if (bits_per_tag == 8 && tags_per_bucket == 4) {
      return hasvalue8(v1, tag) | hasvalue8(v2, tag);
    } else if (bits_per_tag == 12 && tags_per_bucket == 4) {
      return hasvalue12(v1, tag) | hasvalue12(v2, tag);
    } else if (bits_per_tag == 16 && tags_per_bucket == 4) {
      return hasvalue16(v1, tag) | hasvalue16(v2, tag);
    } else {
      for (std::size_t j = 0; j < tags_per_bucket; j++) {
        if ((table.read_tag(filter_data, i1, j) == tag) || (table.read_tag(filter_data, i2, j) == tag)) {
          return true;
        }
      }
      return false;
    }
  }

};


} // namespace internal
} // namespace cuckoofilter
} // namespace dtl


#if defined(__CUDA_ARCH__)
#include "cuckoofilter_table_scalar_cuda.hpp"
#endif
