#pragma once

#ifndef _DTL_SIMD_INCLUDED
#error "Never use <dtl/simd/intrin_sse.hpp> directly; include <dtl/simd.hpp> instead."
#endif

#include "../adept.hpp"

#include "emmintrin.h"
#include "smmintrin.h"
#include "tmmintrin.h"
#include "xmmintrin.h"

namespace dtl {
namespace simd {

/// Define native vector type for SIMD-128: 4 x i32
template<>
struct vs<$i32, 4> : base<$i32, 4> {
  using type = __m128i;
  type data;
};

template<>
struct set<$i32, __m128i, $i32> : vector_fn<$i32, __m128i, $i32> {
  inline __m128i operator()(i32& a) const noexcept {
    return _mm_set1_epi32(a);
  }
};

template<>
struct plus<$i32, __m128i> : vector_fn<$i32, __m128i> {
  inline __m128i operator()(const __m128i& lhs, const __m128i& rhs) const noexcept {
    return _mm_add_epi16(lhs, rhs);
  }
};

} // namespace simd
} // namespace dtl