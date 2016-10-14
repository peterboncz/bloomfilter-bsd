#pragma once

#ifndef _DTL_SIMD_INCLUDED
#error "Never use <dtl/simd/intrin_avx2.hpp> directly; include <dtl/simd.hpp> instead."
#endif

#include "../adept.hpp"

#include "immintrin.h"

namespace dtl {
namespace simd {

namespace {

template<u32 L>
struct mask {
  __m256i data;
  inline u1 all() const { return _mm256_movemask_epi8(data) == -1; };
  inline u1 any() const { return _mm256_movemask_epi8(data) != 0; };
  inline u1 none() const { return _mm256_movemask_epi8(data) == 0; };
  inline void set(u1 value) {
    if (value) {
      data = _mm256_set1_epi64x(-1);
    } else {
      data = _mm256_set1_epi64x(0);
    }
  }
  inline void set(u64 idx, u1 value) {
    i32 v = value ? -1 : 0;
    switch (idx) {
      case 0: data = _mm256_insert_epi32(data, v, 0); break;
      case 1: data = _mm256_insert_epi32(data, v, 1); break;
      case 2: data = _mm256_insert_epi32(data, v, 2); break;
      case 3: data = _mm256_insert_epi32(data, v, 3); break;
      case 4: data = _mm256_insert_epi32(data, v, 4); break;
      case 5: data = _mm256_insert_epi32(data, v, 5); break;
      case 6: data = _mm256_insert_epi32(data, v, 6); break;
      case 7: data = _mm256_insert_epi32(data, v, 7); break;
      default: unreachable();
    }
  };
  inline u1 get(u64 idx) const {
    switch (idx) {
      case 0: return _mm256_extract_epi32(data, 0) != 0;
      case 1: return _mm256_extract_epi32(data, 1) != 0;
      case 2: return _mm256_extract_epi32(data, 2) != 0;
      case 3: return _mm256_extract_epi32(data, 3) != 0;
      case 4: return _mm256_extract_epi32(data, 4) != 0;
      case 5: return _mm256_extract_epi32(data, 5) != 0;
      case 6: return _mm256_extract_epi32(data, 6) != 0;
      case 7: return _mm256_extract_epi32(data, 7) != 0;
      default: unreachable();
    }
  };
  inline mask bit_and(const mask& o) const { return mask { _mm256_and_si256(data, o.data) }; };
  inline mask bit_or(const mask& o) const { return mask { _mm256_or_si256(data, o.data) }; };
  inline mask bit_xor(const mask& o) const { return mask { _mm256_xor_si256(data, o.data) }; };
};

} // anonymous namespace

/// Define native vector type for SIMD-256: 8 x i32
template<>
struct vs<$i32, 8> : base<$i32, 8> {
  using type = __m256i;
  using mask_type = mask<8>;
  type data;
};

/// Define native vector type for SIMD-256: 4 x i64
template<>
struct vs<$i64, 4> : base<$i64, 4> {
  using type = __m256i;
  type data;
};


template<>
struct set<$i32, __m256i, $i32> : vector_fn<$i32, __m256i, $i32> {
  inline __m256i operator()(const $i32& a) const noexcept {
    return _mm256_set1_epi32(a);
  }
};


// Load
template<>
struct gather<$i32, __m256i, $i32> : vector_fn<$i32, __m256i, $i32> {
  inline __m256i operator()(i32* const base_addr, const __m256i& idxs) const noexcept {
    return _mm256_i32gather_epi32(base_addr, idxs, 4);
  }
};

template<>
struct gather<$i64, __m256i, $i64> : vector_fn<$i64, __m256i, $i64> {
  inline __m256i operator()(i64* const base_addr, const __m256i& idxs) const noexcept {
    const auto b = reinterpret_cast<const long long int *>(base_addr);
    return _mm256_i64gather_epi64(b, idxs, 8);
  }
};


// Store
template<>
struct scatter<$i32, __m256i, $i32> : vector_fn<$i32, __m256i, $i32> {
  inline __m256i operator()($i32* const base_addr, const __m256i& idxs, const __m256i& what) const noexcept {
    i32* i = reinterpret_cast<i32*>(&idxs);
    i32* w = reinterpret_cast<i32*>(&what);
    for ($u64 j = 0; j < 8; j++) {
      base_addr[i[j]] = w[j];
    }
  }
};

template<>
struct scatter<$i64, __m256i, $i64> : vector_fn<$i64, __m256i, $i64> {
  inline __m256i operator()($i64* const base_addr, const __m256i& idxs, const __m256i& what) const noexcept {
    i64* i = reinterpret_cast<i64*>(&idxs);
    i64* w = reinterpret_cast<i64*>(&what);
    for ($u64 j = 0; j < 4; j++) {
      base_addr[i[j]] = w[j];
    }
  }
};



// Arithmetic
template<>
struct plus<$i32, __m256i> : vector_fn<$i32, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return _mm256_add_epi32(lhs, rhs);
  }
};

template<>
struct minus<$i32, __m256i> : vector_fn<$i32, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return _mm256_sub_epi32(lhs, rhs);
  }
};

template<>
struct multiplies<$i32, __m256i> : vector_fn<$i32, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return _mm256_mullo_epi32(lhs, rhs);
  }
};

template<>
struct multiplies<$i64, __m256i> : vector_fn<$i64, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    const __m256i hi_lhs = _mm256_srli_epi64(lhs, 32);
    const __m256i hi_rhs = _mm256_srli_epi64(rhs, 32);
    const __m256i t1 = _mm256_mul_epu32(lhs, hi_rhs);
    const __m256i t2 = _mm256_mul_epu32(lhs, rhs);
    const __m256i t3 = _mm256_mul_epu32(hi_lhs, rhs);
    const __m256i t4 = _mm256_add_epi64(_mm256_slli_epi64(t3, 32), t2);
    const __m256i t5 = _mm256_add_epi64(_mm256_slli_epi64(t1, 32), t4);
    return t5;
  }
};


// Shift
template<>
struct shift_left<$i32, __m256i, i32> : vector_fn<$i32, __m256i, i32> {
  inline __m256i operator()(const __m256i& lhs, i32& count) const noexcept {
    return _mm256_slli_epi32(lhs, count);
  }
};

template<>
struct shift_left_var<$i32, __m256i,  __m256i> : vector_fn<$i32, __m256i, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& count) const noexcept {
    return _mm256_sllv_epi32(lhs, count);
  }
};

template<>
struct shift_right<$i32, __m256i, i32> : vector_fn<$i32, __m256i, i32> {
  inline __m256i operator()(const __m256i& lhs, i32& count) const noexcept {
    return _mm256_srli_epi32(lhs, count);
  }
};

template<>
struct shift_right_var<$i32, __m256i,  __m256i> : vector_fn<$i32, __m256i, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& count) const noexcept {
    return _mm256_srlv_epi32(lhs, count);
  }
};



//  Bitwise operators
template<typename Tp /* ignored for bitwise operations */>
struct bit_and<Tp, __m256i> : vector_fn<Tp, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return _mm256_and_si256(lhs, rhs);
  }
};

template<typename Tp /* ignored for bitwise operations */>
struct bit_or<Tp, __m256i> : vector_fn<Tp, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return _mm256_or_si256(lhs, rhs);
  }
};

template<typename Tp /* ignored for bitwise operations */>
struct bit_xor<Tp, __m256i> : vector_fn<Tp, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return _mm256_xor_si256(lhs, rhs);
  }
};



// Comparison
template<>
struct less<$i32, __m256i, __m256i, mask<8>> : vector_fn<$i32, __m256i, __m256i, mask<8>> {
  inline mask<8> operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return mask<8> { _mm256_cmpgt_epi32(rhs, lhs) };
  }
};

template<>
struct greater<$i32, __m256i, __m256i, mask<8>> : vector_fn<$i32, __m256i, __m256i, mask<8>> {
  inline mask<8> operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return mask<8> { _mm256_cmpgt_epi32(lhs, rhs) };
  }
};

template<>
struct equal<$i32, __m256i, __m256i, mask<8>> : vector_fn<$i32, __m256i, __m256i, mask<8>> {
  inline mask<8> operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return mask<8> { _mm256_cmpeq_epi32(lhs, rhs) };
  }
};

template<>
struct not_equal<$i32, __m256i, __m256i, mask<8>> : vector_fn<$i32, __m256i, __m256i, mask<8>> {
  inline mask<8> operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return mask<8> { _mm256_andnot_si256(_mm256_cmpeq_epi32(lhs, rhs), _mm256_set1_epi32(-1))};
  }
};

} // namespace simd
} // namespace dtl