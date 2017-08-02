#pragma once

#ifndef _DTL_SIMD_INCLUDED
#error "Never use <dtl/simd/intrin_avx2.hpp> directly; include <dtl/simd.hpp> instead."
#endif

#include "../adept.hpp"
#include <dtl/bits.hpp>

#include "immintrin.h"

namespace dtl {
namespace simd {

namespace internal {
namespace avx2 {

//template<u32 L>
struct mask {
  __m256i data;
  forceinline u1 all() const { return _mm256_movemask_epi8(data) == -1; };
  forceinline u1 any() const { return _mm256_movemask_epi8(data) != 0; };
  forceinline u1 none() const { return _mm256_movemask_epi8(data) == 0; };
  forceinline void set(u1 value) {
    if (value) {
      data = _mm256_set1_epi64x(-1);
    } else {
      data = _mm256_set1_epi64x(0);
    }
  }
  forceinline void set(u64 idx, u1 value) {
    i32 v = value ? -1 : 0;
    switch (idx) {
      // TODO support other types than 32-bit integer
      case 0: data = _mm256_insert_epi32(data, v, 0); break;
      case 1: data = _mm256_insert_epi32(data, v, 1); break;
      case 2: data = _mm256_insert_epi32(data, v, 2); break;
      case 3: data = _mm256_insert_epi32(data, v, 3); break;
      case 4: data = _mm256_insert_epi32(data, v, 4); break;
      case 5: data = _mm256_insert_epi32(data, v, 5); break;
      case 6: data = _mm256_insert_epi32(data, v, 6); break;
      case 7: data = _mm256_insert_epi32(data, v, 7); break;
//      default: unreachable();
    }
  };
  forceinline u1 get(u64 idx) const {
    switch (idx) {
      case 0: return _mm256_extract_epi32(data, 0) != 0;
      case 1: return _mm256_extract_epi32(data, 1) != 0;
      case 2: return _mm256_extract_epi32(data, 2) != 0;
      case 3: return _mm256_extract_epi32(data, 3) != 0;
      case 4: return _mm256_extract_epi32(data, 4) != 0;
      case 5: return _mm256_extract_epi32(data, 5) != 0;
      case 6: return _mm256_extract_epi32(data, 6) != 0;
      case 7: return _mm256_extract_epi32(data, 7) != 0;
//      default: unreachable();
    }
  };
  forceinline mask bit_and(const mask& o) const { return mask { _mm256_and_si256(data, o.data) }; };
  forceinline mask bit_or(const mask& o) const { return mask { _mm256_or_si256(data, o.data) }; };
  forceinline mask bit_xor(const mask& o) const { return mask { _mm256_xor_si256(data, o.data) }; };
  forceinline mask bit_not() const { return mask { _mm256_andnot_si256(data, _mm256_set1_epi64x(-1)) }; };

//  forceinline $u64
//  to_positions($u32* positions, $u32 offset) const {
//    const __m256i zero = _mm256_setzero_si256();
//    if (_mm256_testc_si256(zero, data)) return 0;
//
//    // TODO consider SIMDfication using a LUT
//    $u32 bitmask = _mm256_movemask_ps(reinterpret_cast<__m256>(data));
////    if (bitmask == 0) return 0;
//    $u32* writer = positions;
//    for ($u32 m = _mm_popcnt_u32(bitmask); m > 0; m--) {
//      $u32 bit_pos = dtl::bits::tz_count(bitmask);
//      *writer = bit_pos + offset;
//      bitmask = _blsr_u32(bitmask);
//      writer++;
//    }
//    return writer - positions;
//  }

  forceinline $u64
  to_positions($u32* positions, $u32 offset) const {
    // only makes sence for unselective queries
//    const __m256i zero = _mm256_setzero_si256();
//    if (_mm256_testc_si256(zero, data)) return 0;

    const __m256i offset_vec = _mm256_set1_epi32(offset);
    i32 bitmask = _mm256_movemask_ps(reinterpret_cast<__m256>(data));
    const dtl::r256 match_pos_vec = { .i = { _mm256_cvtepi16_epi32(dtl::simd::lut_match_pos[bitmask].i) } };
    const __m256i pos_vec = _mm256_add_epi32(offset_vec, match_pos_vec.i);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(positions), pos_vec);
    // TODO consider using popcnt instead
    return dtl::simd::lut_match_cnt[bitmask];
  }

};

} // avx2 namespace
} // internal namespace


namespace {
using mask = internal::avx2::mask;
}


/// Define native vector type for SIMD-256: 8 x i32
template<>
struct vs<$i32, 8> : base<$i32, 8> {
  using type = __m256i;
  using mask_type = mask; //<8>;
  type data;
};
template<>
struct vs<$u32, 8> : base<$u32, 8> {
  using type = __m256i;
  using mask_type = mask; //<8>;
  type data;
};

/// Define native vector type for SIMD-256: 4 x i64
template<>
struct vs<$i64, 4> : base<$i64, 4> {
  using type = __m256i;
  using mask_type = mask;
  type data;
};
template<>
struct vs<$u64, 4> : base<$u64, 4> {
  using type = __m256i;
  using mask_type = mask;
  type data;
};


// --- broadcast / set

#define __GENERATE(Tp, Tv, Ta, IntrinFn) \
template<>                                             \
struct broadcast<Tp, Tv, Ta> : vector_fn<Tp, Tv, Ta> { \
  using fn = vector_fn<Tp, Tv, Ta>;                    \
  forceinline typename fn::vector_type                 \
  operator()(const typename fn::value_type& a) const noexcept { \
    return IntrinFn(a);                                \
  }                                                    \
  forceinline typename fn::vector_type                 \
  operator()(const typename fn::value_type& a,         \
             const typename fn::vector_type& src,      \
             const mask m) const noexcept {       \
    return _mm256_blendv_epi8(src, IntrinFn(a), m.data); \
  }                                                    \
};

__GENERATE($i32, __m256i, $i32, _mm256_set1_epi32)
__GENERATE($u32, __m256i, $u32, _mm256_set1_epi32)
__GENERATE($i64, __m256i, $i64, _mm256_set1_epi64x)
__GENERATE($u64, __m256i, $u64, _mm256_set1_epi64x)
#undef __GENERATE


#define __GENERATE_BLEND(Tp, Tv, Ta, IntrinFnMask)     \
template<>                                             \
struct blend<Tp, Tv, Ta> : vector_fn<Tp, Tv, Ta> {     \
  using fn = vector_fn<Tp, Tv, Ta>;                    \
  forceinline typename fn::vector_type                 \
  operator()(const typename fn::vector_type& a,        \
             const typename fn::vector_type& b,        \
             const mask m) const noexcept {            \
    return IntrinFnMask(a, b, m.data);                 \
  }                                                    \
};

__GENERATE_BLEND($i32, __m256i, __m256i, _mm256_blendv_epi8)
__GENERATE_BLEND($u32, __m256i, __m256i, _mm256_blendv_epi8)
__GENERATE_BLEND($i64, __m256i, __m256i, _mm256_blendv_epi8)
__GENERATE_BLEND($u64, __m256i, __m256i, _mm256_blendv_epi8)
#undef __GENERATE


template<>
struct set<$i32, __m256i, $i32> : vector_fn<$i32, __m256i, $i32> {
  forceinline __m256i operator()(const $i32& a) const noexcept {
    return _mm256_set1_epi32(a);
  }
};


// Load
template<>
struct gather<$i32, __m256i, __m256i> : vector_fn<$i32, __m256i, __m256i, __m256i> {
  inline __m256i operator()(const i32* const base_addr, const __m256i& idxs) const noexcept {
    return _mm256_i32gather_epi32(base_addr, idxs, 4);
  }
};

template<>
struct gather<$u32, __m256i, __m256i> : vector_fn<$u32, __m256i, __m256i, __m256i> {
  inline __m256i operator()(const u32* const base_addr, const __m256i& idxs) const noexcept {
    return _mm256_i32gather_epi32(reinterpret_cast<const i32*>(base_addr), idxs, 4);
  }
};

template<>
struct gather<$i64, __m256i, __m256i> : vector_fn<$i64, __m256i, __m256i, __m256i> {
  inline __m256i operator()(const i64* const base_addr, const __m256i& idxs) const noexcept {
    const auto b = reinterpret_cast<const long long int *>(base_addr);
    return _mm256_i64gather_epi64(b, idxs, 8);
  }
};

template<>
struct gather<$u64, __m256i, __m256i> : vector_fn<$u64, __m256i, __m256i, __m256i> {
  inline __m256i operator()(const u64* const base_addr, const __m256i& idxs) const noexcept {
    const auto b = reinterpret_cast<const long long int *>(base_addr);
    return _mm256_i64gather_epi64(b, idxs, 8);
  }
};


// Store
//template<>
//struct scatter<$i32, __m256i, __m256i> : vector_fn<$i32, __m256i, __m256i, __m256i> {
//  inline __m256i operator()($i32* base_addr, const __m256i& idxs, const __m256i& what) const noexcept {
//    i32* i = reinterpret_cast<i32*>(&idxs);
//    i32* w = reinterpret_cast<i32*>(&what);
//    for ($u64 j = 0; j < 8; j++) {
//      base_addr[i[j]] = w[j];
//    }
//  }
//};
//
//template<>
//struct scatter<$u32, __m256i, __m256i> : vector_fn<$u32, __m256i, __m256i, __m256i> {
//  inline __m256i operator()($u32* base_addr, const __m256i& idxs, const __m256i& what) const noexcept {
//    u32* i = reinterpret_cast<u32*>(&idxs);
//    u32* w = reinterpret_cast<u32*>(&what);
//    for ($u64 j = 0; j < 8; j++) {
//      base_addr[i[j]] = w[j];
//    }
//  }
//};
//
//template<>
//struct scatter<$i64, __m256i, __m256i> : vector_fn<$i64, __m256i, __m256i, __m256i> {
//  inline __m256i operator()($i64* base_addr, const __m256i& idxs, const __m256i& what) const noexcept {
//    i64* i = reinterpret_cast<i64*>(&idxs);
//    i64* w = reinterpret_cast<i64*>(&what);
//    for ($u64 j = 0; j < 4; j++) {
//      base_addr[i[j]] = w[j];
//    }
//  }
//};
//
//template<>
//struct scatter<$u64, __m256i, __m256i> : vector_fn<$u64, __m256i, __m256i, __m256i> {
//  inline __m256i operator()($u64* base_addr, const __m256i& idxs, const __m256i& what) const noexcept {
//    u64* i = reinterpret_cast<u64*>(&idxs);
//    u64* w = reinterpret_cast<u64*>(&what);
//    for ($u64 j = 0; j < 4; j++) {
//      base_addr[i[j]] = w[j];
//    }
//  }
//};



// Arithmetic
template<>
struct plus<$i32, __m256i> : vector_fn<$i32, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return _mm256_add_epi32(lhs, rhs);
  }
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs,
                            const __m256i& src, const mask& m) const noexcept {
    return _mm256_blendv_epi8(src, _mm256_add_epi32(lhs, rhs), m.data);
  }
};

template<>
struct plus<$u32, __m256i> : vector_fn<$u32, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return _mm256_add_epi32(lhs, rhs);
  }
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs,
                            const __m256i& src, const mask& m) const noexcept {
    return _mm256_blendv_epi8(src, _mm256_add_epi32(lhs, rhs), m.data);
  }
};

template<>
struct plus<$i64, __m256i> : vector_fn<$i64, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return _mm256_add_epi64(lhs, rhs);
  }
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs,
                            const __m256i& src, const mask& m) const noexcept {
    return _mm256_blendv_epi8(src, _mm256_add_epi64(lhs, rhs), m.data);
  }
};

template<>
struct plus<$u64, __m256i> : vector_fn<$u64, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return _mm256_add_epi64(lhs, rhs);
  }
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs,
                            const __m256i& src, const mask& m) const noexcept {
    return _mm256_blendv_epi8(src, _mm256_add_epi64(lhs, rhs), m.data);
  }
};

template<>
struct minus<$i32, __m256i> : vector_fn<$i32, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return _mm256_sub_epi32(lhs, rhs);
  }
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs,
                            const __m256i& src, const mask& m) const noexcept {
    return _mm256_blendv_epi8(src, _mm256_sub_epi32(lhs, rhs), m.data);
  }
};

template<>
struct multiplies<$i32, __m256i> : vector_fn<$i32, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return _mm256_mullo_epi32(lhs, rhs);
  }
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs,
                            const __m256i& src, const mask& m) const noexcept {
    return _mm256_blendv_epi8(src, _mm256_mullo_epi32(lhs, rhs), m.data);
  }
};

template<>
struct multiplies<$u32, __m256i> : vector_fn<$u32, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return _mm256_mullo_epi32(lhs, rhs);
  }
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs,
                            const __m256i& src, const mask& m) const noexcept {
    return _mm256_blendv_epi8(src, _mm256_mullo_epi32(lhs, rhs), m.data);
  }
};

template<>
struct multiplies<$i64, __m256i> : vector_fn<$i64, __m256i> {
  inline __m256i mul(const __m256i& lhs, const __m256i& rhs) const noexcept {
    const __m256i hi_lhs = _mm256_srli_epi64(lhs, 32);
    const __m256i hi_rhs = _mm256_srli_epi64(rhs, 32);
    const __m256i t1 = _mm256_mul_epu32(lhs, hi_rhs);
    const __m256i t2 = _mm256_mul_epu32(lhs, rhs);
    const __m256i t3 = _mm256_mul_epu32(hi_lhs, rhs);
    const __m256i t4 = _mm256_add_epi64(_mm256_slli_epi64(t3, 32), t2);
    const __m256i t5 = _mm256_add_epi64(_mm256_slli_epi64(t1, 32), t4);
    return t5;
  }
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return mul(lhs, rhs);
  }
  inline __m256i operator()(const __m256i& lhs, const __m256i& rhs,
                            const __m256i& src, const mask& m) const noexcept {
    return _mm256_blendv_epi8(src, mul(lhs, rhs), m.data);
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
struct shift_left_var<$u32, __m256i,  __m256i> : vector_fn<$u32, __m256i, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& count) const noexcept {
    return _mm256_sllv_epi32(lhs, count);
  }
};

template<>
struct shift_left_var<$u64, __m256i,  __m256i> : vector_fn<$u64, __m256i, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& count) const noexcept {
    return _mm256_sllv_epi64(lhs, count);
  }
};

template<>
struct shift_right<$i32, __m256i, i32> : vector_fn<$i32, __m256i, i32> {
  inline __m256i operator()(const __m256i& lhs, i32& count) const noexcept {
    return _mm256_srli_epi32(lhs, count);
  }
};

template<>
struct shift_right_var<$u32, __m256i,  __m256i> : vector_fn<$u32, __m256i, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& count) const noexcept {
    return _mm256_srlv_epi32(lhs, count);
  }
};

template<>
struct shift_right_var<$u64, __m256i,  __m256i> : vector_fn<$u64, __m256i, __m256i> {
  inline __m256i operator()(const __m256i& lhs, const __m256i& count) const noexcept {
    return _mm256_srlv_epi64(lhs, count);
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
struct less<$i32, __m256i, __m256i, mask> : vector_fn<$i32, __m256i, __m256i, mask> {
  inline mask operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return mask { _mm256_cmpgt_epi32(rhs, lhs) };
  }
};

template<>
struct greater<$i32, __m256i, __m256i, mask> : vector_fn<$i32, __m256i, __m256i, mask> {
  inline mask operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return mask { _mm256_cmpgt_epi32(lhs, rhs) };
  }
};

template<>
struct equal<$i32, __m256i, __m256i, mask> : vector_fn<$i32, __m256i, __m256i, mask> {
  inline mask operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return mask { _mm256_cmpeq_epi32(lhs, rhs) };
  }
};

template<>
struct equal<$u32, __m256i, __m256i, mask> : vector_fn<$u32, __m256i, __m256i, mask> {
  inline mask operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return mask { _mm256_cmpeq_epi32(lhs, rhs) };
  }
};

template<>
struct equal<$u64, __m256i, __m256i, mask> : vector_fn<$u64, __m256i, __m256i, mask> {
  inline mask operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return mask { _mm256_cmpeq_epi64(lhs, rhs) };
  }
};

template<>
struct not_equal<$i32, __m256i, __m256i, mask> : vector_fn<$i32, __m256i, __m256i, mask> {
  inline mask operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return mask { _mm256_andnot_si256(_mm256_cmpeq_epi32(lhs, rhs), _mm256_set1_epi32(-1))};
  }
};

template<>
struct not_equal<$u32, __m256i, __m256i, mask> : vector_fn<$u32, __m256i, __m256i, mask> {
  inline mask operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return mask { _mm256_andnot_si256(_mm256_cmpeq_epi32(lhs, rhs), _mm256_set1_epi32(-1))};
  }
};

template<>
struct not_equal<$i64, __m256i, __m256i, mask> : vector_fn<$i64, __m256i, __m256i, mask> {
  inline mask operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return mask { _mm256_andnot_si256(_mm256_cmpeq_epi64(lhs, rhs), _mm256_set1_epi64x(-1))};
  }
};

template<>
struct not_equal<$u64, __m256i, __m256i, mask> : vector_fn<$u64, __m256i, __m256i, mask> {
  inline mask operator()(const __m256i& lhs, const __m256i& rhs) const noexcept {
    return mask { _mm256_andnot_si256(_mm256_cmpeq_epi64(lhs, rhs), _mm256_set1_epi64x(-1))};
  }
};

} // namespace simd
} // namespace dtl