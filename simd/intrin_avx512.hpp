#pragma once

#ifndef _DTL_SIMD_INCLUDED
#error "Never use <dtl/simd/intrin_avx512.hpp> directly; include <dtl/simd.hpp> instead."
#endif

#include "../adept.hpp"
#include "immintrin.h"

namespace dtl {
namespace simd {

namespace {

struct mask16 {
  __mmask16 data;
  inline u1 all() const { return data == __mmask16(-1); };
  inline u1 any() const { return data != __mmask16(0); };
  inline u1 none() const { return data == __mmask16(0); };
  inline void set(u1 value) {
    data = __mmask16(0) - value;
  }
  inline void set(u64 idx, u1 value) {
    data = __mmask16(1) << idx;
  };
  inline u1 get(u64 idx) const {
    return (data & (__mmask16(1) << idx)) != __mmask16(0);
  };
  inline mask16 bit_and(const mask16& o) const { return mask16 { _mm512_kand(data, o.data) }; };
  inline mask16 bit_or(const mask16& o) const { return mask16 { _mm512_kor(data, o.data) }; };
  inline mask16 bit_xor(const mask16& o) const { return mask16 { _mm512_kxor(data, o.data) }; };
  inline mask16 bit_not() const { return mask16 { _mm512_knot(data) }; };
};

//    __mmask8

} // anonymous namespace


// --- vector types

template<>
struct vs<$i32, 16> : base<$i32, 16> {
  using type = __m512i;
  using mask_type = mask16;
  type data;
};

template<>
struct vs<$u32, 16> : base<$u32, 16> {
  using type = __m512i;
  using mask_type = mask16;
  type data;
};

template<>
struct vs<$i64, 8> : base<$i64, 8> {
  using type = __m512i;
  using mask_type = mask16;
  type data;
};

template<>
struct vs<$u64, 8> : base<$u64, 8> {
  using type = __m512i;
  using mask_type = mask16;
  type data;
};


// --- broadcast / set

#define __GENERATE(Tp, Tv, Ta, IntrinFn, IntrinFnMask) \
template<>                                             \
struct broadcast<Tp, Tv, Ta> : vector_fn<Tp, Tv, Ta> { \
  using fn = vector_fn<Tp, Tv, Ta>;                    \
  inline typename fn::vector_type                      \
  operator()(const typename fn::value_type& a) const noexcept { \
    return IntrinFn(a);                                \
  }                                                    \
  inline typename fn::vector_type                      \
  operator()(const typename fn::value_type& a,         \
             const typename fn::vector_type& src,      \
             const mask16 mask) const noexcept {       \
    return IntrinFnMask(src, mask.data, a);            \
  }                                                    \
};

__GENERATE($i32, __m512i, $i32, _mm512_set1_epi32, _mm512_mask_set1_epi32)
__GENERATE($u32, __m512i, $u32, _mm512_set1_epi32, _mm512_mask_set1_epi32)
__GENERATE($i64, __m512i, $i64, _mm512_set1_epi64, _mm512_mask_set1_epi64)
__GENERATE($u64, __m512i, $u64, _mm512_set1_epi64, _mm512_mask_set1_epi64)
#undef __GENERATE


#define __GENERATE_BLEND(Tp, Tv, Ta, IntrinFnMask) \
template<>                                             \
struct blend<Tp, Tv, Ta> : vector_fn<Tp, Tv, Ta> { \
  using fn = vector_fn<Tp, Tv, Ta>;                    \
  inline typename fn::vector_type                      \
  operator()(const typename fn::vector_type& a,        \
             const typename fn::vector_type& b,        \
             const mask16 mask) const noexcept {       \
    return IntrinFnMask(mask.data, a, b);              \
  }                                                    \
};

__GENERATE_BLEND($i32, __m512i, __m512i, _mm512_mask_blend_epi32)
__GENERATE_BLEND($u32, __m512i, __m512i, _mm512_mask_blend_epi32)
__GENERATE_BLEND($i64, __m512i, __m512i, _mm512_mask_blend_epi64)
__GENERATE_BLEND($u64, __m512i, __m512i, _mm512_mask_blend_epi64)
#undef __GENERATE


// --- Load

#define __GENERATE(Tp, Tv, Ta, IntrinFn, IntrinFnMask) \
template<>                                             \
struct gather<Tp, Tv, Ta> : vector_fn<Tp, Tv, Ta> { \
  using fn = vector_fn<Tp, Tv, Ta>;                    \
  inline typename fn::vector_type                      \
  operator()(const typename fn::vector_type& idx) const noexcept { \
    return IntrinFn(idx, 0, 1);                        \
  }                                                    \
  inline typename fn::vector_type                      \
  operator()(const typename fn::vector_type& idx,      \
             const typename fn::vector_type& src,      \
             const mask16 mask) const noexcept {       \
    return IntrinFnMask(src, mask.data, idx, 0, 1);    \
  }                                                    \
};

// TODO ???
//__GENERATE($i32, __m512i, $i32, _mm512_i64gather_epi32, _mm512_mask_i64gather_epi32)
//__GENERATE($u32, __m512i, $u32, _mm512_i64gather_epi32, _mm512_mask_i64gather_epi32)
__GENERATE($i64, __m512i, $i64, _mm512_i64gather_epi64, _mm512_mask_i64gather_epi64)
__GENERATE($u64, __m512i, $u64, _mm512_i64gather_epi64, _mm512_mask_i64gather_epi64)
#undef __GENERATE


// --- Store

#define __GENERATE(Tp, Tv, Ta, IntrinFn, IntrinFnMask) \
template<>                                             \
struct scatter<Tp, Tv, Ta> : vector_fn<Tp, Tv, Ta> {   \
  using fn = vector_fn<Tp, Tv, Ta>;                    \
  inline void                                          \
  operator()(const typename fn::vector_type& idx,      \
             const typename fn::vector_type& a) const noexcept { \
    IntrinFn(0, idx, a, 1);                            \
  }                                                    \
  inline void                                          \
  operator()(const typename fn::vector_type& idx,      \
             const typename fn::vector_type& a,        \
             const mask16 mask) const noexcept {       \
    IntrinFnMask(0, mask.data, idx, a, 1);             \
  }                                                    \
};

// TODO ???
//__GENERATE($i32, __m512i, $i32, _mm512_i64scatter_epi32, _mm512_mask_i64scatter_epi32)
//__GENERATE($u32, __m512i, $u32, _mm512_i64scatter_epi32, _mm512_mask_i64scatter_epi32)
__GENERATE($i64, __m512i, $i64, _mm512_i64scatter_epi64, _mm512_mask_i64scatter_epi64)
__GENERATE($u64, __m512i, $u64, _mm512_i64scatter_epi64, _mm512_mask_i64scatter_epi64)
#undef __GENERATE


// --- Arithmetic

#define __GENERATE_ARITH(Op, Tp, Tv, Ta, IntrinFn, IntrinFnMask) \
template<>                                             \
struct Op<Tp, Tv, Ta> : vector_fn<Tp, Tv, Ta> {        \
  using fn = vector_fn<Tp, Tv, Ta>;                    \
  inline typename fn::vector_type                      \
  operator()(const typename fn::vector_type& lhs,      \
             const typename fn::vector_type& rhs) const noexcept { \
    return IntrinFn(lhs, rhs);                         \
  }                                                    \
  inline typename fn::vector_type                      \
  operator()(const typename fn::vector_type& lhs,      \
             const typename fn::vector_type& rhs,      \
             const typename fn::vector_type& src,      \
             const mask16 mask) const noexcept {       \
    return IntrinFnMask(src, mask.data, lhs, rhs);     \
  }                                                    \
};

inline __m512i
_mm512_mul_epi64(const __m512i& lhs, const __m512i& rhs) {
  const __m512i hi_lhs = _mm512_srli_epi64(lhs, 32);
  const __m512i hi_rhs = _mm512_srli_epi64(rhs, 32);
  const __m512i t1 = _mm512_mul_epu32(lhs, hi_rhs);
  const __m512i t2 = _mm512_mul_epu32(lhs, rhs);
  const __m512i t3 = _mm512_mul_epu32(hi_lhs, rhs);
  const __m512i t4 = _mm512_add_epi64(_mm512_slli_epi64(t3, 32), t2);
  const __m512i t5 = _mm512_add_epi64(_mm512_slli_epi64(t1, 32), t4);
  return t5;
}


__GENERATE_ARITH(plus, $i32, __m512i, $i32, _mm512_add_epi32, _mm512_mask_add_epi32)
__GENERATE_ARITH(plus, $u32, __m512i, $u32, _mm512_add_epi32, _mm512_mask_add_epi32)
__GENERATE_ARITH(plus, $i64, __m512i, $i64, _mm512_add_epi64, _mm512_mask_add_epi64)
__GENERATE_ARITH(plus, $u64, __m512i, $u64, _mm512_add_epi64, _mm512_mask_add_epi64)
__GENERATE_ARITH(minus, $i32, __m512i, $i32, _mm512_sub_epi32, _mm512_mask_sub_epi32)
__GENERATE_ARITH(minus, $u32, __m512i, $u32, _mm512_sub_epi32, _mm512_mask_sub_epi32)
__GENERATE_ARITH(minus, $i64, __m512i, $i64, _mm512_sub_epi64, _mm512_mask_sub_epi64)
__GENERATE_ARITH(minus, $u64, __m512i, $u64, _mm512_sub_epi64, _mm512_mask_sub_epi64)
__GENERATE_ARITH(multiplies, $i32, __m512i, $i32, _mm512_mullo_epi32, _mm512_mask_mullo_epi32)
__GENERATE_ARITH(multiplies, $u32, __m512i, $u32, _mm512_mullo_epi32, _mm512_mask_mullo_epi32)
__GENERATE_ARITH(multiplies, $i64, __m512i, $i64, _mm512_mul_epi64, _mm512_mask_mul_epi64) // TODO fix
__GENERATE_ARITH(multiplies, $u64, __m512i, $u64, _mm512_mul_epi64, _mm512_mask_mul_epi64) // TODO fix
#undef __GENERATE_ARITH


// --- Shift

#define __GENERATE_SHIFT(Op, Tp, Tv, Ta, IntrinFn, IntrinFnMask) \
template<>                                             \
struct Op<Tp, Tv, Ta> : vector_fn<Tp, Tv, Ta> {        \
  using fn = vector_fn<Tp, Tv, Ta>;                    \
  inline typename fn::vector_type                      \
  operator()(const typename fn::vector_type& a,        \
             const typename fn::argument_type& count) const noexcept { \
    return IntrinFn(a, count);                         \
  }                                                    \
  inline typename fn::vector_type                      \
  operator()(const typename fn::vector_type& a,        \
             const typename fn::argument_type& count,  \
             const typename fn::vector_type& src,      \
             const mask16 mask) const noexcept {       \
    return IntrinFnMask(src, mask.data, a, count);     \
  }                                                    \
};

__GENERATE_SHIFT(shift_left, $i32, __m512i, $i32, _mm512_slli_epi32, _mm512_mask_slli_epi32)
__GENERATE_SHIFT(shift_left, $u32, __m512i, $u32, _mm512_slli_epi32, _mm512_mask_slli_epi32)
__GENERATE_SHIFT(shift_right, $i32, __m512i, $i32, _mm512_srli_epi32, _mm512_mask_srli_epi32)
__GENERATE_SHIFT(shift_right, $u32, __m512i, $u32, _mm512_srli_epi32, _mm512_mask_srli_epi32)
__GENERATE_SHIFT(shift_left_var, $i32, __m512i, __m512i, _mm512_sllv_epi32, _mm512_mask_sllv_epi32)
__GENERATE_SHIFT(shift_left_var, $u32, __m512i, __m512i, _mm512_sllv_epi32, _mm512_mask_sllv_epi32)
__GENERATE_SHIFT(shift_right_var, $i32, __m512i, __m512i, _mm512_srlv_epi32, _mm512_mask_srlv_epi32)
__GENERATE_SHIFT(shift_right_var, $u32, __m512i, __m512i, _mm512_srlv_epi32, _mm512_mask_srlv_epi32)

__GENERATE_SHIFT(shift_left, $i64, __m512i, $i64, _mm512_slli_epi64, _mm512_mask_slli_epi64)
__GENERATE_SHIFT(shift_left, $u64, __m512i, $u64, _mm512_slli_epi64, _mm512_mask_slli_epi64)
__GENERATE_SHIFT(shift_right, $i64, __m512i, $i64, _mm512_srli_epi64, _mm512_mask_srli_epi64)
__GENERATE_SHIFT(shift_right, $u64, __m512i, $u64, _mm512_srli_epi64, _mm512_mask_srli_epi64)
__GENERATE_SHIFT(shift_left_var, $i64, __m512i, __m512i, _mm512_sllv_epi64, _mm512_mask_sllv_epi64)
__GENERATE_SHIFT(shift_left_var, $u64, __m512i, __m512i, _mm512_sllv_epi64, _mm512_mask_sllv_epi64)
__GENERATE_SHIFT(shift_right_var, $i64, __m512i, __m512i, _mm512_srlv_epi64, _mm512_mask_srlv_epi64)
__GENERATE_SHIFT(shift_right_var, $u64, __m512i, __m512i, _mm512_srlv_epi64, _mm512_mask_srlv_epi64)
#undef __GENERATE_SHIFT


// --- Bitwise operators

#define __GENERATE_BITWISE(Op, Tp, Tv, IntrinFn, IntrinFnMask) \
template<>                                             \
struct Op<Tp, Tv> : vector_fn<Tp, Tv> {                \
  using fn = vector_fn<Tp, Tv>;                        \
  inline typename fn::vector_type                      \
  operator()(const typename fn::vector_type& lhs,      \
             const typename fn::vector_type& rhs) const noexcept { \
    return IntrinFn(lhs, rhs);                         \
  }                                                    \
  inline typename fn::vector_type                      \
  operator()(const typename fn::vector_type& lhs,      \
             const typename fn::vector_type& rhs,      \
             const typename fn::vector_type& src,      \
             const mask16 mask) const noexcept {       \
    return IntrinFnMask(src, mask.data, lhs, rhs);     \
  }                                                    \
};

// TODO find a better way to implement bitwise operations
__GENERATE_BITWISE(bit_and, $i32, __m512i, _mm512_and_epi32, _mm512_mask_and_epi32)
__GENERATE_BITWISE(bit_and, $u32, __m512i, _mm512_and_epi32, _mm512_mask_and_epi32)
__GENERATE_BITWISE(bit_or,  $i32, __m512i, _mm512_or_epi32,  _mm512_mask_or_epi32)
__GENERATE_BITWISE(bit_or,  $u32, __m512i, _mm512_or_epi32,  _mm512_mask_or_epi32)
__GENERATE_BITWISE(bit_xor, $i32, __m512i, _mm512_xor_epi32, _mm512_mask_xor_epi32)
__GENERATE_BITWISE(bit_xor, $u32, __m512i, _mm512_xor_epi32, _mm512_mask_xor_epi32)

__GENERATE_BITWISE(bit_and, $i64, __m512i, _mm512_and_epi64, _mm512_mask_and_epi64)
__GENERATE_BITWISE(bit_and, $u64, __m512i, _mm512_and_epi64, _mm512_mask_and_epi64)
__GENERATE_BITWISE(bit_or,  $i64, __m512i, _mm512_or_epi64,  _mm512_mask_or_epi64)
__GENERATE_BITWISE(bit_or,  $u64, __m512i, _mm512_or_epi64,  _mm512_mask_or_epi64)
__GENERATE_BITWISE(bit_xor, $i64, __m512i, _mm512_xor_epi64, _mm512_mask_xor_epi64)
__GENERATE_BITWISE(bit_xor, $u64, __m512i, _mm512_xor_epi64, _mm512_mask_xor_epi64)
#undef __GENERATE_BITWISE


// --- Comparison

#define __GENERATE_CMP(Op, Tp, Tv, IntrinFn, IntrinFnMask) \
template<>                                             \
struct Op<Tp, Tv, Tv, mask16> : vector_fn<Tp, Tv, Tv, mask16> { \
  using fn = vector_fn<Tp, Tv, Tv, mask16>;            \
  inline mask16                                        \
  operator()(const typename fn::vector_type& lhs,      \
             const typename fn::vector_type& rhs) const noexcept { \
    return mask16{ IntrinFn(lhs, rhs) };               \
  }                                                    \
  inline mask16                                        \
  operator()(const typename fn::vector_type& lhs,      \
             const typename fn::vector_type& rhs,      \
             const mask16 mask) const noexcept {       \
    return mask16{ IntrinFnMask(mask.data, lhs, rhs) }; \
  }                                                    \
};

#define __GENERATE_NE(T) \
inline __mmask16 _mm512_cmpne_##T##_mask(__m512i a, __m512i b) { \
  const __mmask16 eq = _mm512_cmpeq_##T##_mask(a, b);            \
  return _mm512_knot(eq);                                        \
}                                                                \
inline __mmask16 _mm512_mask_cmpne_##T##_mask(__mmask16 k1, __m512i a, __m512i b) { \
  const __mmask16 eq = _mm512_mask_cmpeq_##T##_mask(k1, a, b);   \
  return _mm512_kand(k1, _mm512_knot(eq));                       \
}

__GENERATE_NE(epi32)
__GENERATE_NE(epu32)
__GENERATE_NE(epi64)
__GENERATE_NE(epu64)
#undef __GENERATE_NE

__GENERATE_CMP(less,          $i32, __m512i, _mm512_cmplt_epi32_mask, _mm512_mask_cmplt_epi32_mask)
__GENERATE_CMP(less_equal,    $i32, __m512i, _mm512_cmple_epi32_mask, _mm512_mask_cmple_epi32_mask)
__GENERATE_CMP(greater,       $i32, __m512i, _mm512_cmpgt_epi32_mask, _mm512_mask_cmpgt_epi32_mask)
__GENERATE_CMP(greater_equal, $i32, __m512i, _mm512_cmpge_epi32_mask, _mm512_mask_cmpge_epi32_mask)
__GENERATE_CMP(equal,         $i32, __m512i, _mm512_cmpeq_epi32_mask, _mm512_mask_cmpeq_epi32_mask)
__GENERATE_CMP(not_equal,     $i32, __m512i, _mm512_cmpne_epi32_mask, _mm512_mask_cmpne_epi32_mask)

__GENERATE_CMP(less,          $u32, __m512i, _mm512_cmplt_epu32_mask, _mm512_mask_cmplt_epu32_mask)
__GENERATE_CMP(less_equal,    $u32, __m512i, _mm512_cmple_epu32_mask, _mm512_mask_cmple_epu32_mask)
__GENERATE_CMP(greater,       $u32, __m512i, _mm512_cmpgt_epu32_mask, _mm512_mask_cmpgt_epu32_mask)
__GENERATE_CMP(greater_equal, $u32, __m512i, _mm512_cmpge_epu32_mask, _mm512_mask_cmpge_epu32_mask)
__GENERATE_CMP(equal,         $u32, __m512i, _mm512_cmpeq_epu32_mask, _mm512_mask_cmpeq_epu32_mask)
__GENERATE_CMP(not_equal,     $u32, __m512i, _mm512_cmpne_epu32_mask, _mm512_mask_cmpne_epu32_mask)

__GENERATE_CMP(less,          $i64, __m512i, _mm512_cmplt_epi64_mask, _mm512_mask_cmplt_epi64_mask)
__GENERATE_CMP(less_equal,    $i64, __m512i, _mm512_cmple_epi64_mask, _mm512_mask_cmple_epi64_mask)
__GENERATE_CMP(greater,       $i64, __m512i, _mm512_cmpgt_epi64_mask, _mm512_mask_cmpgt_epi64_mask)
__GENERATE_CMP(greater_equal, $i64, __m512i, _mm512_cmpge_epi64_mask, _mm512_mask_cmpge_epi64_mask)
__GENERATE_CMP(equal,         $i64, __m512i, _mm512_cmpeq_epi64_mask, _mm512_mask_cmpeq_epi64_mask)
__GENERATE_CMP(not_equal,     $i64, __m512i, _mm512_cmpne_epi64_mask, _mm512_mask_cmpne_epi64_mask)

__GENERATE_CMP(less,          $u64, __m512i, _mm512_cmplt_epu64_mask, _mm512_mask_cmplt_epu64_mask)
__GENERATE_CMP(less_equal,    $u64, __m512i, _mm512_cmple_epu64_mask, _mm512_mask_cmple_epu64_mask)
__GENERATE_CMP(greater,       $u64, __m512i, _mm512_cmpgt_epu64_mask, _mm512_mask_cmpgt_epu64_mask)
__GENERATE_CMP(greater_equal, $u64, __m512i, _mm512_cmpge_epu64_mask, _mm512_mask_cmpge_epu64_mask)
__GENERATE_CMP(equal,         $u64, __m512i, _mm512_cmpeq_epu64_mask, _mm512_mask_cmpeq_epu64_mask)
__GENERATE_CMP(not_equal,     $u64, __m512i, _mm512_cmpne_epu64_mask, _mm512_mask_cmpne_epu64_mask)

#undef __GENERATE_CMP


} // namespace simd
} // namespace dtl