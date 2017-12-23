#pragma once

#include <functional>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/math.hpp>
#include <dtl/mem.hpp>
#include <dtl/simd.hpp>


namespace dtl {
namespace internal {
// FIXME: works for AVX2 only

//===----------------------------------------------------------------------===//
// TODO should be part of dtl::vec
template<typename Tsrc, typename Tdst, u64 n> // the vector length
struct vector_convert {};


template<u64 n> // the vector length
struct vector_convert<uint32_t, uint32_t, n> {
  __forceinline__
  static vec<uint32_t, n>
  convert(const vec<uint32_t, n>& src) noexcept {
    return src;
  }
};


template<u64 n> // the vector length
struct vector_convert<uint32_t, uint64_t, n> {
  __forceinline__
  static vec<uint64_t, n>
  convert(const vec<uint32_t, n>& src) noexcept {
    vec<uint64_t, n> dst;
    const auto s = reinterpret_cast<const __m128i*>(&src.data);
    auto d = reinterpret_cast<__m256i*>(&dst.data);
    for (std::size_t i = 0; i < dst.nested_vector_cnt; i++) {
      d[i] = _mm256_cvtepu32_epi64(s[i]);
    }
    return dst;
  }
};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// TODO should be part of dtl::vec
template<
    typename Tp, // the primitive type to load
    typename Ti, // the index type
    u64 n        // the vector length
>
struct vector_gather {};

template<u64 n> // the vector length
struct vector_gather<uint32_t, uint32_t, n> {
  __forceinline__ static
      vec<uint32_t, n>
  gather(const u32* const base_addr,
         const vec<uint32_t, n>& idxs) noexcept {
    vec<uint32_t, n> result;
    const auto i = reinterpret_cast<const __m256i*>(&idxs.data);
    auto r = reinterpret_cast<__m256i*>(&result.data);
    const auto b = reinterpret_cast<const int *>(base_addr);
    for (std::size_t j = 0; j < idxs.nested_vector_cnt; j++) {
      r[j] = _mm256_i32gather_epi32(b, i[j], 4);
    }
    return result;
  }

};

template<u64 n> // the vector length
struct vector_gather<uint64_t, uint32_t, n> {
  __forceinline__ static
      vec<uint64_t, n>
  gather(const u64* const base_addr,
         const vec<uint32_t, n>& idxs) noexcept {
    vec<uint64_t, n> result;
    const auto i = reinterpret_cast<const __m128i*>(&idxs.data);
    auto r = reinterpret_cast<__m256i*>(&result.data);
    const auto b = reinterpret_cast<const long long int *>(base_addr);
    for (std::size_t j = 0; j < result.nested_vector_cnt; j++) {
      r[j] = _mm256_i32gather_epi64(b, i[j], 8);
    }
    return result;
  }

};


template<u64 n> // the vector length
struct vector_gather<uint32_t, uint64_t, n> {
  __forceinline__ static
      vec<uint32_t, n>
  gather(const vec<uint64_t, n>& idxs) noexcept {
    vec<uint32_t, n> result;
    const auto i = reinterpret_cast<const __m256i*>(&idxs.data);
    auto r = reinterpret_cast<__m128i*>(&result.data);
    for (std::size_t j = 0; j < idxs.nested_vector_cnt; j++) {
      r[j] = _mm256_i64gather_epi32(0, i[j], 1);
    }
    return result;
  }

};


template<u64 n> // the vector length
struct vector_gather<uint64_t, uint64_t, n> {
  __forceinline__ static
      vec<uint64_t, n>
  gather(const vec<uint64_t, n>& addrs) noexcept {
    vec<uint64_t, n> result;
    const auto a = reinterpret_cast<const __m256i*>(&addrs.data);
    auto r = reinterpret_cast<__m256i*>(&result.data);
    for (std::size_t i = 0; i < result.nested_vector_cnt; i++) {
      r[i] = _mm256_i64gather_epi64(0, a[i], 1);
    }
    return result;
  }

};
//===----------------------------------------------------------------------===//


} // namespace internal
} // namespace dtl