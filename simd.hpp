#pragma once

// used generate compiler errors if native implementations are included directly
#ifndef _DTL_SIMD_INCLUDED
#define _DTL_SIMD_INCLUDED
#endif

#include "adept.hpp"

namespace dtl {
namespace simd {

  struct bitwidth {
    static const u64
#ifdef __AVX512F__
      value = 512;
#elif __AVX2__
      value = 256;
#elif __SSE2__
      value = 256; // TODO reset to 128
#else
      value = 256; // emulated
#endif
  };



  template<typename T>
  static constexpr u64 lane_count = bitwidth::value / (sizeof(T) * 8);

  template<typename T>
  struct lane {
    enum : u64 {
      count = bitwidth::value / (sizeof(T) * 8)
    };
  };

} // namespace simd
} // namespace dtl


#include "simd/vec.hpp"


// populate the dlt namespace
namespace dtl {

template<typename L, typename R>
struct super {
  // should work for now
  using type = typename std::conditional<sizeof(L) < sizeof(R), R, L>::type;
};


template<typename T, u64 N>
using vec = dtl::simd::v<T, N>;

} // namespace dtl