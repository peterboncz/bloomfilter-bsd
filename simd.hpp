#pragma once

#include "adept.hpp"

namespace simd {

  struct bitwidth {
    static const u64
#ifdef __AVX512F__
      value = 512;
#elif __AVX2__
      value = 256;
#elif __SSE2__
      value = 128;
#else
      value = 64;
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

}

template<typename L, typename R>
struct super {
  // should work for now
  using type = typename std::conditional<sizeof(L) < sizeof(R), R, L>::type;
};
