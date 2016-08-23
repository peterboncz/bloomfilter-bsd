#pragma once

namespace simd {

  struct bitwidth {
    enum : uint32_t {
#ifdef __AVX512F__
      value = 512
#elif __AVX2__
      value = 256
#elif __SSE2__
      value = 128
#else
      value = 64
#endif
    };
  };

  template<typename T>
  struct lane {
    enum : uint32_t {
      count = bitwidth::value / (sizeof(T) * 8)
    };
  };

}

template<typename L, typename R>
struct super {
  // should work for now
  using type = typename std::conditional<sizeof(L) < sizeof(R), R, L>::type;
};
