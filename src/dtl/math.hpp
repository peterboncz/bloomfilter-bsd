#pragma once

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>

#include <array>
#include <bitset>
#include <type_traits>

namespace dtl {


constexpr bool is_power_of_two(size_t x) {
  return x == 1 ? true : (x && ((x & (x - 1)) == 0));
}

constexpr u64 next_power_of_two(u64 value) {
  return 1ull << ((sizeof(u64) << 3) - __builtin_clzll(value - 1));
}

constexpr u64 prev_power_of_two(u64 value) {
  return next_power_of_two(value) >> 1;
}

struct trunc {
  static size_t byte(uint64_t max) {
    if (!max) return 0;
    const uint64_t r = (8 - (__builtin_clzll(max) >> 3));
    return next_power_of_two(r);
  }
  static size_t bit(uint64_t max) {
    return max ? (64 - (__builtin_clzll(max))) : 0;
  }
};

__forceinline__ __host__ __device__
constexpr u32
log_2(const u32 n) {
  return 8 * sizeof(u32) - dtl::bits::lz_count(n) - 1;
};

__forceinline__ __host__ __device__
constexpr u64
log_2(const u64 n) {
  return 8 * sizeof(u64) - dtl::bits::lz_count(n) - 1;
};


/// Compile-time template expansions
namespace ct {

  /// Computes N! at compile time.
  template<size_t N>
  struct factorial {
    enum : size_t {
      value = N * factorial<N - 1>::value
    };
  };

  template<>
  struct factorial<0> {
    enum : size_t {
      value = 1
    };
  };


  template<size_t N, size_t K>
  struct n_choose_k {
    static const size_t n = N;
    static const size_t k = (2 * K > N) ? N - K : K;
    enum : size_t {
      value = n_choose_k<N - 1, K - 1>::value + n_choose_k<N - 1, K>::value
    };
  };

  template<size_t N>
  struct n_choose_k<N, 0> {
    enum : size_t {
      value = 1
    };
  };

  template<size_t K>
  struct n_choose_k<0, K> {
    enum : size_t {
      value = 0
    };
  };

  template<size_t N>
  struct n_choose_k<N, N> {
    enum : size_t {
      value = 1
    };
  };


  template<size_t N>
  struct catalan_number {
    enum : size_t {
      value = n_choose_k<2 * N, N>::value / (N + 1)
    };
  };

  template<size_t i, size_t j>
  struct ballot_number {
    static const size_t n = i + 1;
    static const size_t k = (i + j) / 2 + 1;
    enum : size_t {
      value = static_cast<size_t>(((static_cast<double>(j) + 1) / (static_cast<double>(i) + 1)) * n_choose_k<n, k>::value)
    };
  };

  /// Computes the number of paths from (i,j) to (2n,0)
  template<size_t n, size_t i, size_t j>
  struct number_of_paths {
    enum : size_t {
      value = ballot_number<2 * n - i, j>::value
    };
  };

  template<u64 n>
  struct lz_count_u32 {
    static constexpr u64 value = __builtin_clz(n);
  };

  template<u64 n>
  struct lz_count_u64 {
    static constexpr u64 value = __builtin_clzll(n);
  };

  template<size_t n>
  struct log_2{
    enum : size_t {
      value = 8 * sizeof(size_t) - lz_count_u64<n>::value - 1
    };
  };

  template<u32 n>
  struct log_2_u32{
    enum : u32 {
      value = n == 0 ? 0 : 8 * sizeof(u32) - lz_count_u32<n>::value - 1
    };
  };

  template<u64 n>
  struct log_2_u64{
    enum : u64 {
      value = n == 0 ? 0 : 8 * sizeof(u64) - lz_count_u64<n>::value - 1
    };
  };


  template<u64 n>
  struct pop_count {
    static constexpr u64 value = (n & 1) + pop_count<(n >> 1)>::value;
  };

  template<>
  struct pop_count<0ull> {
    static constexpr u64 value = 0;
  };

}


} // namespace dtl
