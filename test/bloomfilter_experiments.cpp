#include "gtest/gtest.h"
#include "../adept.hpp"
#include "../bloomfilter.hpp"
#include "../hash.hpp"
#include "../simd.hpp"
#include <chrono>

using namespace dtl;

struct xorshift32 {
  $u32 x32;
  xorshift32() : x32(314159265) { };
  xorshift32(u32 seed) : x32(seed) { };
  inline u32 operator()() {
    x32 ^= x32 << 13;
    x32 ^= x32 >> 17;
    x32 ^= x32 << 5;
    return x32;
  }

  template<typename T>
  static inline void next(T& x32) {
    x32 ^= x32 << 13;
    x32 ^= x32 >> 17;
    x32 ^= x32 << 5;
  }

};

inline auto timing(std::function<void()> fn) {
  auto start = std::chrono::high_resolution_clock::now();
  fn();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

TEST(bloom, prng_performance) {
  u64 repeat_cnt = 1u << 20;
  xorshift32 prng;
  double duration = timing([&] {
    for ($u64 i = 0; i < repeat_cnt; i++) {
      prng();
    }
  });
  double perf = (repeat_cnt) / (duration / 1000.0);
  std::cout << perf << " [prn/sec]    (" << prng.x32 << ")" << std::endl;
}


TEST(bloom, hash_performance) {
  u64 repeat_cnt = 1u << 28;
  xorshift32 prng;
//  using hash_fn = dtl::hash::xorshift_64<u32>;
//  using hash_fn = dtl::hash::murmur1_32<u32>;
//  using hash_fn = dtl::hash::crc32<u32>;
  using hash_fn = dtl::hash::knuth<u32>;
  $u32 hash = 0;
  double duration = timing([&] {
    for ($u64 i = 0; i != repeat_cnt; i++) {
//      prng();
//      hash ^= hash_fn::hash(prng.x32);
      hash ^= hash_fn::hash(i);
    }
  });
  double perf = (repeat_cnt) / (duration / 1000.0);
  std::cout << perf << " [hashes/sec]    (" << prng.x32 << ")" << std::endl;
}


//using bf_t = dtl::bloomfilter<$u32, dtl::hash::xorshift_64>;
template<typename T>
using murmur = dtl::hash::murmur1_32<T>;
using bf_t = dtl::bloomfilter<$u32, dtl::hash::knuth, murmur>;

TEST(bloom, filter_performance) {
  u64 repeat_cnt = 1u << 28;
  bf_t bf((1ull << 24) -1);
  {
//    xorshift32 prng;

    double duration = timing([&] {
      for ($u64 i = 0; i < repeat_cnt; i++) {
//        prng();
//        bf.insert(prng.x32);
        bf.insert(i);
      }
    });
    double perf = (repeat_cnt) / (duration / 1000.0);
//    std::cout << perf << " [inserts/sec]    (" << prng.x32 << ")" << std::endl;
    std::cout << perf << " [inserts/sec]" << std::endl;
  }
  {
//    xorshift32 prng;
    $u64 found = 0;
    double duration = timing([&] {
      for ($u64 i = 0; i < repeat_cnt; i++) {
//        prng();
//        found += bf.contains(prng.x32);
        found += bf.contains(i);
      }
    });
    double perf = (repeat_cnt) / (duration / 1000.0);
    std::cout << perf << " [probes/sec]    (" << found << ")" << std::endl;
  }

}
