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

  inline u32
  operator()() {
    x32 ^= x32 << 13;
    x32 ^= x32 >> 17;
    x32 ^= x32 << 5;
    return x32;
  }

  template<typename T>
  static inline void
  next(T& x32) {
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


template<typename hash_fn_t>
void run_hash_benchmark(hash_fn_t hash_fn,
                        const std::size_t input_size = 1ull << 10 /* prevent auto-vectorization*/ ) {
  // prepare input
  std::vector<$u32> input;
  xorshift32 prng;
  for (std::size_t i = 0; i < input_size; i++) {
    input.push_back(prng());
  }

  // run benchmark
  u64 repeat_cnt = 100000;
  $u64 chksum = 0;
  double duration = timing([&] {
    for ($u64 r = 0; r != repeat_cnt; r++) {
      for ($u64 i = 0; i != input_size; i++) {
        chksum += hash_fn.hash(input[i]);
      }
    }
  });
  u64 perf = (input_size * repeat_cnt) / (duration / 1000.0);
  std::cout << perf << " [hashes/sec]    (chksum: " << chksum << ")" << std::endl;
}


template<typename hash_fn_t>
void run_hash_benchmark_autovec(hash_fn_t hash_fn) {
  // prepare input
  const std::size_t input_size = 1ull << 10;
  std::vector<$u32> input;
  xorshift32 prng;
  for (std::size_t i = 0; i < input_size; i++) {
    input.push_back(prng());
  }

  // run benchmark
  u64 repeat_cnt = 100000;
  $u64 chksum = 0;
  u64 duration = timing([&] {
    for ($u64 r = 0; r != repeat_cnt; r++) {
      for ($u64 i = 0; i != input_size; i++) {
        chksum += hash_fn.hash(input[i]);
      }
    }
  });
  u64 perf = (input_size * repeat_cnt) / (duration / 1000.0);
  std::cout << perf << " [hashes/sec]    (chksum: " << chksum << ")" << std::endl;
}


TEST(bloom, hash_performance) {
  std::cout.setf(std::ios::fixed);
  std::cout << "xorshift   " << std::endl;
  dtl::hash::xorshift_64<u32> xorshift_64;
  run_hash_benchmark(xorshift_64);
  run_hash_benchmark_autovec(xorshift_64);
  std::cout << "murmur1_32 " << std::endl;
  dtl::hash::murmur1_32<u32> murmur1_32;
  run_hash_benchmark(murmur1_32);
  run_hash_benchmark_autovec(murmur1_32);
  std::cout << "crc32      " << std::endl;
  dtl::hash::crc32<u32> crc32;
  run_hash_benchmark(crc32);
  run_hash_benchmark_autovec(crc32);
  std::cout << "knuth      " << std::endl;
  dtl::hash::knuth<u32> knuth;
  run_hash_benchmark(knuth);
  run_hash_benchmark_autovec(knuth);
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
