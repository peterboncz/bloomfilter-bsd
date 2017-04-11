#include "gtest/gtest.h"
#include "../adept.hpp"
#include "../bloomfilter.hpp"
#include "../hash.hpp"
#include "../simd.hpp"
#include <chrono>

#include "../thread.hpp"

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
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

static constexpr std::chrono::seconds sec(1);
static constexpr double nano_to_sec = std::chrono::duration_cast<std::chrono::nanoseconds>(sec).count();

TEST(bloom, prng_performance) {
  u64 repeat_cnt = 1u << 20;
  xorshift32 prng;
  f64 duration = timing([&] {
    for ($u64 i = 0; i < repeat_cnt; i++) {
      prng();
    }
  });
  u64 perf = (repeat_cnt) / (duration / nano_to_sec);
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
  f64 duration = timing([&] {
    for ($u64 r = 0; r != repeat_cnt; r++) {
      for ($u64 i = 0; i != input_size; i++) {
        chksum += hash_fn.hash(input[i]);
      }
    }
  });
  u64 perf = (input_size * repeat_cnt) / (duration / nano_to_sec);
  std::cout << "scalar:    " << perf << " [hashes/sec]    (chksum: " << chksum << ")" << std::endl;
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
  u64 perf = (input_size * repeat_cnt) / (duration / nano_to_sec);
  std::cout << "auto-vec.: " << perf << " [hashes/sec]    (chksum: " << chksum << ")" << std::endl;
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


using bf_t = dtl::bloomfilter<$u32, dtl::hash::knuth>;

void run_filter_benchmark(u64 bf_size) {
  dtl::thread_affinitize(0);
  u64 repeat_cnt = 1u << 28;
  bf_t bf(bf_size);
  {
    f64 duration = timing([&] {
      for ($u64 i = 0; i < repeat_cnt; i++) {
        bf.insert(i);
      }
    });
    u64 perf = (repeat_cnt) / (duration / nano_to_sec);
    std::cout << perf << " [inserts/sec]" << std::endl;
  }
  {
    $u64 found = 0;
    f64 duration = timing([&] {
      for ($u64 i = 0; i < repeat_cnt; i++) {
        found += bf.contains(i);
      }
    });
    u64 perf = (repeat_cnt) / (duration / nano_to_sec);
    std::cout << perf << " [probes/sec]    (matchcnt: " << found << ")" << std::endl;
  }

}


TEST(bloom, filter_performance) {
  u64 bf_size_lo = 1ull << 11;
  u64 bf_size_hi = 1ull << 31;
  for ($u64 bf_size = bf_size_lo; bf_size <= bf_size_hi; bf_size <<= 2) {
    std::cout << "size " << (bf_size / 8) << " [bytes]" << std::endl;
    run_filter_benchmark(bf_size);
  }

}
