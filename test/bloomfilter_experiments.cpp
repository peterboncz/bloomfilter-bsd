#include "gtest/gtest.h"
#include "../adept.hpp"
#include "../bloomfilter.hpp"
#include "../hash.hpp"
#include "../simd.hpp"
#include <chrono>

//#include "benchmark/benchmark_api.h"

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
  u64 repeat_cnt = 1u << 25;
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

/*
static void benchmark_bloomfilter_probe(benchmark::State& state) {
  u64 element_cnt = 1 << (state.range(0) / 8);
  std::unique_ptr<bf_t> bf;

  // setup
  if (state.thread_index == 0) {
    bf = std::make_unique<bf_t>(1ull << state.range(0));
    xorshift32 prng;
    for ($u64 i = 0; i < element_cnt; i++) {
      prng();
      bf->insert(prng.x32);
    }
  }

  $u64 found = 0;
  xorshift32 prng(state.thread_index);
  while (state.KeepRunning()) {
    prng();
    found += bf->contains(prng.x32);
  }
  state.SetItemsProcessed(state.iterations());

  // teardown
  if (state.thread_index == 0) {
  }

}

static void benchmark_bloomfilter_probe_vectorized(benchmark::State& state) {
  u64 vector_len = 8;
  using bf_t = dtl::bloomfilter<$i32, dtl::hash::xorshift_64>;
  u64 element_cnt = 1 << (state.range(0) / 8);
  std::unique_ptr<bf_t> bf;

  // setup
  if (state.thread_index == 0) {
    bf = std::make_unique<bf_t>(1ull << state.range(0));
    xorshift32 prng;
    for ($u64 i = 0; i < element_cnt; i++) {
      prng();
      bf->insert(prng.x32);
    }
  }


//  vec<$u32, vector_len> found;
//  vec<$u32, vector_len> rnd_keys;
  vec<$i32, vector_len> found;
  vec<$i32, vector_len> rnd_keys;
  xorshift32 prng(state.thread_index);
  for ($u64 i = 0; i < vector_len; i++) {
    found.insert(i, 0);
    rnd_keys.insert(i, prng.x32);
    prng();
  }

  while (state.KeepRunning()) {
    xorshift32::next(rnd_keys);

    //found +=
        bf->contains(rnd_keys);
  }
  state.SetItemsProcessed(state.iterations() * vector_len);

  // teardown
  if (state.thread_index == 0) {

  }

}

BENCHMARK(benchmark_bloomfilter_probe)->DenseRange(8, 32)->ThreadPerCpu()->UseRealTime();
BENCHMARK(benchmark_bloomfilter_probe_vectorized)->DenseRange(8, 32)->ThreadPerCpu()->UseRealTime();
BENCHMARK_MAIN();
 */