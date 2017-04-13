#include "gtest/gtest.h"
#include "../adept.hpp"
#include "../bloomfilter.hpp"
#include "../hash.hpp"
#include "../simd.hpp"

#include <atomic>
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
        bf.insert(dtl::hash::crc32<u32>::hash(i));
      }
    });
    u64 perf = (repeat_cnt) / (duration / nano_to_sec);
    std::cout << perf << " [inserts/sec]" << std::endl;
  }
  {
    $u64 found = 0;
    f64 duration = timing([&] {
      for ($u64 i = 0; i < repeat_cnt; i++) {
        found += bf.contains(dtl::hash::crc32<u32, 7331>::hash(i));
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


void run_filter_benchmark_in_parallel(u64 bf_size, u64 thread_cnt) {
  dtl::thread_affinitize(std::thread::hardware_concurrency() - 1);
  u64 key_cnt = (1ull << 28) * thread_cnt;
  bf_t bf(bf_size);
  {
    f64 duration = timing([&] {
      for ($u64 i = 0; i < bf_size >> 4; i++) {
        bf.insert(dtl::hash::crc32<u32>::hash(i));
      }
    });
    u64 perf = (key_cnt) / (duration / nano_to_sec);
//    std::cout << perf << " [inserts/sec]" << std::endl;
  }

  // prepare the input
  std::vector<$u32> keys;
  keys.resize(key_cnt);
  for ($u64 i = 0; i < key_cnt; i++) {
    keys[i] = dtl::hash::crc32<u32, 7331>::hash(i);
  }

  // size of a work item (dispatched to a thread)
  u64 preferred_grain_size = 1ull << 16;
  u64 grain_size = std::min(preferred_grain_size, key_cnt);

  std::vector<$u64> matches_found;
  matches_found.resize(thread_cnt, 0);
  std::atomic<$u64> grain_cntr(0);

  auto worker_fn = [&](u32 thread_id) {
    $u64 found = 0;
    while (true) {
      u64 read_from = grain_cntr.fetch_add(grain_size);
      u64 read_to = std::min(key_cnt, read_from + grain_size);
      if (read_from >= key_cnt) break;
      for ($u64 i = read_from; i < read_to; i++) {
        found += bf.contains(keys[i]);
      }
    }
    matches_found[thread_id] = found;
  };


  f64 duration = timing([&] {
    dtl::run_in_parallel(worker_fn, thread_cnt);
  });

  $u64 found = 0;
  for ($u64 i = 0; i < thread_cnt; i++) {
    found += matches_found[i];
  }
  u64 perf = (key_cnt) / (duration / nano_to_sec);
  std::cout << "bf_size: " << (bf_size / 8) << " [bytes], "
            << "thread_cnt: " << thread_cnt << ", "
            << "key_cnt: " << key_cnt << ", "
            << "grain_size: " << grain_size << ", "
            << "performance: " << perf << " [1/s]  (matchcnt: " << found << ")" << std::endl;
}


TEST(bloom, filter_performance_parallel) {
//  u64 bf_size_lo = 1ull << 11;
//  u64 bf_size_hi = 1ull << 31;
  u64 bf_size_lo = 1ull << 30;
  u64 bf_size_hi = 1ull << 30;

  for ($u64 bf_size = bf_size_lo; bf_size <= bf_size_hi; bf_size <<= 2) {
//    std::cout << "size " << (bf_size / 8) << " [bytes]" << std::endl;
    for ($u64 t = 1; t <= std::thread::hardware_concurrency(); t++) {
      run_filter_benchmark_in_parallel(bf_size, t);
    }
  }
}
