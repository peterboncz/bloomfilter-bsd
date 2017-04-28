#include "gtest/gtest.h"

#include "../adept.hpp"
#include "../bloomfilter.hpp"
#include "../bloomfilter_vec.hpp"
#include "../hash.hpp"
#include "../simd.hpp"
#include "../env.hpp"

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

// --- compiletime settings ---

using bf_t = dtl::bloomfilter<$u32, dtl::hash::knuth, $u32>;
using bf_vt = dtl::bloomfilter_vec<$u32, dtl::hash::knuth, $u32>;

static const u64 vec_unroll_factor = 4;

// --- runtime settings ---

// the grain size for parallel experiments
static u64 preferred_grain_size = 1ull << dtl::env<$i32>::get("GRAIN_SIZE", 16);

// set the bloomfilter size: m in [2^lo, 2^hi]
static i32 bf_size_lo_exp = dtl::env<$i32>::get("BF_SIZE_LO", 11);
static i32 bf_size_hi_exp = dtl::env<$i32>::get("BF_SIZE_HI", 31);

// repeats the benchmark with different concurrency settings
static i32 thread_cnt_lo = dtl::env<$i32>::get("THREAD_CNT_LO", 1);
static i32 thread_cnt_hi = dtl::env<$i32>::get("THREAD_CNT_HI", std::thread::hardware_concurrency());

// 1 = linear, 2 = exponential
static i32 thread_step_mode = dtl::env<$i32>::get("THREAD_STEP_MODE", 1);
static i32 thread_step = dtl::env<$i32>::get("THREAD_STEP", 1);

static auto inc_thread_cnt = [&](u64 i) {
  if (thread_step_mode == 1) {
    // linear
    return i + thread_step;
  }
  else {
    // exponential
    auto step = thread_step > 1 ? thread_step : 2;
    return i * step;
  }
};

static u64 key_cnt_per_thread = 1ull << dtl::env<$i32>::get("KEY_CNT", 24);

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
  u64 bf_size_lo = 1ull << bf_size_lo_exp;
  u64 bf_size_hi = 1ull << bf_size_hi_exp;
  for ($u64 bf_size = bf_size_lo; bf_size <= bf_size_hi; bf_size <<= 2) {
    std::cout << "size " << (bf_size / 8) << " [bytes]" << std::endl;
    run_filter_benchmark(bf_size);
  }
}


void run_filter_benchmark_in_parallel(u64 bf_size, u64 thread_cnt) {
  dtl::thread_affinitize(std::thread::hardware_concurrency() - 1);
  u64 key_cnt = key_cnt_per_thread * thread_cnt;
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
  aligned_vector<$u32> keys;
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
  u64 bf_size_lo = 1ull << bf_size_lo_exp;
  u64 bf_size_hi = 1ull << bf_size_hi_exp;

  for ($u64 bf_size = bf_size_lo; bf_size <= bf_size_hi; bf_size <<= 2) {
    for ($u64 t = thread_cnt_lo; t <= thread_cnt_hi; t = inc_thread_cnt(t)) {
      run_filter_benchmark_in_parallel(bf_size, t);
    }
  }
}


void run_filter_benchmark_in_parallel_vec(u64 bf_size, u64 thread_cnt) {
  dtl::thread_affinitize(std::thread::hardware_concurrency() - 1);
  u64 key_cnt = key_cnt_per_thread * thread_cnt;
  bf_t bf(bf_size);
  bf_vt bf_vec { bf };
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
  using key_t = $u32;
  aligned_vector<key_t> keys;
  keys.resize(key_cnt);
  for ($u64 i = 0; i < key_cnt; i++) {
    keys[i] = dtl::hash::crc32<u32, 7331>::hash(i);
  }

  // size of a work item (dispatched to a thread)
  u64 grain_size = std::min(preferred_grain_size, key_cnt);

  std::vector<$u64> matches_found;
  matches_found.resize(thread_cnt, 0);
  std::atomic<$u64> grain_cntr(0);

  auto worker_fn = [&](u32 thread_id) {
    u64 vlen = dtl::simd::lane_count<key_t> * vec_unroll_factor;
    using key_vt = dtl::vec<key_t, vlen>;

//    dtl::vec<$u64, vlen> found_vec = 0;
    key_vt found_vec = 0;
    while (true) {
      u64 read_from = grain_cntr.fetch_add(grain_size);
      u64 read_to = std::min(key_cnt, read_from + grain_size);
      if (read_from >= key_cnt) break;
      for ($u64 i = read_from; i < read_to; i += vlen) {
        const key_vt* k = reinterpret_cast<const key_vt*>(&keys[i]);
        auto mask = bf_vec.contains<vlen>(*k);
        found_vec[mask] += 1;
//        found += bf.contains();
      }
    }
    $u64 found = 0;
    for (std::size_t i = 0; i < vlen; i++) {
      found += found_vec[i];
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


TEST(bloom, filter_performance_parallel_vec) {
  u64 bf_size_lo = 1ull << bf_size_lo_exp;
  u64 bf_size_hi = 1ull << bf_size_hi_exp;

  for ($u64 bf_size = bf_size_lo; bf_size <= bf_size_hi; bf_size <<= 2) {
    for ($u64 t = thread_cnt_lo; t <= thread_cnt_hi; t = inc_thread_cnt(t)) {
      run_filter_benchmark_in_parallel_vec(bf_size, t);
    }
  }
}
