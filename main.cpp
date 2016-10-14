#include "adept.hpp"
#include "bloomfilter.hpp"
#include "hash.hpp"
#include "mem.hpp"
#include "simd.hpp"

#include <functional>
#include <chrono>
#include <iostream>

struct xorshift32 {
  $i32 x32;
  xorshift32() : x32(314159265) { };
  xorshift32(i32 seed) : x32(seed) { };
  inline i32 operator()() {
    x32 ^= x32 << 13;
    x32 ^= x32 >> 17;
    x32 ^= x32 << 5;
    return x32;
  }
};

inline auto timing(std::function<void()> fn) {
  auto start = std::chrono::high_resolution_clock::now();
  fn();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

using bf_t = dtl::bloomfilter<$i32, dtl::hash::xorshift_64>;

int main() {
  u64 repeat_cnt = 1u << 25;
  bf_t bf(1ull << 32);
  {
    xorshift32 prng;

    double duration = timing([&] {
      for ($u64 i = 0; i < repeat_cnt; i++) {
        prng();
        bf.insert(prng.x32);
      }
    });
    double perf = ((repeat_cnt) / (duration / 1000.0))/1024;
    std::cout << perf << " [inserts/sec]    (" << prng.x32 << ")" << std::endl;
  }
  {
    xorshift32 prng;
    $u64 found = 0;
    double duration = timing([&] {
      for ($u64 i = 0; i < repeat_cnt; i++) {
        prng();
        found += bf.contains(prng.x32);
        prng();
        found += bf.contains(prng.x32);
        prng();
        found += bf.contains(prng.x32);
        prng();
        found += bf.contains(prng.x32);

      }
    });
    double perf = ((repeat_cnt * 4) / (duration / 1000.0))/1024;
    std::cout << perf << " [probes/sec]    (" << found << ")" << std::endl;
  }

  return 0;
}
