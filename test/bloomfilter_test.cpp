#include "gtest/gtest.h"

#include <bitset>

#include <dtl/dtl.hpp>
#include <dtl/bloomfilter_runtime.hpp>
#include <dtl/bloomfilter.hpp>
#include <dtl/bloomfilter_vec.hpp>
#include <dtl/bloomfilter2.hpp>
#include <dtl/bloomfilter2_vec.hpp>
#include <dtl/hash.hpp>
#include <dtl/mem.hpp>
#include <dtl/simd.hpp>

using namespace dtl;

namespace dtl {
namespace test {

using key_t = $u32;
using word_t = $u32;

template<typename bf_t>
void
print_info(const bf_t& bf) {
  std::cout << "-- bloomfilter parameters --" << std::endl;
  std::cout << "static" << std::endl;
  std::cout << "  k:                    " << bf_t::k << std::endl;
  std::cout << "  word bitlength:       " << bf_t::word_bitlength << std::endl;
  std::cout << "  hash value bitlength: " << bf_t::hash_value_bitlength << std::endl;
  std::cout << "  sectorized:           " << (bf_t::sectorized ? "true" : "false") << std::endl;
  std::cout << "  sector count:         " << bf_t::sector_cnt << std::endl;
  std::cout << "  sector bitlength:     " << bf_t::sector_bitlength << std::endl;
  std::cout << "  hash bits per sector: " << bf_t::sector_bitlength_log2 << std::endl;
  std::cout << "  hash bits per word:   " << (bf_t::k * bf_t::sector_bitlength_log2) << std::endl;
  std::cout << "  hash bits wasted:     " << (bf_t::sectorized ? (bf_t::word_bitlength - (bf_t::sector_bitlength * bf_t::k)) : 0) << std::endl;
  std::cout << "  remaining hash bits:  " << bf_t::remaining_hash_bit_cnt << std::endl;
  std::cout << "  max m:                " << bf_t::max_m << std::endl;
  std::cout << "  max size [MiB]:       " << (bf_t::max_m / 8.0 / 1024.0 / 1024.0 ) << std::endl;
  std::cout << "dynamic" << std::endl;
  std::cout << "  actual m:             " << (bf.length_mask + 1) << std::endl;
  std::cout << "  actual size [MiB]:    " << ((bf.length_mask + 1) / 8.0 / 1024.0 / 1024.0 ) << std::endl;
  std::cout << "  population count:     " << bf.popcnt() << std::endl;
  std::cout << "  load factor:          " << bf.load_factor() << std::endl;
}

TEST(bloomfilter, sectorization_compile_time_asserts) {
  {
    using bf_t = dtl::bloomfilter<key_t, dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 1, true>;
    static_assert(bf_t::sector_cnt == bf_t::k, "Sector count must equal to k.");
  }
  {
    using bf_t = dtl::bloomfilter<key_t, dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 2, true>;
    static_assert(bf_t::sector_cnt == bf_t::k, "Sector count must equal to k.");
  }
  {
    using bf_t = dtl::bloomfilter<key_t, dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 3, true>;
    static_assert(bf_t::sector_cnt >= bf_t::k, "Sector count must be greater or equal to k.");
  }
  {
    using bf_t = dtl::bloomfilter<key_t, dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 4, true>;
    static_assert(bf_t::sector_cnt == bf_t::k, "Sector count must equal to k.");
  }
  {
    using bf_t = dtl::bloomfilter<key_t, dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 5, true>;
    static_assert(bf_t::sector_cnt >= bf_t::k, "Sector count must be greater or equal to k.");
  }
  {
    using bf_t = dtl::bloomfilter<key_t, dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 6, true>;
    static_assert(bf_t::sector_cnt >= bf_t::k, "Sector count must be greater or equal to k.");
  }
  {
    using bf_t = dtl::bloomfilter<key_t, dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 7, true>;
    static_assert(bf_t::sector_cnt >= bf_t::k, "Sector count must be greater or equal to k.");
    static_assert(bf_t::sector_cnt == dtl::next_power_of_two(bf_t::k), "Sector count must equal to next_pow_of_two(k).");
  }
  {
    using bf_t = dtl::bloomfilter<key_t, dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 8, true>;
    static_assert(bf_t::sector_cnt == bf_t::k, "Sector count must equal to k.");
  }

}

TEST(bloomfilter, k1) {
  using bf_t = dtl::bloomfilter<key_t, dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 7, true>;
  bf_t bf(1024);
  print_info(bf);
}

template<typename T>
struct null_hash {
  using Ty = typename std::remove_cv<T>::type;

  static inline Ty
  hash(const Ty& key) {
    return 0;
  }
};

TEST(bloomfilter, k2) {
  using bf_t = dtl::bloomfilter2<key_t,dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, 7, true>;
//  using bf_t = dtl::bloomfilter<key_t,dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 4, false>;
  u32 m = 1024;
  bf_t bf(m);
  print_info(bf);
  std::cout << std::bitset<32>(bf.which_bits(0, 0)) << std::endl;
  std::cout << std::bitset<32>(bf.which_bits(~0, 0)) << std::endl;
  std::cout << std::bitset<32>(bf.which_bits(0, ~0)) << std::endl;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<$u32> dis(0, m - 1);
  for ($u64 i = 0; i < 100; i++) {
    u32 key = dis(gen);
    bf.insert(key);
    ASSERT_TRUE(bf.contains(key));
  }
  std::cout << "popcount: " << bf.popcnt() << std::endl;
  for (word_t word : bf.word_array) {
    std::cout << std::bitset<bf_t::word_bitlength>(word) << std::endl;
  }

}

TEST(bloomfilter, vectorized_probe) {
  u32 k = 5;
  u1 sectorize = false;
  using bf_t = dtl::bloomfilter2<key_t, dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, k, sectorize>;
  using bf_vt = dtl::bloomfilter2_vec<key_t, dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, k, sectorize>;

  u32 key_cnt = 1024;
  u32 m = key_cnt * k * 2;
  bf_t bf(m);

  std::vector<key_t> keys;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<key_t> dis(0, m - 1);
  for ($u64 i = 0; i < key_cnt; i++) {
    const key_t key = dis(gen);
    keys.push_back(key);
    bf.insert(key);
  }

  std::cout << "popcount: " << bf.popcnt() << std::endl;
  for (word_t word : bf.word_array) {
    std::cout << std::bitset<bf_t::word_bitlength>(word) << std::endl;
  }

  std::vector<key_t> match_pos;
  match_pos.resize(keys.size(), -1);

  bf_vt bf_v { bf };
  auto match_cnt = bf_v.batch_contains(&keys[0], key_cnt, &match_pos[0], 0);
  ASSERT_EQ(key_cnt, match_cnt);
}



TEST(bloomfilter, wrapper) {
  for ($u32 i = 1; i <= 8; i++) {
    auto bf_wrapper = dtl::bloomfilter_runtime_t::construct(i, 1024);
    ASSERT_FALSE(bf_wrapper.contains(1337)) << "k = " << i;
    bf_wrapper.insert(1337);
    ASSERT_TRUE(bf_wrapper.contains(1337)) << "k = " << i;
    bf_wrapper.print_info();
    std::cout << std::endl;
    bf_wrapper.destruct();
  }
}

} // namespace test
} // namespace dtl