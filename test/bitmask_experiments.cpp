#include "gtest/gtest.h"
#include "../adept.hpp"
#include "../bitmask.hpp"
#include <bitset>
#include <functional>
#include <iostream>
#include <random>

using namespace bitmask;

static std::vector<$u1> bitvector(const std::string bit_string) {
  std::vector<$u1> bv;
  for ($u64 i = 0; i < bit_string.size(); i++) {
    bv.push_back(bit_string[i] != '0');
  }
  return bv;
}

static u64 rnd(u64 min, u64 max) {
  static std::random_device rd;
  static std::mt19937 rng(rd());
  std::uniform_int_distribution<u64> uni(min,max);
  return uni(rng);;
}

template<u64 size>
static std::bitset<size> make_bitmask(u64 match_cnt) {
  std::bitset<size> bitmask;
  for ($u64 i = 0; i < match_cnt; i++) {
    bitmask.set(i);
  }
  for ($u64 i = size - 1; i != 1; i--) {
    u64 j = rnd(0, i);
    u1 t = bitmask[i];
    bitmask[i] = bitmask[j];
    bitmask[j] = t;
  }
  return bitmask;
}

template<u64 size, u64 compressed_size, template<u64> class mask_impl>
static u64 match_cnt_after_compression(const std::bitset<size> bitmask){
  std::bitset<compressed_size> compressed_bitmask = mask_impl<size>::template compress<compressed_size>(bitmask);
  std::bitset<size> decompressed_bitmask = mask_impl<size>::decode(compressed_bitmask);
  return decompressed_bitmask.count();
};

template<u64 size, u64 max_match_cnt, template<u64> class mask_impl>
static void run() {
  u64 repeat_cnt = 10;

  std::cout << "actual_match_cnt|returned_match_cnt_64bit_compressed|returned_match_cnt_128bit_compressed|returned_match_cnt_256bit_compressed|returned_match_cnt_512bit_compressed" << std::endl;
  for ($u64 match_cnt = 0; match_cnt < max_match_cnt; match_cnt++) {
    for($u64 repeat = 0; repeat < repeat_cnt; repeat++) {
      std::bitset<size> bitmask = make_bitmask<size>(match_cnt);
      std::cout << match_cnt;
      std::cout << "|" << match_cnt_after_compression<size, 64, mask_impl>(bitmask);
      std::cout << "|" << match_cnt_after_compression<size, 128, mask_impl>(bitmask);
      std::cout << "|" << match_cnt_after_compression<size, 256, mask_impl>(bitmask);
      std::cout << "|" << match_cnt_after_compression<size, 512, mask_impl>(bitmask);
      std::cout << std::endl;
    }
  }
};

TEST(bitmask_experiment, tree_mask_uniform_match_distribution) {
  run<2048, 42, tree_mask>();
}

TEST(bitmask_experiment, zone_mask_uniform_match_distribution) {
  run<2048, 42, zone_mask>();
}
