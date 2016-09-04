#include "gtest/gtest.h"
#include "../adept.hpp"
#include "../bitmask.hpp"
#include <bitset>
#include <functional>
#include <iostream>
#include <random>

using namespace tree;

static std::vector<$u1> bitvector(const std::string bit_string) {
  std::vector<$u1> bv;
  for ($u64 i = 0; i < bit_string.size(); i++) {
    bv.push_back(bit_string[i] != '0');
  }
  return bv;
}

TEST(bitmask, tree_encode) {
  ASSERT_EQ(bitvector("11010010"), tree_mask<16>::encode(std::bitset<16>("0000000011111111")));
  ASSERT_EQ(bitvector("11010010"), tree_mask<8>::encode(std::bitset<8>("00001111")));
  ASSERT_EQ(bitvector("11010010"), tree_mask<2>::encode(std::bitset<2>("01")));
  ASSERT_EQ(bitvector("100"), tree_mask<4>::encode(std::bitset<4>("0000")));
  ASSERT_EQ(bitvector("101"), tree_mask<4>::encode(std::bitset<4>("1111")));
  ASSERT_EQ(bitvector("1110100100101"), tree_mask<8>::encode(std::bitset<8>("11110011")));
  ASSERT_EQ(bitvector("1110100100101"), tree_mask<4>::encode(std::bitset<4>("1101")));
  ASSERT_EQ(bitvector("111010011010001010"), tree_mask<4>::encode(std::bitset<4>("0101")));
  ASSERT_EQ(bitvector("111010011010001010"), tree_mask<32>::encode(std::bitset<32>("00000000111111110000000011111111")));
  //ASSERT_EQ(bitvector(""), tree::tree_mask<>::encode(std::bitset<>("")));
}

TEST(bitmask, tree_decode) {
  ASSERT_EQ(std::bitset<16>("0000000011111111"), tree_mask<16>::decode(bitvector("11010010")));
  ASSERT_EQ(std::bitset<8>("00001111"), tree_mask<8>::decode(bitvector("11010010")));
  ASSERT_EQ(std::bitset<2>("01"), tree_mask<2>::decode(bitvector("11010010")));
  ASSERT_EQ(std::bitset<4>("0000"), tree_mask<4>::decode(bitvector("100")));
  ASSERT_EQ(std::bitset<4>("1111"), tree_mask<4>::decode(bitvector("101")));
  ASSERT_EQ(std::bitset<8>("11110011"), tree_mask<8>::decode(bitvector("1110100100101")));
  ASSERT_EQ(std::bitset<4>("1101"), tree_mask<4>::decode(bitvector("1110100100101")));
  ASSERT_EQ(std::bitset<4>("0101"), tree_mask<4>::decode(bitvector("111010011010001010")));
  ASSERT_EQ(std::bitset<32>("00000000111111110000000011111111"), tree_mask<32>::decode(bitvector("111010011010001010")));
}

template<u64 N>
static void test_enc_dec(std::bitset<N> bitmask) {
  ASSERT_EQ(bitmask, tree_mask<N>::decode(tree_mask<N>::encode(bitmask)));
};

TEST(bitmask, tree_encode_decode) {
  test_enc_dec(std::bitset<16>("1111111111111111"));
  test_enc_dec(std::bitset<16>("0000000011111111"));
  test_enc_dec(std::bitset<16>("1111111100000000"));
  test_enc_dec(std::bitset<16>("1111000011110000"));
  test_enc_dec(std::bitset<16>("0000111100001111"));
  test_enc_dec(std::bitset<16>("0011001100110011"));
  test_enc_dec(std::bitset<16>("1100110011001100"));
  test_enc_dec(std::bitset<16>("1010101010101010"));
  test_enc_dec(std::bitset<16>("0101010101010101"));
  test_enc_dec(std::bitset<16>("0000000000000000"));
  test_enc_dec(std::bitset<4>("1111"));
  test_enc_dec(std::bitset<4>("1100"));
  test_enc_dec(std::bitset<4>("0011"));
  test_enc_dec(std::bitset<4>("0101"));
  test_enc_dec(std::bitset<4>("1010"));
  test_enc_dec(std::bitset<4>("0000"));
  test_enc_dec(std::bitset<1>("1"));
  test_enc_dec(std::bitset<1>("0"));
}

template<u64 LEN>
static void assert_no_false_negatives(std::bitset<LEN> exact_bitmask, std::bitset<LEN> other_bitmask) {
  ASSERT_EQ(exact_bitmask, exact_bitmask & other_bitmask);
}

/// Typical blocks contain 2^17 values of type 1, 2 or 4 byte integers.
/// As we need at most cache-line granularity, bitmasks are of size 2^11, 2^12 or 2^13.
constexpr u64 N = 1 << 11; // 12 13;

template<u64 M>
static void test_compression(const std::bitset<N> bitmask) {
  assert_no_false_negatives(bitmask, tree_mask<N>::decode<M>(tree_mask<N>::compress<M>(bitmask)));
  assert_no_false_negatives(bitmask, zone_mask<N>::decode<M>(zone_mask<N>::compress<M>(bitmask)));
};

TEST(bitmask, compression) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<u64> dis(1, 100);

  std::bitset<N> bitmask;
  for ($u64 i = 0; i < N; i++) {
    if (dis(gen) <= 10) {
      bitmask.set(i);
    }
  }
  test_compression<1024>(bitmask);
  test_compression<512>(bitmask);
  test_compression<256>(bitmask);
  test_compression<128>(bitmask);
  test_compression<64>(bitmask);
}