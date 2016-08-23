#include "gtest/gtest.h"

#include <bitset>
#include "../bitmask.hpp"


TEST(bitmask, tree_compress) {
  std::bitset<8> bitmask;//("01100011");
  bitmask.set(1);
  bitmask.set(2);
  bitmask.set(6);
  bitmask.set(7);
  std::bitset<64> compressed_bitmask = compress<8, 64>(bitmask);
}
