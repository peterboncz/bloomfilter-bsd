#include "gtest/gtest.h"

#include <dtl/dtl.hpp>
#include <dtl/bitpack.hpp>


void run_test(u64 k, u64 N) {
  std::vector<$u64> in;
  for (std::size_t i = 0; i < N; i++) {
    in.push_back(i);
  }

  auto bitpacked = dtl::bitpack_horizontal(k, in);
  auto out = dtl::bitunpack_horizontal<$u64>(bitpacked);

  for (std::size_t i = 0; i < N; i++) {
    ASSERT_EQ(in[i] % (1ull << k), out[i]);
  }
}


TEST(bitpack, horizontal) {
  for ($u64 k = 1; k <= 64; k++) {
    run_test(k, 1ull << 16);
  }
}
