#include "gtest/gtest.h"

#include <dtl/dtl.hpp>
#include <dtl/bloomfilter.hpp>
#include <dtl/simd.hpp>

using namespace dtl;

TEST(bloom, vectorization) {
  using key_t = $u32;
  using key_vt = dtl::vec<key_t>;
  std::cout << "vector length: " << key_vt::length << std::endl;
//  using bf_t = dtl::bloomfilter<$u32, dtl::hash::knuth>;
}
