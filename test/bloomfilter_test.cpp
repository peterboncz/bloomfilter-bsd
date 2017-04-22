#include "gtest/gtest.h"
#include "../adept.hpp"
#include "../bloomfilter.hpp"
#include "../hash.hpp"
#include "../simd.hpp"

#include <atomic>
#include <chrono>

#include "../thread.hpp"

using namespace dtl;


TEST(bloom, vectorization) {
  using key_t = $u32;
  using key_vt = dtl::vec<key_t>;
  std::cout << "vector length: " << key_vt::length << std::endl;
//  using bf_t = dtl::bloomfilter<$u32, dtl::hash::knuth>;
}
