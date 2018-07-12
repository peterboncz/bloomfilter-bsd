#include "gtest/gtest.h"

#include <iostream>
#include <fstream>
#include <cstdio>

#include <dtl/dtl.hpp>

#include <dtl/filter/platform.hpp>


using namespace dtl::filter;

//===----------------------------------------------------------------------===//
TEST(platform, read_cache_sizes) {
  auto sizes = platform::get_instance().get_cache_sizes();
  ASSERT_TRUE(sizes.size() > 0);
}
//===----------------------------------------------------------------------===//
