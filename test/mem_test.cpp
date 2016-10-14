#include "gtest/gtest.h"
#include "../adept.hpp"
#include "../mem.hpp"
#include "../simd.hpp"

#include "immintrin.h"

#include <stdlib.h>

#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <typeinfo>

#include <numa.h>

TEST(mem, numa) {
  $i64 node_free;
  i64 node_size = numa_node_size(1, &node_free);
  std::cout << node_size << " " << node_free << std::endl;
}
