#include "gtest/gtest.h"

#include <iostream>
#include <fstream>
#include <cstdio>

#include <dtl/dtl.hpp>

#include "benchmark.hpp"
#include "calibration_data.hpp"
#include "timing.hpp"

using namespace dtl::filter::model;

//===----------------------------------------------------------------------===//
TEST(filter_benchmark, basic_test) {
  benchmark benchmark;
  dtl::blocked_bloomfilter_config bbf_config;
  bbf_config.k = 4;
  timing bbf_timing = benchmark(bbf_config, 8ull * 1024 * 8);
  std::cout << "bbf: cycles=" << bbf_timing.cycles_per_lookup << ", nanos=" << bbf_timing.nanos_per_lookup << std::endl;
  dtl::cuckoofilter::config cf_config;
  timing cf_timing = benchmark(cf_config, 8ull * 1024 * 8);
  std::cout << "cf:  cycles=" << cf_timing.cycles_per_lookup << ", nanos=" << cf_timing.nanos_per_lookup << std::endl;

}
//===----------------------------------------------------------------------===//

