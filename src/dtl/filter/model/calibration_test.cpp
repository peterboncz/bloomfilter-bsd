#include "gtest/gtest.h"

#include <iostream>
#include <fstream>
#include <cstdio>

#include <dtl/dtl.hpp>

#include "dtl/filter/model/calibration.hpp"
#include "dtl/filter/model/calibration_data.hpp"
#include "dtl/filter/model/timing.hpp"

using namespace dtl::filter::model;

//===----------------------------------------------------------------------===//
TEST(model_calibration, cache_sizes) {
  // create a config file
  const std::string filename = "/tmp/calibration_data_cache_sizes";
  calibration_data cd(filename);
  calibration c(cd);
  c.calibrate_tuning_params();
  c.calibrate_cache_sizes();

  std::remove(filename.c_str());
}
//===----------------------------------------------------------------------===//
