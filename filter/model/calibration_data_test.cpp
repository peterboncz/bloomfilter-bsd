#include "gtest/gtest.h"

#include <iostream>
#include <fstream>
#include <cstdio>

#include <dtl/dtl.hpp>

#include "dtl/filter/model/calibration_data.hpp"
#include "dtl/filter/model/timing.hpp"

using namespace dtl::filter::model;

//===----------------------------------------------------------------------===//
TEST(model_calibration_data, basic_persistence) {
  const std::string filename = "/tmp/calibration_data";
  std::remove(filename.c_str());

  dtl::blocked_bloomfilter_config bbf_config_1;
  bbf_config_1.k = 1;
  dtl::blocked_bloomfilter_config bbf_config_2;
  bbf_config_2.k = 2;
  dtl::cuckoofilter::config cf_config_1;
  cf_config_1.tags_per_bucket = 1;
  dtl::cuckoofilter::config cf_config_2;
  cf_config_1.tags_per_bucket = 2;

  const std::vector<timing> delta_timings_1 = {{1,2}, {3,4}, {5,6}, {7,8}};
  const std::vector<timing> delta_timings_2 = {{9,10}, {11,12}, {13,14}, {15,16}};

  {
    calibration_data cd(filename);

    cd.set_cache_sizes({10,20,30});
    ASSERT_EQ(10, cd.get_cache_size(1));
    ASSERT_EQ(20, cd.get_cache_size(2));
    ASSERT_EQ(30, cd.get_cache_size(3));

    cd.set_filter_sizes({5,15,25,35});
    ASSERT_EQ(5, cd.get_filter_size(1));
    ASSERT_EQ(15, cd.get_filter_size(2));
    ASSERT_EQ(25, cd.get_filter_size(3));
    ASSERT_EQ(35, cd.get_filter_size(4));

    ASSERT_EQ(4, cd.get_mem_levels());

    cd.put(bbf_config_1, delta_timings_1);
    cd.put(bbf_config_2, delta_timings_2);
    cd.put(cf_config_1, delta_timings_1);
    cd.put(cf_config_2, delta_timings_2);

    auto actual_bbf_timings_1 = cd.get(bbf_config_1);
    ASSERT_EQ(delta_timings_1, actual_bbf_timings_1);
    auto actual_bbf_timings_2 = cd.get(bbf_config_2);
    ASSERT_EQ(delta_timings_2, actual_bbf_timings_2);
    auto actual_cf_timings_1 = cd.get(cf_config_1);
    ASSERT_EQ(delta_timings_1, actual_cf_timings_1);
    auto actual_cf_timings_2 = cd.get(cf_config_2);
    ASSERT_EQ(delta_timings_1, actual_cf_timings_1);

    ASSERT_TRUE(cd.changed());

    cd.persist();
  }

  {
    calibration_data cd(filename);

    ASSERT_EQ(4, cd.get_mem_levels());

    ASSERT_EQ(10, cd.get_cache_size(1));
    ASSERT_EQ(20, cd.get_cache_size(2));
    ASSERT_EQ(30, cd.get_cache_size(3));

    ASSERT_EQ(5, cd.get_filter_size(1));
    ASSERT_EQ(15, cd.get_filter_size(2));
    ASSERT_EQ(25, cd.get_filter_size(3));
    ASSERT_EQ(35, cd.get_filter_size(4));

    auto received_timings_1 = cd.get(bbf_config_1);
    ASSERT_EQ(delta_timings_1, received_timings_1);
    auto received_timings_2 = cd.get(bbf_config_2);
    ASSERT_EQ(delta_timings_2, received_timings_2);

    ASSERT_FALSE(cd.changed());
  }

}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
TEST(model_calibration_data, add_config) {
  const std::string filename = "/tmp/calibration_data";
  std::remove(filename.c_str());

  dtl::blocked_bloomfilter_config bbf_config_1;
  bbf_config_1.k = 1;
  dtl::blocked_bloomfilter_config bbf_config_2;
  bbf_config_2.k = 2;
  dtl::cuckoofilter::config cf_config_1;
  cf_config_1.tags_per_bucket = 1;
  dtl::cuckoofilter::config cf_config_2;
  cf_config_1.tags_per_bucket = 2;

  const std::vector<timing> delta_timings_1 = {{1,2}, {3,4}, {5,6}, {7,8}};
  const std::vector<timing> delta_timings_2 = {{9,10}, {11,12}, {13,14}, {15,16}};

  {
    // create a config file
    calibration_data cd(filename);
    cd.set_cache_sizes({10,20,30});
    cd.set_filter_sizes({5,15,25,35});

    // put first config
    cd.put(bbf_config_1, delta_timings_1);

    // write to file
    cd.persist();
  }

  {
    // re-open
    calibration_data cd(filename);

    // put second config
    cd.put(bbf_config_2, delta_timings_2);

    // write to file
    cd.persist();
  }

  {
    // re-open
    calibration_data cd(filename);

    // check if first config exists
    auto received_timings_1 = cd.get(bbf_config_1);
    ASSERT_EQ(delta_timings_1, received_timings_1);

    // check if second config exists
    auto received_timings_2 = cd.get(bbf_config_2);
    ASSERT_EQ(delta_timings_2, received_timings_2);
  }

}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
TEST(model_calibration_data, update_config) {
  const std::string filename = "/tmp/calibration_data";
  std::remove(filename.c_str());

  dtl::blocked_bloomfilter_config bbf_config_1;
  bbf_config_1.k = 1;
  dtl::blocked_bloomfilter_config bbf_config_2;
  bbf_config_2.k = 2;
  dtl::cuckoofilter::config cf_config_1;
  cf_config_1.tags_per_bucket = 1;
  dtl::cuckoofilter::config cf_config_2;
  cf_config_1.tags_per_bucket = 2;

  const std::vector<timing> delta_timings_1 = {{1,2}, {3,4}, {5,6}, {7,8}};
  const std::vector<timing> delta_timings_2 = {{9,10}, {11,12}, {13,14}, {15,16}};

  {
    // create a config file
    calibration_data cd(filename);
    cd.set_cache_sizes({10,20,30});
    cd.set_filter_sizes({5,15,25,35});

    // put first config
    cd.put(bbf_config_1, delta_timings_1);
    cd.put(cf_config_1, delta_timings_1);

    // write to file
    cd.persist();
  }

  {
    // re-open
    calibration_data cd(filename);

    // update first config
    cd.put(bbf_config_1, delta_timings_2);
    cd.put(cf_config_1, delta_timings_2);

    // write to file
    cd.persist();
  }

  {
    // re-open
    calibration_data cd(filename);

    // check first config
    auto received_timings_1 = cd.get(bbf_config_1);
    ASSERT_EQ(delta_timings_2, received_timings_1);

    // check if second config exists
    auto received_timings_2 = cd.get(cf_config_1);
    ASSERT_EQ(delta_timings_2, received_timings_2);
  }

}
//===----------------------------------------------------------------------===//
