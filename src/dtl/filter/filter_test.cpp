#include "gtest/gtest.h"

#include <iostream>
#include <fstream>
#include <cstdio>
#include <sstream>

#include <dtl/dtl.hpp>
#include <dtl/src/dtl/barrier.hpp>

#include "filter.hpp"

using namespace dtl::filter;

//===----------------------------------------------------------------------===//
TEST(filter_api, basic_build_probe_test) {
  $u1 rc = false;
  // build
  dtl::blocked_bloomfilter_config config;
  auto f = filter(config, 1024);
  std::vector<$u32> keys {13, 37, 42, 88};
  rc = f.insert(keys.begin(), keys.end());
  ASSERT_TRUE(rc);

  // probe
  auto probe = f.probe();
  std::vector<$u32> match_pos(keys.size(), 0);
  $u32 match_count = 0;
  match_count = probe(keys.begin(), keys.end(), &match_pos[0], 0);
  ASSERT_EQ(4, match_count);
  ASSERT_EQ(0, match_pos[0]);
  ASSERT_EQ(1, match_pos[1]);
  ASSERT_EQ(2, match_pos[2]);
  ASSERT_EQ(3, match_pos[3]);

  std::vector<$u32> probe_keys {0, 1, 42, 2};
  match_count = probe(probe_keys.begin(), probe_keys.end(), &match_pos[0], 1000);
  ASSERT_EQ(1, match_count);
  ASSERT_EQ(1002, match_pos[0]);
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
TEST(filter_api, numa_replication_test) {
  auto& platform = platform::get_instance();

  $u1 rc = false;
  // build
  dtl::blocked_bloomfilter_config config;
  auto f = filter(config, 1024);
  std::vector<$u32> keys {13, 37, 42, 88};
  rc = f.insert(keys.begin(), keys.end());
  ASSERT_TRUE(rc);

  auto thread_cnt = platform.get_thread_count();
  dtl::busy_barrier_one_shot barrier(thread_cnt);
  auto thread_fn = [&](u32 thread_id) {
    barrier.wait();
    // probe
    auto probe = f.probe();
    std::stringstream str;
    str << "thread=" << thread_id;
    str << ", preferred_node=" << platform.get_memory_node_of_this_thread();
    str << ", node_of_filter_data=" << probe.get_numa_node_id();
    str << std::endl;
    std::cout << str.str();

    ASSERT_EQ(platform.get_memory_node_of_this_thread(), probe.get_numa_node_id());

    std::vector<$u32> match_pos(keys.size(), 0);
    $u32 match_count = 0;
    match_count = probe(keys.begin(), keys.end(), &match_pos[0], 0);
    ASSERT_EQ(4, match_count);
    ASSERT_EQ(0, match_pos[0]);
    ASSERT_EQ(1, match_pos[1]);
    ASSERT_EQ(2, match_pos[2]);
    ASSERT_EQ(3, match_pos[3]);

  };
  dtl::run_in_parallel(thread_fn, platform.get_cpu_mask(), thread_cnt);

}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
TEST(filter_api, construct_using_skyline) {
  u64 n = 10000; // elements
  u64 tw = 100;  // cycles
  auto f = filter(n, tw);
  std::vector<$u32> keys {13, 37, 42, 88};
  $u1 rc = false;
  rc = f.insert(keys.begin(), keys.end());
  ASSERT_TRUE(rc);

  // probe
  auto probe = f.probe();
  std::vector<$u32> match_pos(keys.size(), 0);
  $u32 match_count = 0;
  match_count = probe(keys.begin(), keys.end(), &match_pos[0], 0);
  ASSERT_EQ(4, match_count);
  ASSERT_EQ(0, match_pos[0]);
  ASSERT_EQ(1, match_pos[1]);
  ASSERT_EQ(2, match_pos[2]);
  ASSERT_EQ(3, match_pos[3]);

  std::vector<$u32> probe_keys {0, 1, 42, 2};
  match_count = probe(probe_keys.begin(), probe_keys.end(), &match_pos[0], 1000);
  ASSERT_EQ(1, match_count);
  ASSERT_EQ(1002, match_pos[0]);
}
//===----------------------------------------------------------------------===//


