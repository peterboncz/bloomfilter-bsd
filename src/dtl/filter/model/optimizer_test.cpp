#include "gtest/gtest.h"

#include <iostream>
#include <fstream>
#include <cstdio>

#include <dtl/dtl.hpp>

#include "dtl/filter/model/calibration_data.hpp"
#include "dtl/filter/model/optimizer.hpp"
#include "dtl/filter/model/timing.hpp"

#include "util.hpp"

using namespace dtl::filter::model;

//===----------------------------------------------------------------------===//
TEST(model_optimizer, basic) {
  const std::string filename = "/tmp/calibration_data_opt";
  std::remove(filename.c_str());

  const std::vector<timing> delta_timings = {{  2.209,  0.667},
                                             {  0.145,  0.044},
                                             {  8.905,  2.688},
                                             { 33.377, 10.078}};

  calibration_data cd(filename);

  const auto configs = get_valid_bbf_configs();
  for (auto c : configs) {
    cd.put_timings(c, delta_timings);
  }

  u64 n = 1000000;
  timing tw { 1000.0, 1000.0 };
  u64 m_lo = 8ull * 1024 * 8;
  u64 m_hi = 256ull * 1024 * 1024 * 8;


  std::vector<timing> overheads;
  std::vector<$u64> ms;
  for (auto c : configs) {
    optimizer opt(n, tw, m_lo, m_hi, c, cd);
    while (opt.step()) { }
    overheads.push_back(opt.get_result_overhead());
    ms.push_back(opt.get_result_m());
  }

  std::vector<std::size_t> rank(configs.size());
  std::iota(rank.begin(), rank.end(), 0);
  std::sort(rank.begin(), rank.end(), [&](auto a, auto b) {
    return overheads[a] < overheads[b];
  });
  for (std::size_t i = 0; i < std::min(30ul, configs.size()); i++) {
    std::cout << "rank " << i
                         << ": config="
                         << configs[rank[i]]
                         << ", m=" << (ms[rank[i]] / 1024 / 8) << " [KiB]"
                         << ", overhead=" << overheads[rank[i]].cycles_per_lookup
                         << std::endl;
  }

  std::remove(filename.c_str());
}
//===----------------------------------------------------------------------===//
