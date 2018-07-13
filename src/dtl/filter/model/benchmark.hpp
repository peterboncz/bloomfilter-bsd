#pragma once

#include <cstddef>

#include <dtl/dtl.hpp>
#include <dtl/mem.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp>
#include <dtl/filter/cuckoofilter/cuckoofilter_config.hpp>
#include <dtl/filter/filter.hpp>

#include "timing.hpp"

namespace dtl {
namespace filter {
namespace model {

//===----------------------------------------------------------------------===//
class benchmark {

  using key_t = $u32;

  std::vector<key_t, dtl::mem::numa_allocator<key_t>> probe_keys;

  timing
  run(dtl::filter::filter& filter, $u32 thread_cnt);

public:
  benchmark();

  timing
  operator()(const dtl::blocked_bloomfilter_config& filter_config, u64 m, u32 thread_cnt = 0);

  timing
  operator()(const dtl::cuckoofilter::config& filter_config, u64 m, u32 thread_cnt = 0);

};
//===----------------------------------------------------------------------===//

} // namespace model
} // namespace filter
} // namespace dtl
