#pragma once

#include <limits>
#include <dtl/dtl.hpp>
#include <amsfilter/config.hpp>

namespace amsfilter {
namespace model {
//===----------------------------------------------------------------------===//
struct SkylineEntry {
  amsfilter::Config filter_config;
  $u64 m = 0;
  $f64 overhead = std::numeric_limits<$f64>::max();

  $u1
  operator<(const SkylineEntry& other) const {
    return overhead < other.overhead;
  }
};
//===----------------------------------------------------------------------===//
} // namespace model
} // namespace amsfilter
