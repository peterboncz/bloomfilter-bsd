#pragma once

#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp>

namespace amsfilter {
//===----------------------------------------------------------------------===//
using Config = dtl::blocked_bloomfilter_config;
//===----------------------------------------------------------------------===//
/// Returns true if the given configuration is valid, false otherwise.
$u1
is_config_valid(const Config& c);
//===----------------------------------------------------------------------===//
} // namespace amsfilter
