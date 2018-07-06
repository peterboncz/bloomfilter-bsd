#pragma once

#include <dtl/dtl.hpp>

namespace dtl {
namespace filter {
namespace model {


//===----------------------------------------------------------------------===//
/// Timings (benchmark results)
struct timing {
  $f64 cycles_per_lookup = 0.0;
  $f64 nanos_per_lookup = 0.0;

  $u1 operator==(const timing& other) const {
    return cycles_per_lookup == other.cycles_per_lookup
        && nanos_per_lookup == other.nanos_per_lookup;
  }
};
//===----------------------------------------------------------------------===//


} // namespace model
} // namespace filter
} // namespace dtl