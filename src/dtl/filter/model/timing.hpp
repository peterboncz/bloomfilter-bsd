#pragma once

#include <dtl/dtl.hpp>

namespace dtl {
namespace filter {
namespace model {


//===----------------------------------------------------------------------===//
/// Timings
struct timing {
  $f64 cycles_per_lookup = 0.0;
  $f64 nanos_per_lookup = 0.0;

  $u1
  operator==(const timing& other) const {
    return cycles_per_lookup == other.cycles_per_lookup
        && nanos_per_lookup == other.nanos_per_lookup;
  }

  $u1
  operator<(const timing& other) const {
    return cycles_per_lookup < other.cycles_per_lookup;
  }

  $u1
  operator>(const timing& other) const {
    return cycles_per_lookup > other.cycles_per_lookup;
  }

  timing
  operator*(f64 rhs) const {
    timing t;
    t.cycles_per_lookup = cycles_per_lookup * rhs;
    t.nanos_per_lookup = nanos_per_lookup * rhs;
    return t;
  }

  timing
  operator/(f64 rhs) const {
    timing t;
    t.cycles_per_lookup = cycles_per_lookup / rhs;
    t.nanos_per_lookup = nanos_per_lookup / rhs;
    return t;
  }

  timing
  operator+(f64 rhs) const {
    timing t;
    t.cycles_per_lookup = cycles_per_lookup + rhs;
    t.nanos_per_lookup = nanos_per_lookup + rhs;
    return t;
  }

  timing&
  operator+=(f64 rhs) {
    cycles_per_lookup += rhs;
    nanos_per_lookup += rhs;
    return *this;
  }

  timing
  operator+(const timing& rhs) const {
    timing t;
    t.cycles_per_lookup = cycles_per_lookup + rhs.cycles_per_lookup;
    t.nanos_per_lookup = nanos_per_lookup + rhs.nanos_per_lookup;
    return t;
  }

  timing&
  operator+=(const timing& rhs) {
    cycles_per_lookup += rhs.cycles_per_lookup;
    nanos_per_lookup += rhs.nanos_per_lookup;
    return *this;
  }

  timing
  operator-(const timing& rhs) const {
    timing t;
    t.cycles_per_lookup = cycles_per_lookup - rhs.cycles_per_lookup;
    t.nanos_per_lookup = nanos_per_lookup - rhs.nanos_per_lookup;
    return t;
  }

};
//===----------------------------------------------------------------------===//


} // namespace model
} // namespace filter
} // namespace dtl