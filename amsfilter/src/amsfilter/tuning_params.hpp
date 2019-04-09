#pragma once

#include <dtl/dtl.hpp>

namespace amsfilter {
//===----------------------------------------------------------------------===//
/// Encapsulates hardware related tuning parameters.
struct tuning_params {
  /// The SIMD unrolling factor.
  $u32 unroll_factor = 1;
  /// The maximum unrolling factor.
  static constexpr u32 max_unroll_factor = 4;

  tuning_params() = default;
  ~tuning_params() = default;
  tuning_params(const tuning_params& other) = default;
  tuning_params(tuning_params&& other) = default;

  tuning_params& operator=(const tuning_params& rhs) = default;
  tuning_params& operator=(tuning_params&& rhs) = default;

  $u1
  operator==(const tuning_params& other) const {
    return unroll_factor == other.unroll_factor;
  }

  void
  print(std::ostream& os) const {
    os <<  "u=" << unroll_factor;
  }

};
//===----------------------------------------------------------------------===//
} // namespace amsfilter
