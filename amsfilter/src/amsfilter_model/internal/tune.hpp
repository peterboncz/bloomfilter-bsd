#pragma once

#include <amsfilter/amsfilter.hpp>

namespace amsfilter {
namespace model {
//===----------------------------------------------------------------------===//
/// Runs a micro-benchmark to determine the tuning parameters.
TuningParams
tune(const Config& c);
//===----------------------------------------------------------------------===//
} // namespace model
} // namespace amsfilter
