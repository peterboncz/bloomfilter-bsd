#pragma once

#include <dtl/dtl.hpp>
#include <dtl/mem.hpp>
#include <amsfilter/amsfilter.hpp>
#include <amsfilter_model/execution_env.hpp>
#include <amsfilter_model/internal/timing.hpp>

namespace amsfilter {
namespace model {
//===----------------------------------------------------------------------===//
/// Base class for the benchmark runner that determines the lookup time for a
/// given filter configuration.
class benchmark {

public:

  /// Runs the benchmark.
  virtual timing
  run(const Config& filter_config, u64 m,
      const Env& exec_env, const TuningParams& tuning_params) = 0;

};
//===----------------------------------------------------------------------===//
} // namespace model
} // namespace amsfilter
