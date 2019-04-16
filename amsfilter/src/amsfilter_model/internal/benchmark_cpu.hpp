#pragma once

#include <dtl/dtl.hpp>
#include <dtl/mem.hpp>
#include <amsfilter/amsfilter.hpp>
#include <amsfilter_model/internal/benchmark.hpp>
#include <amsfilter_model/internal/timing.hpp>
#include <amsfilter_model/execution_env.hpp>

namespace amsfilter {
namespace model {
//===----------------------------------------------------------------------===//
/// Performs a micro-benchmark to determine the actual lookup time for a
/// given filter configuration.
class benchmark_cpu : public benchmark {

  std::vector<key_t, dtl::mem::numa_allocator<key_t>> probe_keys;

public:
  benchmark_cpu();

  /// Runs the benchmark on the CPU using the specified number of threads.
  timing
  run(const Config& filter_config, u64 m,
      const Env& exec_env, const TuningParams& tuning_params);

};
//===----------------------------------------------------------------------===//
} // namespace model
} // namespace amsfilter
