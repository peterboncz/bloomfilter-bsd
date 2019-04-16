#pragma once

#include <vector>
#include <dtl/dtl.hpp>
#include <dtl/mem.hpp>
#include <amsfilter/amsfilter.hpp>
#include <amsfilter_model/internal/benchmark.hpp>
#include <amsfilter_model/internal/timing.hpp>
#include <amsfilter_model/execution_env.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


namespace amsfilter {
namespace model {
//===----------------------------------------------------------------------===//
/// Performs a micro-benchmark to determine the actual lookup time for a
/// given filter configuration.
class benchmark_gpu : public benchmark {

  std::vector<key_t> probe_keys;

  using device_word_vector = thrust::device_vector<amsfilter::word_t>;
  using device_key_vector = thrust::device_vector<amsfilter::key_t>;

  // One for each GPU.
  std::vector<device_word_vector*> device_bitmap_vectors;
  std::vector<device_key_vector*> device_key_vectors;

public:
  benchmark_gpu();
  ~benchmark_gpu();

  /// Runs the benchmark on the specified GPU.
  timing
  run(const Config& filter_config, u64 m,
      const Env& exec_env, const TuningParams& tuning_params);

};
//===----------------------------------------------------------------------===//
} // namespace model
} // namespace amsfilter
