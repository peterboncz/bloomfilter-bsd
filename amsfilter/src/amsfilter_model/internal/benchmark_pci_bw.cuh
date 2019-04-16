#pragma once

#include <utility>
#include <amsfilter_model/execution_env.hpp>
#include <amsfilter_model/internal/timing.hpp>

namespace amsfilter {
namespace model {
//===----------------------------------------------------------------------===//
/// Performs a micro-benchmark to determine the PCI bandwidth between the host
/// and a CUDA device.
class benchmark_pci_bw {

public:
  benchmark_pci_bw();
  ~benchmark_pci_bw();

  /// Runs the benchmark for the specified GPU and returns the measured
  /// bandwidth in MiB/s and MiB/cycle.
  std::pair<$f64, $f64>
  run(const Env& exec_env);

};
//===----------------------------------------------------------------------===//
} // namespace model
} // namespace amsfilter
