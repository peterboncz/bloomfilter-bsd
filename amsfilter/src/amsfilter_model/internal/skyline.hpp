#pragma once

#include <set>
#include <vector>
#include <amsfilter_model/internal/perf_db.hpp>
#include <amsfilter_model/internal/skyline_entry.hpp>
#include <amsfilter_model/execution_env.hpp>

namespace amsfilter {
namespace model {
//===----------------------------------------------------------------------===//
class Skyline {

  std::shared_ptr<PerfDb> perf_db_;
  Env exec_env_;

public:

  explicit
  Skyline(const std::shared_ptr<PerfDb>& perf_db, const Env& exec_env)
      : perf_db_(perf_db), exec_env_(exec_env) {}

  /// Returns the candidate filter configurations which are likely to appear
  /// in the skyline. The calibration data for these configurations is supposed
  /// to be refined before the actual skyline matrix is computed.
  std::set<amsfilter::Config>
  determine_candidate_filter_configurations();

  /// Computes the skyline matrix and stores it in the database.
  void compute();

  /// Returns the candidate configuration.
  SkylineEntry
  lookup(const std::size_t n, f64 tw);

};
//===----------------------------------------------------------------------===//
} // namespace model
} // namespace amsfilter
