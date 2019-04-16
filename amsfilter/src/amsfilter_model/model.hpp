#pragma once

#include <limits>
#include <memory>
#include <string>

#include <dtl/dtl.hpp>

#include <amsfilter/config.hpp>
#include <amsfilter_model/internal/perf_db.hpp>
#include <amsfilter_model/execution_env.hpp>

namespace amsfilter {
//===----------------------------------------------------------------------===//
class Model;
//===----------------------------------------------------------------------===//
/// Contains information on how to parameterize the AMS-Filter.
class Params {
  friend class Model;

  Config filter_config_;
  TuningParams tuning_params_;
  std::size_t filter_size_bits_;
  $f64 max_selectivity_;

  Params(const Config& config, const TuningParams& tuning_params,
      const std::size_t m, f64 max_selectivity)
      : filter_config_(config),
        tuning_params_(tuning_params),
        filter_size_bits_(m),
        max_selectivity_(max_selectivity) {}

public:

  Params(Params&&) noexcept = default;
  Params(const Params& other) = default;
  Params& operator=(const Params& other) = default;
  Params& operator=(Params&& other) = default;
  ~Params() = default;

  /// Returns the filter configuration.
  Config
  get_filter_config() const {
    return filter_config_;
  }

  /// Returns the tuning parameters.
  TuningParams
  get_tuning_params() const {
    return tuning_params_;
  }

  /// Returns the filter size in bits.
  std::size_t
  get_filter_size() const {
    return filter_size_bits_;
  }

  /// Returns the selectivity (probability of a true hit) where filtering
  /// becomes unprofitable. Thus, a filter should only be used when the actual
  /// selectivity is less than max_selectivity.
  $f64
  get_max_selectivity() const {
    return max_selectivity_;
  }

};
//===----------------------------------------------------------------------===//
/// Determines performance optimal filter configurations.
class Model {

  /// The database with the performance data.
  std::shared_ptr<amsfilter::model::PerfDb> perf_db_;

public:

  explicit
  Model(std::string db_filename)
      : perf_db_(std::make_shared<amsfilter::model::PerfDb>(db_filename)) {}
  Model();
  Model(Model&&) noexcept = default;
  Model(const Model& other) = default;
  Model& operator=(const Model& other) = default;
  Model& operator=(Model&& other) = default;
  ~Model() = default;

  /// Determines the (close to) performance optimal filter configuration based
  /// on the given problem size n and the work time (tw), which is the execution
  /// time in nanoseconds that is saved when an element is filtered out.
  Params
  determine_filter_params(const amsfilter::model::Env& exec_env, u64 n, f64 tw);

};
//===----------------------------------------------------------------------===//
} // namespace amsfilter
