#include <amsfilter_model/internal/skyline.hpp>
#include "model.hpp"

using namespace amsfilter::model;

namespace amsfilter {
//===----------------------------------------------------------------------===//
Model::Model() : Model(PerfDb::get_default_filename()) {}
//===----------------------------------------------------------------------===//
Params
Model::determine_filter_params(const Env& exec_env, u64 n,
    f64 tw) {
  Skyline skyline(perf_db_, exec_env);
  SkylineEntry entry = skyline.lookup(n, tw);
  f64 max_sel = (entry.overhead < tw)
      ? (1.0 - (entry.overhead / tw))
      : 0.0;
  auto tuning_params = perf_db_->get_tuning_params(entry.filter_config);
  Params params(entry.filter_config, tuning_params, entry.m, max_sel);
  return params;
}
//===----------------------------------------------------------------------===//
} // namespace amsfilter
