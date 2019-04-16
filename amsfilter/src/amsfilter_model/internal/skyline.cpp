#include "skyline.hpp"

#include <atomic>
#include <cmath>
#include <set>
#include <queue>

#include <dtl/dtl.hpp>
#include <dtl/thread.hpp>

#include <amsfilter_model/internal/common.hpp>
#include <amsfilter_model/internal/cost_fn.hpp>
#include <amsfilter_model/internal/util.hpp>

namespace amsfilter {
namespace model {
//===----------------------------------------------------------------------===//
// Limits for the skyline matrix.
static u64 n_lo_log2 = 10;
static u64 n_lo = 1ull << n_lo_log2;
static u64 n_hi_log2 = 31;
static u64 n_hi = 1ull << n_hi_log2;
static u64 tw_lo_log2 = 0;
static u64 tw_hi_log2 = 31;
//===----------------------------------------------------------------------===//
// Bits-per-element rate
static u64 b_lo = 1;
static u64 b_hi = 64;
//===----------------------------------------------------------------------===//
/// Returns all values for n which are considered in the skyline matrix.
auto
get_n_values() {
  const auto n_values = [&]() {
    std::set<$u64> n_s;

    for ($u64 n_log2 = n_lo_log2; n_log2 <= n_hi_log2; n_log2++) {
      const std::vector<$f64> exp {
          n_log2 +  0 * 0.0625,
          n_log2 +  1 * 0.0625,
          n_log2 +  2 * 0.0625,
          n_log2 +  3 * 0.0625,
          n_log2 +  4 * 0.0625,
          n_log2 +  5 * 0.0625,
          n_log2 +  6 * 0.0625,
          n_log2 +  7 * 0.0625,
          n_log2 +  8 * 0.0625,
          n_log2 +  9 * 0.0625,
          n_log2 + 10 * 0.0625,
          n_log2 + 11 * 0.0625,
          n_log2 + 12 * 0.0625,
          n_log2 + 13 * 0.0625,
          n_log2 + 14 * 0.0625,
          n_log2 + 15 * 0.0625,
          n_log2 + 16 * 0.0625,
      };
//      const std::vector<$f64> exp {
//          n_log2 + 0 * 0.125,
//          n_log2 + 1 * 0.125,
//          n_log2 + 2 * 0.125,
//          n_log2 + 3 * 0.125,
//          n_log2 + 4 * 0.125,
//          n_log2 + 5 * 0.125,
//          n_log2 + 6 * 0.125,
//          n_log2 + 7 * 0.125,
//          n_log2 + 8 * 0.125,
//      };
//      const std::vector<$f64> exp {
//          n_log2 + 0 * 0.25,
//          n_log2 + 1 * 0.25,
//          n_log2 + 2 * 0.25,
//          n_log2 + 3 * 0.25,
//          n_log2 + 4 * 0.25,
//      };

      for (auto e : exp) {
        u64 n = std::pow(2, e);
        if ((n * b_lo) > m_max) continue; // make sure to not exceed the max filter size
        n_s.insert(n);
      }
    }
    std::vector<$u64> ret_val(n_s.begin(), n_s.end());
    return ret_val;
  }();
  return n_values;
}
//===----------------------------------------------------------------------===//
/// Returns all values for tw which are considered in the skyline matrix.
auto
get_tw_values() {
  const std::vector<$u64> tw_values = [&]() {
    std::set<$u64> tw_vals;

    for ($u64 tw_log2 = tw_lo_log2; tw_log2 <= tw_hi_log2; tw_log2++) {
      const std::vector<$f64> exp {
          tw_log2 +  0 * 0.0625,
          tw_log2 +  1 * 0.0625,
          tw_log2 +  2 * 0.0625,
          tw_log2 +  3 * 0.0625,
          tw_log2 +  4 * 0.0625,
          tw_log2 +  5 * 0.0625,
          tw_log2 +  6 * 0.0625,
          tw_log2 +  7 * 0.0625,
          tw_log2 +  8 * 0.0625,
          tw_log2 +  9 * 0.0625,
          tw_log2 + 10 * 0.0625,
          tw_log2 + 11 * 0.0625,
          tw_log2 + 12 * 0.0625,
          tw_log2 + 13 * 0.0625,
          tw_log2 + 14 * 0.0625,
          tw_log2 + 15 * 0.0625,
          tw_log2 + 16 * 0.0625,
      };
//      const std::vector<$f64> exp {
//          tw_log2 + 0 * 0.125,
//          tw_log2 + 1 * 0.125,
//          tw_log2 + 2 * 0.125,
//          tw_log2 + 3 * 0.125,
//          tw_log2 + 4 * 0.125,
//          tw_log2 + 5 * 0.125,
//          tw_log2 + 6 * 0.125,
//          tw_log2 + 7 * 0.125,
//          tw_log2 + 8 * 0.125,
//      };
//      const std::vector<$f64> exp {
//          tw_log2 + 0 * 0.25,
//          tw_log2 + 1 * 0.25,
//          tw_log2 + 2 * 0.25,
//          tw_log2 + 3 * 0.25,
//          tw_log2 + 4 * 0.25,
//      };
//      const std::vector<$f64> exp {
//          tw_log2 + 0 * 0.5,
//          tw_log2 + 1 * 0.5,
//      };
//      const std::vector<$f64> exp {
//          tw_log2 + 0 * 0.5,
//      };
      for (auto e : exp) {
        u64 tw = std::pow(2, e);
        tw_vals.insert(tw);
      }
    }
    std::vector<$u64> ret_val(tw_vals.begin(), tw_vals.end());
    return ret_val;
  }();
  return tw_values;
}
//===----------------------------------------------------------------------===//
static std::set<amsfilter::Config>
determine_candidate_filter_configs(PerfDb& perf_db, const Env& exec_env) {

  // The n,tw values for which the skyline is computed.
  const auto n_values = get_n_values();
  const auto tw_values = get_tw_values();

  // The considered filter configurations.
  const auto valid_configs = get_valid_configs();

  // Instantiate the function to minimize the overhead.
  std::vector<Optimizer> optimizers;
  optimizers.reserve(valid_configs.size());
  for (std::size_t i = 0; i < valid_configs.size(); ++i) {
    optimizers.emplace_back(valid_configs[i], perf_db, exec_env);
  }

  // The candidate filter configuration that will show up in the skyline.
  std::set<amsfilter::Config> skyline_candidate_configs;
  std::mutex skyline_candidate_configs_mutex;

  const std::size_t matrix_size = n_values.size() * tw_values.size();
//  const std::size_t n_cnt = n_values.size();
  const std::size_t tw_cnt = tw_values.size();

  std::atomic<std::size_t> cntr { 0 };
  const std::size_t batch_size = 128;
  auto thread_fn = [&](u32 thread_id) {
    while (true) {
      // Grab work.
      const auto idx_begin = cntr.fetch_add(batch_size);
      const auto idx_end = std::min(idx_begin + batch_size, matrix_size);
      if (idx_begin >= matrix_size) break;

      for (std::size_t i = idx_begin; i < idx_end; ++i) {
        const auto n = n_values[i / tw_cnt];
        const auto tw = tw_values[i % tw_cnt];
//          std::cout << "n=" << n << ", tw=" << tw << " -> " << std::flush;
        std::priority_queue<SkylineEntry> candidates;
        amsfilter::Config opt_c;
        for (auto& optimizer : optimizers) {
          const auto res = optimizer(n, tw);
          SkylineEntry candidate;
          candidate.filter_config = optimizer.get_config();
          candidate.m = std::get<0>(res);
          candidate.overhead = std::get<1>(res);
          // Keep the top-3 for every n,tw pair.
          if (candidates.size() < 3) {
            candidates.push(candidate);
          }
          else {
            if (candidates.top().overhead > candidate.overhead) {
              candidates.pop();
              candidates.push(candidate);
            }
          }
        }
        /* Critical section */ {
          // Add the candidates to the result set.
          std::lock_guard<std::mutex> guard(skyline_candidate_configs_mutex);
          while (!candidates.empty()) {
            const auto& c = candidates.top();
            skyline_candidate_configs.insert(c.filter_config);
            candidates.pop();
          }
        }
//      std::cout << "m=" << opt_m << ", o=" << opt_o << ", c=" << opt_c << std::endl;
      }
    }
  };
  dtl::run_in_parallel(thread_fn);
  return skyline_candidate_configs;
}
//===----------------------------------------------------------------------===//
std::set<amsfilter::Config>
Skyline::determine_candidate_filter_configurations() {
  return determine_candidate_filter_configs(*perf_db_, exec_env_);
}
//===----------------------------------------------------------------------===//
void Skyline::compute() {
  perf_db_->begin();
  // The n,tw values for which the skyline is computed.
  const auto n_values = get_n_values();
  const auto tw_values = get_tw_values();

  // The considered filter configurations.
  const auto valid_configs = get_valid_configs();

  // Instantiate the function to minimize the overhead.
  std::vector<Optimizer> optimizers;
  optimizers.reserve(valid_configs.size());
  for (std::size_t i = 0; i < valid_configs.size(); ++i) {
    optimizers.emplace_back(valid_configs[i], *perf_db_, exec_env_);
  }

  const std::size_t matrix_size = n_values.size() * tw_values.size();
//  const std::size_t n_cnt = n_values.size();
  const std::size_t tw_cnt = tw_values.size();

  std::atomic<std::size_t> cntr { 0 };
  const std::size_t batch_size = 128;
  auto thread_fn = [&](u32 thread_id) {
    while (true) {
      // Grab work.
      const auto idx_begin = cntr.fetch_add(batch_size);
      const auto idx_end = std::min(idx_begin + batch_size, matrix_size);
      if (idx_begin >= matrix_size) break;

      for (std::size_t i = idx_begin; i < idx_end; ++i) {
        const auto n = n_values[i / tw_cnt];
        const auto tw = tw_values[i % tw_cnt];

        amsfilter::Config opt_c;
        $u64 opt_m = 0;
        $f64 opt_o = std::numeric_limits<$f64>::max();

        for (auto& optimizer : optimizers) {
          const auto res = optimizer(n, tw);
          const auto c = optimizer.get_config();
          const auto m = std::get<0>(res);
          const auto o = std::get<1>(res);
          if (o < opt_o) {
            opt_c = c;
            opt_m = m;
            opt_o = o;
          }
        }
        perf_db_->put_skyline_entry(exec_env_, n, tw, opt_c, opt_m, opt_o);
      }
    }
  };
  dtl::run_in_parallel(thread_fn);
  perf_db_->commit();
}
//===----------------------------------------------------------------------===//
SkylineEntry
Skyline::lookup(const std::size_t n, f64 tw) {
  const auto candidates = perf_db_->find_skyline_entries(exec_env_, n, tw);
  std::set<Config> candidate_configs;
  for (auto& candidate : candidates) {
    candidate_configs.insert(candidate.filter_config);
  }

  SkylineEntry result;
  for (auto& candidate_config : candidate_configs) {
    Optimizer optimizer(candidate_config, *perf_db_, exec_env_);
    const auto m_o = optimizer(n, tw);
    const auto m = std::get<0>(m_o);
    const auto o = std::get<1>(m_o);
    if (o < result.overhead) {
      result.filter_config = candidate_config;
      result.m = m;
      result.overhead = o;
    }
  }
  return result;
}
//===----------------------------------------------------------------------===//
} // namespace model
} // namespace amsfilter
