#pragma once

#include <limits>
#include <tuple>

#include <dtl/dtl.hpp>
#include <amsfilter/config.hpp>
#include <amsfilter_model/internal/perf_db.hpp>
#include <amsfilter_model/internal/common.hpp>
#include <amsfilter_model/fpr.hpp>


namespace amsfilter {
namespace model {
//===----------------------------------------------------------------------===//
/// Function to compute the lookup costs based on the filter size. The costs
/// are approximated using linear interpolation on reference data points stored
/// in the database.
class LookupCostFn {

  PerfDb::tl_table tl_table_;

public:

  LookupCostFn(const Config& config, PerfDb& perf_db, const Env& env) {
    // Load the reference points from the DB.
    tl_table_ = perf_db.get_tls(config, env);
    if (tl_table_.m.empty()) {
      throw std::runtime_error("No reference data points available.");
    }
  }

  LookupCostFn(LookupCostFn&&) noexcept = default;
  LookupCostFn(const LookupCostFn& other) = default;
  LookupCostFn& operator=(const LookupCostFn& other) = default;
  LookupCostFn& operator=(LookupCostFn&& other) = default;
  ~LookupCostFn() = default;

  $f64
  operator()(const std::size_t m) const {
    auto search = std::lower_bound(tl_table_.m.begin(), tl_table_.m.end(), m);
    if (search == tl_table_.m.end()) {
      return tl_table_.nanos.back();
    }
    if (search == tl_table_.m.begin()) {
      return tl_table_.nanos.front();
    }
    const auto i_hi = std::distance(tl_table_.m.begin(), search);
    const auto i_lo = i_hi - 1;
    if (tl_table_.m[i_hi] == m) {
      return tl_table_.nanos[i_hi];
    }

    const auto delta_m = tl_table_.m[i_hi] - tl_table_.m[i_lo];
    const auto delta_tl = tl_table_.nanos[i_hi] - tl_table_.nanos[i_lo];
    const auto gradient = delta_tl / delta_m;

    const auto t = tl_table_.nanos[i_lo] - gradient * tl_table_.m[i_lo];
    const auto nanos = gradient * m + t;
    return nanos;
  }
};
//===----------------------------------------------------------------------===//
/// Function to compute the lookup costs based on the filter size. The costs
/// are approximated using linear interpolation on reference data points stored
/// in the database.
class OverheadFn {

  LookupCostFn tl_;
  Config config_;

public:

  OverheadFn(const Config& config, PerfDb& perf_db, const Env& env)
      : tl_(config, perf_db, env), config_(config) {
  }

  OverheadFn(OverheadFn&&) noexcept = default;
  OverheadFn(const OverheadFn& other) = default;
  OverheadFn& operator=(const OverheadFn& other) = default;
  OverheadFn& operator=(OverheadFn&& other) = default;
  ~OverheadFn() = default;

  $f64
  operator()(const std::size_t m, const std::size_t n, f64 tw) const {
    f64 bits_per_element = (m * 1.0) / n;
    if (bits_per_element < 1.0) {
      return std::numeric_limits<$f64>::max();
    }
    return tl_(m) + fpr_fast(config_, bits_per_element) * tw;
//    return tl_(m) + fpr(config_, bits_per_element) * tw;
  }
};
//===----------------------------------------------------------------------===//
/// Determines the performance-optimal filter size for a given filter
/// configuration and the given execution environment.
class Optimizer {

  LookupCostFn tl_;
  OverheadFn rho_;
  Config config_;

  PerfDb::tl_table tl_table_;

public:

  Optimizer(const Config& config, PerfDb& perf_db, const Env& env)
      : tl_(config, perf_db, env),
        rho_(config, perf_db, env),
        config_(config) {
    // Load the reference points from the DB.
    tl_table_ = perf_db.get_tls(config, env);
    if (tl_table_.m.empty()) {
      throw std::runtime_error("No reference data points available.");
    }
  }

  Optimizer(Optimizer&&) noexcept = default;
  Optimizer(const Optimizer& other) = default;
  Optimizer& operator=(const Optimizer& other) = default;
  Optimizer& operator=(Optimizer&& other) = default;
  ~Optimizer() = default;

  /// Returns the configuration.
  Config get_config() const { return config_; }

  /// Returns the optimal m and the overhead for the given n and tw values.
  auto
  operator()(u64 n, f64 tw) const {

    // Derive the overhead function.
    auto d = [&](u64 m, u64 h) {
      // Numerical differentiation (symmetric difference quotient)
      auto a = rho_(m + h, n, tw);
      auto b = rho_(m + h, n, tw);
      f64 df = (a - b) / (2 * h);
      return df;
    };

    if (config_.addr_mode == dtl::block_addressing::POWER_OF_TWO) {
      const auto m_min_log2 = dtl::log_2(m_min);
      const auto m_max_log2 = dtl::log_2(m_max);
      $u64 opt_m = 1ull << m_min_log2;
      $f64 overhead = rho_(opt_m, n, tw);
//      std::cout << overhead << std::endl;
      for ($u64 curr_m_log2 = m_min_log2 + 1; curr_m_log2 <= m_max_log2;
           ++curr_m_log2) {
        f64 curr_overhead = rho_(1ull << curr_m_log2, n, tw);
//        std::cout << overhead << std::endl;
        if (curr_overhead > overhead + 0.00001) break;
        opt_m = 1ull << curr_m_log2;
        overhead = curr_overhead;
      }
      return std::make_tuple(opt_m, rho_(opt_m, n, tw));
    }
    else {
      $u64 opt_m_idx = 0;
      $u64 opt_m = opt_m_idx;
      $f64 overhead = rho_(opt_m, n, tw);
      for (std::size_t i = 1; i < tl_table_.m.size(); ++i) {
        u64 curr_m = tl_table_.m[i];
        f64 curr_overhead = rho_(curr_m, n, tw);
        if (curr_overhead > overhead) break;
        opt_m_idx = i;
        opt_m = curr_m;
        overhead = curr_overhead;
      }
      if (opt_m_idx == 0 || opt_m_idx == (tl_table_.m.size() - 1)) {
        return std::make_tuple(opt_m, rho_(opt_m, n, tw));
      }
      $u64 search_range_m_lo = tl_table_.m[opt_m_idx - 1];
      $u64 search_range_m_hi = tl_table_.m[opt_m_idx + 1];
      while (search_range_m_hi - search_range_m_lo < 1024) {
        u64 span = ((search_range_m_hi - search_range_m_lo) / 2);
        u64 m_mid = search_range_m_lo + span;
        f64 dm_mid = d(m_mid, span / 4);
        if (dm_mid < 0) {
          search_range_m_lo = m_mid;
        }
        else if (dm_mid > 0) {
          search_range_m_hi = m_mid;
        }
        else {
          break;
        }
      }
      opt_m = search_range_m_lo;
      return std::make_tuple(opt_m, rho_(opt_m, n, tw));
    }
  }
};
//===----------------------------------------------------------------------===//
} // namespace model
} // namespace amsfilter
