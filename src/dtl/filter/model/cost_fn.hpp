#pragma once

#include <iostream>
#include <map>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/math.hpp>
#include <dtl/mem.hpp>
#include <dtl/simd.hpp>
#include <dtl/thread.hpp>
#include <dtl/filter/blocked_bloomfilter/block_addressing_logic.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp>
#include <dtl/filter/blocked_bloomfilter/fpr.hpp>
#include <dtl/filter/cuckoofilter/cuckoofilter_config.hpp>

#include "timing.hpp"
#include "calibration_data.hpp"

namespace dtl {
namespace filter {
namespace model {


//===----------------------------------------------------------------------===//
class cost_fn {
  // TODO implement cost function for Cuckoo filter


  using bbf_config = dtl::blocked_bloomfilter_config;
  using cf_config = dtl::cuckoofilter::config;


  //===----------------------------------------------------------------------===//
  /// Compute the probability of a cache miss.
  /// Note: Filter and cache sizes must have the same unit. (Typically bits)
  static f64
  p_miss(u64 level, u64 filter_size, const std::vector<$u64>& cache_sizes) {
    if (level == 0) return 1.0;
    f64 cache_size = cache_sizes[level - 1];
    if (cache_size >= filter_size) return 0.0;
    f64 p_hit = cache_size / filter_size;
    return 1 - p_hit;
  };
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Sanitize the timings (e.g., L2 should not be faster than L1)
  static std::vector<timing>
  sanitize_timings(const std::vector<timing>& timings) {
    std::vector<timing> sanitized(timings.begin(), timings.end());
    for (std::size_t i = 1; i < sanitized.size(); i++) {
      if (sanitized[i].cycles_per_lookup < sanitized[i - 1].cycles_per_lookup) {
        // Simple fix
        sanitized[i] = sanitized[i - 1];
      }
    }
    return sanitized;
  };
  //===----------------------------------------------------------------------===//


public:

  // Pure static
  cost_fn() = delete;
  ~cost_fn() = delete;


  //===----------------------------------------------------------------------===//
  /// Compute the delta t_l values based on the measured lookup costs.
  /// delta t_{l,i} = t_{l,i} - delta t_{l,i-1} ; for 2 <= i <= # mem levels
  /// delta_t_{l,1} = t_{l,1}                   ; otherwise
  ///
  /// Note: Filter and cache sizes must have the same unit. (Typically bits)
  static std::vector<timing>
  compute_delta_tls(const std::vector<$u64>& cache_sizes,
                    const std::vector<$u64>& filter_sizes, /* the filter sizes used to determine the lookup costs for the individual mem levels */
                    const std::vector<timing>& timings /* the timings for the individual mem levels */
                    ) {
    u64 mem_levels = cache_sizes.size() + 1;

    // cache miss probability
    const auto p = [&](u64 level, u64 filter_size) {
      return p_miss(level, filter_size, cache_sizes);
    };

    // lookup costs (aka 't_l hat' in the paper)
    const auto tl = [&](u64 level) {
      if (level == 0) return timing();
      return timings[level - 1];
    };

    // filter size (aka 'm hat' in the paper)
    const auto m = [&](u64 level) {
      assert(level > 0);
      assert(level <= filter_sizes.size());
      return filter_sizes[level - 1];
    };

    std::vector<timing> deltas;

    std::function<timing(u64, u64)> delta_rec = [&](u64 level, u64 filter_size) {
      if (level == 0) return timing();
      return deltas[level - 1] * p(level - 1, filter_size) + delta_rec(level - 1, filter_size);
    };

    // delta_t_{l,1} = t_{l,1}
    deltas.push_back(tl(1)); // Level 1
    for (std::size_t mem_level = 2; mem_level <= mem_levels; mem_level++) {
      u64 filter_size = m(mem_level);
      timing d = (tl(mem_level) - delta_rec(mem_level - 1, filter_size)) / p(mem_level - 1, filter_size);
      assert(d.cycles_per_lookup >= 0.0 && d.nanos_per_lookup >= 0.0);
      deltas.push_back(d);
    }
    return deltas;
  };
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Compute the estimated t_lookup for the given filter size m
  /// based on the delta t_l values and the cache sizes.
  static timing
  lookup_costs(u64 m,
               const std::vector<$u64>& cache_sizes,
               const std::vector<timing>& delta_tls) {
    u64 mem_levels = cache_sizes.size() + 1;
    timing res;
    for (std::size_t i = 1; i <= mem_levels; i++) {
      auto d = delta_tls[i - 1];
      auto p = p_miss(i - 1, m, cache_sizes);
      res += d * p;
    }
    return res;
  };
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Compute the estimated lookup costs for the given filter configuration.
  static timing
  lookup_costs(u64 m, const bbf_config& c, const calibration_data& calibration_data) {
    std::vector<timing> delta_timings = calibration_data.get_timings(c);
    if (delta_timings.empty() || delta_timings[0].cycles_per_lookup == 0.0 || delta_timings[0].nanos_per_lookup == 0.0) {
      throw std::runtime_error("Failed to calculate lookup costs due to unknown filter configuration.");
    }
    auto filter_size_bytes = m / 8;
    auto t_lookup = lookup_costs(filter_size_bytes, calibration_data.get_cache_sizes(), delta_timings);
    return t_lookup;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Compute the estimated overhead for the given filter configuration.
  static timing
  overhead(u64 n, const timing& tw,
           u64 m, const bbf_config& c,
           const calibration_data& calibration_data) {
    const auto f = dtl::bloomfilter::fpr(m, n, c);
    const timing tl = lookup_costs(m, c, calibration_data);
    const timing overhead = tl + (tw * f);
    return overhead;
  }
  //===----------------------------------------------------------------------===//


};
//===----------------------------------------------------------------------===//


} // namespace model
} // namespace filter
} // namespace dtl
