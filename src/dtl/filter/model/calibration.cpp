#include <algorithm>
#include <set>
#include <vector>

#include <dtl/dtl.hpp>

#include "benchmark.hpp"
#include "calibration.hpp"
#include "cost_fn.hpp"
#include "util.hpp"

#include "../bbf_32.hpp"
#include "../bbf_64.hpp"
#include "../cf.hpp"
#include "../zbbf_32.hpp"
#include "../zbbf_64.hpp"

namespace dtl {
namespace filter {
namespace model {

// TODO cleanup

//===----------------------------------------------------------------------===//
void calibration::calibrate_tuning_params() {
  std::cout << "Determining the best performing filter implementations:" << std::endl;
  std::cout << " - Benchmarking blocked Bloom filter (32-bit words)" << std::endl;
  bbf_32::calibrate();
//  std::cout << " - Benchmarking blocked Bloom filter (64-bit words)" << std::endl;
//  bbf_64::calibrate();
  std::cout << " - Benchmarking zoned blocked Bloom filter (32-bit words)" << std::endl;
  zbbf_32::calibrate();
//  std::cout << " - Benchmarking zoned blocked Bloom filter (64-bit words)" << std::endl;
//  zbbf_64::calibrate();
  std::cout << " - Benchmarking Cuckoo filter" << std::endl;
  cf::calibrate();
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
void
calibration::calibrate_cache_sizes() {
  // in general: unit of size = [bit]

  benchmark benchmark_runner;
  u64 thread_cnt = 0; // 0 = let the platform instance decide

  // TODO warm up CPU before the actual calibration starts (and try to detect fluctuations).

  std::cout << "Determining the effective cache sizes:" << std::endl;

  // The configuration used to determine the cache sizes.
  blocked_bloomfilter_config bbf_config;
  bbf_config.k = 8;
  bbf_config.addr_mode = dtl::block_addressing::MAGIC;

  std::cout << " - Collecting performance data for varying filter sizes" << std::endl;
  std::vector<$u64> filter_sizes;
  std::vector<timing> filter_lookup_costs;
  for ($u64 m_desired = 8ull * 1024 * 8; m_desired < 256ull * 1024 * 1024 * 8; m_desired *= 1.0625) {
    std::cout << "." << std::flush;
    u64 m = dtl::is_power_of_two(m_desired) ? m_desired + 8 : m_desired; // enforce Magic Modulo
    const auto t = benchmark_runner(bbf_config, m, thread_cnt);
    filter_sizes.push_back(m);
    filter_lookup_costs.push_back(t);
  }
  std::cout << std::endl;

  // second run
  for ($u64 i = 0; i < filter_sizes.size(); i++) {
    std::cout << "." << std::flush;
    auto m = filter_sizes[i];
    auto t = benchmark_runner(bbf_config, m, thread_cnt);
    if (t < filter_lookup_costs[i]) {
      filter_lookup_costs[i] = t;
    }
  }
  std::cout << std::endl;

  // refinement run(s)
  for ($u64 r = 0; r < 10; r++) {
    std::set<$u64> rerun;
    for ($u64 i = 1; i < filter_lookup_costs.size(); i++) {
      if (filter_lookup_costs[i] < filter_lookup_costs[i - 1]) {
        rerun.insert(i - 1);
      }
    }
    if (rerun.size() == 0) break;

    for ($u64 i : rerun) {
      std::cout << "." << std::flush;
      auto m = filter_sizes[i];
      auto t = benchmark_runner(bbf_config, m, thread_cnt);
      if (t < filter_lookup_costs[i]) {
        filter_lookup_costs[i] = t;
      }
    }
    std::cout << std::endl;
  }

  // TODO sanitize timings

  std::cout << " - Read (physical) cache sizes" << std::endl;
  const auto& platform = dtl::filter::platform::get_instance();
  std::vector<$u64> cache_sizes;
  for (auto s : platform.get_cache_sizes()) {
    // convert size from bytes to bits
    cache_sizes.push_back(s * 8);
  }
  const auto cache_levels = cache_sizes.size();
  const auto mem_levels = cache_sizes.size() + 1;
  for (std::size_t i = 0; i < cache_levels; i++) {
    std::cout << "L" << (i + 1) << "=" << (cache_sizes[i] / 8 / 1024) << " KiB" << std::endl;
  }

  std::cout << " - Determining effective cache sizes" << std::endl;

  //===----------------------------------------------------------------------===//
  // the function to minimize
  auto compute_deviation = [&](const std::vector<$u64>& cache_sizes,
                               const std::vector<$u64>& reference_filter_sizes, // one per mem level
                               const std::vector<timing>& reference_filter_lookup_costs // the corresponding lookup costs
                              ) {
    // compute the delta lookup costs (delta tl)
    const auto delta_tls = cost_fn::compute_delta_tls(cache_sizes, reference_filter_sizes, reference_filter_lookup_costs);

    // compute the deviation of the modelled and actual lookup costs (determined earlier).
    $f64 sum = 0.0;
    for (std::size_t i = 0; i < filter_sizes.size(); i++) {
      const auto tl_actual = filter_lookup_costs[i];
      const auto tl_model = cost_fn::lookup_costs(filter_sizes[i], cache_sizes, delta_tls);
      sum += std::abs(100.0 - ((100.0 / tl_actual.cycles_per_lookup) * tl_model.cycles_per_lookup));
    }
    return sum;
  };
  //===----------------------------------------------------------------------===//


  // pick reference points for each memory level
  std::vector<$u64> reference_filter_sizes;
  std::vector<timing> reference_filter_lookup_costs;
  // ... cache levels
  for (std::size_t current_cache_level = 0; current_cache_level < cache_levels; current_cache_level++) {

    auto desired_filter_size = cache_sizes[current_cache_level] / 2;
    for (std::size_t i = 0; i < current_cache_level; i++) {
      desired_filter_size += cache_sizes[i];
    }

    auto search = std::lower_bound(filter_sizes.begin(), filter_sizes.end(), desired_filter_size);
    if (search == filter_sizes.end()) {
      throw std::runtime_error("Failed to pick a reference point for L" + std::to_string(current_cache_level + 1) + ".");
    }
    auto idx = std::distance(filter_sizes.begin(), search);
    if (filter_sizes[idx] == cache_sizes[current_cache_level]
        && idx > 0) {
      idx--;
    }
    reference_filter_sizes.push_back(filter_sizes[idx]);
    reference_filter_lookup_costs.push_back(filter_lookup_costs[idx]);
  }
  // ... RAM
  reference_filter_sizes.push_back(filter_sizes.back());
  reference_filter_lookup_costs.push_back(filter_lookup_costs.back());
  std::cout << "Reference filter sizes:" << std::endl;
  for (std::size_t i = 0; i < mem_levels; i++) {
    std::cout << (reference_filter_sizes[i] / 1024 / 8) << " KiB" << std::endl;
  }


  std::vector<$u64> adjusted_cache_sizes = cache_sizes;
  for (std::size_t i = 0; i < 10; i++) {
    std::cout << "run " << i << ":" << std::endl;
    for (std::size_t current_cache_level = 0;
         current_cache_level < cache_sizes.size();
         current_cache_level++) {
      const auto step_size = cache_sizes[current_cache_level] / 1000;
      std::cout << "adjusting cache size a level " << (current_cache_level + 1)
                << ". step size = " << step_size << " [bytes]" << std::endl;

      $f64 min_dev = compute_deviation(adjusted_cache_sizes, reference_filter_sizes, reference_filter_lookup_costs);
      $u64 min_dev_cache_size = adjusted_cache_sizes[current_cache_level];

      while (adjusted_cache_sizes[current_cache_level] > step_size
          && adjusted_cache_sizes[current_cache_level] + step_size > filter_sizes[current_cache_level]) {
        adjusted_cache_sizes[current_cache_level] -= step_size;
        const auto d = compute_deviation(adjusted_cache_sizes, reference_filter_sizes, reference_filter_lookup_costs);
        if (d <= min_dev) {
          min_dev = d;
          min_dev_cache_size = adjusted_cache_sizes[current_cache_level];
        }
      }
      adjusted_cache_sizes[current_cache_level] = min_dev_cache_size;

      std::cout << "L" << (current_cache_level + 1) << ": "
                << (cache_sizes[current_cache_level] / 1024 / 8) << " KiB -> "
                << (adjusted_cache_sizes[current_cache_level] / 1024 / 8) << " KiB" << std::endl;
    }
  }

  std::cout << "Summary:" << std::endl;
  for (std::size_t i = 0; i < cache_levels; i++) {
    std::cout << " " << (cache_sizes[i] / 1024 / 8) << " KiB -> " << (adjusted_cache_sizes[i] / 1024 / 8) << " KiB" << std::endl;
  }

  // TODO remove
  std::cout << "CSV data:" << std::endl;
  std::cout << "m,tl_actual,tl_model" << std::endl;
  for (std::size_t i = 0; i < filter_sizes.size(); i++) {
    // compute the delta lookup costs (delta tl)
    const auto delta_tls = cost_fn::compute_delta_tls(adjusted_cache_sizes, reference_filter_sizes, reference_filter_lookup_costs);
    const auto tl_model = cost_fn::lookup_costs(filter_sizes[i], adjusted_cache_sizes, delta_tls);
    std::cout << filter_sizes[i]
              << "," << filter_lookup_costs[i].cycles_per_lookup
              << "," << tl_model.cycles_per_lookup
              << std::endl;
  }

  // Memorize the calibrated cache sizes in the calibration data object.
  data_.set_cache_sizes(adjusted_cache_sizes);

  // Pick reference filter sizes, that are powers of two.
  auto reference_filter_sizes_pow2 = reference_filter_sizes;
  for (std::size_t l = 0; l < cache_levels; l++) {
    if (dtl::is_power_of_two(reference_filter_sizes_pow2[l])) continue;
    reference_filter_sizes_pow2[l] = dtl::next_power_of_two(reference_filter_sizes_pow2[l]);
    if (reference_filter_sizes_pow2[l] >= cache_sizes[l]) {
      reference_filter_sizes_pow2[l] >>= 1;
    }
    if (l > 0 && reference_filter_sizes_pow2[l] <= cache_sizes[l - 1]) {
      throw std::runtime_error("Failed to pick reference filter sizes.");
    }
  }
  if (!dtl::is_power_of_two(reference_filter_sizes_pow2[mem_levels - 1])) {
    reference_filter_sizes_pow2[mem_levels - 1] =
        std::min(u64(256ull * 1024 * 1024 * 8),
                 dtl::next_power_of_two(reference_filter_sizes_pow2[mem_levels - 1]));
  }
  std::cout << "Reference filter sizes:" << std::endl;
  for (std::size_t i = 0; i < mem_levels; i++) {
    std::cout << " " << (reference_filter_sizes_pow2[i] / 1024 / 8) << " KiB" << std::endl;
  }
  // Memorize the calibrated cache and filter sizes in the calibration data object.
  data_.set_filter_sizes(reference_filter_sizes_pow2);
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
void
calibration::calibrate_bbf_costs() {
  benchmark benchmark_runner;
  u32 thread_cnt = 0; // = let the platform instance decide
//  u32 thread_cnt = 1;

  const auto cache_sizes = data_.get_cache_sizes();
  std::cout << "Cache sizes:" << std::endl;
  for (std::size_t i = 0; i < cache_sizes.size(); i++) {
    std::cout << "L" << (i + 1) << ": " << (cache_sizes[i] / 1024 / 8) << " KiB" << std::endl;
  }

  const auto filter_sizes = data_.get_filter_sizes();
  std::cout << "Filter sizes:" << std::endl;
  for (std::size_t i = 0; i < filter_sizes.size(); i++) {
    std::cout << "Level " << (i + 1) << ": " << (filter_sizes[i] / 1024 / 8) << " KiB" << std::endl;
  }

  std::vector<blocked_bloomfilter_config> configs = get_valid_bbf_configs();
  for (auto c : configs) {
    std::cout << "calibrating: " << c << std::endl;

    try {
      // determine t_lookup
      std::vector<timing> tls;
      for (auto m : filter_sizes) {
        assert(dtl::is_power_of_two(m));
        $u64 m_act = (c.addr_mode == dtl::block_addressing::MAGIC) ? m + 128 // enforce MAGIC modulo
                                                                   : m;
        const auto tl = benchmark_runner(c, m_act, thread_cnt);
        tls.push_back(tl);
      }
      std::cout << "config=[" << c << "], tl=[" << tls[0].cycles_per_lookup;
      for (std::size_t i = 1; i < tls.size(); i++) {
        std::cout << "," << tls[i].cycles_per_lookup;
      }
      std::cout << "]" << std::endl;

      // compute delta t_lookup
      auto delta_tls = cost_fn::compute_delta_tls(cache_sizes, filter_sizes, tls);

      // memorize results
      data_.put_timings(c, delta_tls);
    }
    catch (std::exception& ex) {
      std::cout << "Failed: " << ex.what() << std::endl;
    }
  }
}

void
calibration::calibrate_cf_costs() {
  std::vector<dtl::cuckoofilter::config> configs = get_valid_cf_configs();
  // TODO implement calibration for cuckoo filter
}

void
calibration::calibrate_filter_costs() {
  calibrate_bbf_costs();
  calibrate_cf_costs();
}
//===----------------------------------------------------------------------===//


} // namespace model
} // namespace filter
} // namespace dtl
