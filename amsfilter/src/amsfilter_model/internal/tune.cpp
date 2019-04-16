#include <iomanip>
#include <limits>
#include <random>
#include <vector>

#include <amsfilter/amsfilter_lite.hpp>
#include <amsfilter/tuning_params.hpp>

#include <dtl/dtl.hpp>
#include <dtl/batchwise.hpp>
#include <dtl/mem.hpp>

#include "immintrin.h"
#include "tune.hpp"

namespace amsfilter {
namespace model {
//===----------------------------------------------------------------------===//
/// Runs a micro-benchmark to determine the tuning parameters.
TuningParams
tune(const Config& c) {

  // Generate random data to look up.
  static constexpr u32 to_lookup_cnt = 4u*1024*8;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis;

  std::vector<key_t> to_lookup;
  to_lookup.reserve(to_lookup_cnt);
  for (std::size_t i = 0; i < to_lookup_cnt; i++) {
    to_lookup.push_back(dis(gen));
  }

  try {
    std::cerr << "w = " << std::setw(2) << c.word_cnt_per_block << ", "
        << "s = " << std::setw(2) << c.sector_cnt << ", "
        << "z = " << std::setw(2) << c.zone_cnt << ", "
        << "addr = " << std::setw(5)
        << (c.addr_mode == dtl::block_addressing::POWER_OF_TWO
            ? "pow2"
            : "magic")
        << ", " << "k = " <<  std::setw(2) << c.k << ": " << std::flush;

    $f64 cycles_per_lookup_min = std::numeric_limits<$f64>::max();
    $u32 u_opt = 1;

    // The baselines.
    $f64 cycles_per_lookup_scalar = 0.0;
    $f64 cycles_per_lookup_unroll_by_one = 0.0;

    for ($u32 u = 0; u <= TuningParams::max_unroll_factor; u = (u == 0) ? 1 : u * 2) {
      std::cerr << std::setw(2) << "u(" << std::to_string(u) + ") = "<< std::flush;

      u64 desired_filter_size_bits = 4ull * 1024 * 8;
      u64 m = desired_filter_size_bits
          + (128 * static_cast<u32>(c.addr_mode)); // Enforce MAGIC addressing.

      // Instantiate the filter.
      // Note: No need to insert elements, as the filter is branch-free.
      AmsFilterLite filter(c, m);

      // Obtain a probe instance.
      TuningParams tuning_params;
      tuning_params.unroll_factor = u;
      auto probe = filter.batch_probe(dtl::BATCH_SIZE, tuning_params);

      // Run the micro benchmark.
      $u64 rep_cntr = 0;
      auto chrono_start = std::chrono::high_resolution_clock::now();
      auto tsc_start = _rdtsc();
      while (true) {
        dtl::batch_wise(to_lookup.begin(), to_lookup.end(),
            [&](const auto batch_begin, const auto batch_end) {
              probe(&batch_begin[0], batch_end - batch_begin);
            });

        rep_cntr++;
        std::chrono::duration<double> chrono_duration =
            std::chrono::high_resolution_clock::now() - chrono_start;

        // Run micro benchmark for at least 250ms.
        if (chrono_duration.count() > 0.25) break;
      }
      auto tsc_end = _rdtsc();

      auto cycles_per_lookup =
          (tsc_end - tsc_start) / (to_lookup_cnt * rep_cntr * 1.0);

      if (u == 0) cycles_per_lookup_scalar = cycles_per_lookup;
      if (u == 1) cycles_per_lookup_unroll_by_one = cycles_per_lookup;
      std::cerr << std::setprecision(3) << std::setw(4) << std::right
          << cycles_per_lookup << ", ";
      if (cycles_per_lookup < cycles_per_lookup_min) {
        cycles_per_lookup_min = cycles_per_lookup;
        u_opt = u;
      }
    }
    std::cerr << " picked u = " << u_opt
        << " (" << std::setprecision(3) << std::setw(4) << cycles_per_lookup_min
        << " [cycles/lookup])"
        << ", speedup over u(0) = " << std::setprecision(3) << std::setw(4)
        << std::right << (cycles_per_lookup_scalar / cycles_per_lookup_min)
        << ", speedup over u(1) = " << std::setprecision(3) << std::setw(4)
        << std::right << (cycles_per_lookup_unroll_by_one / cycles_per_lookup_min)
        << std::endl;

    // Return results.
    TuningParams tuning_params;
    tuning_params.unroll_factor = u_opt;
    return tuning_params;

  } catch (...) {
    std::cerr << " -> Failed to calibrate: " << c << "." << std::endl;
  }
  throw std::invalid_argument("Calibration failed.");
}
//===----------------------------------------------------------------------===//
} // namespace model
} // namespace amsfilter
