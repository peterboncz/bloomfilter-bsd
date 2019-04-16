#include <algorithm>
#include <functional>
#include <set>
#include <iomanip>
#include <thread>

#include <dtl/dtl.hpp>
#include <dtl/env.hpp>

#include <amsfilter/amsfilter.hpp>
#include <amsfilter/cuda/internal/cuda_api_helper.hpp>
#include <amsfilter/internal/blocked_bloomfilter_template.hpp>
#include <amsfilter_model/internal/benchmark.hpp>
#include <amsfilter_model/internal/benchmark_cpu.hpp>
#include <amsfilter_model/internal/benchmark_gpu.cuh>
#include <amsfilter_model/internal/benchmark_pci_bw.cuh>
#include <amsfilter_model/internal/tune.hpp>
#include <amsfilter_model/internal/util.hpp>
#include <amsfilter_model/internal/perf_db.hpp>
#include <amsfilter_model/execution_env.hpp>
#include <amsfilter_model/internal/skyline.hpp>

//===----------------------------------------------------------------------===//
// Min/Max filter size in bits.
u64 m_min = u64(16) * 1024 * 8;
u64 m_max = u64(512ull) * 1024 * 1024 * 8;
//===----------------------------------------------------------------------===//
std::set<$u64>
get_filter_sizes() {
  // All valid filter sizes.
  const std::set<$u64> m_s = [&]() {
    std::set<$u64> m_s;
    // Power of two filter sizes.
    for ($u64 m = dtl::next_power_of_two(m_min);
         m <= dtl::next_power_of_two(m_max); m *= 2) {
      m_s.insert(m);
    }
//    // Magic filter sizes.
//    for ($u64 m = dtl::next_power_of_two(m_lo); m <= dtl::next_power_of_two(m_hi); m *= 2) {
//      u64 num_inter_steps = 3;
//      u64 step = m / (num_inter_steps + 1);
//      for ($u64 i = 1; i <= num_inter_steps; i++) {
//        if ((m + i * step) <= m_hi) m_s.insert(m + i * step);
//      }
//    }
    return m_s;
  }();
  return m_s;
}
//===----------------------------------------------------------------------===//
std::string
bitlength_human_readable(const std::size_t n) {
  std::stringstream out;
  u64 GiB = 1ull * 1024 * 1024 * 1024 * 8;
  u64 MiB = 1ull * 1024 * 1024 * 8;
  u64 KiB = 1ull * 1024 * 8;
  u64 B   = 1ull * 8;
  if (n >= GiB) {
    out << ((n * 1.0) / GiB) << " GiB";
  }
  else if (n >= MiB) {
    out << ((n * 1.0) / MiB) << " MiB";
  }
  else if (n >= KiB) {
    out << ((n * 1.0) / KiB) << " KiB";
  }
  else {
    out << ((n * 1.0) / B) << " B";
  }
  return out.str();
}
//===----------------------------------------------------------------------===//
/// Measure lookup costs and store it in the database.  The filter sizes are
/// chosen adaptively. Only, if the difference between to data points is too
/// large, intermediate filter sizes are measured.  This significantly reduces
/// the calibration time and improves the models' accuracy, when linear
/// interpolation is used.
void
measure_lookup_costs_recursive(
    amsfilter::model::PerfDb& perf_db,
    const amsfilter::Config& config,
    const amsfilter::model::Env& env,
    amsfilter::model::benchmark& benchmark,
    f64 e) {

  const amsfilter::TuningParams tuning_params =
      perf_db.get_tuning_params(config);

  std::size_t performed_pow2_measurements_cntr = 0;
  std::size_t performed_magic_measurements_cntr = 0;

  // The criterion for recursion.
  auto needs_refinement = [&](const auto tl_lo, const auto tl_hi) {
    const auto delta_tl = tl_hi - tl_lo;
    const auto threshold_nanos =
        tl_lo.nanos_per_lookup * e; // <<-- "max error"
    if (delta_tl.nanos_per_lookup > threshold_nanos) {
      return true;
    }
    return false;
  };

  // Measure performance for the given filter size and store the results.
  auto measure = [&](const std::size_t m) {
    std::cout << " measuring m=" << std::setw(14) << bitlength_human_readable(m)
        << " " << (dtl::is_power_of_two(m) ? "(pow2) " : "(magic)") << " ";
    auto result = benchmark.run(config, m, env, tuning_params);
    std::cout << " - Throughput: "
        << std::setw(8)
        << (1.0 / result.nanos_per_lookup * 1e9) / 1024 / 1024
        << " Mtps" << std::endl;
    perf_db.put_tl(config, m, env, result);
    if (dtl::is_power_of_two(m)) {
      performed_pow2_measurements_cntr++;
    }
    else {
      performed_magic_measurements_cntr++;
    }
  };

  std::function<void(std::size_t, std::size_t)> rec_magic =
      [&](const std::size_t m_lo, const std::size_t m_hi) {
    if (config.addr_mode != dtl::block_addressing::MAGIC) {
      return;
    }
    if ((m_hi - m_lo) < 1ull * 1024 * 8 ) {
      return;
    }
    // Measure the lookup performance for the two given filter sizes.
    if (!perf_db.has_tl(config, m_lo, env)) {
      measure(m_lo);
    }
    if (!perf_db.has_tl(config, m_hi, env)) {
      measure(m_hi);
    }

    auto tl_lo = perf_db.get_tl(config, m_lo, env);
    auto tl_hi = perf_db.get_tl(config, m_hi, env);

    // Sanitize results. tl_lo <= tl_hi must hold.
    if (tl_lo > tl_hi) {
      // Re-run
      measure(m_lo);
      tl_lo = perf_db.get_tl(config, m_lo, env);
      if (tl_lo > tl_hi) {
        // Replicate results.
        tl_lo = tl_hi;
        perf_db.put_tl(config, m_lo, env, tl_lo);
      }
    }

    if (needs_refinement(tl_lo, tl_hi)) {
      // Recurse
      auto m_mid = m_lo + ((m_hi - m_lo) / 2);
      if (m_mid != m_lo && m_mid != m_hi) {
        rec_magic(m_lo, m_mid);
        rec_magic(m_mid, m_hi);
      }
    }
  };

  std::function<void(std::size_t, std::size_t)> rec_pow2 =
      [&](const std::size_t m_lo, const std::size_t m_hi) {
    // Measure the lookup performance for the two given filter sizes.
    if (!perf_db.has_tl(config, m_lo, env)) {
      measure(m_lo);
    }
    if (!perf_db.has_tl(config, m_hi, env)) {
      measure(m_hi);
    }

    auto tl_lo = perf_db.get_tl(config, m_lo, env);
    auto tl_hi = perf_db.get_tl(config, m_hi, env);

    // Sanitize results. tl_lo <= tl_hi must hold.
    if (tl_lo > tl_hi) {
      // Re-run
      measure(m_lo);
      tl_lo = perf_db.get_tl(config, m_lo, env);
      if (tl_lo > tl_hi) {
        // Replicate results.
        tl_lo = tl_hi;
        perf_db.put_tl(config, m_lo, env, tl_lo);
      }
    }

    if (needs_refinement(tl_lo, tl_hi)) {
      // Recurse
      auto m_lo_log2 = dtl::log_2(m_lo);
      auto m_hi_log2 = dtl::log_2(m_hi);
      auto m_mid_log2 = m_lo_log2 + ((m_hi_log2 - m_lo_log2) / 2);
      if (m_mid_log2 == m_lo_log2) {
        m_mid_log2++;
      }
      if (m_mid_log2 == m_hi_log2) {
        // Continue with MAGIC addressing.
        const auto m_mid = m_lo + ((m_hi - m_lo) / 2);
        rec_magic(m_lo, m_mid);
        rec_magic(m_mid, m_hi);
      }
      else {
        const auto m_mid = 1ull << m_mid_log2;
        rec_pow2(m_lo, m_mid);
        rec_pow2(m_mid, m_hi);
      }
    }
  };

  rec_pow2(m_min, m_max);
  std::cout << "# of measurements: "
      << performed_pow2_measurements_cntr << " (pow2) + "
      << performed_magic_measurements_cntr << " (magic) = "
      << performed_pow2_measurements_cntr + performed_magic_measurements_cntr
      << " (total)" << std::endl;
}
//===----------------------------------------------------------------------===//
/// Sanitize timings. Ensure that tl(m) is monotonically increasing.
void
sanitize_lookup_costs(
    amsfilter::model::PerfDb& perf_db,
    const amsfilter::Config& config,
    const amsfilter::model::Env& env,
    amsfilter::model::benchmark& benchmark) {

  const std::vector<std::size_t> filter_sizes = perf_db.get_filter_sizes(
      config, env);
  std::cout << "# of reference measurements: " << filter_sizes.size()
      << std::endl;

  for (std::size_t i = filter_sizes.size() - 1; i > 1; --i) {
    const auto curr_m = filter_sizes[i];
    const auto prec_m = filter_sizes[i - 1];

    const auto curr_result = perf_db.get_tl(config, curr_m, env);
    const auto prec_result = perf_db.get_tl(config, prec_m, env);

    if (prec_result > curr_result) {
//      std::cout << "Re-run" << std::endl;
      // Re-run the measurement with the preceding setting.
      const auto tuning_params = perf_db.get_tuning_params(config);
      auto refined_result = benchmark.run(config, prec_m, env, tuning_params);

      if (refined_result < curr_result) {
        // Write the refined result.
        perf_db.put_tl(config, prec_m, env, refined_result);
      }
      else {
//        std::cout << " Replicate" << std::endl;
        // Replicate the current result (rather than running the measurement
        // again).
        perf_db.put_tl(config, prec_m, env, curr_result);
      }
    }
  }
}
//===----------------------------------------------------------------------===//
$i32
main() {
  // The number of threads to use.
  u32 thread_cnt = dtl::env<$u32>::get("AMSFILTER_THREAD_CNT",
      std::thread::hardware_concurrency() / 2);

  f64 max_error_coarse_calibration = dtl::env<$f64>::get(
      "AMSFILTER_MAX_ERROR_COARSE", 0.25);

  f64 max_error_refine_calibration = dtl::env<$f64>::get(
      "AMSFILTER_MAX_ERROR_REFINE", 0.1);

  // Where to store the performance data.
  const auto perf_db_filename = dtl::env<std::string>::get("AMSFILTER_DB_FILE",
      amsfilter::model::PerfDb::get_default_filename());
  auto perf_db_ptr =
      std::make_shared<amsfilter::model::PerfDb>(perf_db_filename);
  amsfilter::model::PerfDb& perf_db = *perf_db_ptr; // TODO remove

  std::cout << "AMS-Filter calibration tool" << std::endl;
  std::cout << "SIMD architecture = " << amsfilter::simd_arch << std::endl;
  std::cout << "Database file = " << perf_db_filename << std::endl;

  auto configs = amsfilter::model::get_valid_configs();
  std::sort(configs.begin(), configs.end());
  std::cout << "Number of valid configurations = " << configs.size() << std::endl;

  //===--------------------------------------------------------------------===//
  // Determine the best performing SIMD unrolling factors.
  //===--------------------------------------------------------------------===//
  if (true) {
    std::cout << "- Calibrating SIMD unrolling factor:" << std::endl;
    std::size_t tunings_performed = 0;
    for (auto& c : configs) {
      if (!perf_db.has_tuning_params(c)) {
        auto tuning_params = amsfilter::model::tune(c);
        perf_db.put_tuning_params(c, tuning_params);
        ++tunings_performed;
      }
    }
    std::cout << "Done. "
        << (configs.size() - tunings_performed)
        << " configurations were already calibrated before."
        << std::endl;
  }
  // TODO write to binary file


  // The filter sizes for which we perform measurements.
  const auto filter_sizes_set = get_filter_sizes();
  std::vector<std::size_t> filter_sizes_magic(
      filter_sizes_set.begin(), filter_sizes_set.end());
  std::vector<std::size_t> filter_sizes_pow2;
  for (auto m : filter_sizes_set) {
    if (dtl::is_power_of_two(m)) filter_sizes_pow2.push_back(m);
  }

  //===--------------------------------------------------------------------===//
  // Measure the lookup costs on the CPU. (Coarse grained)
  //===--------------------------------------------------------------------===//
  if (true) {
    std::cout << "- Measuring lookup costs (on CPU):" << std::endl;
    const auto env = amsfilter::model::Env::cpu(thread_cnt);
    amsfilter::model::benchmark_cpu benchmark;

    for (auto& config : configs) {
      std::cout << "config = " << config << std::endl;
      // Measure the lookup costs.
      measure_lookup_costs_recursive(perf_db, config, env, benchmark,
          max_error_coarse_calibration);
      // Ensure the lookup costs are monotonically increasing.
      sanitize_lookup_costs(perf_db, config, env, benchmark);
    }
  }

  //===--------------------------------------------------------------------===//
  // Measure the lookup costs on the GPU. (Coarse grained)
  //===--------------------------------------------------------------------===//
  if (true) {
    const auto device_cnt = amsfilter::cuda::get_cuda_device_count();
    for (auto device = 0u; device < device_cnt; ++device) {
      std::cout
          << "- Measuring lookup costs (on GPU '"
          << amsfilter::cuda::get_cuda_device_name(device) << "'):"
          << std::endl;

      const auto env = amsfilter::model::Env::gpu_keys_in_device_memory(device);
      amsfilter::model::benchmark_gpu benchmark;

      for (auto& config : configs) {
        std::cout << "config = " << config << std::endl;
        // Measure the lookup costs.
        measure_lookup_costs_recursive(perf_db, config, env, benchmark,
            max_error_coarse_calibration);
        // Ensure the lookup costs are monotonically increasing.
        sanitize_lookup_costs(perf_db, config, env, benchmark);
      }

    }
  }

  //===--------------------------------------------------------------------===//
  // Measure the PCIe bandwidth, which limits the lookup throughput.
  //===--------------------------------------------------------------------===//
  if (true) {
    const auto device_cnt = amsfilter::cuda::get_cuda_device_count();
    for (auto device = 0u; device < device_cnt; ++device) {
      std::cout
          << "- Measuring the PCIe bandwidth between host and GPU '"
          << amsfilter::cuda::get_cuda_device_name(device) << "':"
          << std::endl;
      amsfilter::model::benchmark_pci_bw benchmark;

      const auto pageable_mem =
          amsfilter::model::Env::gpu_keys_in_pageable_memory(device);
      const auto throughput_pageable_mem = benchmark.run(pageable_mem);

      const auto pinned_mem =
          amsfilter::model::Env::gpu_keys_in_pinned_memory(device);
      const auto throughput_pinned_mem = benchmark.run(pinned_mem);

      amsfilter::model::timing t_pageable;
      t_pageable.nanos_per_lookup = 1e9 // nanos per key
          / ((std::get<0>(throughput_pageable_mem) * 1024 * 1024) // bytes per sec
          / sizeof(amsfilter::key_t)); // keys per sec
      t_pageable.cycles_per_lookup = 1 // cycles per key
          / ((std::get<1>(throughput_pageable_mem) * 1024 * 1024) // bytes per cycles
          / sizeof(amsfilter::key_t)); // keys per cycle

          amsfilter::model::timing t_pinned;
      t_pinned.nanos_per_lookup = 1e9 // nanos per key
          / ((std::get<0>(throughput_pinned_mem) * 1024 * 1024) // bytes per sec
          / sizeof(amsfilter::key_t)); // keys per sec
      t_pinned.cycles_per_lookup = 1 // cycles per key
          / ((std::get<1>(throughput_pinned_mem) * 1024 * 1024) // bytes per cycles
          / sizeof(amsfilter::key_t)); // keys per cycle

      std::cout << "  from pageable memory: "
          << std::get<0>(throughput_pageable_mem) << " [MiB/s]"
          << " = " << t_pageable.nanos_per_lookup << " [ns/key]"
          << " = " << t_pageable.cycles_per_lookup << " [cycles/key]"
          << std::endl;
      std::cout << "  from pinned memory:   "
          << std::get<0>(throughput_pinned_mem) << " [MiB/s]"
          << " = " << t_pinned.nanos_per_lookup << " [ns/key]"
          << " = " << t_pinned.cycles_per_lookup << " [cycles/key]"
          << std::endl;

      perf_db.set_gpu_bandwidth_limit(pageable_mem, t_pageable);
      perf_db.set_gpu_bandwidth_limit(pinned_mem, t_pinned);

      // No limitations when the keys are already in device memory.
      const auto device_mem =
          amsfilter::model::Env::gpu_keys_in_device_memory(device);
      amsfilter::model::timing t_device;
      t_device.nanos_per_lookup = 0.0;
      t_device.cycles_per_lookup = 0.0;
      perf_db.set_gpu_bandwidth_limit(device_mem, t_device);
    }
  }

  //===--------------------------------------------------------------------===//
  // Refine the lookup costs on the CPU.
  //===--------------------------------------------------------------------===//
  if (true) {
    std::cout << "- Refining lookup costs (on CPU):" << std::endl;
    const auto env = amsfilter::model::Env::cpu(thread_cnt);
    amsfilter::model::benchmark_cpu benchmark;
    amsfilter::model::Skyline skyline(perf_db_ptr, env);
    auto candidate_configs = skyline.determine_candidate_filter_configurations();

    for (auto& config : candidate_configs) {
      std::cout << "config = " << config << std::endl;
      // Measure the lookup costs.
      measure_lookup_costs_recursive(perf_db, config, env, benchmark,
          max_error_refine_calibration);
      // Ensure the lookup costs are monotonically increasing.
      sanitize_lookup_costs(perf_db, config, env, benchmark);
    }
  }

  //===--------------------------------------------------------------------===//
  // Refine the lookup costs on the GPU.
  //===--------------------------------------------------------------------===//
  if (true) {
    const auto device_cnt = amsfilter::cuda::get_cuda_device_count();
    for (auto device = 0u; device < device_cnt; ++device) {
      std::cout
          << "- Refining lookup costs (on GPU '"
          << amsfilter::cuda::get_cuda_device_name(device) << "'):"
          << std::endl;

      const auto env = amsfilter::model::Env::gpu_keys_in_device_memory(device);
      amsfilter::model::benchmark_gpu benchmark;
      amsfilter::model::Skyline skyline(perf_db_ptr, env);
      auto candidate_configs = skyline.determine_candidate_filter_configurations();

      for (auto& config : candidate_configs) {
        std::cout << "config = " << config << std::endl;
        // Measure the lookup costs.
        measure_lookup_costs_recursive(perf_db, config, env, benchmark,
          max_error_refine_calibration);
        // Ensure the lookup costs are monotonically increasing.
        sanitize_lookup_costs(perf_db, config, env, benchmark);
      }

    }
  }

  std::cout << "Done with calibration." << std::endl;
  if (true) {
    std::cout << "- Computing the skyline matrix(es):" << std::endl;

    auto compute_skyline = [&](const amsfilter::model::Env& env) {
      std::cout << "  Execution environment: ";
      if (env.is_cpu()) {
        std::cout
            << "CPU, using " << env.get_thread_cnt() << " thread(s)"
            << std::endl;
      }
      else {
        std::cout
            << "GPU '"
            << amsfilter::cuda::get_cuda_device_name(env.get_device()) << "'"
            << " (device no.: " << env.get_device()
            << ", memory: ";
        switch (env.get_probe_key_location()) {
          case amsfilter::model::Memory::HOST_PAGEABLE:
            std::cout << "pageable";
            break;
          case amsfilter::model::Memory::HOST_PINNED:
            std::cout << "pinned";
            break;
          case amsfilter::model::Memory::DEVICE:
            std::cout << "device";
            break;
        }
        std::cout
            << ")"
            << std::endl;
      }

      amsfilter::model::Skyline skyline(perf_db_ptr, env);
      skyline.compute();
    };

    compute_skyline(amsfilter::model::Env::cpu(thread_cnt));
    const auto device_cnt = amsfilter::cuda::get_cuda_device_count();
    for (auto device = 0u; device < device_cnt; ++device) {
      compute_skyline(amsfilter::model::Env::gpu_keys_in_device_memory(
          device));
      compute_skyline(amsfilter::model::Env::gpu_keys_in_pinned_memory(
          device));
      compute_skyline(amsfilter::model::Env::gpu_keys_in_pageable_memory(
          device));
    }
  }

  std::cout << "Done." << std::endl;

  return 0;
}
//===----------------------------------------------------------------------===//
