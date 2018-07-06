#pragma once

#include <chrono>
#include <map>
#include <random>

#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/block_addressing_logic.hpp>

#include "immintrin.h"

#include "cuckoofilter.hpp"
#include "cuckoofilter_tune.hpp"


namespace dtl {
namespace cuckoofilter {


namespace internal {

  struct cuckoofilter_tune_mock : cuckoofilter_tune {

    u32 unroll_factor;

    cuckoofilter_tune_mock(u32 unroll_factor) : unroll_factor(unroll_factor) {};

    $u32
    get_unroll_factor(u32 bits_per_tag,
                      u32 tags_per_bucket,
                      dtl::block_addressing addr_mode) const override {
      return unroll_factor;
    }
  };

} // namespace internal

//===----------------------------------------------------------------------===//
struct cuckoofilter_tune_impl : cuckoofilter_tune {


  static constexpr u32 max_k = 16;
  static constexpr u32 max_unroll_factor = 8;


  std::map<config, tuning_params>
  m;


  void
  set_unroll_factor(u32 bits_per_tag,
                    u32 tags_per_bucket,
                    dtl::block_addressing addr_mode,
                    u32 unroll_factor) override {
    config c { bits_per_tag, tags_per_bucket, addr_mode};
    tuning_params tp { unroll_factor };
    m.insert(std::pair<config, tuning_params>(c, tp));
  }


  $u32
  get_unroll_factor(u32 bits_per_tag,
                    u32 tags_per_bucket,
                    dtl::block_addressing addr_mode) const override {
    config c { bits_per_tag, tags_per_bucket, addr_mode};
    if (m.count(c)) {
      auto p = m.find(c)->second;
      return p.unroll_factor;
    }
    return 1; // defaults to SIMD implementation
  }


  $u32
  tune_unroll_factor(u32 bits_per_tag,
                     u32 tags_per_bucket,
                     dtl::block_addressing addr_mode) override {

    config c { bits_per_tag, tags_per_bucket, addr_mode};
    tuning_params tp = calibrate(c);
    m.insert(std::pair<config, tuning_params>(c, tp));
    return tp.unroll_factor;
  }


  void
  tune_unroll_factor() {
    for ($u32 bits_per_tag : { 4u, 8u, 12u, 16u, 32u } ) {
      for ($u32 tags_per_bucket : { 1u, 2u, 4u } ) {
        for (auto addr_mode : {dtl::block_addressing::POWER_OF_TWO, dtl::block_addressing::MAGIC}) {
          tune_unroll_factor(bits_per_tag, tags_per_bucket, addr_mode);
        }
      }
    }
  }


  //===----------------------------------------------------------------------===//
  tuning_params
  calibrate(const config& c) {
    using key_t = $u32;
    using word_t = $u32;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis;

    static constexpr u32 data_size = 4u*1024*8;
    std::vector<key_t> random_data;
    random_data.reserve(data_size);
    for (std::size_t i = 0; i < data_size; i++) {
      random_data.push_back(dis(gen));
    }

    try {
      std::cerr << "t = " << std::setw(2) << c.bits_per_tag << ", "
                << "b = " << std::setw(2) << c.tags_per_bucket << ", "
                << "addr = " << std::setw(5) << (c.addr_mode == dtl::block_addressing::POWER_OF_TWO ? "pow2" : "magic")
                << ": " << std::flush;

      $f64 cycles_per_lookup_min = std::numeric_limits<$f64>::max();
      $u32 u_min = 1;

      std::size_t match_count = 0;
      uint32_t match_pos[dtl::BATCH_SIZE];

      // base lines
      $f64 cycles_per_lookup_scalar = 0.0;
      $f64 cycles_per_lookup_unroll_by_one = 0.0;

      for ($u32 u = 0; u <= max_unroll_factor; u = (u == 0) ? 1 : u * 2) {
        std::cerr << std::setw(2) << "u(" << std::to_string(u) + ") = "<< std::flush;

        u64 desired_filter_size_bits = 4ull * 1024 * 8;
        const std::size_t m = desired_filter_size_bits
            + (128 * static_cast<u32>(c.addr_mode)); // enforce MAGIC addressing

        // Instantiate bloom filter logic.
        internal::cuckoofilter_tune_mock tune_mock { u };
        cuckoofilter cf(m , c.bits_per_tag, c.tags_per_bucket, tune_mock);

        // Allocate memory.
        dtl::mem::allocator_config alloc_config = dtl::mem::allocator_config::local();
        if (dtl::mem::hbm_available()) {
          // Use HBM if available
          const auto cpu_id = dtl::this_thread::get_cpu_affinity().find_first();
          const auto node_id = dtl::mem::get_node_of_cpu(cpu_id);
          const auto hbm_node_id = dtl::mem::get_nearest_hbm_node(node_id);
          alloc_config = dtl::mem::allocator_config::on_node(hbm_node_id);
        }

        dtl::mem::numa_allocator<word_t> alloc(alloc_config);
        std::vector<word_t, dtl::mem::numa_allocator<word_t>>
            filter_data(cf.size(), 0, alloc); // TODO how to pass different allocator (for KNL/HBM)

        // Note: No need to insert elements, as the BBF is branch-free.

        // Run the micro benchmark.
        $u64 rep_cntr = 0;
        auto start = std::chrono::high_resolution_clock::now();
        auto tsc_start = _rdtsc();
        while (true) {
          dtl::batch_wise(random_data.begin(), random_data.end(),
                          [&](const auto batch_begin, const auto batch_end) {
                            match_count += cf.batch_contains(&filter_data[0], &batch_begin[0], batch_end - batch_begin, match_pos, 0);
                          });

          rep_cntr++;
          std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - start;
          if (diff.count() > 0.25) break; // Run micro benchmark for at least 250ms.
        }
        auto tsc_end = _rdtsc();

        auto cycles_per_lookup = (tsc_end - tsc_start) / (data_size * rep_cntr * 1.0);

        if (u == 0) cycles_per_lookup_scalar = cycles_per_lookup;
        if (u == 1) cycles_per_lookup_unroll_by_one = cycles_per_lookup;
        std::cerr << std::setprecision(3) << std::setw(4) << std::right << cycles_per_lookup << ", ";
        if (cycles_per_lookup < cycles_per_lookup_min) {
          cycles_per_lookup_min = cycles_per_lookup;
          u_min = u;
        }
      }
      std::cerr << " picked u = " << u_min << " (" << cycles_per_lookup_min << " [cycles/lookup])"
                << ", speedup over u(0) = " << std::setprecision(3) << std::setw(4) << std::right << (cycles_per_lookup_scalar / cycles_per_lookup_min)
                << ", speedup over u(1) = " << std::setprecision(3) << std::setw(4) << std::right << (cycles_per_lookup_unroll_by_one / cycles_per_lookup_min)
                << " (chksum: " << match_count << ")" << std::endl;

      return tuning_params { u_min };

    } catch (...) {
      std::cerr<< " -> Failed to calibrate for t=" << c.bits_per_tag << " and b=" << c.tags_per_bucket << "." << std::endl;
    }
    return tuning_params { 1 };
  }
  //===----------------------------------------------------------------------===//

};

} // namespace cuckoofilter
} // namespace dtl