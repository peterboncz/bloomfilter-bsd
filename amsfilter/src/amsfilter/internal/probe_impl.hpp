#pragma once

#define AMSFILTER_PROBE_EXTERN_TEMPLATES

#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_batch_probe.hpp>

#include <amsfilter/amsfilter.hpp>
#include <amsfilter/internal/blocked_bloomfilter_template.hpp>
#include <amsfilter/internal/cuckoofilter_template.hpp>
#include <amsfilter/internal/filter_resolve.hpp>

#ifdef AMSFILTER_PROBE_EXTERN_TEMPLATES
#include <amsfilter/internal/probe_instances/probe_extern_templates.hpp>
#endif

namespace amsfilter {
namespace internal {
//===----------------------------------------------------------------------===//
// TODO
struct probe_impl {

  using key_t = amsfilter::internal::key_t;
  using word_t = amsfilter::internal::word_t;

  /// Pointer to the filter (probe) logic instance.
  std::unique_ptr<amsfilter::internal::filter_batch_probe_base_t>
      filter_batch_probe_logic_;
  /// The blocked Bloom filter parameters.
  Config config_;
  /// The tuning parameters, used for hardware related optimizations.
  TuningParams tuning_params_;
  /// The (desired) bit length of the Bloom filter.
  std::size_t desired_length_;

  /// Construct a probe instance.
  probe_impl(const Config& config, const TuningParams& tuning_params,
      const std::size_t desired_length)
      : config_(config),
        tuning_params_(tuning_params),
        desired_length_(desired_length) {
    // Initialize the filter logic.
    init();
  }

  /// Move c'tor.
  probe_impl(probe_impl&& src) noexcept = default;
  probe_impl(const probe_impl& other) = delete;
  probe_impl& operator=(const probe_impl& other) = delete;
  probe_impl& operator=(probe_impl&& other) = delete;
  ~probe_impl() = default;

  /// Batch-probe the filter.
  void __attribute__((noinline))
  contains(const word_t* __restrict filter_data,
      const key_t* __restrict keys, u32 key_cnt,
      word_t* __restrict result_bitmap) {
    if (key_cnt == 0) return;
    // Execute the kernel.
    filter_batch_probe_logic_->batch_contains_bitmap(
        filter_data, keys, key_cnt, result_bitmap,
        tuning_params_.unroll_factor);
  }

  //===--------------------------------------------------------------------===//
  // Template expansions.
  //===--------------------------------------------------------------------===//
  /// Resolves the (specialized) filter type and constructs an instance (on
  /// heap memory).
  void
  init() {
    amsfilter::internal::get_instance(config_, *this);
  }

  // This function is called by 'get_instance'.
  template<u32 w, u32 s, u32 z, u32 k, dtl::block_addressing a>
  void
  operator()(const dtl::blocked_bloomfilter_config& conf) {
    using resolved_type =
        typename amsfilter::internal::bbf_type<w, s, z, k, a>::type;
    if (resolved_type::word_cnt_per_block != conf.word_cnt_per_block
        || resolved_type::sector_cnt != conf.sector_cnt
        || resolved_type::zone_cnt != conf.zone_cnt
        || resolved_type::k != conf.k
        || resolved_type::addr_mode != conf.addr_mode) {
      throw std::invalid_argument(
          "Failed to construct a blocked Bloom filter with the parameters w="
              + std::to_string(conf.word_cnt_per_block)
              + ", s=" + std::to_string(conf.sector_cnt)
              + ", z=" + std::to_string(conf.zone_cnt)
              + ", k=" + std::to_string(conf.k)
              + ", and, a=" + std::to_string(static_cast<i32>(conf.addr_mode))
              + ".");
    }
    // Instantiate the resolved type.
    resolved_type filter_logic(desired_length_);
    using batch_probe_type = dtl::blocked_bloomfilter_batch_probe<resolved_type>;
    auto batch_probe_logic = std::make_unique<batch_probe_type>(filter_logic);
    filter_batch_probe_logic_ = std::move(batch_probe_logic);
  }
  //===--------------------------------------------------------------------===//

};
//===----------------------------------------------------------------------===//
} // namespace internal
} // namespace amsfilter
