#pragma once

#define AMSFILTER_PROBE_EXTERN_TEMPLATES

#include <dtl/dtl.hpp>

#include <amsfilter/amsfilter.hpp>
#include <amsfilter/internal/blocked_bloomfilter_resolve.hpp>
#include <amsfilter/internal/blocked_bloomfilter_template.hpp>

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

  /// Pointer to the filter logic instance.
  std::unique_ptr<amsfilter::internal::bbf_base_t> filter_logic_;
  /// The blocked Bloom filter parameters.
  Config config_;
  /// The tuning parameters, used for hardware related optimizations.
  TuningParams tuning_params_;
  /// The (desired) bit length of the Bloom filter.
  std::size_t desired_length_;
  /// Pointer to the (specialized) contains functions.
  std::function<void(
      const word_t* __restrict /*filter_data*/,
      const key_t* __restrict /*keys*/, u32 /*key_cnt*/,
      word_t* __restrict /*result bitmap*/)>
    batch_contains_fn_;

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
    batch_contains_fn_(filter_data, keys, key_cnt, result_bitmap);
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
    auto filter_logic = std::make_unique<resolved_type>(desired_length_);

    // Bind the (specialized) execute_kernel function.
    using namespace std::placeholders;
    const auto unroll_factor = tuning_params_.unroll_factor;
    switch (unroll_factor) {
      case 0: {
        constexpr auto vector_len = 0; // use scalar code path
        batch_contains_fn_ = std::bind(
            &resolved_type::template batch_contains_bitmap<vector_len>,
            filter_logic.get(),
            _1, _2, _3, _4);
        break;
      }
      case 1: {
        constexpr auto vector_len = dtl::simd::lane_count<key_t> * 1;
        batch_contains_fn_ = std::bind(
            &resolved_type::template batch_contains_bitmap<vector_len>,
            filter_logic.get(),
            _1, _2, _3, _4);
        break;
      }
      case 2: {
        constexpr auto vector_len = dtl::simd::lane_count<key_t> * 2;
        batch_contains_fn_ = std::bind(
            &resolved_type::template batch_contains_bitmap<vector_len>,
            filter_logic.get(),
            _1, _2, _3, _4);
        break;
      }
      case 4: {
        constexpr auto vector_len = dtl::simd::lane_count<key_t> * 4;
        batch_contains_fn_ = std::bind(
            &resolved_type::template batch_contains_bitmap<vector_len>,
            filter_logic.get(),
            _1, _2, _3, _4);
        break;
      }
      default:
        throw std::invalid_argument(
            "Invalid unrolling factor: " + std::to_string(unroll_factor));
    }
    filter_logic_ = std::move(filter_logic);
  }
  //===--------------------------------------------------------------------===//

};
//===----------------------------------------------------------------------===//
} // namespace internal
} // namespace amsfilter
