#pragma once

#define AMSFILTER_CUDA_PROBE_EXTERN_TEMPLATES

#include <dtl/dtl.hpp>
#include <amsfilter/amsfilter.hpp>
#include <amsfilter/internal/blocked_bloomfilter_resolve.hpp>
#include <amsfilter/internal/blocked_bloomfilter_template.hpp>
#include <cuda_runtime.h>
#include "cuda_helper.cuh"
#include "kernel.cuh"

#ifdef AMSFILTER_CUDA_PROBE_EXTERN_TEMPLATES
#include <amsfilter/cuda/internal/kernel_instances/kernel_extern_templates.cuh>
#endif

namespace amsfilter {
namespace cuda {
namespace internal {
//===----------------------------------------------------------------------===//
// used to switch kernels
struct kernel_default {};
struct kernel_block_prefetch {};
//===----------------------------------------------------------------------===//
/// TODO
struct probe_impl {

  using key_t = amsfilter::internal::key_t;
  using word_t = amsfilter::internal::word_t;

  /// Pointer to the filter logic instance.
  std::unique_ptr<amsfilter::internal::bbf_base_t> filter_logic_;
  /// The blocked Bloom filter parameters.
  Config config_;
  /// The (desired) bit length of the Bloom filter.
  std::size_t desired_length_;
  /// Pointer to the (specialized) contains functions.
  std::function<void(
      const word_t* __restrict /*filter_data*/,
      const key_t* __restrict /*keys*/, u32 /*key_cnt*/,
      word_t* __restrict /*result bitmap*/,
      const cudaStream_t& /*cuda_stream*/)>
    execute_kernel_fn_;

  /// Construct a probe instance.
  probe_impl(const Config& config, const std::size_t desired_length)
      : config_(config), desired_length_(desired_length) {
    // Initialize the filter logic.
    init();
  }

  /// Move c'tor.
  probe_impl(probe_impl&& src) noexcept = default;
  probe_impl(const probe_impl& other) = delete;
  probe_impl& operator=(const probe_impl& other) = delete;
  probe_impl& operator=(probe_impl&& other) = delete;
  ~probe_impl() = default;

  /// Asynchronously batch-probe the filter.
  inline void
  contains(const word_t* __restrict filter_data,
      const key_t* __restrict keys, u32 key_cnt,
      word_t* __restrict result_bitmap,
      const cudaStream_t& cuda_stream) {
    if (key_cnt == 0) return;
    // Execute the kernel.
    execute_kernel_fn_(filter_data, keys, key_cnt, result_bitmap, cuda_stream);
    cuda_check_error();
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

  /// Function template to execute the CUDA kernel.
  template<typename filter_t>
  void __attribute__((noinline))
  execute_kernel(const word_t* __restrict filter_data,
      const key_t* keys, u32 key_cnt,
      word_t* __restrict result_bitmap,
      const cudaStream_t& cuda_stream
      ) {
    const filter_t* filter = static_cast<const filter_t*>(filter_logic_.get());
    i32 block_size = warp_size; // dtl::env<$i32>::get("BLOCK_SIZE", warp_size);
    i32 elements_per_thread = warp_size;
    i32 elements_per_block = block_size * elements_per_thread;
    i32 block_cnt = (key_cnt + elements_per_block - 1) / elements_per_block;
    // Execute the kernel.
    contains_kernel<filter_t>
        <<<block_cnt, block_size, 0, cuda_stream>>>(
        *filter, filter_data, keys, key_cnt, result_bitmap);
  }
  template<typename filter_t>
  void __attribute__((noinline))
  execute_kernel_with_block_prefetch(const word_t* __restrict filter_data,
      const key_t* keys, u32 key_cnt,
      word_t* __restrict result_bitmap,
      const cudaStream_t& cuda_stream
      ) {
    const filter_t* filter = static_cast<const filter_t*>(filter_logic_.get());
    i32 block_size = warp_size; // dtl::env<$i32>::get("BLOCK_SIZE", warp_size);
    i32 elements_per_thread = warp_size;
    i32 elements_per_block = block_size * elements_per_thread;
    i32 block_cnt = (key_cnt + elements_per_block - 1) / elements_per_block;
    // Execute the kernel.
    contains_kernel_with_block_prefetch<filter_t>
        <<<block_cnt, block_size, 0, cuda_stream>>>(
        *filter, filter_data, keys, key_cnt, result_bitmap);
  }

  // This function is called by get_instance().
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
    filter_logic_ = std::make_unique<resolved_type>(desired_length_);
    // Bind the (specialized) execute_kernel function.
    using kernel_type = typename std::conditional<w == 1,
        kernel_default, kernel_block_prefetch>::type;
    bind_kernel_fn<resolved_type>(kernel_type());
  }

  template<typename resolved_type>
  void
  bind_kernel_fn(const kernel_default& /* selector */) {
    using namespace std::placeholders;
    execute_kernel_fn_ = std::bind(
        &probe_impl::template execute_kernel<resolved_type>, this,
        _1, _2, _3, _4, _5);
  }
  template<typename resolved_type>
  void
  bind_kernel_fn(const kernel_block_prefetch& /* selector */) {
    using namespace std::placeholders;
    execute_kernel_fn_ = std::bind(
        &probe_impl::template execute_kernel_with_block_prefetch<resolved_type>, this,
        _1, _2, _3, _4, _5);
  }
  //===--------------------------------------------------------------------===//

};

//===----------------------------------------------------------------------===//
} // namespace internal
} // namespace cuda
} // namespace amsfilter
