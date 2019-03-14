#pragma once

#include <memory>

#include <amsfilter/amsfilter.hpp>
#include <amsfilter/cuda/internal/alloc.hpp>
#include <amsfilter/cuda/internal/cuda_api_helper.hpp>
#include <amsfilter/cuda/internal/replicas.hpp>
#include <amsfilter/cuda/probe.hpp>
#include <amsfilter/internal/buffer.hpp>
#include <amsfilter/bitmap_view.hpp>


namespace amsfilter {
namespace cuda {
//===----------------------------------------------------------------------===//
/// The AMS-Filter probe implementation with a lightweight interface.
///
/// In contrast to the Probe class, ProbeLite takes care of all the memory
/// management. Each instance allocates memory on the host-side as well as on
/// the device side. These memory resources are supposed to be re-used among
/// multiple batch-probes.
class ProbeLite {

  // Typedefs
  using bitmap_storage_t = amsfilter::AmsFilter::word_t;
  using bitmap_device_buffer_t = amsfilter::internal::buffer<bitmap_storage_t,
      internal::cuda_device_allocator<bitmap_storage_t>>;
  using bitmap_host_buffer_t = amsfilter::internal::buffer<bitmap_storage_t,
      internal::cuda_host_allocator<bitmap_storage_t>>;

  using key_t = amsfilter::AmsFilter::key_t;
  using keys_device_buffer_t =
  amsfilter::internal::buffer<key_t, internal::cuda_device_allocator<key_t>>;

  /// The probe instance (with the low-level API).
  Probe probe_instance_;
  /// Pointer to the filter data in device memory.
  internal::replicas::shared_device_vector_t filter_data_;
  std::size_t max_batch_size_;
  /// The CUDA device number, this instance is associated with.
  $u32 cuda_device_no_;
  /// The CUDA stream used for asynchronous execution.
  std::unique_ptr<cudaStream_t> cuda_stream_;
  /// The CUDA event that signals when the kernel terminated.
  std::unique_ptr<cudaEvent_t> cuda_done_event_;
  /// Host memory to where the result bitmap is written.
  bitmap_host_buffer_t host_bitmap_;
  /// Device memory to where the result bitmap is written.
  bitmap_device_buffer_t device_bitmap_;
  /// Device memory to where the probe keys are copied before kernel execution.
  keys_device_buffer_t device_keys_;

  /// Returns the number of words required to store the resulting bitmap.
  static std::size_t
  bitmap_word_cnt(std::size_t max_batch_size) {
    return (max_batch_size + (bitwidth<bitmap_storage_t> - 1))
        / bitwidth<bitmap_storage_t>;
  }

public:

  using result_type = bitmap_view<bitmap_storage_t>;

  explicit
  ProbeLite(const AmsFilter& filter,
      internal::replicas::shared_device_vector_t& filter_data,
      std::size_t max_batch_size,
      $u32 cuda_device_no)
      : probe_instance_(filter),
        filter_data_(filter_data),
        max_batch_size_(max_batch_size),
        cuda_device_no_(cuda_device_no),
        cuda_stream_(std::make_unique<cudaStream_t>()),
        cuda_done_event_(std::make_unique<cudaEvent_t>()),
        host_bitmap_(bitmap_word_cnt(max_batch_size_)),
        device_bitmap_(bitmap_word_cnt(max_batch_size_),
            internal::cuda_device_allocator<bitmap_storage_t>(cuda_device_no_)),
        device_keys_(max_batch_size_,
            internal::cuda_device_allocator<key_t>(cuda_device_no_))
  {

    // Create CUDA stream and event (required for asynchronous kernel execution).
    cudaStreamCreateWithFlags(cuda_stream_.get(),
        cudaStreamNonBlocking & cudaEventDisableTiming);
    cuda_check_error(); // TODO improve error handling
    cudaEventCreate(cuda_done_event_.get());
    cuda_check_error(); // TODO improve error handling

  };
  ProbeLite(const ProbeLite& other) = delete;
  ProbeLite(ProbeLite&& other) noexcept = default;
  ProbeLite& operator=(const ProbeLite& other) = delete;
  ProbeLite& operator=(ProbeLite&& other) = default;
  ~ProbeLite() {
    // Make sure that the kernel is not running.
    wait();
    // TODO use Eyal's wrapper
    if (cuda_done_event_ != nullptr) {
      cudaEventDestroy(*cuda_done_event_);
      cuda_check_error();
    }
    if (cuda_stream_ != nullptr) {
      cudaStreamDestroy(*cuda_stream_);
      cuda_check_error();
    }
  };

  /// Asynchronously batch-probe the filter.
  void __attribute__((noinline))
  operator()(const key_t* keys, const std::size_t key_cnt) {
    if (key_cnt == 0) return;
    if (key_cnt > max_batch_size_) {
      throw std::invalid_argument("The 'key_cnt' argument must not exceed"
          " the max. batch size.");
    }
    cudaSetDevice(cuda_device_no_);
    cuda_check_error();
    // Copy the keys to the pre-allocated device memory.
    cudaMemcpyAsync(device_keys_.begin(), keys, key_cnt * sizeof(key_t),
        cudaMemcpyHostToDevice, *cuda_stream_);
    cuda_check_error();
    // Execute the kernel (using the low-level API).
    probe_instance_(thrust::raw_pointer_cast(filter_data_->data()),
      device_keys_.begin(), key_cnt, device_bitmap_.begin(), *cuda_stream_);
    // Copy the result bitmap from device to pre-allocated host memory.
    const std::size_t bitmap_size_bytes =
        host_bitmap_.size() * sizeof(bitmap_storage_t);
    cudaMemcpyAsync(host_bitmap_.begin(), device_bitmap_.begin(),
        bitmap_size_bytes, cudaMemcpyDeviceToHost, *cuda_stream_);
    cuda_check_error();
    cudaEventRecord(*cuda_done_event_, *cuda_stream_);
    cuda_check_error();
  }

  /// Returns true if the asynchronously executed probe is done, false otherwise.
  u1 __forceinline__
  is_done() {
    return cudaEventQuery(*cuda_done_event_) == cudaSuccess;
  }

  /// Blocks until a asynchronously executed probe is finished.
  void __forceinline__
  wait() {
    if (cuda_done_event_ != nullptr) {
      cudaEventSynchronize(*cuda_done_event_);
      cuda_check_error();
    }
  }

  /// Gives access to the result bitmap.
  result_type
  get_results() {
    return bitmap_view<bitmap_storage_t> {
        host_bitmap_.begin(),
        host_bitmap_.end()
    };
  }

};
//===----------------------------------------------------------------------===//
} // namespace cuda
} // namespace amsfilter
