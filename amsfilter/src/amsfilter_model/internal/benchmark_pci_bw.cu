#include <algorithm>
#include <chrono>
#include <cstddef>
#include <set>

#include <dtl/dtl.hpp>
#include <amsfilter_model/internal/timing.hpp>
#include <amsfilter/cuda/internal/alloc.hpp>
#include <amsfilter/cuda/internal/cuda_api_helper.hpp>
#include <amsfilter/amsfilter.hpp>
#include <dtl/thread.hpp>

#include "benchmark_pci_bw.cuh"

namespace amsfilter {
namespace model {
//===----------------------------------------------------------------------===//
using T = $u32;
const std::size_t SAMPLE_SIZE_MIB = 512;
const std::size_t SAMPLE_SIZE = SAMPLE_SIZE_MIB * 1024 * 1024 / sizeof(T);
//===----------------------------------------------------------------------===//
benchmark_pci_bw::benchmark_pci_bw() {
}
//===----------------------------------------------------------------------===//
benchmark_pci_bw::~benchmark_pci_bw() {
}
//===----------------------------------------------------------------------===//
std::pair<$f64, $f64>
benchmark_pci_bw::run(const Env& exec_env) {

  dtl::this_thread::set_cpu_affinity(0);

  if (!exec_env.is_gpu()) {
    throw std::invalid_argument(
        "The execution environment does not refer to a GPU.");
  }

  const auto device_no = static_cast<u32>(exec_env.get_device());
  const auto device_cnt = amsfilter::cuda::get_cuda_device_count();
  if (device_no >= device_cnt) {
    throw std::invalid_argument("Unknown CUDA device.");
  }
  cudaSetDevice(exec_env.get_device());
  cuda_check_error();

  T* host_buffer;
  T* device_buffer;

  auto allocate = [&]() {
    if (exec_env.get_probe_key_location() == Memory::HOST_PINNED) {
      amsfilter::cuda::internal::cuda_host_allocator<T> alloc;
      host_buffer = alloc.allocate(SAMPLE_SIZE);
    }
    else {
      boost::alignment::aligned_allocator<T, 64> alloc;
      host_buffer = alloc.allocate(SAMPLE_SIZE);
    }

    amsfilter::cuda::internal::cuda_device_allocator<T> device_alloc(
        device_no);
    device_buffer = device_alloc.allocate(SAMPLE_SIZE);
  };

  auto deallocate = [&]() {
    if (host_buffer) {
      if (exec_env.get_probe_key_location() == Memory::HOST_PINNED) {
        amsfilter::cuda::internal::cuda_host_allocator<T> alloc;
        alloc.deallocate(host_buffer, SAMPLE_SIZE);
      }
      else {
        boost::alignment::aligned_allocator<T, 64> alloc;
        alloc.deallocate(host_buffer, SAMPLE_SIZE);
      }
    }
    if (device_buffer) {
      amsfilter::cuda::internal::cuda_device_allocator<T> device_alloc(
          device_no);
      device_alloc.deallocate(device_buffer, SAMPLE_SIZE);
    }
  };

  $f64 max_throughput_per_sec = 0.0;
  $f64 max_throughput_per_cycle = 0.0;
  try {

    const auto buffer_size_bytes = SAMPLE_SIZE * sizeof(T);

    static constexpr u32 thread_cnt = 1;
    std::array<cudaStream_t, thread_cnt> streams;
    std::array<cudaEvent_t, thread_cnt> events;

    for (auto i = 0u; i < thread_cnt; ++i) {
      cudaStreamCreateWithFlags(&streams[i],
          cudaStreamNonBlocking & cudaEventDisableTiming);
      cuda_check_error();
      cudaEventCreate(&events[i]);
      cuda_check_error();
    }

    // The actual measurement.
    for (std::size_t i = 0; i < 10; ++i) {
      // Need to allocate in each iteration to work around the caching
      // mechanism in the CUDA driver.
      allocate();

      const auto time_start = std::chrono::high_resolution_clock::now();
      const auto cycles_start = _rdtsc();

      auto thread_fn = [&](u32 thread_id) {
        cudaSetDevice(exec_env.get_device());
        // Copy the keys to the pre-allocated device memory.
        const std::size_t offset = (SAMPLE_SIZE / thread_cnt) * thread_id;
        cudaMemcpyAsync(device_buffer + offset, host_buffer + offset,
            buffer_size_bytes / thread_cnt, cudaMemcpyHostToDevice,
            streams[thread_id]);
        cudaEventRecord(events[thread_id], streams[thread_id]);
        cudaEventSynchronize(events[thread_id]);
        cuda_check_error();
      };
      dtl::run_in_parallel(thread_fn, thread_cnt);
      const auto time_end = std::chrono::high_resolution_clock::now();
      const auto cycles_end = _rdtsc();

      std::chrono::duration<double> duration = time_end - time_start;
      const auto throughput_per_sec =
          (buffer_size_bytes / 1024.0 / 1024.0) / duration.count();
      const auto throughput_per_cycle =
          (buffer_size_bytes / 1024.0 / 1024.0) / (cycles_end - cycles_start);

      if (max_throughput_per_sec < throughput_per_sec) {
        max_throughput_per_sec = throughput_per_sec;
        max_throughput_per_cycle = throughput_per_cycle;
      }

      deallocate();
    }

    for (auto i = 0u; i < thread_cnt; ++i) {
      cudaStreamDestroy(streams[i]);
      cuda_check_error();
      cudaEventDestroy(events[i]);
      cuda_check_error();
    }

  } catch(...) {
    deallocate();
    throw std::runtime_error("Failed to measure PCIe bandwidth.");
  }
  return std::make_pair(max_throughput_per_sec, max_throughput_per_cycle);
}
//===----------------------------------------------------------------------===//
} // namespace model
} // namespace amsfilter
