#include <algorithm>
#include <chrono>
#include <cstddef>
#include <set>

#include <dtl/dtl.hpp>
#include <dtl/barrier.hpp>
#include <dtl/mem.hpp>
#include <dtl/thread.hpp>
#include <amsfilter/cuda/internal/cuda_api_helper.hpp>
#include <amsfilter/cuda/probe.hpp>
#include <amsfilter_model/internal/platform.hpp>
#include <amsfilter_model/internal/util.hpp>

#include "benchmark_gpu.cuh"

namespace amsfilter {
namespace model {
//===----------------------------------------------------------------------===//
const std::size_t SAMPLE_SIZE = 1ull << 27; // 512 MiB
//===----------------------------------------------------------------------===//
benchmark_gpu::benchmark_gpu() {
  // Generate random probe keys (unique keys).
  probe_keys.reserve(SAMPLE_SIZE);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<key_t> dis;
  auto is_in_sample = new std::bitset<1ull<<32>;
  std::size_t s = 0;
  while (s < SAMPLE_SIZE) {
    auto val = dis(gen);
    if (!(*is_in_sample)[val]) {
      probe_keys.push_back(val);
      (*is_in_sample)[val] = true;
      s++;
    }
  }
  delete is_in_sample;
}
//===----------------------------------------------------------------------===//
benchmark_gpu::~benchmark_gpu() {
  // Free GPU resources.
  for (std::size_t i = 0; i < device_key_vectors.size(); ++i) {
    if (device_bitmap_vectors[i]) delete device_bitmap_vectors[i];
    if (device_key_vectors[i]) delete device_key_vectors[i];
  }
}
//===----------------------------------------------------------------------===//
timing
benchmark_gpu::run(const Config& filter_config, u64 m,
    const Env& exec_env, const TuningParams& tuning_params) {

  if (!exec_env.is_gpu()) {
    throw std::invalid_argument(
        "The execution environment does not refer to a GPU.");
  }

  const auto device_no = exec_env.get_device();
  cudaSetDevice(device_no);
  const auto device_cnt = amsfilter::cuda::get_cuda_device_count();
  device_bitmap_vectors.resize(device_cnt, nullptr);
  device_key_vectors.resize(device_cnt, nullptr);

  // Copy the probe key to the device, if the are not already there.
  if (!device_key_vectors[device_no]) {
    device_key_vectors[device_no] = new device_key_vector(
        probe_keys.begin(), probe_keys.end());

    // Also allocate space for the result bitmap.
    std::size_t bitmap_word_cnt = (probe_keys.size()
        + (bitwidth<amsfilter::word_t> - 1)) / bitwidth<amsfilter::word_t>;
    device_bitmap_vectors[device_no] =
        new device_word_vector(bitmap_word_cnt, 0);
  }

  // Construct the filter.
  amsfilter::AmsFilter filter(filter_config, m);
  thrust::device_vector<amsfilter::word_t> device_filter_data(filter.size(), 0);

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream,
      cudaStreamNonBlocking & cudaEventDisableTiming);
  cuda_check_error();

  cudaEvent_t done_event;
  cudaEventCreate(&done_event);
  cuda_check_error();

  amsfilter::cuda::Probe probe(filter);

  const auto probe_nanos_begin = now_nanos();
  const auto probe_tsc_begin = _rdtsc();

  std::size_t repetition_cntr = 0;
  do {
    // The number of repetitions, that are enqueued at once.
    const auto rep = 5;
    for (std::size_t i = 0; i < rep; ++i) {
      probe(
          thrust::raw_pointer_cast(device_filter_data.data()),
          thrust::raw_pointer_cast(device_key_vectors[device_no]->data()),
          probe_keys.size(),
          thrust::raw_pointer_cast(device_bitmap_vectors[device_no]->data()),
          stream);
    }

    cudaEventRecord(done_event, stream);
    cuda_check_error();
    cudaEventSynchronize(done_event);
    cuda_check_error();
    repetition_cntr += rep;
  } while (now_nanos() - probe_nanos_begin < 250000000);

  const auto probe_tsc_end = _rdtsc();
  const auto probe_nanos_end = now_nanos();

  timing t;
  t.cycles_per_lookup = ((probe_tsc_end - probe_tsc_begin) * 1.0)
      / (probe_keys.size() * repetition_cntr);
  t.nanos_per_lookup = ((probe_nanos_end - probe_nanos_begin) * 1.0)
      / (probe_keys.size() * repetition_cntr);
  return t;
}
//===----------------------------------------------------------------------===//
} // namespace model
} // namespace amsfilter
