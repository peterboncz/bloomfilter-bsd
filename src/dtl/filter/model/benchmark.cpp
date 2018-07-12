#include <algorithm>
#include <chrono>
#include <cstddef>
#include <set>

#include <dtl/dtl.hpp>
#include <dtl/barrier.hpp>
#include <dtl/batchwise.hpp>
#include <dtl/mem.hpp>
#include <dtl/thread.hpp>
#include <dtl/filter/filter.hpp>
#include <dtl/filter/platform.hpp>

#include "benchmark.hpp"

namespace dtl {
namespace filter {
namespace model {

const std::size_t SAMPLE_SIZE = 1ull << 26; // 256 MiB

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//
template<
    typename input_it,
    typename consumer_fn
>
__forceinline__ void
batchwise_contains(input_it begin, input_it end,
                   const filter::probe_t& probe,
                   consumer_fn consumer) {
  // Process the input batch-wise
  dtl::batch_wise(begin, end, [&](const auto batch_begin, const auto batch_end) {
    std::size_t match_count = 0;
    $u32 match_pos[dtl::BATCH_SIZE];
    match_count += probe(batch_begin, batch_end, &match_pos[0], 0u);
    // Call the consumer every time a batch is completed
    consumer(match_pos, match_pos + match_count);
  });
}

u64
now_nanos() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct statistics {
  $f64 match_count;
  $f64 avg_nanos;
  $f64 avg_cycles;
};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
benchmark::benchmark() {

  //===----------------------------------------------------------------------===//
  // Initialize probe sample.
  // Allocate memory, round robin on all NUMA nodes (excluding HBM nodes).
  auto& platform = dtl::filter::platform::get_instance();
  auto cpu_mask = platform.get_cpu_mask();
  auto node_mask = platform.get_cpu_mask();
  node_mask.reset();
  for (auto it = cpu_mask.on_bits_begin(); it != cpu_mask.on_bits_end(); it++) {
    node_mask.set(dtl::mem::get_node_of_cpu(*it));
  }
  std::vector<$u32> numa_nodes;
  for (auto it = node_mask.on_bits_begin(); it != node_mask.on_bits_end(); it++) {
    numa_nodes.push_back(*it);
  }
  std::cerr << "Allocating probe data on NUMA node(s):";
  for (auto nid : numa_nodes) std::cerr << " " << nid;
  std::cerr << std::endl;

  dtl::mem::allocator_config alloc_config = dtl::mem::allocator_config::on_node(numa_nodes);
  dtl::mem::numa_allocator<uint32_t> alloc(alloc_config);
  probe_keys = std::vector<key_t, dtl::mem::numa_allocator<key_t>>(alloc);
  probe_keys.reserve(SAMPLE_SIZE);

  // Generate random probe keys (unique keys).
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
  //===----------------------------------------------------------------------===//

}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
timing
benchmark::run(filter& filter) {
  // prepare the parallel benchmark
  auto& platform = dtl::filter::platform::get_instance();
  u64 thread_cnt = platform.get_thread_count();

  statistics results;
  dtl::busy_barrier_one_shot barrier(thread_cnt);

  std::vector<$u64> lookup_duration_time(thread_cnt);

  std::vector<$u64> probe_nanos_begin(thread_cnt);
  std::vector<$u64> probe_nanos_end(thread_cnt);

  std::vector<$u64> match_counts(thread_cnt);

  std::vector<$u64> probe_tsc_begin(thread_cnt);
  std::vector<$u64> probe_tsc_end(thread_cnt);

  auto thread_fn = [&](u32 thread_id) {

    // Obtain a probe instance.
    const auto probe = filter.probe();

    // Each thread starts reading from a different offset
    // to force accesses to different memory locations
    // and to stress the memory bus.
    const auto start_offset = (thread_id * (std::distance(probe_keys.begin(), probe_keys.end()) / thread_cnt));

    std::size_t match_count = 0;

    // Wait until all threads have spawned and the filter data have been replicated.
    barrier.wait();

    // Begin of measurement.
    const auto probe_nanos_begin_local = now_nanos();
    const auto probe_tsc_begin_local = _rdtsc();

    // Call the batch/simd/data-parallel contains function
    batchwise_contains(probe_keys.begin() + start_offset, probe_keys.end(),
                       probe,
                       [&](u32* matches_begin, u32* matches_end) {
                         match_count += matches_end - matches_begin;
                       });
    batchwise_contains(probe_keys.begin(), probe_keys.begin() + start_offset,
                       probe,
                       [&](u32* matches_begin, u32* matches_end) {
                         match_count += matches_end - matches_begin;
                       });

    const auto probe_tsc_end_local = _rdtsc();
    const auto probe_nanos_end_local = now_nanos();
    // End of measurement.

    probe_nanos_begin[thread_id] = probe_nanos_begin_local;
    probe_nanos_end[thread_id] = probe_nanos_end_local;
    probe_tsc_begin[thread_id] = probe_tsc_begin_local;
    probe_tsc_end[thread_id] = probe_tsc_end_local;

    match_counts[thread_id] = match_count;
  };

  std::size_t repetition_cntr = 0;
  const auto begin_nanos = now_nanos();
  do {
    barrier.reset();
    dtl::run_in_parallel(thread_fn, platform.get_cpu_mask(), thread_cnt);

    // Aggregate results
    statistics stats;
    stats.avg_nanos = 0;
    stats.avg_cycles = 0;
    stats.match_count = 0;
    for (std::size_t i = 0; i < thread_cnt; i++) {
      stats.avg_nanos += probe_nanos_end[i] - probe_nanos_begin[i];
      stats.avg_cycles += probe_tsc_end[i] - probe_tsc_begin[i];
      stats.match_count += match_counts[i];
    }
    stats.avg_nanos /= thread_cnt;
    stats.avg_cycles /= thread_cnt;
    stats.match_count /= thread_cnt;

    if (repetition_cntr == 0) {
      results = stats;
    }
    if (stats.avg_nanos < results.avg_nanos) {
      results = stats;
    }
    repetition_cntr++;
  } while (now_nanos() - begin_nanos < 250000000);

  timing t;
  t.cycles_per_lookup = results.avg_cycles / SAMPLE_SIZE;
  t.nanos_per_lookup = results.avg_nanos / SAMPLE_SIZE;
  return t;
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
timing
benchmark::operator()(const dtl::blocked_bloomfilter_config& filter_config, u64 m) {
  // create an empty filter
  filter filter(filter_config, m);
  return run(filter);
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
timing
benchmark::operator()(const dtl::cuckoofilter::config& filter_config, u64 m) {
  // create an empty filter
  filter filter(filter_config, m);
  return run(filter);
}
//===----------------------------------------------------------------------===//


} // namespace model
} // namespace filter
} // namespace dtl
