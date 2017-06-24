#include "gtest/gtest.h"

#include "../adept.hpp"
#include "../bloomfilter_runtime.hpp"
//#include "../bloomfilter.hpp"
//#include "../bloomfilter_vec.hpp"
#include "../hash.hpp"
#include "../mem.hpp"
#include "../simd.hpp"
#include "../thread.hpp"
#include "../env.hpp"

#include <atomic>

#include <chrono>

#include "immintrin.h"

using namespace dtl;


inline auto timing(std::function<void()> fn) {
  auto start = std::chrono::high_resolution_clock::now();
  fn();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

static constexpr std::chrono::seconds sec(1);
static constexpr double nano_to_sec = std::chrono::duration_cast<std::chrono::nanoseconds>(sec).count();


// --- compile-time settings ---
struct bf {
  using key_t = $u32;
  using word_t = $u32;

  using key_alloc = dtl::mem::numa_allocator<key_t>;
  using word_alloc = dtl::mem::numa_allocator<word_t>;
};


// --- runtime settings ---

// the grain size for parallel experiments
static u64 preferred_grain_size = 1ull << dtl::env<$i32>::get("GRAIN_SIZE", 16);

// set the bloomfilter size: m in [2^lo, 2^hi]
static i32 bf_size_lo_exp = dtl::env<$i32>::get("BF_SIZE_LO", 11);
static i32 bf_size_hi_exp = dtl::env<$i32>::get("BF_SIZE_HI", 31);

// the number of hash functions to use
static i32 bf_k = dtl::env<$i32>::get("BF_K", 1);

// repeats the benchmark with different concurrency settings
static i32 thread_cnt_lo = dtl::env<$i32>::get("THREAD_CNT_LO", 1);
static i32 thread_cnt_hi = dtl::env<$i32>::get("THREAD_CNT_HI", std::thread::hardware_concurrency());

// 1 = linear, 2 = exponential
static i32 thread_step_mode = dtl::env<$i32>::get("THREAD_STEP_MODE", 1);
static i32 thread_step = dtl::env<$i32>::get("THREAD_STEP", 1);

// the number of keys to probe per thread
static u64 key_cnt_per_thread = 1ull << dtl::env<$i32>::get("KEY_CNT", 24);

// the number of repetitions
static u64 repeat_cnt = dtl::env<$i32>::get("REPEAT_CNT", 16);;


// place bloomfilter in HBM?
static u1 use_hbm = dtl::env<$i32>::get("HBM", 1);
// replicate bloomfilter in HBM?
static u1 replicate_bloomfilter = dtl::env<$i32>::get("REPL", 1);

static void
print_env_settings() {
  std::cout
      << "Configuration:\n"
      << "  BF_K=" << bf_k
      << ", BF_SIZE_LO=" << bf_size_lo_exp
      << ", BF_SIZE_HI=" << bf_size_hi_exp
      << "\n  GRAIN_SIZE=" << dtl::env<$i32>::get("GRAIN_SIZE", 16)
      << ", THREAD_CNT_LO=" << thread_cnt_lo
      << ", THREAD_CNT_HI=" << thread_cnt_hi
      << ", THREAD_STEP=" << thread_step
      << ", THREAD_STEP_MODE=" << thread_step_mode << " (1=linear, 2=exponential)"
      << ", KEY_CNT=" << dtl::env<$i32>::get("KEY_CNT", 24) << " (per thread)"
      << ", REPEAT_CNT=" << repeat_cnt
      << "\n  HBM=" << static_cast<u32>(use_hbm) << " (0=no, 1=yes)"
      << ", REPL=" << static_cast<u32>(replicate_bloomfilter) << " (0=interleaved, 1=replicate)"
      << std::endl;
}

static auto inc_thread_cnt = [&](u64 i) {
  if (thread_step_mode == 1) {
    // linear
    return i + thread_step;
  }
  else {
    // exponential
    auto step = thread_step > 1 ? thread_step : 2;
    return i * step;
  }
};



void run_filter_benchmark_in_parallel_vec(u32 k, u32 m, u64 thread_cnt) {
  dtl::thread_affinitize(std::thread::hardware_concurrency() - 1);
  u64 key_cnt = key_cnt_per_thread * thread_cnt;

  using bf_rts_t = dtl::bloomfilter_runtime;
  bf::word_alloc bf_cpu_interleaved_alloc(dtl::mem::allocator_config::interleave_cpu());
  bf::word_alloc bf_hbm_interleaved_alloc(dtl::mem::allocator_config::interleave_hbm());

//  if (use_hbm) {
//    std::cout << "Using HBM for bloomfilter" << std::endl;
//  }
  // TODO  bf_rts_t bf(bf_size, use_hbm ? bf_hbm_interleaved_alloc : bf_cpu_interleaved_alloc);
  auto bf = bf_rts_t::construct(k, m);

  {
    f64 duration = timing([&] {
      for ($u64 i = 0; i < m >> 4; i++) {
        bf.insert(dtl::hash::crc32<u32>::hash(i));
      }
    });
    u64 perf = (key_cnt) / (duration / nano_to_sec);
  }

  // prepare the input (interleaved)
  bf::key_alloc input_interleaved_alloc(dtl::mem::allocator_config::interleave_cpu());
  std::vector<bf::key_t, bf::key_alloc> keys(input_interleaved_alloc);
  keys.resize(key_cnt);
  for ($u64 i = 0; i < key_cnt; i++) {
    keys[i] = dtl::hash::crc32<u32, 7331>::hash(i);
  }

  // create replicas as requested (see env 'HBM' and 'HBM_REPL')
  std::vector<bf_rts_t> bloomfilter_replicas;
  // maps node_id -> replica_id
  std::vector<$u64> bloomfilter_node_map;
  // insert the already existing bloomfilter (as a fallback when numa/hbm is not available)
  bloomfilter_replicas.push_back(std::move(bf));
  // initially, let all nodes refer to the first replica
  bloomfilter_node_map.resize(dtl::mem::get_node_count(), 0);

  if (replicate_bloomfilter) {
    // replicate the bloomfilter to all HBM nodes
    auto replica_nodes = (use_hbm && dtl::mem::hbm_available())
                         ? dtl::mem::get_hbm_nodes()
                         : dtl::mem::get_cpu_nodes();

    for (auto dst_node_id : replica_nodes) {
      // make a copy
      std::cout << "replicate bloomfilter to node " << dst_node_id << std::endl;
      auto alloc_config = dtl::mem::allocator_config::on_node(dst_node_id);
      bf_rts_t replica = bloomfilter_replicas[0].make_copy(alloc_config);
      bloomfilter_replicas.push_back(std::move(replica));
      // update mapping
      bloomfilter_node_map[dst_node_id] = bloomfilter_replicas.size() - 1;
    }
  }

  // size of a work item (dispatched to a thread)
  u64 grain_size = std::min(preferred_grain_size, key_cnt);

  std::vector<$f64> avg_cycles_per_probe;
  avg_cycles_per_probe.resize(thread_cnt, 0.0);

  std::vector<$u64> matches_found;
  matches_found.resize(thread_cnt, 0);
  std::atomic<$u64> grain_cntr(0);

  auto worker_fn = [&](u32 thread_id) {
    // determine NUMA node id
    const auto cpu_mask = dtl::this_thread::get_cpu_affinity();
    const auto cpu_id = cpu_mask.find_first(); // handwaving
    const auto numa_node_id = dtl::mem::get_node_of_cpu(cpu_id);

    // determine nearest HBM node (returns numa_node_id if HBM is not available)
    const auto hbm_node_id = dtl::mem::get_nearest_hbm_node(numa_node_id);

    // obtain the local bloomfilter instance
    const bf_rts_t& _bf = bloomfilter_replicas[bloomfilter_node_map[hbm_node_id]];

    // allocate a match vector
    std::vector<$u32> match_pos;
    match_pos.resize(grain_size, 0);

    $u64 tsc = 0;
    $u64 found = 0;
    $u64 probe_cntr = 0;
    while (true) {
      u64 cntr = grain_cntr.fetch_add(grain_size);
      u64 read_from = cntr % key_cnt;
      u64 read_to = std::min(key_cnt, read_from + grain_size);
      if (cntr >= key_cnt * repeat_cnt) break;
      u64 cnt = read_to - read_from;
      probe_cntr += cnt;
      __sync_synchronize();
      u64 tsc_begin = _rdtsc();
      auto match_cnt = _bf.batch_contains(&keys[read_from], cnt, &match_pos[0], 0);
      __sync_synchronize();
      u64 tsc_end = _rdtsc();
      tsc += tsc_end - tsc_begin;
      found += match_cnt;
    }
    matches_found[thread_id] = found;
    avg_cycles_per_probe[thread_id] = (tsc * 1.0) / probe_cntr;
  };


  $f64 duration = timing([&] {
    dtl::run_in_parallel(worker_fn, thread_cnt);
  });

  duration /= repeat_cnt;

  $u64 found = 0;
  for ($u64 i = 0; i < thread_cnt; i++) {
    found += matches_found[i];
  }
  found /= repeat_cnt;

  f64 cycles_per_probe = std::accumulate(avg_cycles_per_probe.begin(), avg_cycles_per_probe.end(), 0.0) / thread_cnt;

  u64 perf = (key_cnt) / (duration / nano_to_sec);
  std::cout << "bf_size: " << (m / 8) << " [bytes], "
            << "thread_cnt: " << thread_cnt << ", "
            << "key_cnt: " << key_cnt << ", "
            << "grain_size: " << grain_size << ", "
            << "performance: " << perf << " [1/s], "
            << "cycles/probe: " << cycles_per_probe
            << " (matchcnt: " << found << ")"
            << std::endl;
}


TEST(bloom, filter_performance_parallel_vec) {
  print_env_settings();
  u64 bf_size_lo = 1ull << bf_size_lo_exp;
  u64 bf_size_hi = 1ull << bf_size_hi_exp;

  for ($u64 bf_size = bf_size_lo; bf_size <= bf_size_hi; bf_size <<= 1) {
    for ($u64 t = thread_cnt_lo; t <= thread_cnt_hi; t = inc_thread_cnt(t)) {
      run_filter_benchmark_in_parallel_vec(bf_k, bf_size, t);
    }
  }
}
