#pragma once

#include <atomic>
#include <bitset>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <sched.h>
#include <thread>

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>
#include <dtl/bitset.hpp>
#include <dtl/thread.hpp>


namespace dtl {


using cpu_mask = dtl::bitset<CPU_SETSIZE>;


namespace this_thread {


/// pins the current thread to the specified CPU(s)
void
set_cpu_affinity(dtl::cpu_mask& cpu_mask) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  // convert bitset to CPU mask
  for (auto it = dtl::on_bits_begin(cpu_mask);
       it != dtl::on_bits_end(cpu_mask);
       it++) {
    CPU_SET(*it, &mask);
  }
  i32 result = sched_setaffinity(0, sizeof(mask), &mask);
  if (result != 0) {
    std::cout << "Failed to set CPU affinity." << std::endl;
  }
}

/// pins the current thread to a specific CPU
void
set_cpu_affinity(u32 cpu_id) {
  //TODO consider using pthread_setaffinity_np
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(cpu_id % std::thread::hardware_concurrency(), &mask);
  i32 result = sched_setaffinity(0, sizeof(mask), &mask);
  if (result != 0) {
    std::cout << "Failed to set CPU affinity." << std::endl;
  }
}

/// reset the CPU affinity of the current thread
void
reset_cpu_affinity() {
  dtl::cpu_mask m;
  for ($u64 i = 0; i < std::thread::hardware_concurrency(); i++) {
    m[i] = 1;
  }
  set_cpu_affinity(m);
}

/// returns the CPU affinity of the current thread
dtl::cpu_mask
get_cpu_affinity() {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  i32 result = sched_getaffinity(0, sizeof(mask), &mask);
  if (result != 0) {
    std::cout << "Failed to determine CPU affinity." << std::endl;
  }
  // convert c-style mask to std::bitset
  u64 bit_width = sizeof(cpu_set_t) * 8;
  dtl::cpu_mask bm = dtl::to_bitset<bit_width>(&mask);
  return bm;
}


namespace detail {


// used to assign unique ids
std::atomic<$u64> thread_cntr(0);
thread_local $u64 uid = ~0ull;
thread_local $u64 id = ~0ull;


void
init(u32 thread_id, std::function<void()> fn) {
  // affinitize thread
  dtl::this_thread::set_cpu_affinity(thread_id);
  // set the given thread id
  dtl::this_thread::detail::id = thread_id;
  // set the unique id
  dtl::this_thread::detail::uid = dtl::this_thread::detail::thread_cntr.fetch_add(1);
  // run the given function in the current thread
  fn();
};


} // namespace detail


/// returns the given id of the current thread
u64
get_id() {
  return detail::id;
}


/// returns the unique id of the current thread
u64
get_uid() {
  return detail::uid;
}


} // namespace this_thread


/// spawn a new thread
std::thread
thread(u32 thread_id, std::function<void()> fn) {
  return std::thread(dtl::this_thread::detail::init, thread_id, fn);
}

/// spawn a new thread
template<typename Fn, typename... Args>
std::thread
thread(u32 thread_id, Fn&& fn, Args&&... args) {
  return std::thread(dtl::this_thread::detail::init, thread_id,
                     std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...));
}



/// thread affinitizer
/// @deprecated
static void
thread_affinitize(u32 thread_id) {
  dtl::this_thread::set_cpu_affinity(thread_id);
}

void
run_in_parallel(std::function<void()> fn,
                u32 thread_cnt = std::thread::hardware_concurrency()) {

  auto thread_fn = [](u32 thread_id, std::function<void()> fn) {
    thread_affinitize(thread_id);
    fn();
  };

  std::vector<std::thread> workers;
  for (std::size_t i = 0; i < thread_cnt - 1; i++) {
    workers.push_back(std::move(std::thread(thread_fn, i, fn)));
  }
  std::thread(thread_fn, thread_cnt - 1, fn).join();
  for (auto& worker : workers) {
    worker.join();
  }
}

void
run_in_parallel(std::function<void(u32 thread_id)> fn,
                u32 thread_cnt = std::thread::hardware_concurrency()) {

  auto thread_fn = [](u32 thread_id, std::function<void(u32 thread_id)> fn) {
    thread_affinitize(thread_id);
    fn(thread_id);
  };

  std::vector<std::thread> workers;
  for (std::size_t i = 0; i < thread_cnt - 1; i++) {
    workers.push_back(std::move(std::thread(thread_fn, i, fn)));
  }
  std::thread(thread_fn, thread_cnt - 1, fn).join();
  for (auto& worker : workers) {
    worker.join();
  }
}


void
run_in_parallel_async(std::function<void()> fn,
                      std::vector<std::thread>& workers,
                      u32 thread_cnt = std::thread::hardware_concurrency()) {
  workers.reserve(thread_cnt);

  auto thread_fn = [](u32 thread_id, std::function<void()> fn) {
    thread_affinitize(thread_id);
    fn();
  };

  for (std::size_t i = 0; i < thread_cnt; i++) {
    workers.push_back(std::move(std::thread(thread_fn, i, fn)));
  }
}

void
wait_for_threads(std::vector<std::thread>& workers) {
  for (auto& worker : workers) {
    worker.join();
  }
  workers.clear();
}

} // namespace dtl
