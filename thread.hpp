#pragma once

#include "adept.hpp"
#include <iostream>
#include <thread>

namespace dtl {

/// thread affinitizer
static void
thread_affinitize(u32 thread_id) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(thread_id % std::thread::hardware_concurrency(), &mask);
  i32 result = sched_setaffinity(0, sizeof(mask), &mask);
  if (result != 0) {
    std::cout << "Failed to set CPU affinity." << std::endl;
  }
}

void
run_in_parallel(std::function<void()> fn,
                u32 thread_cnt = std::thread::hardware_concurrency()) {

  auto thread_fn = [&](u32 thread_id) {
    thread_affinitize(thread_id);
    fn();
  };

  std::vector<std::thread> workers;
  for (std::size_t i = 0; i < thread_cnt - 1; i++) {
    workers.push_back(std::move(std::thread(thread_fn, i)));
  }
  std::thread(thread_fn, thread_cnt - 1).join();
  for (auto& worker : workers) {
    worker.join();
  }
}

} // namespace dtl
