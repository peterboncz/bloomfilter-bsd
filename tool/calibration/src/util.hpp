#pragma once

#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>
#include <functional>

#include <dtl/dtl.hpp>
#include <dtl/env.hpp>
#include <dtl/thread.hpp>

// Read the CPU affinity for the process.
static const auto cpu_mask = dtl::this_thread::get_cpu_affinity();


template<typename T>
void dispatch(const std::vector<T> tasks,
              std::function<void(const T&, std::ostream&)> fn) {

  i64 thread_cnt = dtl::env<$u64>::get("THREAD_CNT", cpu_mask.count());
  i64 task_cnt = tasks.size();
  i64 min_batch_size = 1;
  i64 max_batch_size = 16;

  const auto time_start = std::chrono::system_clock::now();
  std::atomic<$i64> cntr { 0 };
  auto thread_fn = [&](u32 thread_id) {
    while (true) {
      // Grab work.
      const auto inc = std::min(std::max(min_batch_size, (task_cnt - cntr) / thread_cnt), max_batch_size);
      const auto task_idx_begin = cntr.fetch_add(inc);
      const auto task_idx_end = std::min(task_idx_begin + inc, task_cnt);
      if (task_idx_begin >= task_cnt) break;
      std::stringstream s;
//      s << "thread " << thread_id << " got " << (task_idx_end - task_idx_begin) << " task(s)" << std::endl;
      std::cerr << s.str();

      std::stringstream str;
      for ($i64 ci = task_idx_begin; ci < task_idx_end; ci++) {
        fn(tasks[ci], str);
      }
      std::cout << str.str();

      if (thread_id == 0) {
        i64 i = std::min(i64(cntr), i64(task_cnt));
        i64 r = std::min(task_cnt, task_cnt - i);
        // Estimate time until completion.
        const auto now = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = now - time_start;
        f64 avg_sec_per_task = elapsed_seconds.count() / i;
        u64 remaining_sec = avg_sec_per_task * r;
        u64 h = (remaining_sec / 3600);
        u64 m = (remaining_sec % 3600) / 60;
        std::stringstream str;
        str << "Progress: [" << (i + 1) << "/" << task_cnt << "]";
        str << " - estimated time until completion: " << h << "h " << m << "m" << std::endl;
        std::cerr << str.str();
      }
    }
  };
  dtl::run_in_parallel(thread_fn, cpu_mask, cpu_mask.count());
}
