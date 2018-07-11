#include <algorithm>

#include <dtl/dtl.hpp>
#include <dtl/mem.hpp>

#include "platform.hpp"


namespace dtl {
namespace filter {

//===----------------------------------------------------------------------===//
// Read the CPU affinity for the process.
const dtl::cpu_mask platform::cpu_mask = dtl::this_thread::get_cpu_affinity();
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
platform::platform() :
    thread_cnt(std::max(u64(1), cpu_mask.count() / 2)), // TODO probably not the best choice for everyone
    numa_node_count(static_cast<u32>(dtl::mem::get_node_count())) {
  // initialize thread_id -> node_id mapping.
  const auto max_thread_cnt = cpu_mask.count();
  thread_id_to_node_id_map.resize(max_thread_cnt, 0);
  auto thread_fn = [&](u32 thread_id) {
    const auto cpu_node_id = dtl::mem::get_node_of_cpu(dtl::this_thread::get_cpu_affinity().find_first());
    const auto mem_node_id = dtl::mem::hbm_available() ? dtl::mem::get_nearest_hbm_node(cpu_node_id)
                                                       : cpu_node_id;
    thread_id_to_node_id_map[thread_id] = mem_node_id;
  };
  dtl::run_in_parallel(thread_fn, cpu_mask, max_thread_cnt);
}

u32
platform::get_memory_node_of_thread(u32 thread_id) const {
  const auto node_id = thread_id_to_node_id_map[thread_id % thread_id_to_node_id_map.size()];
  return node_id;
}

u32
platform::get_memory_node_of_this_thread() const {
  u32 thread_id = get_thread_id();
  u32 mem_node_id = get_memory_node_of_thread(thread_id);
  return mem_node_id;
}

u32
platform::get_thread_id() const {
  if (this_thread_is_known) {
    return this_thread_id;
  }
  // Assuming, threads are pinned to a single core.
  return dtl::this_thread::get_cpu_affinity().find_first();
}
//===----------------------------------------------------------------------===//

} // namespace filter
} // namespace dtl
