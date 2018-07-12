#pragma once

#include <dtl/dtl.hpp>
#include <dtl/thread.hpp>


namespace dtl {
namespace filter {

static thread_local $u1 this_thread_is_known = false;
static thread_local $u32 this_thread_id = 0;


//===----------------------------------------------------------------------===//
class platform {

  // CPU affinity for the process.
  static const dtl::cpu_mask cpu_mask;

  // The number of threads to use.
  u32 thread_cnt;

  // The number of NUMA nodes.
  u32 numa_node_count;

  // Maps from a thread ID to the preferred (memory) node ID.
  std::vector<$u32> thread_id_to_node_id_map;

  platform();

public:

  static platform& get_instance() {
    static platform instance;
    return instance;
  }

  const dtl::cpu_mask& get_cpu_mask() const { return cpu_mask; }
  u32 get_numa_node_count() const { return numa_node_count; };
  u32 get_thread_count() const { return thread_cnt; };
  u32 get_memory_node_of_thread(u32 thread_id) const;
  u32 get_memory_node_of_this_thread() const;
  u32 get_thread_id() const;

  // Returns the (data) caches sizes in bytes.
  std::vector<$u64> get_cache_sizes() const;

  platform(platform const&) = delete;
  void operator=(platform const&) = delete;

};
//===----------------------------------------------------------------------===//

} // namespace filter
} // namespace dtl
