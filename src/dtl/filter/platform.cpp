#include <algorithm>
#include <cctype>

#include <dtl/dtl.hpp>
#include <dtl/mem.hpp>
#include <fstream>

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

inline u1
file_exists(const std::string& name) {
  std::ifstream f(name.c_str());
  return f.good();
}

std::vector<$u64>
platform::get_cache_sizes() const {
  // WARNING: non-portable code
  const std::string sys_path = "/sys/devices/system/cpu/cpu0/cache";
  std::vector<$u64> cache_sizes;

  $i64 idx = -1;
  while(true) {
    idx++;
    const std::string type_filename = sys_path + "/index" + std::to_string(idx) + "/type";
    const std::string size_filename = sys_path + "/index" + std::to_string(idx) + "/size";
    if (!file_exists(type_filename) || !file_exists(size_filename)) break;

    std::ifstream type_file(type_filename);
    std::string type_str;
    std::getline(type_file, type_str);
    type_file.close();
    if (type_str == "Instruction") continue; // skip instruction cache

    std::ifstream size_file(size_filename);
    std::string size_str;
    std::getline(size_file, size_str);
    size_file.close();

    $u64 cache_size = 0;

    // parsing inspired from: http://www.cs.columbia.edu/~orestis/vbf.c
    auto iter = size_str.begin();
    while (iter != size_str.end() && isdigit(*iter)) {
      cache_size = (cache_size * 10) + *iter++ - '0';
    }
    if (iter != size_str.end()) {
      const auto unit = *iter;
      switch (unit) {
        case 'K': cache_size <<= 10; break;
        case 'M': cache_size <<= 20; break;
        case 'G': cache_size <<= 30; break;
        default:
          throw std::runtime_error("Failed to parse cache size: " + size_str);
      }
    }
    cache_sizes.push_back(cache_size);
  }

  if (cache_sizes.size() == 0) {
    throw std::runtime_error("Failed to determine the cache sizes.");
  }
  return cache_sizes;
}
//===----------------------------------------------------------------------===//

} // namespace filter
} // namespace dtl
