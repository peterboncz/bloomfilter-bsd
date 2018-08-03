#pragma once

#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/mem.hpp>
#include <dtl/filter/cuckoofilter/cuckoofilter_config.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp>
#include <dtl/type_traits.hpp>
#include <dtl/filter/model/calibration_data.hpp>

#include "filter_base.hpp"
#include "bbf_32.hpp"
#include "bbf_64.hpp"
#include "cf.hpp"
#include "zbbf_32.hpp"
#include "zbbf_64.hpp"
#include "platform.hpp"

namespace dtl {
namespace filter {


//===----------------------------------------------------------------------===//
/// Provides an easy to use API for the filter library.
///
/// Usage:
///   filter f(...); // construct the filter
///   f.insert(...); // populate the filter
///   auto probe = f.probe(); // obtain a probe instance
///   match_cnt = probe(keys...) // probe the filter
///
/// Note, that 'insert()' must not be called after a probe instance
/// has been created. Otherwise, the behavior is undefined.
class filter {

private:
  //===----------------------------------------------------------------------===//
  // Helper functions to construct the actual filter based on a given config.
  static
  std::shared_ptr<filter_base>
  construct(const blocked_bloomfilter_config& config, u64 m) {
    switch (config.word_size) {
      case 4: {
        if (config.zone_cnt == 1) {
          std::shared_ptr<filter_base> instance = std::make_shared<dtl::bbf_32>(m, config.k, config.word_cnt_per_block, config.sector_cnt);
          return instance;
        }
        else {
          std::shared_ptr<filter_base> instance = std::make_shared<dtl::zbbf_32>(m, config.k, config.word_cnt_per_block, config.zone_cnt);
          return instance;
        }
      }
      case 8: {
        if (config.zone_cnt == 1) {
          std::shared_ptr<filter_base> instance = std::make_shared<dtl::bbf_64>(m, config.k, config.word_cnt_per_block, config.sector_cnt);
          return instance;
        }
        else {
          std::shared_ptr<filter_base> instance = std::make_shared<dtl::zbbf_64>(m, config.k, config.word_cnt_per_block, config.zone_cnt);
          return instance;
        }
      }
      default: throw std::runtime_error("Illegal configuration. Word size must be either 4 or 8.");
    }
  }

  static std::shared_ptr<filter_base>
  construct(const cuckoofilter::config& config, u64 m) {
    std::shared_ptr<filter_base> instance = std::make_shared<dtl::cf>(m, config.bits_per_tag, config.tags_per_bucket);
    return instance;
  }

  // constructs a filter for the given n, tw values. // TODO implement refinement with time budget!
  static std::shared_ptr<filter_base>
  construct(u64 n, u64 tw, u64 time_budget_millis, const model::calibration_data& data) {
    auto bbf_config_candidates = data.get_skyline_matrix()->get_candidate_bbf_configs(n, tw);
    if (bbf_config_candidates.size() == 0) {
      throw std::runtime_error("Failed to determine filter configuration due to missing skyline matrix.");
    }
    return construct(bbf_config_candidates[0].config_, bbf_config_candidates[0].m_);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // The shared filter state, intended to be allocated on heap memory.
  struct filter_shared {
    // Pointer to the actual filter instance
    std::shared_ptr<filter_base> filter_instance;

    using data_t = std::vector<$u64, dtl::mem::numa_allocator<$u64>>;

    // The filter data. (allocated on the preferred memory node of the creating thread)
    data_t filter_data;
    // The NUMA node ID where the filter data got allocated.
    i32 filter_data_node;

    // Replicas of the filter data (used on NUMA architectures).
    std::vector<data_t> filter_data_replicas;
    // Locks, that are acquired during replication.
    std::vector<std::mutex> replica_mutexes;

    static auto
    get_default_allocator() {
      const auto& platform = dtl::filter::platform::get_instance();
      u32 mem_node_id = platform.get_memory_node_of_this_thread();
      auto alloc_config = dtl::mem::allocator_config::on_node(mem_node_id);
      dtl::mem::numa_allocator<$u64> allocator(alloc_config);
      return allocator;
    }

    explicit
    filter_shared(std::shared_ptr<filter_base> filter_instance)
        : filter_instance(filter_instance),
          filter_data(filter_instance->size(), 0, get_default_allocator()),
          filter_data_node(dtl::mem::get_node_of_address(&filter_data[0])),
          filter_data_replicas(platform::get_instance().get_numa_node_count()),
          replica_mutexes(platform::get_instance().get_numa_node_count()) { }

    filter_shared(const filter_shared&) = delete;
    filter_shared(filter_shared&&) = delete;
    filter_shared& operator=(const filter_shared&) = delete;
    filter_shared& operator=(filter_shared&&) = delete;

    /// Returns the pointer to the thread-local filter data.
    /// If required, the filter data is replicated.
    u64*
    get_filter_data_ptr() {
      const auto& platform = dtl::filter::platform::get_instance();
      if (platform.get_numa_node_count() == 1) {
        return &filter_data[0];
      }

      u32 mem_node_id = platform.get_memory_node_of_this_thread();
      if (mem_node_id == filter_data_node) {
        return &filter_data[0];
      }

      if (filter_data_replicas[mem_node_id].size() == 0) {
        std::lock_guard<std::mutex> guard(replica_mutexes[mem_node_id]);
        // critical section
        if (filter_data_replicas[mem_node_id].size() == 0) { // double check
          // replicate the filter data
          auto alloc_config = dtl::mem::allocator_config::on_node(mem_node_id);
          dtl::mem::numa_allocator<$u64> allocator(alloc_config);
          data_t replica(filter_data.begin(), filter_data.end(), allocator);
          std::swap(replica, filter_data_replicas[mem_node_id]);
        }
      }
      return &filter_data_replicas[mem_node_id][0];
    }
  };
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Pointer to the shared state.
  std::shared_ptr<filter_shared> shared_filter_instance;
  //===----------------------------------------------------------------------===//

public:

  //===----------------------------------------------------------------------===//
  /// C'tor
  filter(const blocked_bloomfilter_config& config, u64 m)
      : shared_filter_instance(std::make_shared<filter_shared>(construct(config, m))) { }
  filter(const cuckoofilter::config& config, u64 m)
      : shared_filter_instance(std::make_shared<filter_shared>(construct(config, m))) { }
  filter(u64 n, u64 tw, u64 time_budget_millis = 1000)
      : shared_filter_instance(std::make_shared<filter_shared>(construct(n, tw, time_budget_millis, model::calibration_data::get_default_instance()))) { }
  filter(u64 n, u64 tw, const model::calibration_data& calibration_data, u64 time_budget_millis = 1000)
      : shared_filter_instance(std::make_shared<filter_shared>(construct(n, tw, time_budget_millis, calibration_data))) { }

  filter(const filter&) = default;
  filter(filter&&) = default;
  filter& operator=(const filter&) = default;
  filter& operator=(filter&&) = default;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Insert
  u1
  insert(u32* __restrict begin, u32* __restrict end) {
    auto key_cnt = static_cast<u32>(std::distance(begin, end));
    return shared_filter_instance->filter_instance->batch_insert(&shared_filter_instance->filter_data[0], begin, key_cnt);
  }

  template<typename input_it>
  __forceinline__ u1
  insert(input_it begin, input_it end,
         typename std::enable_if<is_iterator<input_it>::value>::type* = 0) {
    using T = typename std::iterator_traits<input_it>::value_type;
    static_assert(std::is_same<$u32, typename std::remove_cv<T>::type>::value, "Unsupported key type.");
    const T* begin_ptr = &(*begin);
    const T* end_ptr = &(*end);
    return insert(begin_ptr, end_ptr);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Probing the filter is encapsulated in a different class.
  class probe_t {

    /// Pointer to the shared filter instance.
    const std::shared_ptr<filter_shared> shared_filter_instance;
    /// Pointer to the (thread-local) filter data.
    u64* filter_data;

  public:

    //===----------------------------------------------------------------------===//
    explicit
    probe_t(const std::shared_ptr<filter_shared> shared_filter_instance) :
        shared_filter_instance(shared_filter_instance),
        filter_data(shared_filter_instance->get_filter_data_ptr()) { }
    ~probe_t() = default;
    probe_t(const probe_t&) = default;
    probe_t(probe_t&&) = default;
    probe_t& operator=(const probe_t&) = default;
    probe_t& operator=(probe_t&&) = default;

    // Returns the NUMA node of the corresponding filter data. (used for testing)
    i32 get_numa_node_id() {
      return dtl::mem::get_node_of_address(filter_data);
    }
    //===----------------------------------------------------------------------===//

    //===----------------------------------------------------------------------===//
    // C-style API
    $u64
    operator()(u32* __restrict begin, u32* __restrict end,
               $u32* __restrict match_pos, $u32 match_offset = 0) const {
      auto key_cnt = static_cast<u32>(std::distance(begin, end));
      return shared_filter_instance->filter_instance->batch_contains(
          &shared_filter_instance->filter_data[0], begin, key_cnt, match_pos, match_offset);
    }

    // C++ style API
    template<typename input_it>
    __forceinline__ $u32
    operator()(const input_it begin, const input_it end,
               $u32* __restrict match_pos, $u32 match_offset = 0) const {
      using T = typename std::iterator_traits<input_it>::value_type;
      static_assert(std::is_same<$u32, typename std::remove_cv<T>::type>::value, "Unsupported key type.");
      const T* begin_ptr = &(*begin);
      const T* end_ptr = &(*end);
      return (*this)(begin_ptr, end_ptr, match_pos, match_offset);
    }
    //===----------------------------------------------------------------------===//
  };

  /// Returns a probe instance.
  probe_t probe() {
    return probe_t(shared_filter_instance);
  }
  //===----------------------------------------------------------------------===//

};

} // namespace filter
} // namespace dtl