#pragma once

#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp>
#include <dtl/filter/cuckoofilter/cuckoofilter_config.hpp>
#include <dtl/filter/blocked_bloomfilter/fpr.hpp>


#include "timing.hpp"

namespace dtl {
namespace filter {
namespace model {


//===----------------------------------------------------------------------===//
class skyline_matrix {

public:

  struct meta_data_t {
    $u64 n_values_count_ = 0;
    $u64 tw_values_count_ = 0;
  };

  struct entry_t {
    blocked_bloomfilter_config config_;
    $u64 m_ = 0;
    timing overhead_;
  };

  meta_data_t meta_data_;

  skyline_matrix() {}

  $u64*
  n_values_begin() {
    return reinterpret_cast<$u64*>((&meta_data_) + 1);
  }

  $u64*
  n_values_end() {
    return n_values_begin() + meta_data_.n_values_count_;
  }

  $u64*
  tw_values_begin() {
    return n_values_end();
  }

  $u64*
  tw_values_end() {
    return tw_values_begin() + meta_data_.tw_values_count_;
  }

  entry_t*
  entries_begin() {
    return reinterpret_cast<entry_t*>(tw_values_end());
  }

  entry_t*
  entries_end() {
    return entries_begin() + (meta_data_.n_values_count_ * meta_data_.tw_values_count_);
  }

  static u64
  size_in_bytes(u64 n_values_count, u64 tw_values_count) {
    return sizeof(meta_data_t)
         + n_values_count * sizeof(u64)
         + tw_values_count * sizeof(u64)
         + n_values_count * tw_values_count * sizeof(entry_t);
  }

  u64
  size_in_bytes() const {
    return sizeof(meta_data_t)
         + meta_data_.n_values_count_ * sizeof(u64)
         + meta_data_.tw_values_count_ * sizeof(u64)
         + meta_data_.n_values_count_ * meta_data_.tw_values_count_ * sizeof(entry_t);
  }

  std::vector<dtl::blocked_bloomfilter_config>
  get_candiate_bbf_configs(u64 n, u64 tw) {
    std::vector<dtl::blocked_bloomfilter_config> ret;
    const auto n_cnt = meta_data_.n_values_count_;
    const auto tw_cnt = meta_data_.tw_values_count_;

    // search n
    if (n_values_begin() == n_values_end()) return ret;
    const auto n_search = std::lower_bound(n_values_begin(), n_values_end(), n);
    auto n_idx = std::distance(n_values_begin(), n_search);
    if (n_idx == n_cnt) {
      n_idx--;
    }

    // search tw
    if (tw_values_begin() == tw_values_end()) return ret;
    const auto tw_search = std::lower_bound(tw_values_begin(), tw_values_end(), tw);
    auto tw_idx = std::distance(tw_values_begin(), tw_search);
    if (tw_idx == tw_cnt) {
      tw_idx--;
    }

    // return the four candidate configurations
    ret.push_back(entries_begin()[(tw_idx * n_cnt) + n_idx].config_);
    if (n_idx + 1 < n_cnt) {
      ret.push_back(entries_begin()[(tw_idx * n_cnt) + (n_idx + 1)].config_);
    }
    if (tw_idx + 1 < tw_cnt) {
      ret.push_back(entries_begin()[((tw_idx + 1) * n_cnt) + n_idx].config_);
    }
    if (n_idx + 1 < n_cnt
        && tw_idx + 1 < tw_cnt) {
      ret.push_back(entries_begin()[((tw_idx + 1) * n_cnt) + (n_idx + 1)].config_);
    }
    return ret;
  }

  void
  serialize($u8* dst) {
    std::memcpy(dst, reinterpret_cast<$u8*>(this), size_in_bytes());
  }

};
//===----------------------------------------------------------------------===//


} // namespace model
} // namespace filter
} // namespace dtl
