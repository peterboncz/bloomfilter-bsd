#pragma once

#include <dtl/dtl.hpp>
#include <dtl/filter/cuckoofilter/cuckoofilter_config.hpp>

#include "filter_base.hpp"
#include "bbf_32.hpp"
#include "bbf_64.hpp"
#include "cf.hpp"
#include "zbbf_32.hpp"
#include "zbbf_64.hpp"

namespace dtl {
namespace filter {

enum class type {
  BLOCKED_BLOOM = 0,
  CUCKOO = 1,
};

class filter {

  type type_;

public:
//  using key_t = $u32;
//  using word_t = $u32;

  //===----------------------------------------------------------------------===//
  // The API functions.
  //===----------------------------------------------------------------------===//
//  $u1
//  insert(word_t* __restrict filter_data, key_t key);
//
//  $u1
//  batch_insert(word_t* __restrict filter_data, const key_t* __restrict keys, u32 key_cnt);
//
//  $u1
//  contains(const word_t* __restrict filter_data, key_t key) const;
//
//  $u64
//  batch_contains(const word_t* __restrict filter_data,
//                 const key_t* __restrict keys, u32 key_cnt,
//                 $u32* __restrict match_positions, u32 match_offset) const;
//
//  std::string
//  name() const;
//
//  std::size_t
//  size_in_bytes() const;
//
//  std::size_t
//  size() const;

//  static void
//  calibrate();

//  static void
//  force_unroll_factor(u32 u);
  //===----------------------------------------------------------------------===//

//  filter(std::size_t m, u32 k, u32 word_cnt_per_block = 1, u32 sector_cnt = 1);
//  ~filter();
//  filter(filter&&);
//  filter(const filter&) = delete;
//  filter& operator=(filter&&);
//  filter& operator=(const filter&) = delete;

  static std::unique_ptr<filter_base>
  construct(const blocked_bloomfilter_config& config, u64 m) {
    switch (config.word_size) {
      case 4: {
        if (config.zone_cnt == 1) {
          std::unique_ptr<filter_base> instance = std::make_unique<dtl::bbf_32>(m, config.k, config.word_cnt_per_block, config.sector_cnt);
          return instance;
        }
        else {
          std::unique_ptr<filter_base> instance = std::make_unique<dtl::zbbf_32>(m, config.k, config.word_cnt_per_block, config.zone_cnt);
          return instance;
        }
      }
      case 8: {
        if (config.zone_cnt == 1) {
          std::unique_ptr<filter_base> instance = std::make_unique<dtl::bbf_64>(m, config.k, config.word_cnt_per_block, config.sector_cnt);
          return instance;
        }
        else {
          std::unique_ptr<filter_base> instance = std::make_unique<dtl::zbbf_64>(m, config.k, config.word_cnt_per_block, config.zone_cnt);
          return instance;
        }
      }
      default: throw std::runtime_error("Illegal configuration. Word size must be either 4 or 8.");
    }
  }

  static std::unique_ptr<filter_base>
  construct(const cuckoofilter::config& config, u64 m) {
    std::unique_ptr<filter_base> instance = std::make_unique<dtl::cf>(m, config.bits_per_tag, config.tags_per_bucket);
    return instance;
  }

};

} // namespace filter
} // namespace dtl