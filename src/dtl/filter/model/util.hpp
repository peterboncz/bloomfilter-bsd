#pragma once

#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp>
#include <dtl/filter/cuckoofilter/cuckoofilter_config.hpp>

namespace dtl {
namespace filter {
namespace model {


//===----------------------------------------------------------------------===//
// Returns all valid BBF configurations.
static std::vector<dtl::blocked_bloomfilter_config>
get_valid_bbf_configs() {
  std::vector<dtl::blocked_bloomfilter_config> bbf_configs;
  for (auto k : {1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u,12u,13u,14u,15u,16u}) {
    for (auto word_size : {4u,8u}) {
      for (auto word_cnt : {1u,2u,4u,8u,16u}) {
        if (word_size * word_cnt > 64) continue;
        for (auto sector_cnt : {1u,2u,4u,8u,16u}) {
          if (sector_cnt > word_cnt) continue;
          if (sector_cnt > k) continue;
          if (k % sector_cnt != 0) continue;
          dtl::blocked_bloomfilter_config c;
          c.k = k;
          c.word_size = word_size;
          c.word_cnt_per_block = word_cnt;
          c.sector_cnt = sector_cnt;
          c.zone_cnt = 1;
          bbf_configs.push_back(c);
        }

        // zoned (sector_cnt = word_cnt)
        if (word_cnt >= 4) {
          const auto sector_cnt = word_cnt;
          for (auto zone_cnt : {2u,4u,8u}) {
            if (zone_cnt >= sector_cnt) continue;
            if (zone_cnt > k) continue;
            if (k % zone_cnt != 0) continue;
            dtl::blocked_bloomfilter_config c;
            c.k = k;
            c.word_size = word_size;
            c.word_cnt_per_block = word_cnt;
            c.sector_cnt = sector_cnt;
            c.zone_cnt = zone_cnt;
            bbf_configs.push_back(c);
          }
        }

      }
    }
  }
  return bbf_configs;
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Returns all valid CF configurations.
static std::vector<dtl::cuckoofilter::config>
get_valid_cf_configs() {
  std::vector<dtl::cuckoofilter::config> cf_configs;
  for (auto bits_per_tag : {4u,8u,12u,16u}) {
    for (auto tags_per_bucket : {1u,2u,4u}) {
      dtl::cuckoofilter::config c;
      c.bits_per_tag = bits_per_tag;
      c.tags_per_bucket = tags_per_bucket;
      cf_configs.push_back(c);
    }
  }
  return cf_configs;
}
//===----------------------------------------------------------------------===//


} // namespace model
} // namespace filter
} // namespace dtl
