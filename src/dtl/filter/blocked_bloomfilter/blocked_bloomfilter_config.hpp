#pragma once

#include <sstream>

#include <dtl/dtl.hpp>
#include "block_addressing_logic.hpp"

namespace dtl {

//===----------------------------------------------------------------------===//
struct blocked_bloomfilter_config {
  $u32 k = 8;
  $u32 word_size = 4; // [byte]
  $u32 word_cnt_per_block = 1;
  $u32 sector_cnt = 1;
  dtl::block_addressing addr_mode = dtl::block_addressing::POWER_OF_TWO;
  $u32 zone_cnt = 1;

  bool
  operator<(const blocked_bloomfilter_config& o) const {
    return k < o.k
        || (k == o.k && word_size  < o.word_size)
        || (k == o.k && word_size == o.word_size && word_cnt_per_block  < o.word_cnt_per_block)
        || (k == o.k && word_size == o.word_size && word_cnt_per_block == o.word_cnt_per_block && sector_cnt < o.sector_cnt)
        || (k == o.k && word_size == o.word_size && word_cnt_per_block == o.word_cnt_per_block && sector_cnt == o.sector_cnt && addr_mode < o.addr_mode)
        || (k == o.k && word_size == o.word_size && word_cnt_per_block == o.word_cnt_per_block && sector_cnt == o.sector_cnt && addr_mode == o.addr_mode && zone_cnt < o.zone_cnt);
  }

  bool
  operator==(const blocked_bloomfilter_config& o) const {
    return k == o.k && word_size == o.word_size && word_cnt_per_block == o.word_cnt_per_block && sector_cnt == o.sector_cnt && addr_mode == o.addr_mode && zone_cnt == o.zone_cnt;
  }

  bool
  operator!=(const blocked_bloomfilter_config& o) const {
    return !(*this == o);
  }

  void print(std::ostream& os) const {
    std::stringstream str;
    str << "{\"type\":\"bbf\""
        << ",\"word_size\":" << word_size
        << ",\"word_cnt_per_block\":" << word_cnt_per_block
        << ",\"sector_cnt\":" << sector_cnt
        << ",\"zone_cnt\":" << zone_cnt
        << ",\"k\":" << k
        << ",\"addr\":" << (addr_mode == dtl::block_addressing::POWER_OF_TWO
          ? "\"pow2\""
          : "\"magic\"")
        << "}";
    os << str.str();
  }

};
//===----------------------------------------------------------------------===//

} // namespace dtl