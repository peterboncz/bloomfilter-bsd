#pragma once

#include <sstream>

#include <dtl/dtl.hpp>
#include <dtl/bloomfilter/block_addressing_logic.hpp>

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
  operator<(const blocked_bloomfilter_config &o) const {
    return k < o.k
        || (k == o.k && word_size  < o.word_size)
        || (k == o.k && word_size == o.word_size && word_cnt_per_block  < o.word_cnt_per_block)
        || (k == o.k && word_size == o.word_size && word_cnt_per_block == o.word_cnt_per_block && sector_cnt < o.sector_cnt)
        || (k == o.k && word_size == o.word_size && word_cnt_per_block == o.word_cnt_per_block && sector_cnt == o.sector_cnt && addr_mode < o.addr_mode)
        || (k == o.k && word_size == o.word_size && word_cnt_per_block == o.word_cnt_per_block && sector_cnt == o.sector_cnt && addr_mode == o.addr_mode && zone_cnt < o.zone_cnt);
  }

  void print(std::ostream& os) const {
    std::stringstream str;
    str << "bbf"
        << ",k=" << k
        << ",word_size=" << word_size
        << ",word_cnt=" << word_cnt_per_block
        << ",sector_cnt=" << sector_cnt
        << ",addr=" << (addr_mode == dtl::block_addressing::POWER_OF_TWO ? "pow2" : "magic")
        << ",zone_cnt=" << zone_cnt;
    os << str.str();
  }

};
//===----------------------------------------------------------------------===//

} // namespace dtl