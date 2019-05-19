#pragma once

#include <vector>
#include <amsfilter/amsfilter.hpp>
#include <chrono>

namespace amsfilter {
namespace model {
//===----------------------------------------------------------------------===//
/// Returns all valid filter configurations.
static std::vector<Config>
get_valid_configs() {
  std::vector<Config> configs;
  for (auto k : {1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u,12u,13u,14u,15u,16u}) {
    for (auto word_size : {4u,8u}) {
      for (auto word_cnt : {1u,2u,4u,8u,16u,32u,64u}) {
        for (auto sector_cnt : {1u,2u,4u,8u,16u,32u,64u}) {
          for (auto zone_cnt : {1u,2u,4u,8u,16u,32u,64u}) {
            Config c;
            c.k = k;
            c.word_size = word_size;
            c.word_cnt_per_block = word_cnt;
            c.sector_cnt = sector_cnt;
            c.zone_cnt = zone_cnt;
            c.addr_mode = dtl::block_addressing::POWER_OF_TWO;
            try {
              AmsFilter instance(c, 1024);
              configs.push_back(c);
            } catch (...) {}
//            c.addr_mode = dtl::block_addressing::MAGIC;
//            try {
//              AmsFilter instance(c, 1024);
//              configs.push_back(c);
//            } catch (...) {}
          }
        }
      }
    }
  }
  return configs;
}
//===----------------------------------------------------------------------===//
static u64
now_nanos() {
  return static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}
//===----------------------------------------------------------------------===//
static u32
config_to_int(const Config& config) {
  $u32 enc = 0;
  const auto w_log2 = dtl::log_2(config.word_cnt_per_block);
  const auto s_log2 = dtl::log_2(config.sector_cnt);
  const auto z_log2 = dtl::log_2(config.zone_cnt);
  enc |= w_log2;
  enc |= s_log2 << 3;
  enc |= z_log2 << 6;
  enc |= (config.k == 16 ? 0 : config.k) << 9; // FIXME patched, as we don't want to invalidate existing databases.
  const auto addr = (config.addr_mode == dtl::block_addressing::MAGIC ? 1 : 0);
  enc |= addr << 13;
  return enc;
}
//===----------------------------------------------------------------------===//
static Config
config_from_int(u32 enc) {
  Config c;
  c.word_cnt_per_block = 1u << dtl::bits::extract(enc, 0, 3);
  c.sector_cnt = 1u << dtl::bits::extract(enc, 3, 3);
  c.zone_cnt = 1u << dtl::bits::extract(enc, 6, 3);
  c.k = dtl::bits::extract(enc, 9, 4);
  if (c.k == 0) c.k = 16; // FIXME patched, as we don't want to invalidate existing databases.
  c.addr_mode = dtl::bits::extract(enc, 13, 1) == 0
      ? dtl::block_addressing::POWER_OF_TWO
      : dtl::block_addressing::MAGIC;
  return c;
}
//===----------------------------------------------------------------------===//
} // namespace model
} // namespace amsfilter
