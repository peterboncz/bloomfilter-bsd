#pragma once

#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/block_addressing_logic.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_batch_probe_base.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_logic.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_block_logic_sgew.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_block_logic_sltw.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_block_logic_zoned.hpp>
#include <dtl/filter/blocked_bloomfilter/hash_family.hpp>

namespace amsfilter {
namespace internal {
//===----------------------------------------------------------------------===//
// Typedefs. - Currently, only 32-bit keys are supported.
using key_t = $u32;
using hash_value_t = $u32;
using word_t = $u32;
//===----------------------------------------------------------------------===//
/// The base class for all blocked Bloom filters.
using filter_base_t = dtl::blocked_bloomfilter_logic_base;
using filter_batch_probe_base_t = dtl::blocked_bloomfilter_batch_probe_base;
//===----------------------------------------------------------------------===//
} // namespace internal
} // namespace amsfilter
