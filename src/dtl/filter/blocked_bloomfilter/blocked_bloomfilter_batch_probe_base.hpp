#pragma once

#include <dtl/dtl.hpp>

namespace dtl {
//===----------------------------------------------------------------------===//
struct blocked_bloomfilter_batch_probe_base {

  using key_t = $u32;
  using word_t = $u32;

  virtual void
  batch_contains_bitmap(const word_t* __restrict filter_data,
      const key_t* __restrict keys, u32 key_cnt,
      word_t* __restrict bitmap, u32 unroll_factor) const = 0;

  virtual
  ~blocked_bloomfilter_batch_probe_base() = default;

};
//===----------------------------------------------------------------------===//
} // namespace dtl