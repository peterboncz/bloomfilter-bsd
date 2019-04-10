#pragma once

#include <dtl/dtl.hpp>

namespace dtl {
//===----------------------------------------------------------------------===//
/// The base class for all blocked Bloom filter.
struct blocked_bloomfilter_logic_base {

  using key_t = $u32;
  using word_t = $u32;

  virtual void
  insert(word_t* __restrict filter_data, const key_t key) noexcept = 0;

  __host__ __device__
  virtual u1
  contains(const word_t* __restrict filter_data, const key_t key) const
    noexcept = 0;

  virtual void
  batch_contains_bitmap(const word_t* __restrict filter_data,
      const key_t* __restrict keys, u32 key_cnt,
      word_t* __restrict bitmap, u32 unroll_factor) const = 0;

  virtual std::size_t
  word_cnt() const noexcept = 0;

  virtual
  ~blocked_bloomfilter_logic_base() = default;

};
//===----------------------------------------------------------------------===//
} // namespace dtl