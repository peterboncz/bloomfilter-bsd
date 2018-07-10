#pragma once

#include <dtl/dtl.hpp>

namespace dtl {

//===----------------------------------------------------------------------===//
// Pimpl wrapper to reduce compilation time.
//===----------------------------------------------------------------------===//
class zbbf_32 {
  class impl;
  std::unique_ptr<impl> pimpl;

public:

  using key_t = $u32;
  using word_t = $u32;

  //===----------------------------------------------------------------------===//
  // The API functions.
  //===----------------------------------------------------------------------===//
  void
  insert(word_t* __restrict filter_data, key_t key);

  void
  batch_insert(word_t* __restrict filter_data, const key_t* keys, u32 key_cnt);

  $u1
  contains(const word_t* __restrict filter_data, key_t key) const;

  $u64
  batch_contains(const word_t* __restrict filter_data,
                 const key_t* keys, u32 key_cnt,
                 $u32* match_positions, u32 match_offset) const;

  std::string
  name() const;

  std::size_t
  size_in_bytes() const;

  std::size_t
  size() const;

  static void
  calibrate();

  static void
  force_unroll_factor(u32 u);
  //===----------------------------------------------------------------------===//

  zbbf_32(std::size_t m, u32 k, u32 word_cnt_per_block = 4, u32 zone_cnt = 2);
  ~zbbf_32();
  zbbf_32(zbbf_32&&) noexcept;
  zbbf_32(const zbbf_32&) = delete;
  zbbf_32& operator=(zbbf_32&&);
  zbbf_32& operator=(const zbbf_32&) = delete;

};

} // namespace dtl
