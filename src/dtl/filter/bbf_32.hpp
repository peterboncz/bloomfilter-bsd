#pragma once

#include <dtl/dtl.hpp>
#include "filter_base.hpp"

namespace dtl {

//===----------------------------------------------------------------------===//
// PImpl wrapper to reduce compilation time.
//===----------------------------------------------------------------------===//
class bbf_32 : public dtl::filter::filter_base {
  class impl;
  std::unique_ptr<impl> pimpl;

public:

  using key_t = $u32;
  using word_t = $u64; // internally, 32-bit words are used

  //===----------------------------------------------------------------------===//
  // The API functions.
  //===----------------------------------------------------------------------===//
  $u1
  insert(word_t* __restrict filter_data, key_t key);

  $u1
  batch_insert(word_t* __restrict filter_data, const key_t* __restrict keys, u32 key_cnt);

  $u1
  contains(const word_t* __restrict filter_data, key_t key) const;

  $u64
  batch_contains(const word_t* __restrict filter_data,
                 const key_t* __restrict keys, u32 key_cnt,
                 $u32* __restrict match_positions, u32 match_offset) const;

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

  bbf_32(std::size_t m, u32 k, u32 word_cnt_per_block = 1, u32 sector_cnt = 1);
  ~bbf_32() override;
  bbf_32(bbf_32&&) noexcept;
  bbf_32(const bbf_32&) = delete;
  bbf_32& operator=(bbf_32&&);
  bbf_32& operator=(const bbf_32&) = delete;

};

} // namespace dtl
