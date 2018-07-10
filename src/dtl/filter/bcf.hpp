#pragma once

#include <dtl/dtl.hpp>
#include "filter_base.hpp"

namespace dtl {

//===----------------------------------------------------------------------===//
// PImpl wrapper to reduce compilation time.
//===----------------------------------------------------------------------===//
class bcf /*: public dtl::filter::filter_base*/ {
  class impl;
  std::unique_ptr<impl> pimpl;

public:

  using key_t = $u32;
  using word_t = $u32;

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

  void print();

  //===----------------------------------------------------------------------===//

  explicit
  bcf(size_t m, u32 block_size_bytes = 64, u32 tag_size_bits = 16, u32 associativity = 4);
  ~bcf();
  bcf(bcf&&) noexcept;
  bcf(const bcf&) = delete;
  bcf& operator=(bcf&&);
  bcf& operator=(const bcf&) = delete;

};

} // namespace dtl
