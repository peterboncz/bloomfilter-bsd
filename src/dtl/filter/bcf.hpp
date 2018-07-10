#pragma once

#include <dtl/dtl.hpp>

namespace dtl {

//===----------------------------------------------------------------------===//
// Pimpl wrapper to reduce compilation time.
//===----------------------------------------------------------------------===//
class bcf {
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

  void print();

  //===----------------------------------------------------------------------===//

  bcf(size_t m, u32 block_size_bytes = 64, u32 tag_size_bits = 16, u32 associativity = 4);
  ~bcf();
  bcf(bcf&&);
  bcf(const bcf&) = delete;
  bcf& operator=(bcf&&);
  bcf& operator=(const bcf&) = delete;

};

} // namespace dtl
