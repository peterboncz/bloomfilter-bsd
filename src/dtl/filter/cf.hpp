#pragma once

#include <dtl/dtl.hpp>

namespace dtl {

//===----------------------------------------------------------------------===//
// Pimpl wrapper to reduce compilation time.
//===----------------------------------------------------------------------===//
class cf {
  class impl;
  std::unique_ptr<impl> pimpl;
  std::size_t bits_per_tag;
  std::size_t tags_per_bucket;

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

  // Cuckoo specific functions
  std::size_t
  get_bits_per_tag() const {
    return bits_per_tag;
  };

  std::size_t
  get_tags_per_bucket() const {
    return tags_per_bucket;
  };

  std::size_t
  get_bucket_count() const;

  std::size_t
  count_occupied_slots(const word_t* __restrict filter_data) const;

  //===----------------------------------------------------------------------===//

  cf(std::size_t m, u32 bits_per_tag = 16, u32 tags_per_bucket = 4);
  ~cf();
  cf(cf&&);
  cf(const cf&) = delete;
  cf& operator=(cf&&);
  cf& operator=(const cf&) = delete;

};

} // namespace dtl
