#pragma once

#include <dtl/dtl.hpp>

namespace dtl {
namespace filter {

//===----------------------------------------------------------------------===//
// The base class for all filters.
//===----------------------------------------------------------------------===//
class filter_base {

public:
  using key_t = $u32;
  using word_t = $u64;

  //===----------------------------------------------------------------------===//
  // The API functions.
  //===----------------------------------------------------------------------===//
  /// Inserts a single key. - Note: Use batched inserts for better performance.
  virtual $u1
  insert(word_t* __restrict filter_data, key_t key) = 0;

  /// Inserts a batch of keys.
  virtual $u1
  batch_insert(word_t* __restrict filter_data, const key_t* __restrict keys, u32 key_cnt) = 0;

  /// Probes the filter for a single key.
  /// Note: Probing a single key at-a-time is significantly(!) slower than
  ///       batch-probing the filter.
  virtual $u1
  contains(const word_t* __restrict filter_data, key_t key) const = 0;

  /// Batch-probes the filter for the given keys.
  /// The return value is the number of matches and the match positions
  /// are written to the 'match_positions' buffer.
  ///
  /// Example: Assuming the keys at position 0, 1, and 42 are positive
  ///          and the 'match_offset' is 1000, then the following values
  ///          are written to 'match_positions': [1000, 1001, 1042]
  virtual $u64
  batch_contains(const word_t* __restrict filter_data,
                 const key_t* __restrict keys, u32 key_cnt,
                 $u32* __restrict match_positions, u32 match_offset) const = 0;

  /// Returns the name (and the parameters) of the filter as a string in JSON format.
  virtual std::string
  name() const = 0;

  /// Returns the filter size in bytes.
  virtual std::size_t
  size_in_bytes() const = 0;

  /// Returns the filter size in number of 64-bit words.
  virtual std::size_t
  size() const = 0;

  //===----------------------------------------------------------------------===//

//  filter_base() = default;
  virtual ~filter_base() = default;
//  filter_base(filter_base&&) noexcept = 0;
//  filter_base(filter_base&&) noexcept = default;
//  filter_base(const filter_base&) = delete;
  virtual filter_base& operator=(filter_base&&) = default;
//  filter_base& operator=(const filter_base&) = default;

};

} // namespace filter
} // namespace dtl