#pragma once

namespace dtl {
namespace cuckoofilter {
namespace internal {


template<std::size_t bits_per_tag, std::size_t tags_per_bucket>
struct find_tag_in_buckets_simd {

  // Used to determine whether a SIMD implementation is available.
  static constexpr u1 vectorized = false;

};


} // namespace internal
} // namespace cuckoofilter
} // namespace dtl
