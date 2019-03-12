#pragma once

#include <dtl/dtl.hpp>

namespace amsfilter {
//===----------------------------------------------------------------------===//
template<typename T>
struct bitmap_view {
  static_assert(std::is_integral<T>::value,
      "The template parameter must a of an integral type.");

  using storage_type = typename std::remove_cv<T>::type;

  storage_type* bitmap_begin;
  storage_type* bitmap_end;

  inline u1
  test(const std::size_t idx) const noexcept {
    const std::size_t word_idx = idx / bitwidth<storage_type>;
    const std::size_t bit_idx = idx % bitwidth<storage_type>;
    return ((bitmap_begin[word_idx] >> bit_idx) & 1) == 1;
  }

  inline u1
  operator[](const std::size_t idx) const noexcept {
    return test(idx);
  }

  inline storage_type* begin() { return bitmap_begin; }
  inline storage_type* end() { return bitmap_end; }

};
//===----------------------------------------------------------------------===//
} // namespace amsfilter
