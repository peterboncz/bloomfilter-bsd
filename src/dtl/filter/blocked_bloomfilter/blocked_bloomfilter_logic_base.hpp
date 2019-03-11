#pragma once

#include <dtl/dtl.hpp>

namespace dtl {
//===----------------------------------------------------------------------===//
template<
    typename Tk  // the key type
>
struct blocked_bloomfilter_logic_base {

  using key_t = typename std::remove_cv<Tk>::type;

  void
  insert(word_t* __restrict filter_data, const key_t key) noexcept;

  u1
  contains(const word_t* __restrict filter_data,
      const key_t key) const noexcept;

  };
//===----------------------------------------------------------------------===//
} // namespace dtl