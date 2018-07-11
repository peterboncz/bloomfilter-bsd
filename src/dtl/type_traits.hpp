#pragma once

#include <iterator>
#include <type_traits>

#include <dtl/dtl.hpp>

namespace dtl {


//===----------------------------------------------------------------------===//
// source: https://stackoverflow.com/questions/12032771/how-to-check-if-an-arbitrary-type-is-an-iterator
template<typename T, typename = void>
struct is_iterator {
  static constexpr bool value = false;
};

template<typename T>
struct is_iterator<T, typename std::enable_if<!std::is_same<typename std::iterator_traits<T>::value_type, void>::value>::type> {
  static constexpr bool value = true;
};
//===----------------------------------------------------------------------===//


} // namespace dtl