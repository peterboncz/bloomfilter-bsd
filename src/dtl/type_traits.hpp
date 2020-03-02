#pragma once
//===----------------------------------------------------------------------===//
#include <dtl/dtl.hpp>

#include <iterator>
#include <type_traits>
//===----------------------------------------------------------------------===//
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
// Polyfill for C++20
// adapted from https://en.cppreference.com/w/cpp/experimental/is_detected
struct nonesuch {
  ~nonesuch() = delete;
  nonesuch(nonesuch const&) = delete;
  void operator=(nonesuch const&) = delete;
};

namespace detail {
template <class Default, class AlwaysVoid,
    template<class...> class Op, class... Args>
struct detector {
  using value_t = std::false_type;
  using type = Default;
};

template <class Default, template<class...> class Op, class... Args>
struct detector<Default, std::void_t<Op<Args...>>, Op, Args...> {
  using value_t = std::true_type;
  using type = Op<Args...>;
};

} // namespace detail

template <template<class...> class Op, class... Args>
using is_detected = typename detail::detector<nonesuch, void, Op, Args...>::value_t;

template <template<class...> class Op, class... Args>
using detected_t = typename detail::detector<nonesuch, void, Op, Args...>::type;

template <class Default, template<class...> class Op, class... Args>
using detected_or = detail::detector<Default, void, Op, Args...>;

template<class T>
using copy_assign_t = decltype(std::declval<T&>() = std::declval<const T&>());
//===----------------------------------------------------------------------===//
} // namespace dtl