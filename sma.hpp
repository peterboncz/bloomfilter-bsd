#pragma once

#include "adept.hpp"
#include "index.hpp"
#include <limits>

namespace dtl {

template<typename T>
class sma {

public:
  T min_value;
  T max_value;

  sma() {
    min_value = std::numeric_limits<T>::min();
    max_value = std::numeric_limits<T>::max();
  }

  inline void
  update(const T *const values, const size_t n) noexcept {
    min_value = std::numeric_limits<T>::max();
    max_value = std::numeric_limits<T>::min();
    for (uint32_t i = 0; i != n; i++) {
      const T v = values[i];
      if (v < min_value) {
        min_value = v;
      }
      if (v > max_value) {
        max_value = v;
      }
    }
  }

  // query: x between value_lower and value_upper
  inline bool
  lookup(const op p, const T value_lower, const T value_upper) const noexcept {
    const bool left_inclusive = p == op::BETWEEN || p == op::BETWEEN_RO;
    const bool right_inclusive = p == op::BETWEEN || p == op::BETWEEN_LO;

    const T lo = value_lower + !left_inclusive;
    const T hi = value_upper - !right_inclusive;

    return lo >= min_value && hi <= max_value && lo <= hi;
  }

  // query: x op value
  inline bool
  lookup(const op p, const T value) const noexcept {
    switch (p) {
      case op::EQ:
        return lookup(op::BETWEEN_O, value, value);
      case op::LT:
        return lookup(op::BETWEEN_LO, std::numeric_limits<T>::min(), value);
      case op::LE:
        return lookup(op::BETWEEN_O, std::numeric_limits<T>::min(), value);
      case op::GT:
        return lookup(op::BETWEEN_RO, value, std::numeric_limits<T>::max());
      case op::GE:
        return lookup(op::BETWEEN_O, value, std::numeric_limits<T>::max());
    }
    return true;
  }

  inline bool
  lookup(const predicate& p) const noexcept {
    T value = *reinterpret_cast<T*>(p.value_ptr);
    T second_value; // in case of between predicates
    switch (p.comparison_operator) {
      case op::EQ:
      case op::LT:
      case op::LE:
      case op::GT:
      case op::GE:
        return lookup(p.comparison_operator, value);
      case op::BETWEEN:
      case op::BETWEEN_LO:
      case op::BETWEEN_RO:
      case op::BETWEEN_O:
        second_value = *reinterpret_cast<T*>(p.second_value_ptr);
        return lookup(p.comparison_operator, value, second_value);
    }

  }

};

} // namespace dtl