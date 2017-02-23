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

  inline void update(const T *const values, const size_t n) noexcept {
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
  inline bool lookup(const op p, const T value_lower, const T value_upper) const noexcept {
    const bool left_inclusive = p == op::BETWEEN_LO || p == op::BETWEEN_O;
    const bool right_inclusive = p == op::BETWEEN_RO || p == op::BETWEEN_O;

    const T lo = value_lower + !left_inclusive;
    const T hi = value_upper - !right_inclusive;

    return lo >= min_value && hi <= max_value && lo <= hi;
  }

  // query: x op value
  inline bool lookup(const op p, const T value) const noexcept {
    uint32_t b = 0;
    uint32_t e = 0;
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

};

} // namespace dtl