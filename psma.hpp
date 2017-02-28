#pragma once

#include "index.hpp"
#include <algorithm>
#include <cstring>
#include <type_traits>
#include <vector>

#include <immintrin.h>

namespace dtl {


/// PSMA implementation based on the paper of Lang et al. 'Data Blocks: Hybrid OLTP and OLAP on Compressed Storage
/// using both Vectorization and Compilation'
template<typename T>
class psma {
public:

  // a PSMA consists of 256 range entries for each byte of T
  static constexpr uint32_t size = 256 * sizeof(T);

  range table[size];

  // c'tor
  psma() noexcept {
    // initialize the lookup table with empty range
    for (uint32_t i = 0; i < size; i++) {
      table[i] = {0, 0};
    }
  }

  // compute the PSMA slot for a given value
  inline uint32_t get_slot(const T value) const noexcept {
    // number of remaining bytes (note: clz is undefined for 0)
    const uint64_t r = value ? (7 - (__builtin_clzll(value) >> 3)) : 0;
    // the index of the most significant non-zero byte
    const uint64_t m = (value >> (r << 3));
    // return the slot in lookup table
    return static_cast<uint32_t>(m + (r << 8));
  }

  // update ranges
  inline void update(const T* const values, const size_t n) noexcept {
    for (uint32_t i = 0; i != n; i++) {
      auto &range = table[get_slot(values[i])];
      if (range.is_empty()) {
        range = {i, i + 1};
      }
      else {
        range.end = i + 1;
      }
    }
  }

  // query: x op value
  inline range lookup(const op p, const T value) const noexcept {
    const uint32_t s = get_slot(value);
    range r = table[s];
    if (p == op::EQ) return r;

    uint32_t b = 0;
    uint32_t e = 0;
    switch (p) {
      case op::LT:
      case op::LE:
        b = 0;
        e = s;
        break;
      case op::GT:
      case op::GE:
        b = s + 1;
        e = size;
        break;
    }
    for (size_t i = b; i < e; i++) {
      r = r | table[i];
    }
    return r;
  }

  // query: x between value_lower and value_upper
  inline range lookup(const op /*p*/, const T value_lower, const T value_upper) const noexcept {
    // note: the between predicate type is ignored here
    const uint32_t b = get_slot(value_lower);
    const uint32_t e = get_slot(value_upper);
    range r = table[b];
    for (size_t i = b + 1; i <= e; i++) {
      r = r | table[i];
    }
    return r;
  }

  // query: x op value
  inline range lookup(const predicate& p) const noexcept {
    T value = *reinterpret_cast<T*>(p.value_ptr);
    T second_value; // in case of between predicates

    const uint32_t s = get_slot(value);
    range r = table[s];
    if (p.comparison_operator == op::EQ) return r;

    uint32_t b = 0;
    uint32_t e = 0;
    switch (p.comparison_operator) {
      case op::LT:
      case op::LE:
        b = 0;
        e = s;
        break;
      case op::GT:
      case op::GE:
        b = s + 1;
        e = size;
        break;
      case op::BETWEEN:
      case op::BETWEEN_LO:
      case op::BETWEEN_RO:
      case op::BETWEEN_O:
        second_value = *reinterpret_cast<T*>(p.second_value_ptr);
        b = get_slot(value);
        e = get_slot(second_value);
        break;
    }
    for (size_t i = b; i < e; i++) {
      r = r | table[i];
    }
    return r;
  }

};

} // namespace dtl