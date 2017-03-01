#pragma once

#include "index.hpp"
#include "tree_mask.hpp"
#include "zone_mask.hpp"
#include <algorithm>
#include <cstring>
#include <type_traits>
#include <vector>

#include <immintrin.h>

namespace dtl {

/// A PSMA lookup table for the type T. Each table entry consists of an instance of V (e.g., a range in the
/// default implementation).
template<typename T, typename V>
class psma_table {
public:

  // a PSMA consists of 256 range entries for each byte of T
  static constexpr uint32_t size = 256 * sizeof(T);

  V entries[size];

  // compute the PSMA slot for a given value
  inline uint32_t get_slot(const T value) const noexcept {
    // number of remaining bytes (note: clz is undefined for 0)
    const uint64_t r = value ? (7 - (__builtin_clzll(value) >> 3)) : 0;
    // the index of the most significant non-zero byte
    const uint64_t m = (value >> (r << 3));
    // return the slot in lookup table
    return static_cast<uint32_t>(m + (r << 8));
  }


};

/// PSMA implementation based on the paper of Lang et al. 'Data Blocks: Hybrid OLTP and OLAP on Compressed Storage
/// using both Vectorization and Compilation'
template<typename T>
class psma {
public:

  using table_t = psma_table<T, range>;
  static constexpr uint32_t size = table_t::size;

  table_t table;

  // c'tor
  psma() noexcept {
    // initialize the lookup table with empty range
    for (uint32_t i = 0; i < size; i++) {
      table.entries[i] = {0, 0};
    }
  }

  // update ranges
  inline void update(const T* const values, const size_t n) noexcept {
    for (uint32_t i = 0; i != n; i++) {
      auto &range = table.entries[table.get_slot(values[i])];
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
    const uint32_t s = table.get_slot(value);
    range r = table.entries[s];
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
      r = r | table.entries[i];
    }
    return r;
  }

  // query: x between value_lower and value_upper
  inline range lookup(const op /*p*/, const T value_lower, const T value_upper) const noexcept {
    // note: the between predicate type is ignored here
    const uint32_t b = table.get_slot(value_lower);
    const uint32_t e = table.get_slot(value_upper);
    range r = table.entries[b];
    for (size_t i = b + 1; i <= e; i++) {
      r = r | table.entries[i];
    }
    return r;
  }

  // query: x op value
  inline range lookup(const predicate& p) const noexcept {
    T value = *reinterpret_cast<T*>(p.value_ptr);
    T second_value; // in case of between predicates

    const uint32_t s = table.get_slot(value);
    range r = table.entries[s];
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
        b = table.get_slot(value);
        e = table.get_slot(second_value);
        break;
    }
    for (size_t i = b; i < e; i++) {
      r = r | table.entries[i];
    }
    return r;
  }

};

/// A combination of a PSMA lookup table and a Zone Mask.
/// M = the number of bits per table entry.
template<typename T, u64 N, u64 M>
class psma_zone_mask {
public:

  using mask_t = zone_mask<N, M>;
  using table_t = psma_table<T, mask_t>;
  static constexpr uint32_t size = table_t::size;

  table_t table;

  inline void
  update(const T* const values, const size_t n) noexcept {
    for (uint32_t i = 0; i != n; i++) {
      auto& entry = table.entries[table.get_slot(values[i])];
      entry.set(i);
    }
  }

//  // query: x op value
//  inline mask_t
//  lookup(const op p, const T value) const noexcept {
//    const uint32_t s = table.get_slot(value);
//    auto r = table.entries[s];
//    if (p == op::EQ) return r;
//
//    uint32_t b = 0;
//    uint32_t e = 0;
//    switch (p) {
//      case op::LT:
//      case op::LE:
//        b = 0;
//        e = s;
//        break;
//      case op::GT:
//      case op::GE:
//        b = s + 1;
//        e = size;
//        break;
//    }
//    for (size_t i = b; i < e; i++) {
//      r = r | table.entries[i];
//    }
//    return r;
//  }

//  // query: x between value_lower and value_upper
//  inline mask_t
//  lookup(const op /*p*/, const T value_lower, const T value_upper) const noexcept {
//    // note: the between predicate type is ignored here
//    const uint32_t b = table.get_slot(value_lower);
//    const uint32_t e = table.get_slot(value_upper);
//    auto r = table.entries[b];
//    for (size_t i = b + 1; i <= e; i++) {
//      r = r | table.entries[i];
//    }
//    return r;
//  }

  // query: x op value
  inline mask_t
  lookup(const predicate& p) const noexcept {
    T value = *reinterpret_cast<T*>(p.value_ptr);
    T second_value; // in case of between predicates

    const uint32_t s = table.get_slot(value);
    auto r = table.entries[s];
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
        b = table.get_slot(value);
        e = table.get_slot(second_value);
        break;
    }
    for (size_t i = b; i <= e; i++) {
      r = r | table.entries[i];
    }
    return r;
  }

};

template<typename T, u64 N>
using psma_bitmask = psma_zone_mask<T, N, N>;


/// A combination of a PSMA lookup table and a Zone Mask.
/// M = the number of bits per table entry.
template<typename T, u64 N, u64 M>
class psma_tree_mask {
public:

  using mask_t = tree_mask<N, M>;
  using table_t = psma_table<T, mask_t>;
  static constexpr uint32_t size = table_t::size;

  table_t table;

  inline void
  update(const psma_bitmask<T, N>& src) noexcept {
    for (uint32_t i = 0; i != table_t::size; i++) {
      table.entries[i].set(src.table.entries[i].data);
    }
  }

  // query: x op value
  inline std::bitset<N>
  lookup(const op p, const T value) const noexcept {
    const uint32_t s = table.get_slot(value);
    auto r = table.entries[s].get();
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
      r = r | table.entries[i].get();
    }
    return r;
  }

  // query: x between value_lower and value_upper
  inline std::bitset<N>
  lookup(const op /*p*/, const T value_lower, const T value_upper) const noexcept {
    // note: the between predicate type is ignored here
    const uint32_t b = table.get_slot(value_lower);
    const uint32_t e = table.get_slot(value_upper);
    auto r = table.entries[b].get();
    for (size_t i = b + 1; i <= e; i++) {
      r = r | table.entries[i].get();
    }
    return r;
  }

  // query: x op value
  inline std::bitset<N>
  lookup(const predicate& p) const noexcept {
    T value = *reinterpret_cast<T*>(p.value_ptr);
    T second_value; // in case of between predicates

    const uint32_t s = table.get_slot(value);
    auto r = table.entries[s].get();
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
        b = table.get_slot(value);
        e = table.get_slot(second_value);
        break;
    }
    for (size_t i = b; i < e; i++) {
      r = r | table.entries[i].get();
    }
    return r;
  }

};


} // namespace dtl