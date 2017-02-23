#pragma once

namespace dtl {

enum class op {
  GE, GT, EQ, LT, LE,
  IS_NULL, IS_NOT_NULL,
  IN,
  BETWEEN, BETWEEN_LO, BETWEEN_RO, BETWEEN_O
};


/// a monadic predicate (e.g., attr OP const)
struct predicate {
  op comparision_operator;
  void* value_ptr;
  void* second_value_ptr = nullptr; // in case of BETWEEN
};


struct range {
  $u32 begin;
  $u32 end;

  inline bool is_empty() const {
    return begin == end;
  }

  inline range operator|(const range &other) {
    if (is_empty()) return other;
    if (other.is_empty()) return range{begin, end};
    return range{std::min(begin, other.begin), std::max(end, other.end)};
  }

  inline range operator&(const range &other) {
    if (is_empty()) return range{0, 0};
    if (other.is_empty()) return range{0, 0};
    range r{std::max(begin, other.begin), std::min(end, other.end)};
    if (r.begin >= r.end) r = {0, 0};
    return r;
  }
};

} // namespace dtl