#pragma once

enum class pred {
  GE, GT, EQ, LT, LE,
  BETWEEN, BETWEEN_LO, BETWEEN_RO, BETWEEN_O
};

struct range {
  uint32_t begin;
  uint32_t end;

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
