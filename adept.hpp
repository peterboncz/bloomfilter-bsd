#pragma once

#include <iostream>
#include <cstdint>

using i1 = const bool;
using u1 = const bool;
using i8 = const int8_t;
using u8 = const uint8_t;
using i16 = const int16_t;
using u16 = const uint16_t;
using i32 = const int32_t;
using u32 = const uint32_t;
using i64 = const int64_t;
using u64 = const uint64_t;

using $i1 = bool;
using $u1 = bool;
using $i8 = int8_t;
using $u8 = uint8_t;
using $i16 = int16_t;
using $u16 = uint16_t;
using $i32 = int32_t;
using $u32 = uint32_t;
using $i64 = int64_t;
using $u64 = uint64_t;

using f32 = const float;
using f64 = const double;

using $f32 = float;
using $f64 = double;


// polyfill until C++14
template<u64...>
struct integer_sequence {};

template<u64 N, u64... Ints>
struct make_integer_sequence : make_integer_sequence<N - 1, N - 1, Ints...> {};

template<u64... Ints>
struct make_integer_sequence<0, Ints...> : integer_sequence<Ints...> {};


// Compiler hints
#define assume_aligned(address, byte) __builtin_assume_aligned(address, byte)
#define unreachable() __builtin_unreachable();


// add missing operator function objects
namespace std {

  template<typename T>
  struct bit_shift_left {
    constexpr T operator()(const T &lhs, const T &rhs) const {
      return lhs << rhs;
    }
  };

  template<typename T>
  struct bit_shift_right {
    constexpr T operator()(const T &lhs, const T &rhs) const {
      return lhs >> rhs;
    }
  };

  template<typename T>
  struct post_increment {
    constexpr T operator()(const T &lhs) const {
      return lhs++;
    }
  };

  template<typename T>
  struct post_decrement {
    constexpr T operator()(const T &lhs) const {
      return lhs--;
    }
  };

}

#include <bitset>
#include <vector>

template<size_t n>
void print(const std::bitset<n>& b) {
  for (size_t i = 0; i < n; i++) {
    std::cout << b[i];
  }
  std::cout << std::endl;
}

template<size_t n>
void print(const std::bitset<n>& b, const size_t l) {
  const size_t x = 1 << l;
  for (size_t i = 0; i < n; i++) {
    if (i % x == 0) {
      std::cout << b[i];
    }
    else {
      std::cout << "_";
    }
  }
  std::cout << std::endl;
}

template<typename T>
void print(const std::vector<T>& v, const size_t l) {
  const size_t x = 1 << l;
  for (size_t i = 0; i < v.size(); i++) {
    if (i % x == 0) {
//      std::cout << v[v.size() - (i + 1)] << ", ";
      std::cout << v[i] << ", ";
    }
    else {
      std::cout << "_, ";
    }
  }
  std::cout << std::endl;
}

static void print(const std::vector<bool>& v, const size_t l) {
  const size_t x = 1 << l;
  for (size_t i = 0; i < v.size(); i++) {
    if (i % x == 0) {
      std::cout << v[i];
    }
    else {
      std::cout << "_, ";
    }
  }
  std::cout << std::endl;
}
