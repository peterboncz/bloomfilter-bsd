#pragma once

#include <bitset>
#include "math.hpp"

#include <iostream> //TODO remove

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

template<size_t num_input_bits, size_t num_output_bits>
std::bitset<num_output_bits> compress(const std::bitset<num_input_bits>& bitmask) {
  static_assert(is_power_of_two(num_input_bits), "Template parameter 'num_input_bits'  must be a power of two.");
  static_assert(is_power_of_two(num_output_bits), "Template parameter 'num_output_bits'  must be a power of two.");

  const size_t n = ct::log_2<num_input_bits>::value;

  size_t num_nodes = 2 * num_input_bits - 1;

  std::cout << std::endl;

  std::bitset<num_input_bits> b(bitmask);
  std::bitset<num_output_bits> compressed_bitmask;
  std::vector<uint32_t> false_positives(num_input_bits, 0);
  std::vector<uint32_t> num_leaves(num_input_bits, 0);
  for (size_t i = 1; i < n; i++) {
    std::cout << "b "; print(b, i-1);
    std::bitset<num_input_bits> t = b >> i;
    std::cout << "t "; print(t, 0);
    std::bitset<num_input_bits> d = b ^ t;
    std::cout << "d "; print(d, i);
    b |= t;
    std::cout << "b "; print(b, i);
    for (size_t j = 0; j < num_input_bits; j += 1 << i) {
      false_positives[j] = d[j] * (1 << (i-1))  + (false_positives[j] + false_positives[j + (1 << (i-1))]);
      num_leaves[j] = !d[j] * (num_leaves[j] + num_leaves[j + (1 << (i-1))]) + 2* d[j];
    }
    std::cout << "false positives "; print(false_positives, i);
    std::cout << "num leaves      "; print(num_leaves, 0);
  }
  std::cout << false_positives[0] << std::endl;
  std::cout << compressed_bitmask[0] << std::endl;

  return compressed_bitmask;
}
