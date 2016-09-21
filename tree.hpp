#pragma once

#include "adept.hpp"
#include "math.hpp"
#include <bitset>

namespace dbb {

  template<u64 N>
  class full_binary_tree {
    static_assert(is_power_of_two(N), "Template parameter 'N' must be a power of two.");

  public:
    static constexpr u64 max_node_cnt = 2 * N - 1;
    static constexpr u64 height = ct::log_2<N>::value;

  private:

    std::bitset<max_node_cnt> _is_inner_node;

  public:

    full_binary_tree() {
      // initialize a complete binary tree
      // ... all the inner nodes have two children
      for ($u64 i = 0; i < max_node_cnt / 2; i++) {
        _is_inner_node[i] = true;
      }
    }

    static inline u64 parent_of(u64 node_idx) {
      return (node_idx - 1) / 2;
    }

    static inline u64 left_child_of(u64 node_idx) {
      return 2 * node_idx + 1;
    }

    static inline u64 right_child_of(u64 node_idx) {
      return 2 * node_idx + 2;
    }

    static inline u64 level_of(u64 node_idx) {
      return log_2(node_idx + 1);
    }

    inline u1 is_inner_node(u64 node_idx) {
      return _is_inner_node[node_idx];
    }

    inline u1 is_leaf_node(u64 node_idx) {
      return ! _is_inner_node[node_idx];
    }

    inline void set_leaf(u64 node_idx) {
      _is_inner_node[node_idx] = false;
    }

    inline void set_inner(u64 node_idx) {
      _is_inner_node[node_idx] = true;
    }

  };

}
