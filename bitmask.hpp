#pragma once

#include "adept.hpp"
#include "math.hpp"
#include <array>
#include <bitset>
#include <functional>
#include <vector>

namespace tree {

  template<u64 N>
  class tree_mask {
  private:
    static_assert(is_power_of_two(N), "Template parameter 'N' must be a power of two.");

    static constexpr u64 length = 2 * N - 1;
    static constexpr u64 height = ct::log_2<N>::value;
    std::bitset<length> is_inner_node;
    std::bitset<length> bit;
    std::array<$u32, length> false_positive_cnt {};


    static inline u64 parent_of(u64 idx) {
      return (idx - 1) / 2;
    }
    static inline u64 left_child_of(u64 idx) {
      return 2 * idx + 1;
    }
    static inline u64 right_child_of(u64 idx) {
      return 2 * idx + 2;
    }
    static inline u64 level_of(u64 idx) {
      return log_2(idx + 1);
    }

    explicit tree_mask(const std::bitset<N> bitmask) {
      false_positive_cnt.fill(0);
      // initialize a complete binary tree
      // ... all the inner nodes have two children
      for ($u64 i = 0; i < length / 2; i++) {
        is_inner_node[i] = true;
      }
      // ... the leaf nodes are labelled with the given bitmask
      for ($u64 i = length / 2; i < length; i++) {
        is_inner_node[i] = false;
        bit[i] = bitmask[i - length / 2];
      }
      // propagate the mask bits along the tree (bottom-up)
      for ($u64 i = 0; i < length - 1; i++) {
        u64 node_idx = length - i - 1;
        bit[parent_of(node_idx)] = bit[parent_of(node_idx)] | bit[node_idx];
      }
      // bottom-up pruning (loss-less)
      for ($u64 i = 0; i < length - 1; i += 2) {
        u64 left_node_idx = length - i - 2;
        u64 right_node_idx = left_node_idx + 1;

        u1 left_bit = bit[left_node_idx];
        u1 right_bit = bit[right_node_idx];

        u64 parent_node_idx = parent_of(left_node_idx);
        false_positive_cnt[parent_node_idx] = false_positive_cnt[left_node_idx] + false_positive_cnt[right_node_idx];

        u1 prune_causes_false_positives = left_bit ^ right_bit;
        u1 both_nodes_are_leaves = !is_inner_node[left_node_idx] & !is_inner_node[right_node_idx];
        u1 prune = both_nodes_are_leaves & !prune_causes_false_positives;
        if (prune) {
          is_inner_node[parent_node_idx] = false;
        }
        else {
          if (prune_causes_false_positives) {
            u64 left_fp = !left_bit * (1 << (height - level_of(left_node_idx)));
            u64 right_fp = !right_bit * (1 << (height - level_of(right_node_idx)));
            false_positive_cnt[parent_node_idx] = false_positive_cnt[left_node_idx] + false_positive_cnt[right_node_idx]
                                                  + left_fp + right_fp;
          }
        }
      }
      assert(bitmask.count() == 0 || false_positive_cnt[0] == N - bitmask.count());
    }

    std::vector<$u1> encode() {
      std::vector<$u1> structure;
      std::vector<$u1> labels;
      std::function<void(u64)> encode_recursively = [&](u64 idx) {
        u1 is_inner = is_inner_node[idx];
        structure.push_back(true);
        if (is_inner) {
          encode_recursively(left_child_of(idx));
          encode_recursively(right_child_of(idx));
        }
        else {
          labels.push_back(bit[idx]);
        }
        structure.push_back(false);
      };
      encode_recursively(0);
      // append the labels
      std::copy(labels.begin(), labels.end(), std::back_inserter(structure));
      return structure;
    }

    void compress(u64 target_bit_cnt) {
      assert(target_bit_cnt > 2);

      std::function<u64(u64)> node_cnt = [&](u64 node_idx) -> u64 {
        if (!is_inner_node[node_idx]) return 1;
        return node_cnt(left_child_of(node_idx)) + node_cnt(right_child_of(node_idx));
      };

      std::function<u64(u64, u64)> prune_single = [&](u64 node_idx, u64 safe_bits) {
        assert(is_inner_node[node_idx]);

        u64 left_child_idx = left_child_of(node_idx);
        u64 right_child_idx = right_child_of(node_idx);

        u64 left_sub_tree_size = node_cnt(left_child_idx);
        u64 right_sub_tree_size = node_cnt(right_child_idx);

        if (left_sub_tree_size == 1 && right_sub_tree_size == 1) {
          is_inner_node[node_idx] = false;
          return node_idx;
        }

        if (left_sub_tree_size == 1) {
          return prune_single(right_child_idx, safe_bits);
        }

        if (right_sub_tree_size == 1) {
          return prune_single(left_child_idx, safe_bits);
        }

        u64 left_false_positive_cnt = false_positive_cnt[left_child_idx];
        u64 right_false_positive_cnt = false_positive_cnt[right_child_idx];

        f64 left_trade = left_sub_tree_size / (left_false_positive_cnt + 0.01);
        f64 right_trade = right_sub_tree_size / (right_false_positive_cnt + 0.01);

        return prune_single(left_child_idx + (right_trade > left_trade), safe_bits);
      };

      std::function<void(u64)> prune_clean = [&](u64 node_idx) {
        assert(!is_inner_node[node_idx]);
        $u64 current_node_idx = node_idx;
        while (current_node_idx != 0) {
          current_node_idx = parent_of(current_node_idx);
          u64 left_child_idx = left_child_of(node_idx);
          u64 right_child_idx = right_child_of(node_idx);
          if (bit[left_child_idx] == bit[right_child_idx]) {
            is_inner_node[current_node_idx] = false;
          }
          else {
            break;
          }
        }
      };

      while (true) {
        auto enc = encode();
        u64 current_bit_cnt = enc.size();
        if (current_bit_cnt <= target_bit_cnt) break;
        prune_clean(prune_single(0, current_bit_cnt - target_bit_cnt));
      }
    }

    static u64 find_close(const std::vector<$u1> bitstring, u64 idx) {
      if (!bitstring[idx]) return idx;
      $u64 cntr = 1;
      for ($u64 i = idx + 1; i < bitstring.size(); i++) {
        bitstring[i] ? cntr++ : cntr--;
        if (cntr == 0) return i;
      }
      return idx;
    }

    template<u64 M>
    static u64 find_close(const std::bitset<M> bitstring, u64 idx) {
      if (!bitstring[idx]) return idx;
      $u64 cntr = 1;
      for ($u64 i = idx + 1; i < M; i++) {
        bitstring[i] ? cntr++ : cntr--;
        if (cntr == 0) return i;
      }
      return idx;
    }

    static inline void write(std::bitset<N>& bitmask, u64 offset, u1 bit, u64 cnt) {
      for ($u64 i = 0; i < cnt; i++) {
        bitmask[offset + i] = bit;
      }
    }


  public:

    /// Encodes the given bitmask as a full binary tree using balanced parentheses representation.
    /// @returns a bit vector of variable size containing the encoded 'tree mask'
    static std::vector<$u1> encode(const std::bitset<N> bitmask) {
      auto t = tree_mask(bitmask);
      return t.encode();
    }

    /// Decodes the given 'tree mask'.
    /// @returns a bitmask of fixed size
    static std::bitset<N> decode(const std::vector<$u1> code) {
      std::bitset<N> bitmask;
      u64 labels_offset = find_close(code, 0) + 1;
      u64 height = ct::log_2<N>::value;
      $u64 write_pos = 0;
      $u64 level = 0;
      $u1 previous_bit = code[0];
      for ($u64 i = 1, j = labels_offset; i < labels_offset; i++) {
        u1 current_bit = code[i];
        u1 is_leaf = ! current_bit & previous_bit;
        if (is_leaf) {
          u64 n = 1 << (height - level);
          u1 label = code[j];
          write(bitmask, write_pos, label, n);
          write_pos += n;
          j++;
        }
        current_bit ? level++ : level--;
        previous_bit = current_bit;
      }
      return bitmask;
    }

    /// Encodes and compresses the given bitmask as a full binary tree using balanced parentheses representation.
    /// The length of encoded tree mask is guaranteed to be less or equal to M.
    /// Note, that the compression can lead to an information loss. However, the following holds: m == m & d(e(m)
    /// @returns a bit set of size M containing the encoded 'tree mask'
    template<u64 M>
    static std::bitset<M> compress(const std::bitset<N>& bitmask) {
      auto tree = tree_mask(bitmask);
      tree.compress(M);
      auto compressed_bitvector = tree.encode();
      std::bitset<M> compressed_bitmask;
      for ($u64 i = 0; i < compressed_bitvector.size(); i++) {
        compressed_bitmask[i] = compressed_bitvector[i];
      }
      return compressed_bitmask;
    }


    /// Decodes the given 'tree mask' of length M into a bitmask of length N.
    /// @returns a bitmask of fixed size
    template<u64 M>
    static std::bitset<N> decode(const std::bitset<M> code) {
      std::bitset<N> bitmask;
      u64 labels_offset = find_close(code, 0) + 1;
      u64 height = ct::log_2<N>::value;
      $u64 write_pos = 0;
      $u64 level = 0;
      $u1 previous_bit = code[0];
      for ($u64 i = 1, j = labels_offset; i < labels_offset; i++) {
        u1 current_bit = code[i];
        u1 is_leaf = ! current_bit & previous_bit;
        if (is_leaf) {
          u64 n = 1 << (height - level);
          u1 label = code[j];
          write(bitmask, write_pos, label, n);
          write_pos += n;
          j++;
        }
        current_bit ? level++ : level--;
        previous_bit = current_bit;
      }
      return bitmask;
    }

  };


  template<u64 N>
  class zone_mask {
    static_assert(is_power_of_two(N), "Template parameter 'N' must be a power of two.");

  public:

    template<u64 M>
    static std::bitset<M> compress(const std::bitset<N> bitmask) {
      static_assert(is_power_of_two(M), "Template parameter 'M' must be a power of two.");
      u64 zone_size = N / M;
      u64 zone_cnt = N / zone_size;

      std::bitset<N> tmp_mask;
      for ($u64 i = 0; i < zone_size; i++) {
        tmp_mask |= bitmask >> i;
      }

      std::bitset<M> zone_mask;
      for ($u64 i = 0; i < zone_cnt; i++) {
        zone_mask[i] = tmp_mask[i * zone_size];
      }
      return zone_mask;
    }

    template<u64 M>
    static std::bitset<N> decode(const std::bitset<M> compressed_bitmask) {
      static_assert(is_power_of_two(M), "Template parameter 'M' must be a power of two.");
      u64 zone_size = N / M;
      u64 zone_cnt = N / zone_size;
      std::bitset<N> bitmask;
      for ($u64 i = 0; i < zone_cnt; i++) {
        if (!compressed_bitmask[i]) continue;
        for ($u64 j = 0; j < zone_size; j++) {
          bitmask[i * zone_size + j] = true;
        }
      }
      return bitmask;
    }

  };

}
