#pragma once
#include "adept.hpp"
#include "tree.hpp"
#include <bitset>
#include <vector>
#include "math.hpp"

#include <iostream> //TODO remove

template<u64 input_bit_cnt, u64 output_bit_cnt>
std::bitset<output_bit_cnt> compress(const std::bitset<input_bit_cnt>& bitmask) {
  static_assert(is_power_of_two(input_bit_cnt), "Template parameter 'input_bit_cnt'  must be a power of two.");
  //static_assert(is_power_of_two(output_bit_cnt), "Template parameter 'output_bit_cnt'  must be a power of two.");

  // the tree height
  u64 h = ct::log_2<input_bit_cnt>::value;

  u64 node_cnt = 2 * input_bit_cnt - 1;

  std::cout << std::endl;

  // a mutable copy of the original bitmask
  std::bitset<input_bit_cnt> b(bitmask);
  // keep track of the cumulative false positives
  std::vector<$u32> false_positive_cnt(input_bit_cnt, 0);
  // keep track of the number of leave for the individual subtrees
  std::vector<$u32> leaf_cnts(input_bit_cnt, 0);

  // bottom-up pruning (loss less)
  for ($u64 i = 1; i < h; i++) {
    std::cout << "b "; print(b, i-1);
    std::bitset<input_bit_cnt> t = b >> i;
    std::cout << "t "; print(t, 0);
    std::bitset<input_bit_cnt> d = b ^ t;
    std::cout << "d "; print(d, i);
    b |= t;
    std::cout << "b "; print(b, i);
    for ($u64 j = 0; j < input_bit_cnt; j += 1 << i) {
      false_positive_cnt[j] = d[j] * (1 << (i-1))  + (false_positive_cnt[j] + false_positive_cnt[j + (1 << (i-1))]);
      leaf_cnts[j] = !d[j] * (leaf_cnts[j] + leaf_cnts[j + (1 << (i-1))]) + 2* d[j];
    }
    std::cout << "false positives "; print(false_positive_cnt, i);
    std::cout << "num leaves      "; print(leaf_cnts, 0);
  }
  std::cout << false_positive_cnt[0] << std::endl;


  std::vector<$u1> bar = tree::encode_structure_succinct(leaf_cnts);
  print(bar,0);

  // top-down pruning (lossy)
  std::cout << "---- top-down pruning (lossy) ----" << std::endl;
  u32 target_leaf_cnts = (output_bit_cnt - 1) / 2;
  std::vector<$u64> to_prune(input_bit_cnt, 0);
  to_prune[0] = leaf_cnts[0] - std::min(leaf_cnts[0], target_leaf_cnts);
  std::cout << "target num leaves " << target_leaf_cnts << std::endl;
  std::cout << "to prune        "; print(to_prune, 0);
  for ($u64 i = 1; i < (h - 1); i++) {
    std::cout << "level " << i << std::endl;
    for ($u64 j = 0; j < 1 << (h - 1); j += 1 << (h - i) ) {
      if (to_prune[j] == 0) continue;
      u64 leaf_cnts_current = leaf_cnts[j];
      u64 false_positive_cnt_current = false_positive_cnt[j];
      if (leaf_cnts_current == to_prune[j] << 1) {
        std::cout << "hit" << std::endl;
        to_prune[j] = 0;
        leaf_cnts[j] = 0;
        continue;
      }
      if (leaf_cnts_current > 0) {
        u64 leaf_cnt_right_child = leaf_cnts[1 << ((h - 1) - i)];
        u64 leaf_cnt_left_child = leaf_cnts[j] = leaf_cnts_current - leaf_cnt_right_child;
        u64 num_false_positive_cnt_right = false_positive_cnt[1 << ((h - 1) - i)];
        u64 num_false_positive_cnt_left = false_positive_cnt[j] = false_positive_cnt_current - num_false_positive_cnt_right;

        f64 trade_right = leaf_cnt_right_child / (num_false_positive_cnt_right + 0.01);
        f64 trade_left = leaf_cnt_left_child / (num_false_positive_cnt_left + 0.01);

        u64 to_prune_right = (trade_right > trade_left)
                             ? std::min(leaf_cnt_right_child, to_prune[j])
                             : to_prune[j] - std::min(leaf_cnt_left_child, to_prune[j]);
        u64 to_prune_left = (trade_right <= trade_left)
                             ? std::min(leaf_cnt_left_child, to_prune[j])
                             : to_prune[j] - std::min(leaf_cnt_right_child, to_prune[j]);

        to_prune[j] = to_prune_left;
        to_prune[1 << ((h - 1) - i)] = to_prune_right;

        std::cout << "- nl: " << leaf_cnt_left_child << " " << leaf_cnt_right_child;
        std::cout << " - fp: " << num_false_positive_cnt_left << " " << num_false_positive_cnt_right;
        std::cout << " - tr: " << trade_left << " " << trade_right << std::endl;
      }
      else {
        std::cout << " no " << j << std::endl;
        leaf_cnts[j] = 0;
      }
    }
    std::cout << "false positives "; print(false_positive_cnt, 0);
    std::cout << "num leaves      "; print(leaf_cnts, 0);
    std::cout << "to prune        "; print(to_prune, 0);
  }

  std::vector<$u1> foo = tree::encode_structure_succinct(leaf_cnts);
  print(foo,0);

  std::bitset<output_bit_cnt> compressed_bitmask;
  std::cout << compressed_bitmask[0] << std::endl;
  return compressed_bitmask;
}
