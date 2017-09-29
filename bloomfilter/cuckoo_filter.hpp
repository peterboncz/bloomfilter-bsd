#pragma once

#include <cstdlib>

#include <dtl/dtl.hpp>
#include <dtl/hash.hpp>
#include <dtl/math.hpp>

#include <dtl/bloomfilter/bloomfilter_addressing_logic.hpp>
#include <dtl/bloomfilter/cuckoo_filter_word_table.hpp>
#include <dtl/bloomfilter/cuckoo_filter_cacheline_table.hpp>
#include <dtl/bloomfilter/cuckoo_filter_helper.hpp>


namespace dtl {


struct cuckoo_filter {

//  using table_t = cuckoo_filter_word_table;
  using table_t = cuckoo_filter_cacheline_table;
  using hasher = dtl::hash::knuth_32<uint32_t>;
  using key_t = uint32_t;

  static constexpr uint32_t capacity = table_t::capacity;


  // for compatibility with block addressing logic
  static constexpr uint32_t block_bitlength = sizeof(table_t) * 8;
  // ---

  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  table_t table;
  //===----------------------------------------------------------------------===//

  __forceinline__
  uint32_t
  alternative_bucket_idx(const uint32_t bucket_idx, const uint32_t tag) const {
//    return (bucket_idx ^ (tag * 0x5bd1e995u)) % table_t::bucket_count;
    return (bucket_idx ^ tag) % table_t::bucket_count;
  }


  __forceinline__
  void
  insert(const key_t& key) {
    auto hash_value = hasher::hash(key);
//    auto bucket_idx = hash_value >> (32 - table_t::bucket_count_log2);
//    auto tag = (hash_value >> (32 - table_t::bucket_count_log2 - table_t::tag_size_bits)) & table_t::tag_mask;
    auto bucket_idx = (hash_value >> (32 - table_t::bucket_addressing_bits)) % table_t::bucket_count;
    auto tag = (hash_value >> (32 - table_t::bucket_addressing_bits - table_t::tag_size_bits)) & table_t::tag_mask;
    tag += (tag == 0); // tag must not be zero
    insert(bucket_idx, tag);
  }


  __forceinline__
  void
  insert(const uint32_t bucket_idx, const uint32_t tag) {
    uint32_t current_idx = bucket_idx;
    uint32_t current_tag = tag;
    uint32_t old_tag;

    // Try to insert without kicking other tags out.
    old_tag = table.insert_tag(current_idx, current_tag);
    if (old_tag == table_t::null_tag) { return; } // successfully inserted
    if (old_tag == table_t::overflow_tag) { return; } // hit an overflowed bucket (always return true)

    // Re-try at the alternative bucket.
    current_idx = alternative_bucket_idx(current_idx, current_tag);

    for (uint32_t count = 0; count < 500; count++) {
      old_tag = table.insert_tag_kick_out(current_idx, current_tag);
      if (old_tag == table_t::null_tag) { return; } // successfully inserted
      if (old_tag == table_t::overflow_tag) { return; } // hit an overflowed bucket (always return true)
//      std::cout << ".";
      current_tag = old_tag;
      current_idx = alternative_bucket_idx(current_idx, current_tag);
    }
    // Failed to find a place for the current tag through partial-key cuckoo hashing.
    // Introduce an overflow bucket.
//    std::cout << "!";
    table.overflow(current_idx);
  }


  __forceinline__
  bool
  contains(const key_t& key) const {
    const auto hash_value = hasher::hash(key);
//    const auto bucket_idx = hash_value >> (32 - table_t::bucket_count_log2);
//    auto tag = (hash_value >> (32 - table_t::bucket_count_log2 - table_t::tag_size_bits)) & table_t::tag_mask;
    auto bucket_idx = (hash_value >> (32 - table_t::bucket_addressing_bits)) % table_t::bucket_count;
    auto tag = (hash_value >> (32 - table_t::bucket_addressing_bits - table_t::tag_size_bits)) & table_t::tag_mask;
    tag += (tag == 0); // tag must not be zero
    const auto alt_bucket_idx = alternative_bucket_idx(bucket_idx, tag);
    return table.find_tag_in_buckets(bucket_idx, alt_bucket_idx, tag);
  }


};


//===----------------------------------------------------------------------===//


struct blocked_cuckoo_filter {

  using block_t = cuckoo_filter;
  using key_t = uint32_t;
  using hash_value_t = uint32_t;
  using hasher = dtl::hash::knuth_32_alt<hash_value_t>;
  using addr_t = bloomfilter_addressing_logic<block_addressing::POWER_OF_TWO, hash_value_t, block_t>;

  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  const addr_t addr;
  std::vector<block_t> blocks;
  //===----------------------------------------------------------------------===//

  explicit
  blocked_cuckoo_filter(const std::size_t length) : addr(length), blocks(addr.block_cnt) { }

  __forceinline__
  void
  insert(const key_t& key) {
    auto h = hasher::hash(key);
    auto i = addr.get_block_idx(h);
    blocks[i].insert(key);
  }


  __forceinline__
//  __attribute__((noinline))
  bool
  contains(const key_t& key) const {
    auto h = hasher::hash(key);
    auto i = addr.get_block_idx(h);
    return blocks[i].contains(key);
  }


};



} // namespace dtl