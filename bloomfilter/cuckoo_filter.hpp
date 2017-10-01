#pragma once

#include <cstdlib>

#include <dtl/dtl.hpp>
#include <dtl/hash.hpp>
#include <dtl/math.hpp>

#include <dtl/bloomfilter/bloomfilter_addressing_logic.hpp>
#include <dtl/bloomfilter/cuckoo_filter_word_table.hpp>
#include <dtl/bloomfilter/cuckoo_filter_multiword_table.hpp>
#include <dtl/bloomfilter/cuckoo_filter_helper.hpp>


namespace dtl {


struct cuckoo_filter {

//  using table_t = cuckoo_filter_word_table;
  using table_t = cuckoo_filter_multiword_table;
  using hash_value_t = uint32_t;
  using hasher = dtl::hash::knuth_32<hash_value_t>;
//  using hasher = dtl::hash::murmur_32<uint32_t>;
  using key_t = uint32_t;

  static constexpr uint32_t capacity = table_t::capacity;
  static constexpr uint32_t required_hash_bits = table_t::required_hash_bits;

  // for compatibility with block addressing logic
  static constexpr uint32_t block_bitlength = sizeof(table_t) * 8;
  // ---

//private:

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
      old_tag = table.insert_tag_relocate(current_idx, current_tag);
      if (old_tag == table_t::null_tag) { return; } // successfully inserted
      if (old_tag == table_t::overflow_tag) { return; } // hit an overflowed bucket (always return true)
      std::cout << ".";
      current_tag = old_tag;
      current_idx = alternative_bucket_idx(current_idx, current_tag);
    }
    // Failed to find a place for the current tag through partial-key cuckoo hashing.
    // Introduce an overflow bucket.
    std::cout << "!";
    table.mark_overflow(current_idx);
  }


public:

  __forceinline__
  void
  insert_hash(const hash_value_t& hash_value) {
    auto bucket_idx = (hash_value >> (32 - table_t::bucket_addressing_bits)) % table_t::bucket_count;
    auto tag = (hash_value >> (32 - table_t::bucket_addressing_bits - table_t::tag_size_bits)) & table_t::tag_mask;
    tag += (tag == 0); // tag must not be zero
    insert(bucket_idx, tag);
  }


  __forceinline__
  void
  insert_key(const key_t& key) {
    auto hash_value = hasher::hash(key);
    insert_hash(hash_value);
  }


  __forceinline__
  bool
  contains_hash(const hash_value_t& hash_value) const {
    auto bucket_idx = (hash_value >> (32 - table_t::bucket_addressing_bits)) % table_t::bucket_count;
    auto tag = (hash_value >> (32 - table_t::bucket_addressing_bits - table_t::tag_size_bits)) & table_t::tag_mask;
    tag += (tag == 0); // tag must not be zero
    const auto alt_bucket_idx = alternative_bucket_idx(bucket_idx, tag);
    return table.find_tag_in_buckets(bucket_idx, alt_bucket_idx, tag);
  }


  __forceinline__
  bool
  contains_key(const key_t& key) const {
    const auto hash_value = hasher::hash(key);
    return contains_hash(hash_value);
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

  blocked_cuckoo_filter(const blocked_cuckoo_filter&) noexcept = default;

  blocked_cuckoo_filter(blocked_cuckoo_filter&&) noexcept = default;

  ~blocked_cuckoo_filter() noexcept = default;


  __forceinline__
  void
  insert(const key_t& key) {
    auto h = hasher::hash(key);
    auto i = addr.get_block_idx(h);
    if ((addr.get_required_addressing_bits() + block_t::required_hash_bits) <= (sizeof(hash_value_t) * 8)) {
      blocks[i].insert_hash(h << addr.get_required_addressing_bits());
    }
    else {
      blocks[i].insert_key(key);
    }
  }


  __attribute__((noinline))
//  __forceinline__
  uint64_t
  batch_insert(const key_t* keys, const uint32_t key_cnt) {
    for (uint32_t i = 0; i < key_cnt; i++) {
      insert(keys[i]);
    }
  }


//  __attribute__((noinline))
  __forceinline__
  bool
  contains(const key_t& key) const {
    auto h = hasher::hash(key);
    auto i = addr.get_block_idx(h);
    if ((addr.get_required_addressing_bits() + block_t::required_hash_bits) <= (sizeof(hash_value_t) * 8)) {
      return blocks[i].contains_hash(h << addr.get_required_addressing_bits());
    }
    else {
      return blocks[i].contains_key(key);
    }
  }


  __attribute__((noinline))
//  __forceinline__
  uint64_t
  batch_contains(const key_t* keys, const uint32_t key_cnt,
                 uint32_t* match_positions, const uint32_t match_offset) const {
    uint32_t* match_writer = match_positions;
    for (uint32_t i = 0; i < key_cnt; i++) {
      auto is_contained = contains(keys[i]);
      *match_writer = i + match_offset;
      match_writer += is_contained;
    }
    return match_writer - match_positions;
  }


};


} // namespace dtl