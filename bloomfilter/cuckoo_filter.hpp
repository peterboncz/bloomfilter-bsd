#pragma once

#include <cstdlib>

#include <dtl/dtl.hpp>
#include <dtl/hash.hpp>
#include <dtl/math.hpp>
#include <dtl/mem.hpp>

#include <dtl/bloomfilter/bloomfilter_addressing_logic.hpp>
#include <dtl/bloomfilter/cuckoo_filter_word_table.hpp>
#include <dtl/bloomfilter/cuckoo_filter_multiword_table.hpp>
#include <dtl/bloomfilter/cuckoo_filter_helper.hpp>


namespace dtl {
namespace cuckoo_filter {
namespace internal {


// A statically sized cuckoo filter (used for blocking).
template<
    typename __key_t = uint32_t,
    typename __table_t = cuckoo_filter_multiword_table<uint64_t, 64, 16, 4>
>
struct cuckoo_filter {

  using key_t = __key_t;
  using table_t = __table_t;
  using hash_value_t = uint32_t;
  using hasher = dtl::hash::knuth_32<hash_value_t>;

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
//      std::cout << ".";
      current_tag = old_tag;
      current_idx = alternative_bucket_idx(current_idx, current_tag);
    }
    // Failed to find a place for the current tag through partial-key cuckoo hashing.
    // Introduce an overflow bucket.
//    std::cout << "!";
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


// A blocked cuckoo filter.
template<
    typename __key_t = uint32_t,
    typename __block_t = cuckoo_filter<__key_t>,
    block_addressing __block_addressing = block_addressing::POWER_OF_TWO
>
struct blocked_cuckoo_filter {

  using key_t = __key_t;
  using block_t = __block_t;
  using hash_value_t = uint32_t;
  using hasher = dtl::hash::knuth_32_alt<hash_value_t>;
  using addr_t = bloomfilter_addressing_logic<__block_addressing, hash_value_t, block_t>;


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


  __forceinline__
  uint64_t
  batch_insert(const key_t* keys, const uint32_t key_cnt) {
    for (uint32_t i = 0; i < key_cnt; i++) {
      insert(keys[i]);
    }
  }


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


  /// Performs a batch-probe
  __forceinline__ __host__
  std::size_t
  batch_contains(const key_t* keys, u32 key_cnt, $u32* match_positions, u32 match_offset) const {
    $u32* match_writer = match_positions;
    if ((addr.get_required_addressing_bits() + block_t::required_hash_bits) <= (sizeof(hash_value_t) * 8)) {
      for (uint32_t j = 0; j < key_cnt; j++) {
        auto h = hasher::hash(keys[j]);
        auto i = addr.get_block_idx(h);
        auto is_contained = blocks[i].contains_hash(h << addr.get_required_addressing_bits());
        *match_writer = j + match_offset;
        match_writer += is_contained;
      }
    }
    else {
      for (uint32_t j = 0; j < key_cnt; j++) {
        auto k = keys[j];
        auto h = hasher::hash(k);
        auto i = addr.get_block_idx(h);
        auto is_contained = blocks[i].contains_key(k);
        *match_writer = j + match_offset;
        match_writer += is_contained;
      }
    }
    return match_writer - match_positions;
  }


};


static constexpr uint64_t cache_line_size = 64;


} // namespace internal
} // namespace cuckoo_filter


//===----------------------------------------------------------------------===//
// Instantiations of some reasonable cuckoo filters
//===----------------------------------------------------------------------===//
template<typename __key_t, typename __derived>
struct blocked_cuckoo_filter_base {

  using key_t = __key_t;

  __forceinline__ void
  insert(const key_t& key) {
    return static_cast<__derived*>(this)->filter.insert(key);
  }

  __forceinline__ uint64_t
  batch_insert(const key_t* keys, const uint32_t key_cnt) {
    return static_cast<__derived*>(this)->filter.batch_insert(keys, key_cnt);
  }

  __forceinline__ bool
  contains(const key_t& key) const {
    return static_cast<const __derived*>(this)->filter.contains(key);
  }

    __forceinline__ uint64_t
  batch_contains(const key_t* __restrict keys, const uint32_t key_cnt,
                 uint32_t* __restrict match_positions, const uint32_t match_offset) const {
    return static_cast<const __derived*>(this)->filter.batch_contains(keys, key_cnt, match_positions, match_offset);
  };

};


template<uint32_t bits_per_element, uint32_t associativity>
struct blocked_cuckoo_filter {};

template<>
struct blocked_cuckoo_filter<16, 4> : blocked_cuckoo_filter_base<uint32_t, blocked_cuckoo_filter<16, 4>> {
  using key_type = uint32_t;
  using table_type = cuckoo_filter_multiword_table<uint64_t, cuckoo_filter::internal::cache_line_size, 16, 4>;
  using block_type = cuckoo_filter::internal::cuckoo_filter<key_type, table_type>;
  using filter_type = cuckoo_filter::internal::blocked_cuckoo_filter<uint32_t, block_type, block_addressing::POWER_OF_TWO>;

  filter_type filter; // the actual filter instance

  explicit blocked_cuckoo_filter(const std::size_t length) : filter(length) { }

};

template<>
struct blocked_cuckoo_filter<16, 2> : blocked_cuckoo_filter_base<uint32_t, blocked_cuckoo_filter<16, 2>> {
  using key_type = uint32_t;
  using table_type = cuckoo_filter_multiword_table<uint64_t, cuckoo_filter::internal::cache_line_size, 16, 2>;
  using block_type = cuckoo_filter::internal::cuckoo_filter<key_type, table_type>;
  using filter_type = cuckoo_filter::internal::blocked_cuckoo_filter<uint32_t, block_type, block_addressing::POWER_OF_TWO>;

  filter_type filter; // the actual filter instance

  explicit blocked_cuckoo_filter(const std::size_t length) : filter(length) { }

};

template<>
struct blocked_cuckoo_filter<12, 4> : blocked_cuckoo_filter_base<uint32_t, blocked_cuckoo_filter<12, 4>> {
  using key_type = uint32_t;
  using table_type = cuckoo_filter_multiword_table<uint64_t, cuckoo_filter::internal::cache_line_size, 12, 4>;
  using block_type = cuckoo_filter::internal::cuckoo_filter<key_type, table_type>;
  using filter_type = cuckoo_filter::internal::blocked_cuckoo_filter<uint32_t, block_type, block_addressing::POWER_OF_TWO>;

  filter_type filter; // the actual filter instance

  explicit blocked_cuckoo_filter(const std::size_t length) : filter(length) { }

};

template<>
struct blocked_cuckoo_filter<10, 6> : blocked_cuckoo_filter_base<uint32_t, blocked_cuckoo_filter<10, 6>> {
  using key_type = uint32_t;
  using table_type = cuckoo_filter_multiword_table<uint64_t, cuckoo_filter::internal::cache_line_size, 10, 6>;
  using block_type = cuckoo_filter::internal::cuckoo_filter<key_type, table_type>;
  using filter_type = cuckoo_filter::internal::blocked_cuckoo_filter<uint32_t, block_type, block_addressing::POWER_OF_TWO>;

  filter_type filter; // the actual filter instance

  explicit blocked_cuckoo_filter(const std::size_t length) : filter(length) { }

};


template<>
struct blocked_cuckoo_filter<8, 8> : blocked_cuckoo_filter_base<uint32_t, blocked_cuckoo_filter<8, 8>> {
  using key_type = uint32_t;
  using table_type = cuckoo_filter_multiword_table<uint64_t, cuckoo_filter::internal::cache_line_size, 8, 8>;
  using block_type = cuckoo_filter::internal::cuckoo_filter<key_type, table_type>;
  using filter_type = cuckoo_filter::internal::blocked_cuckoo_filter<uint32_t, block_type, block_addressing::POWER_OF_TWO>;

  filter_type filter; // the actual filter instance

  explicit blocked_cuckoo_filter(const std::size_t length) : filter(length) { }

};


template<>
struct blocked_cuckoo_filter<8, 4> : blocked_cuckoo_filter_base<uint32_t, blocked_cuckoo_filter<8, 4>> {
  using key_type = uint32_t;
  using table_type = cuckoo_filter_multiword_table<uint32_t, cuckoo_filter::internal::cache_line_size, 8, 4>;
  using block_type = cuckoo_filter::internal::cuckoo_filter<key_type, table_type>;
  using filter_type = cuckoo_filter::internal::blocked_cuckoo_filter<uint32_t, block_type, block_addressing::POWER_OF_TWO>;

  filter_type filter; // the actual filter instance

  explicit blocked_cuckoo_filter(const std::size_t length) : filter(length) { }

};


} // namespace dtl