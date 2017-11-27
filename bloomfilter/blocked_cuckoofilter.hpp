#pragma once

#include <cstdlib>

#include <dtl/dtl.hpp>
#include <dtl/hash.hpp>
#include <dtl/math.hpp>
#include <dtl/mem.hpp>

#include <dtl/bloomfilter/block_addressing_logic.hpp>
#include <dtl/bloomfilter/blocked_cuckoofilter_multiword_table.hpp>
#include <dtl/bloomfilter/blocked_cuckoofilter_simd.hpp>
#include <dtl/bloomfilter/blocked_cuckoofilter_util.hpp>
#include <dtl/bloomfilter/blocked_cuckoofilter_word_table.hpp>


namespace dtl {
namespace cuckoofilter {
namespace internal {


//===----------------------------------------------------------------------===//
// A statically sized cuckoo filter (used for blocking).
//===----------------------------------------------------------------------===//
template<
    typename __key_t = uint32_t,
    typename __table_t = blocked_cuckoofilter_multiword_table<uint64_t, 64, 16, 4>
>
struct blocked_cuckoofilter_block_logic {

  using key_t = __key_t;
  using table_t = __table_t;
  using hash_value_t = uint32_t;
  using hasher = dtl::hash::knuth_32<hash_value_t>;

  static constexpr uint32_t capacity = table_t::capacity;
  static constexpr uint32_t required_hash_bits = table_t::required_hash_bits;

//private:

  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  table_t table;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  __forceinline__
  static uint32_t
  get_bucket_idx(const hash_value_t hash_value) {
    return (hash_value >> (32 - table_t::bucket_addressing_bits)); //% table_t::bucket_count;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  static dtl::vec<hash_value_t, dtl::vector_length<Tv>::value>
  get_bucket_idxs(const Tv& hash_values) {
    return (hash_values >> (32 - table_t::bucket_addressing_bits)); //% table_t::bucket_count;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  __forceinline__
  static uint32_t
  get_alternative_bucket_idx(const uint32_t bucket_idx, const uint32_t tag) {
    return (bucket_idx ^ ((tag * 0x5bd1e995u) >> (32 - table_t::bucket_addressing_bits)));// & ((1u << table_t::bucket_addressing_bits) - 1);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  static dtl::vec<hash_value_t, dtl::vector_length<Tv>::value>
  get_alternative_bucket_idxs(const Tv& bucket_idxs, const Tv& tags) {
    return (bucket_idxs ^ ((tags * 0x5bd1e995u) >> (32 - table_t::bucket_addressing_bits)));// & ((1u << table_t::bucket_addressing_bits) - 1);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
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
    current_idx = get_alternative_bucket_idx(current_idx, current_tag);

    for (uint32_t count = 0; count < 50; count++) {
      old_tag = table.insert_tag_relocate(current_idx, current_tag);
      if (old_tag == table_t::null_tag) { return; } // successfully inserted
      if (old_tag == table_t::overflow_tag) { return; } // hit an overflowed bucket (always return true)
//      std::cout << ".";
      current_tag = old_tag;
      current_idx = get_alternative_bucket_idx(current_idx, current_tag);
    }
    // Failed to find a place for the current tag through partial-key cuckoo hashing.
    // Introduce an overflow bucket.
//    std::cout << "!";
    table.mark_overflow(current_idx);
  }
  //===----------------------------------------------------------------------===//


public:

  //===----------------------------------------------------------------------===//
  __forceinline__
  void
  insert_hash(const hash_value_t& hash_value) {
    auto bucket_idx = get_bucket_idx(hash_value);
    auto tag = (hash_value >> (32 - table_t::bucket_addressing_bits - table_t::tag_size_bits)) & table_t::tag_mask;
    tag += (tag == 0); // tag must not be zero
    insert(bucket_idx, tag);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  __forceinline__
  void
  insert_key(const key_t& key) {
    auto hash_value = hasher::hash(key);
    insert_hash(hash_value);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  __forceinline__
  bool
  contains_hash(const hash_value_t& hash_value) const {
    auto bucket_idx = get_bucket_idx(hash_value);
    auto tag = (hash_value >> (32 - table_t::bucket_addressing_bits - table_t::tag_size_bits)) & table_t::tag_mask;
    tag += (tag == 0); // tag must not be zero
    const auto alt_bucket_idx = get_alternative_bucket_idx(bucket_idx, tag);
    return table.find_tag_in_buckets(bucket_idx, alt_bucket_idx, tag);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  __forceinline__
  bool
  contains_key(const key_t& key) const {
    const auto hash_value = hasher::hash(key);
    return contains_hash(hash_value);
  }
  //===----------------------------------------------------------------------===//

};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// A blocked cuckoo filter template.
//===----------------------------------------------------------------------===//
template<
    typename __key_t = uint32_t,
    typename __block_t = blocked_cuckoofilter_block_logic<__key_t>,
    block_addressing __block_addressing = block_addressing::POWER_OF_TWO
>
struct blocked_cuckoofilter {

  using key_t = __key_t;
  using block_t = __block_t;
  using hash_value_t = uint32_t;
  using hasher = dtl::hash::knuth_32_alt<hash_value_t>;
  using addr_t = block_addressing_logic<__block_addressing>;


  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  const addr_t addr;
  std::vector<block_t, dtl::mem::numa_allocator<block_t>> blocks;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  explicit
  blocked_cuckoofilter(const std::size_t length) : addr(length, block_t::block_bitlength), blocks(addr.get_block_cnt()) { }

  blocked_cuckoofilter(const blocked_cuckoofilter&) noexcept = default;

  blocked_cuckoofilter(blocked_cuckoofilter&&) noexcept = default;

  ~blocked_cuckoofilter() noexcept = default;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
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
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  __forceinline__
  uint64_t
  batch_insert(const key_t* keys, const uint32_t key_cnt) {
    for (uint32_t i = 0; i < key_cnt; i++) {
      insert(keys[i]);
    }
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
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
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Performs a batch-probe
  __forceinline__ __unroll_loops__ __host__
  std::size_t
  batch_contains(const key_t* __restrict keys, u32 key_cnt,
                 $u32* __restrict match_positions, u32 match_offset) const {
    constexpr u32 mini_batch_size = 16;
    const u32 mini_batch_cnt = key_cnt / mini_batch_size;

    $u32* match_writer = match_positions;
    if ((addr.get_required_addressing_bits() + block_t::required_hash_bits) <= (sizeof(hash_value_t) * 8)) {

      for ($u32 mb = 0; mb < mini_batch_cnt; mb++) {
        for (uint32_t j = mb * mini_batch_size; j < ((mb + 1) * mini_batch_size); j++) {
          auto h = hasher::hash(keys[j]);
          auto i = addr.get_block_idx(h);
          auto is_contained = blocks[i].contains_hash(h << addr.get_required_addressing_bits());
          *match_writer = j + match_offset;
          match_writer += is_contained;
        }
      }
      for (uint32_t j = (mini_batch_cnt * mini_batch_size); j < key_cnt; j++) {
        auto h = hasher::hash(keys[j]);
        auto i = addr.get_block_idx(h);
        auto is_contained = blocks[i].contains_hash(h << addr.get_required_addressing_bits());
        *match_writer = j + match_offset;
        match_writer += is_contained;
      }
    }

    else {
      for ($u32 mb = 0; mb < mini_batch_cnt; mb++) {
        for (uint32_t j = mb * mini_batch_size; j < ((mb + 1) * mini_batch_size); j++) {
          auto k = keys[j];
          auto h = hasher::hash(k);
          auto i = addr.get_block_idx(h);
          auto is_contained = blocks[i].contains_key(k);
          *match_writer = j + match_offset;
          match_writer += is_contained;
        }
      }
      for (uint32_t j = (mini_batch_cnt * mini_batch_size); j < key_cnt; j++) {
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
  //===----------------------------------------------------------------------===//


};
//===----------------------------------------------------------------------===//


static constexpr uint64_t cache_line_size = 64;


} // namespace internal
} // namespace cuckoofilter


//===----------------------------------------------------------------------===//
// Instantiations of some reasonable cuckoo filters
//===----------------------------------------------------------------------===//
template<typename __key_t, typename __derived>
struct blocked_cuckoofilter_base {

  using key_t = __key_t;
  using word_t = typename __derived::word_t;

  __forceinline__ __host__ __device__
  void
  insert(word_t* __restrict filter_data, const key_t& key) {
    return static_cast<__derived*>(this)->filter.insert(filter_data, key);
  }
  //===----------------------------------------------------------------------===//


  __forceinline__ __host__
  uint64_t
  batch_insert(word_t* __restrict filter_data, const key_t* keys, const uint32_t key_cnt) {
    return static_cast<__derived*>(this)->filter.batch_insert(filter_data, keys, key_cnt);
  }
  //===----------------------------------------------------------------------===//


  __forceinline__ __host__ __device__
  bool
  contains(const word_t* __restrict filter_data, const key_t& key) const {
    return static_cast<const __derived*>(this)->filter.contains(filter_data, key);
  }
  //===----------------------------------------------------------------------===//


  __forceinline__  __host__
  uint64_t
  batch_contains(const word_t* __restrict filter_data,
                 const key_t* __restrict keys, const uint32_t key_cnt,
                 uint32_t* __restrict match_positions, const uint32_t match_offset) const {
    return static_cast<const __derived*>(this)->filter.batch_contains(filter_data, keys, key_cnt, match_positions, match_offset);
  };
  //===----------------------------------------------------------------------===//

};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
template<uint32_t block_size_bytes, uint32_t bits_per_element, uint32_t associativity, block_addressing addressing>
struct blocked_cuckoofilter {};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
template<uint32_t block_size_bytes, block_addressing addressing>
struct blocked_cuckoofilter<block_size_bytes, 16, 4, addressing>
    : blocked_cuckoofilter_base<uint32_t, blocked_cuckoofilter<block_size_bytes, 16, 4, addressing>> {

  using key_t = uint32_t;
  using word_t = uint64_t;
  using table_t = cuckoofilter::blocked_cuckoofilter_multiword_table<word_t, block_size_bytes, 16, 4>;
  using block_t = cuckoofilter::internal::blocked_cuckoofilter_block_logic<key_t, table_t>;
  using filter_t = cuckoofilter::internal::blocked_cuckoofilter<uint32_t, block_t, addressing>;

  filter_t filter; // the actual filter instance

  explicit blocked_cuckoofilter(const std::size_t length) : filter(length) { }

  // use SIMD implementation
  __forceinline__ uint64_t
  batch_contains(const key_t* __restrict keys, const uint32_t key_cnt,
                 uint32_t* __restrict match_positions, const uint32_t match_offset) const {
    return dtl::cuckoofilter::simd_batch_contains_16_4(*this, keys, key_cnt, match_positions, match_offset);
  };

};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
template<uint32_t block_size_bytes, block_addressing addressing>
struct blocked_cuckoofilter<block_size_bytes, 16, 2, addressing>
    : blocked_cuckoofilter_base<uint32_t, blocked_cuckoofilter<block_size_bytes, 16, 2, addressing>> {

  using key_t = uint32_t;
  using word_t = uint64_t;
  using table_t = cuckoofilter::blocked_cuckoofilter_multiword_table<word_t, block_size_bytes, 16, 2>;
  using block_t = cuckoofilter::internal::blocked_cuckoofilter_block_logic<key_t, table_t>;
  using filter_t = cuckoofilter::internal::blocked_cuckoofilter<uint32_t, block_t, addressing>;

  filter_t filter; // the actual filter instance

  explicit blocked_cuckoofilter(const std::size_t length) : filter(length) { }

};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
template<uint32_t block_size_bytes, block_addressing addressing>
struct blocked_cuckoofilter<block_size_bytes, 12, 4, addressing>
    : blocked_cuckoofilter_base<uint32_t, blocked_cuckoofilter<block_size_bytes, 12, 4, addressing>> {

  using key_t = uint32_t;
  using word_t = uint64_t;
  using table_t = cuckoofilter::blocked_cuckoofilter_multiword_table<word_t, block_size_bytes, 12, 4>;
  using block_t = cuckoofilter::internal::blocked_cuckoofilter_block_logic<key_t, table_t>;
  using filter_t = cuckoofilter::internal::blocked_cuckoofilter<uint32_t, block_t, addressing>;

  filter_t filter; // the actual filter instance

  explicit blocked_cuckoofilter(const std::size_t length) : filter(length) { }

};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
template<uint32_t block_size_bytes, block_addressing addressing>
struct blocked_cuckoofilter<block_size_bytes, 10, 6, addressing>
    : blocked_cuckoofilter_base<uint32_t, blocked_cuckoofilter<block_size_bytes, 10, 6, addressing>> {

  using key_t = uint32_t;
  using word_t = uint64_t;
  using table_t = cuckoofilter::blocked_cuckoofilter_multiword_table<word_t, block_size_bytes, 10, 6>;
  using block_t = cuckoofilter::internal::blocked_cuckoofilter_block_logic<key_t, table_t>;
  using filter_t = cuckoofilter::internal::blocked_cuckoofilter<uint32_t, block_t, addressing>;

  filter_t filter; // the actual filter instance

  explicit blocked_cuckoo_filter(const std::size_t length) : filter(length) { }

};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
template<uint32_t block_size_bytes, block_addressing addressing>
struct blocked_cuckoofilter<block_size_bytes, 8, 8, addressing>
    : blocked_cuckoofilter_base<uint32_t, blocked_cuckoofilter<block_size_bytes, 8, 8, addressing>> {

  using key_t = uint32_t;
  using word_t = uint64_t;
  using table_t = cuckoofilter::blocked_cuckoofilter_multiword_table<word_t, block_size_bytes, 8, 8>;
  using block_t = cuckoofilter::internal::blocked_cuckoofilter_block_logic<key_t, table_t>;
  using filter_t = cuckoofilter::internal::blocked_cuckoofilter<uint32_t, block_t, addressing>;

  filter_t filter; // the actual filter instance

  explicit blocked_cuckoofilter(const std::size_t length) : filter(length) { }

};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
template<uint32_t block_size_bytes, block_addressing addressing>
struct blocked_cuckoofilter<block_size_bytes, 8, 4, addressing>
    : blocked_cuckoofilter_base<uint32_t, blocked_cuckoofilter<block_size_bytes, 8, 4, addressing>> {

  using key_t = uint32_t;
  using word_t = uint32_t;
  using table_t = cuckoofilter::blocked_cuckoofilter_multiword_table<word_t, block_size_bytes, 8, 4>;
  using block_t = cuckoofilter::internal::blocked_cuckoofilter_block_logic<key_t, table_t>;
  using filter_t = cuckoofilter::internal::blocked_cuckoofilter<uint32_t, block_t, addressing>;

  filter_t filter; // the actual filter instance

  explicit blocked_cuckoofilter(const std::size_t length) : filter(length) { }

  // use SIMD implementation
  __forceinline__ uint64_t
  batch_contains(const word_t* __restrict filter_data,
                 const key_t* __restrict keys, const uint32_t key_cnt,
                 uint32_t* __restrict match_positions, const uint32_t match_offset) const {
    return dtl::cuckoofilter::simd_batch_contains_8_4(*this, keys, key_cnt, match_positions, match_offset);
  };

};
//===----------------------------------------------------------------------===//


} // namespace dtl