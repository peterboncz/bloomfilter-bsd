#pragma once

//===----------------------------------------------------------------------===//
// This file is based on the original Cuckoo filter implementation,
// that can be found on GitHub: https://github.com/efficient/cuckoofilter
// Usable under the terms in the Apache License, Version 2.0.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2013, Carnegie Mellon University and Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

// Change log:
// 2017-2018 | H. Lang (TUM), P. Boncz (CWI):
//  - memory needs to be managed by the caller
//  - configurable associativity (number of tags per bucket)
//  - the filter size is no longer restricted to powers of two (see "dtl/filter/block_addressing_logic.hpp" for details)
//  - an interface for batched lookups
//  - SIMDized implementations for batched lookups (AVX2, AVX-512, and experimental CUDA)
//  - faster hashing (multiplicative)
//  - removed 'deletion' support

#include <bitset>

#include <dtl/dtl.hpp>
#include <dtl/math.hpp>
#include <dtl/mem.hpp>
#include <dtl/simd.hpp>
#include <dtl/filter/blocked_bloomfilter/block_addressing_logic.hpp>
#include <dtl/filter/blocked_bloomfilter/hash_family.hpp>

#include "cuckoofilter_table.hpp"

namespace dtl {
namespace cuckoofilter {


// Dispatch batch_contains calls either to batch_contains_scalar or batch_contains_vec.
namespace internal {

//===----------------------------------------------------------------------===//
// Batch-wise Contains (SIMD)
//===----------------------------------------------------------------------===//
template<
    typename filter_t,
    u64 vector_len
>
struct batch_contains {

  // Typedefs
  using key_t = typename filter_t::key_t;
  using word_t = typename filter_t::word_t;

  __attribute__ ((__noinline__))
  static $u64
  dispatch(const filter_t& filter,
           const word_t* __restrict filter_data,
           const key_t* __restrict keys, u32 key_cnt,
           $u32* __restrict match_positions, u32 match_offset) {
    return filter.template batch_contains_vec<vector_len>(filter_data, keys,
        key_cnt, match_positions, match_offset);
  }

};


//===----------------------------------------------------------------------===//
// Batch-wise Contains (no SIMD)
//===----------------------------------------------------------------------===//
template<
    typename filter_t
>
struct batch_contains<filter_t, 0> {

  // Typedefs
  using key_t = typename filter_t::key_t;
  using word_t = typename filter_t::word_t;

  __attribute__ ((__noinline__))
  static $u64
  dispatch(const filter_t& filter,
           const word_t* __restrict filter_data,
           const key_t* __restrict keys, u32 key_cnt,
           $u32* __restrict match_positions, u32 match_offset) {
    return filter.batch_contains_scalar(filter_data, keys, key_cnt,
        match_positions, match_offset);
  }

};

} // namespace internal


//===----------------------------------------------------------------------===//
// A cuckoo filter class exposes a Bloomier filter interface,
// providing methods of insert, Delete, contain. It takes three
// template parameters:
//   bits_per_tag:    how many bits each item is hashed into
//   tags_per_bucket: how items fit into a single bucket
//   table_t:         the storage of the table
//===----------------------------------------------------------------------===//
template <std::size_t bits_per_tag,
          std::size_t tags_per_bucket,
          template <std::size_t, std::size_t> class table_t = cuckoofilter_table,
          dtl::block_addressing block_addressing = dtl::block_addressing::POWER_OF_TWO,
          u1 has_victim_cache = true
>
class cuckoofilter_logic {

  // Maximum number of cuckoo kicks before claiming failure.
  static constexpr std::size_t max_relocation_count = 500;

public:
  using key_t = uint32_t;
  using word_t = typename table_t<bits_per_tag, tags_per_bucket>::word_t;

  static constexpr uint32_t tag_mask =
      static_cast<uint32_t>((1ull << bits_per_tag) - 1);

  // An overflow entry for a single item (used when the filter became full)
  typedef struct {
    uint32_t index;
    uint32_t tag;
    bool used;
  } victim_cache_t;

  // The block addressing logic (either MAGIC or POWER_OF_TWO).
  // In the context of cuckoo filters a 'block' refers to a 'bucket'.
  using addr_t = dtl::block_addressing_logic<block_addressing>;
  const addr_t block_addr;

  // The table logic.
  const table_t<bits_per_tag, tags_per_bucket> table;

  // The victim cache is stored at the very end of the (externally managed)
  // filter data.
  const std::size_t victim_cache_offset;

  static constexpr std::size_t bucket_bitlength =
      table_t<bits_per_tag, tags_per_bucket>::bytes_per_bucket * 8;

  //===----------------------------------------------------------------------===//
  explicit
  cuckoofilter_logic(const std::size_t desired_bitlength)
      : block_addr((desired_bitlength + (bucket_bitlength - 1)) / bucket_bitlength),
        table(block_addr.get_block_cnt()),
        victim_cache_offset(has_victim_cache ? table.word_cnt() : 0) {
  }
  cuckoofilter_logic(cuckoofilter_logic&&) noexcept = default;
  cuckoofilter_logic(const cuckoofilter_logic&) = default;
  ~cuckoofilter_logic() = default;
  cuckoofilter_logic& operator=(cuckoofilter_logic&&) noexcept = default;
  cuckoofilter_logic& operator=(const cuckoofilter_logic&) = default;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Maps a 32-bit hash value to a bucket index.
  __forceinline__ __host__ __device__
  uint32_t
  index_hash(uint32_t hash_val) const {
    return block_addr.get_block_idx(hash_val);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Derives a (valid) tag from the given 32 bit hash value.
  __forceinline__ __host__ __device__
  uint32_t
  tag_hash(uint32_t hash_value) const {
    uint32_t tag;
    tag = hash_value & tag_mask;
    tag += (tag == 0);
    return tag;
  }

  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  dtl::vec<uint32_t, dtl::vector_length<Tv>::value>
  tag_hash(const Tv& hash_value) const {
    const auto m = tag_mask;
    auto tag = hash_value & m;
    tag[tag == 0] += 1;
    return tag;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Hash the key and derive the tag and the bucket index.
  __forceinline__ __host__ __device__
  void
  generate_index_tag_hash(const key_t& key, uint32_t* index, uint32_t* tag) const {
    *index = index_hash(dtl::hasher<uint32_t, 0>::hash(key));
    *tag = tag_hash(dtl::hasher<uint32_t, 1>::hash(key));
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Compute the alternative bucket index
  /// where # of buckets is a power of two.
  __forceinline__ __host__ __device__
  uint32_t
  alt_index(const uint32_t bucket_idx, const uint32_t tag,
      const dtl::block_addressing_logic<dtl::block_addressing::POWER_OF_TWO>&
      addr_logic) const {
    // NOTE(binfan): originally we use:
    // bucket_idx ^ HashUtil::BobHash((const void*) (&tag), 4)) & table->INDEXMASK;
    // now doing a quick-n-dirty way:
    // 0x5bd1e995 is the hash constant from MurmurHash2
    return (bucket_idx ^ (tag * 0x5bd1e995)) & block_addr.block_cnt_mask;
  }
  // vectorized
  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  dtl::vec<uint32_t, dtl::vector_length<Tv>::value>
  alt_index(const Tv& bucket_idx, const Tv& tag,
      const dtl::block_addressing_logic<dtl::block_addressing::POWER_OF_TWO>&
      addr_logic) const {
    return (bucket_idx ^ (tag * 0x5bd1e995)) & block_addr.block_cnt_mask;
  }

  /// Compute the alternative bucket index
  /// where # of buckets is NOT a power of two (aka MAGIC addressing)
  __forceinline__ __host__ __device__
  uint32_t
  alt_index(const uint32_t bucket_idx, const uint32_t tag,
      const dtl::block_addressing_logic<dtl::block_addressing::MAGIC>&
      addr_logic) const {
    // If the number of buckets is not a power of two, we cannot use XOR anymore.
    // Instead we use the following self-inverse function:
    // f_sig(x) = - (s + x) mod m
    // where m denotes the number of buckets.
    return block_addr.get_block_idx(-(bucket_idx + (tag * 0x5bd1e995)));
  }
  // vectorized
  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  dtl::vec<uint32_t, dtl::vector_length<Tv>::value>
  alt_index(const Tv& bucket_idx, const Tv& tag,
      const dtl::block_addressing_logic<dtl::block_addressing::MAGIC>&
      addr_logic) const {
    return block_addr.get_block_idxs(Tv::make(0) - (bucket_idx + (tag * 0x5bd1e995))); // TODO make(0) can be optimized
  }

  /// Compute the alternative bucket index.
  __forceinline__ __host__ __device__
  uint32_t
  alt_index(const uint32_t bucket_idx, const uint32_t tag) const {
    // Dispatch based on the addressing type.
    // Note: This doesn't induce runtime overhead.
    return alt_index(bucket_idx, tag, block_addr);
  }
  // vectorized
  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  dtl::vec<uint32_t, dtl::vector_length<Tv>::value>
  alt_index(const Tv& bucket_idx, const Tv& tag) const {
    return alt_index(bucket_idx, tag, block_addr);
  }
  //===----------------------------------------------------------------------===//


public:

  //===----------------------------------------------------------------------===//
  // Insert
  //===----------------------------------------------------------------------===//
  bool
  insert(word_t* __restrict filter_data, const key_t& key) {
    uint32_t i;
    uint32_t tag;
    victim_cache_t& victim =
        *reinterpret_cast<victim_cache_t*>(&filter_data[victim_cache_offset]);
    if (has_victim_cache) {
      if (victim.used) {
        return false;
      }
    }

    generate_index_tag_hash(key, &i, &tag);
    uint32_t curindex = i;
    uint32_t curtag = tag;
    uint32_t oldtag;

    for (uint32_t count = 0; count < max_relocation_count; count++) {
      bool kickout = count > 0;
      oldtag = 0;
      if (table.insert_tag_to_bucket(filter_data, curindex, curtag, kickout, oldtag)) {
        return true;
      }
      if (kickout) {
        curtag = oldtag;
      }
      assert(alt_index(alt_index(curindex, curtag), curtag) == curindex);
      curindex = alt_index(curindex, curtag);
    }

    if (has_victim_cache) {
      victim.index = curindex;
      victim.tag = curtag;
      victim.used = true;
    }
    return true;
  };
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Batch Insert
  //===----------------------------------------------------------------------===//
  bool
  batch_insert(word_t* __restrict filter_data,
               const key_t* keys, u32 key_cnt) noexcept {
    $u1 success = true;
    for (std::size_t i = 0; i < key_cnt; i++) {
      success &= insert(filter_data, keys[i]);
    }
    return success;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains
  //===----------------------------------------------------------------------===//

  // Scalar code path.
  __forceinline__ __host__ __device__
  bool
  contains(const word_t* __restrict filter_data, const key_t& key) const {
    uint32_t i1, i2;
    uint32_t tag;

    generate_index_tag_hash(key, &i1, &tag);
    i2 = alt_index(i1, tag);

    assert(i1 == alt_index(i2, tag));

    if (has_victim_cache) {
      const victim_cache_t& victim =
          *reinterpret_cast<const victim_cache_t*>(&filter_data[victim_cache_offset]);
      bool found = victim.used && (tag == victim.tag) &&
          (i1 == victim.index || i2 == victim.index);
      return found | table.find_tag_in_buckets(filter_data, i1, i2, tag);
    }
    else {
      return table.find_tag_in_buckets(filter_data, i1, i2, tag);
    }

  }

  // Vectorized code path.
  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  typename dtl::vec<uint32_t, dtl::vector_length<Tv>::value>::mask
  contains_vec(const word_t* __restrict filter_data, const Tv& keys) const {

    Tv i1 = block_addr.get_block_idxs(dtl::hasher<Tv, 0>::hash(keys));
    Tv tags = tag_hash(dtl::hasher<Tv, 1>::hash(keys));
    Tv i2 = alt_index(i1, tags);

    if (has_victim_cache) {
      auto found_mask = Tv::mask::make_none_mask();
      const victim_cache_t& victim =
          *reinterpret_cast<const victim_cache_t*>(&filter_data[victim_cache_offset]);
      if (victim.used) {
        auto found_in_victim_cache = (tags == victim.tag)
            & ((i1 == victim.index) | (i2 == victim.index));
        found_mask |= found_in_victim_cache;
      }
      found_mask |= table.find_tag_in_buckets(filter_data, i1, i2, tags);
      return found_mask;
    }
    else {
      return table.find_tag_in_buckets(filter_data, i1, i2, tags);
    }
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Batch-wise Contains
  //===----------------------------------------------------------------------===//
  template<u64 vector_len = dtl::simd::lane_count<u32>>
  $u64
  batch_contains(const word_t* __restrict filter_data,
                 const key_t* __restrict keys, u32 key_cnt,
                 $u32* __restrict match_positions, u32 match_offset) const {
    // determine whether a SIMD implementation is available
    constexpr u64 actual_vector_len =
        table_t<bits_per_tag, tags_per_bucket>::is_vectorized
        ? vector_len
        : 0;
    return internal::batch_contains<cuckoofilter_logic, actual_vector_len>
                   ::dispatch(*this, filter_data,
                              keys, key_cnt,
                              match_positions, match_offset);
  }


  $u64
  batch_contains_scalar(const word_t* __restrict filter_data,
                        const key_t* keys, u32 key_cnt,
                        $u32* match_positions, u32 match_offset) const {
    $u32* match_writer = match_positions;
    $u32 i = 0;
    if (key_cnt >= 4) {
      for (; i < key_cnt - 4; i += 4) {
        u1 is_match_0 = contains(filter_data, keys[i]);
        u1 is_match_1 = contains(filter_data, keys[i + 1]);
        u1 is_match_2 = contains(filter_data, keys[i + 2]);
        u1 is_match_3 = contains(filter_data, keys[i + 3]);
        *match_writer = i + match_offset;
        match_writer += is_match_0;
        *match_writer = (i + 1) + match_offset;
        match_writer += is_match_1;
        *match_writer = (i + 2) + match_offset;
        match_writer += is_match_2;
        *match_writer = (i + 3) + match_offset;
        match_writer += is_match_3;
      }
    }
    for (; i < key_cnt; i++) {
      u1 is_match = contains(filter_data, keys[i]);
      *match_writer = i + match_offset;
      match_writer += is_match;
    }
    return match_writer - match_positions;
  }


  template<
      std::size_t vector_len = dtl::simd::lane_count<u32>
  >
  __attribute_noinline__ $u64
  batch_contains_vec(const word_t* __restrict filter_data,
                     const key_t* keys, u32 key_cnt,
                     $u32* match_positions, u32 match_offset) const {
    using vec_t = dtl::vec<key_t, vector_len>;

    const key_t* reader = keys;
    $u32* match_writer = match_positions;

    // Determine the number of keys that need to be probed sequentially, due to alignment.
    u64 required_alignment_bytes = vec_t::byte_alignment;
    u64 t = dtl::mem::is_aligned(reader)  // should always be true
            ? (required_alignment_bytes - (reinterpret_cast<uintptr_t>(reader) % required_alignment_bytes)) / sizeof(key_t) // FIXME first elements are processed sequentially even if aligned
            : key_cnt;
    u64 unaligned_key_cnt = std::min(static_cast<$u64>(key_cnt), t);
    // Process the unaligned keys sequentially.
    $u64 read_pos = 0;
    for (; read_pos < unaligned_key_cnt; read_pos++) {
      u1 is_match = contains(filter_data, *reader);
      *match_writer = static_cast<$u32>(read_pos) + match_offset;
      match_writer += is_match;
      reader++;
    }
    // Process the aligned keys vectorized.
    u64 aligned_key_cnt = ((key_cnt - unaligned_key_cnt) / vector_len) * vector_len;
    for (; read_pos < (unaligned_key_cnt + aligned_key_cnt); read_pos += vector_len) {
      const auto mask = contains_vec(filter_data, *reinterpret_cast<const vec_t*>(reader));
      u64 match_cnt = mask.to_positions(match_writer, read_pos + match_offset);
      match_writer += match_cnt;
      reader += vector_len;
    }
    // Process remaining keys sequentially.
    for (; read_pos < key_cnt; read_pos++) {
      u1 is_match = contains(filter_data, *reader);
      *match_writer = static_cast<$u32>(read_pos) + match_offset;
      match_writer += is_match;
      reader++;
    }
    return match_writer - match_positions;
  }
  //===----------------------------------------------------------------------===//


  /// Returns the size of the filter in number of words.
  __forceinline__ __host__ __device__
  std::size_t
  word_cnt() const {
    if (has_victim_cache) {
      return table.word_cnt()
          + ((sizeof(victim_cache_t) + (sizeof(word_t) - 1)) / sizeof(word_t));
    }
    else {
      return table.word_cnt();
    }
  }

  std::size_t
  size_in_bytes() const {
    return table_t<bits_per_tag, tags_per_bucket>::bytes_per_bucket
        * table.num_buckets_;
  }

  std::size_t
  count_occupied_slots(const word_t* __restrict filter_data) const {
    if (has_victim_cache) {
      const victim_cache_t& victim = *reinterpret_cast<const victim_cache_t*>(
          &filter_data[victim_cache_offset]);
      return table.count_occupied_entires(filter_data) + victim.used;
    }
    else {
      return table.count_occupied_entires(filter_data);
    }
  }

  std::size_t
  get_bucket_count() const {
    return table.num_buckets();
  }

  std::size_t
  get_slot_count() const {
    if (has_victim_cache) {
      return table.size_in_tags() + 1 /* victim cache */;
    }
    else{
      return table.size_in_tags();
    }
  }

  std::size_t
  get_tags_per_bucket() const {
    return tags_per_bucket;
  }

  std::size_t
  get_bits_per_tag() const {
    return bits_per_tag;
  }

};
//===----------------------------------------------------------------------===//


}  // namespace cuckoofilter
}  // namespace dtl
