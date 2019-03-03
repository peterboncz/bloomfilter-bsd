#pragma once

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>
#include <dtl/math.hpp>
#include <dtl/mem.hpp>
#include <dtl/simd.hpp>

namespace dtl {
namespace internal { // TODO should be in bloom filter namespace

//===----------------------------------------------------------------------===//
// Batch-wise Contains (SIMD)
//===----------------------------------------------------------------------===//
template<
    typename filter_t,
    u64 vector_len
>
struct dispatch {

  // Typedefs
  using key_t = typename filter_t::key_t;
  using word_t = typename filter_t::word_t;
  using vec_t = vec<key_t, vector_len>;
  using mask_t = typename vec<key_t, vector_len>::mask;


  static $u64 __attribute__ ((__noinline__))
  batch_contains(const filter_t& filter,
                 const word_t* __restrict filter_data,
                 const key_t* __restrict keys, u32 key_cnt,
                 $u32* __restrict match_positions, u32 match_offset) {

    const key_t* reader = keys;
    $u32* match_writer = match_positions;

    // Determine the number of keys that need to be probed sequentially, due to alignment
    u64 required_alignment_bytes = vec_t::byte_alignment;
    u1 is_aligned = (reinterpret_cast<uintptr_t>(reader) % alignof(key_t)) == 0; // TODO use dtl instead
//    u1 is_aligned = dtl::mem::is_aligned(reader)
    u64 t = is_aligned  // should always be true
            ? (required_alignment_bytes - (reinterpret_cast<uintptr_t>(reader) % required_alignment_bytes)) / sizeof(key_t) // FIXME first elements are processed sequentially even if aligned
            : key_cnt;
    u64 unaligned_key_cnt = std::min(static_cast<$u64>(key_cnt), t);
    // process the unaligned keys sequentially
    $u64 read_pos = 0;
    for (; read_pos < unaligned_key_cnt; read_pos++) {
      u1 is_match = filter.contains(filter_data, *reader);
      *match_writer = static_cast<$u32>(read_pos) + match_offset;
      match_writer += is_match;
      reader++;
    }
    // Process the aligned keys vectorized
    u64 aligned_key_cnt = ((key_cnt - unaligned_key_cnt) / vector_len) * vector_len;
    for (; read_pos < (unaligned_key_cnt + aligned_key_cnt); read_pos += vector_len) {
      const auto mask = filter.template contains_vec<vector_len>(filter_data, *reinterpret_cast<const vec_t*>(reader));
      u64 match_cnt = mask.to_positions(match_writer, read_pos + match_offset);
      match_writer += match_cnt;
      reader += vector_len;
    }
    // process remaining keys sequentially
    for (; read_pos < key_cnt; read_pos++) {
      u1 is_match = filter.contains(filter_data, *reader);
      *match_writer = static_cast<$u32>(read_pos) + match_offset;
      match_writer += is_match;
      reader++;
    }
    return match_writer - match_positions;
  }

  static void __attribute__ ((__noinline__))
  batch_get_block_idx(const filter_t& filter,
                      const key_t* __restrict keys,
                      u32 key_cnt,
                      $u32* __restrict block_idxs) {
    const key_t* reader = keys;
    $u32* writer = block_idxs;

    // Determine the number of keys that need to be probed sequentially, due to alignment
    u64 required_alignment_bytes = vec_t::byte_alignment;
    u1 is_aligned = (reinterpret_cast<uintptr_t>(reader) % alignof(key_t)) == 0; // TODO use dtl instead
    u64 t = is_aligned  // should always be true
            ? (required_alignment_bytes - (reinterpret_cast<uintptr_t>(reader) % required_alignment_bytes)) / sizeof(key_t) // FIXME first elements are processed sequentially even if aligned
            : key_cnt;
    u64 unaligned_key_cnt = std::min(static_cast<$u64>(key_cnt), t);
    // process the unaligned keys sequentially
    $u64 read_pos = 0;
    for (; read_pos < unaligned_key_cnt; read_pos++) {
      *writer = filter.get_block_idx(*reader);
      writer++;
      reader++;
    }
    // Process the aligned keys vectorized
    u64 aligned_key_cnt = ((key_cnt - unaligned_key_cnt) / vector_len) * vector_len;
    for (; read_pos < (unaligned_key_cnt + aligned_key_cnt); read_pos += vector_len) {
      const auto b = filter.template get_block_idx_vec<vector_len>(
          *reinterpret_cast<const vec_t*>(reader));
      b.storeu(writer);
      writer += vector_len;
      reader += vector_len;
    }
    // process remaining keys sequentially
    for (; read_pos < key_cnt; read_pos++) {
      *writer = filter.get_block_idx(*reader);
      writer++;
      reader++;
    }
  }

};


//===----------------------------------------------------------------------===//
// Batch-wise Contains (no SIMD)
//===----------------------------------------------------------------------===//
template<
    typename filter_t
>
struct dispatch<filter_t, 0> {

  // Typedefs
  using key_t = typename filter_t::key_t;
  using word_t = typename filter_t::word_t;

  static $u64
  batch_contains(const filter_t& filter,
                 const word_t* __restrict filter_data,
                 const key_t* __restrict keys, u32 key_cnt,
                 $u32* __restrict match_positions, u32 match_offset) {
    $u32* match_writer = match_positions;
    $u32 i = 0;
    if (key_cnt >= 4) {
      for (; i < key_cnt - 4; i += 4) {
        u1 is_match_0 = filter.contains(filter_data, keys[i]);
        u1 is_match_1 = filter.contains(filter_data, keys[i + 1]);
        u1 is_match_2 = filter.contains(filter_data, keys[i + 2]);
        u1 is_match_3 = filter.contains(filter_data, keys[i + 3]);
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
      u1 is_match = filter.contains(filter_data, keys[i]);
      *match_writer = i + match_offset;
      match_writer += is_match;
    }
    return match_writer - match_positions;
  }

  static void
  batch_get_block_idx(const filter_t& filter,
                      const key_t* __restrict keys,
                      u32 key_cnt,
                      $u32* __restrict block_idxs) {
    $u32* match_writer = block_idxs;
    $u32 i = 0;
    if (key_cnt >= 4) {
      for (; i < key_cnt - 4; i += 4) {
        u32 block_idx_0 = filter.get_block_idx(keys[i]);
        u32 block_idx_1 = filter.get_block_idx(keys[i + 1]);
        u32 block_idx_2 = filter.get_block_idx(keys[i + 2]);
        u32 block_idx_3 = filter.get_block_idx(keys[i + 3]);
        match_writer[0] = block_idx_0;
        match_writer[1] = block_idx_1;
        match_writer[2] = block_idx_2;
        match_writer[3] = block_idx_3;
        match_writer += 4;
      }
    }
    for (; i < key_cnt; i++) {
      *match_writer = filter.get_block_idx(keys[i]);
      match_writer++;
    }
  }

};

} // namespace internal
} // namespace dtl