#pragma once

#include <functional>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/bloomfilter.hpp>
#include <dtl/math.hpp>
#include <dtl/mem.hpp>
#include <dtl/simd.hpp>

#include "immintrin.h"

namespace dtl {

template<typename Tk, template<typename Ty> class hash_fn, typename Tw = u64, typename Alloc = std::allocator<Tw>>
struct bloomfilter_vec {

  using bloomfilter_t = dtl::bloomfilter<Tk, hash_fn, Tw, Alloc>;
  const bloomfilter_t& bf;

  using key_t = typename bloomfilter_t::key_t;
  using word_t = typename bloomfilter_t::word_t;

  template<u64 vector_len>
  forceinline typename vec<key_t, vector_len>::mask_t
  contains(const vec<key_t, vector_len>& keys) const {
    assert(dtl::mem::is_aligned(&keys, 32));
    using key_vt = vec<key_t, vector_len>;
    using word_vt = vec<typename bloomfilter_t::word_t, vector_len>;
    const key_vt hash_vals = hash_fn<key_vt>::hash(keys);
    const key_vt bit_idxs = hash_vals & bf.length_mask;
    const key_vt word_idxs = bit_idxs >> bloomfilter_t::word_bitlength_log2;
    const word_vt words = dtl::gather(bf.word_array.data(), word_idxs);
    const key_vt in_word_idxs = bit_idxs & (bloomfilter_t::word_bitlength - 1);
    const key_vt second_in_word_idxs = hash_vals >> (32 - bloomfilter_t::word_bitlength_log2);
    const word_vt search_masks = (word_vt::make(1) << in_word_idxs) | (word_vt::make(1) << second_in_word_idxs); // implement vec(x) constructor
    return (words & search_masks) == search_masks;
  }

  /// Performs a batch-probe
  template<u64 vector_len = dtl::simd::lane_count<key_t>>
  forceinline $u64
  contains(const key_t* keys, u32 key_cnt, $u32* match_positions, u32 match_offset) const {
    const key_t* reader = keys;
    $u32* match_writer = match_positions;

    // determine the number of keys that need to be probed sequentially, due to alignment
    u64 required_alignment_bytes = 64;
    u64 unaligned_key_cnt = dtl::mem::is_aligned(reader)
                            ? (required_alignment_bytes - (reinterpret_cast<uintptr_t>(reader) % required_alignment_bytes)) / sizeof(key_t)
                            : key_cnt;
    // process the unaligned keys sequentially
    $u64 read_pos = 0;
    for (; read_pos < unaligned_key_cnt; read_pos++) {
      u1 is_match = bf.contains(*reader);
      *match_writer = static_cast<$u32>(read_pos) + match_offset;
      match_writer += is_match;
      reader++;
    }
    // process the aligned keys vectorized
    using vec_t = vec<key_t, vector_len>;
    using mask_t = typename vec<key_t, vector_len>::mask;
    u64 aligned_key_cnt = ((key_cnt - unaligned_key_cnt) / vector_len) * vector_len;
    for (; read_pos < (unaligned_key_cnt + aligned_key_cnt); read_pos += vector_len) {
      assert(dtl::mem::is_aligned(reader, 32));
      const mask_t mask = contains<vector_len>(*reinterpret_cast<const vec_t*>(reader));
      u64 match_cnt = mask.to_positions(match_writer, read_pos + match_offset);
      match_writer += match_cnt;
      reader += vector_len;
    }
    // process remaining keys sequentially
    for (; read_pos < key_cnt; read_pos++) {
      u1 is_match = bf.contains(*reader);
      *match_writer = static_cast<$u32>(read_pos) + match_offset;
      match_writer += is_match;
      reader++;
    }
    return match_writer - match_positions;
  }

};

} // namespace dtl