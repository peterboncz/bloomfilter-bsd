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

  using bf_t = dtl::bloomfilter<Tk, hash_fn, Tw, Alloc>;
  const bf_t& bf;

  using key_t = typename bf_t::key_t;
  using word_t = typename bf_t::word_t;
  using size_t = typename bf_t::size_t;
  using hash_value_t = typename bf_t::hash_value_t;

//  static constexpr word_t sector_mask = bf_t::sector_mask;


  template<u64 vector_len>
  forceinline vec<size_t, vector_len>
  which_word(const vec<hash_value_t, vector_len>& hash_val) const noexcept{
    const vec<size_t, vector_len> word_idx = hash_val >> (bf_t::hash_value_bitlength - bf.word_cnt_log2);
//    assert((word_idx < ((bf.length_mask + 1) / bf_t::word_bitlength)).all());
    return word_idx;
  }


  template<u64 n> // the vector length
  forceinline unroll_loops vec<word_t, n>
  which_bits(const vec<hash_value_t, n>& hash_val) const noexcept {
    vec<word_t, n> words = 0;
    for (size_t i = 0; i < bf_t::k; i++) {
      const vec<$u32, n> bit_idxs = (hash_val >> (i * bf_t::sector_bitlength_log2)) & bf_t::sector_mask();
      words |= vec<$u32, n>::make(1) << bit_idxs;
    }
    return words;
  }


  template<u64 n> // the vector length
  forceinline typename vec<key_t, n>::mask_t
  contains(const vec<key_t, n>& keys) const noexcept {
    assert(dtl::mem::is_aligned(&keys, 32));
    using key_vt = vec<key_t, n>;
    using word_vt = vec<typename bf_t::word_t, n>;

    const key_vt hash_vals = hash_fn<key_vt>::hash(keys);
    const key_vt word_idxs = which_word(hash_vals);
    const word_vt words = dtl::gather(bf.word_array.data(), word_idxs);
    const word_vt search_masks = which_bits(hash_vals);
    return (words & search_masks) == search_masks;
  }

//  template<u64 vector_len>
//  forceinline typename vec<key_t, vector_len>::mask_t
//  contains(const vec<key_t, vector_len>& keys) const {
//    assert(dtl::mem::is_aligned(&keys, 32));
//    using key_vt = vec<key_t, vector_len>;
//    using word_vt = vec<typename bf_t::word_t, vector_len>;
//    const key_vt hash_vals = hash_fn<key_vt>::hash(keys);
//    const key_vt bit_idxs = hash_vals & bf.length_mask;
//    const key_vt word_idxs = bit_idxs >> bf_t::word_bitlength_log2;
//    const word_vt words = dtl::gather(bf.word_array.data(), word_idxs);
//    const key_vt in_word_idxs = bit_idxs & (bf_t::word_bitlength - 1);
//    const key_vt second_in_word_idxs = hash_vals >> (32 - bf_t::word_bitlength_log2);
//    const word_vt search_masks = (word_vt::make(1) << in_word_idxs) | (word_vt::make(1) << second_in_word_idxs); // implement vec(x) constructor
//    return (words & search_masks) == search_masks;
//  }

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