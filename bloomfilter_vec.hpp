#pragma once

#include <functional>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/bloomfilter.hpp>
#include <dtl/math.hpp>
#include <dtl/simd.hpp>

#include "immintrin.h"

namespace dtl {

template<typename Tk, template<typename Ty> class hash_fn, typename Tw = u64>
struct bloomfilter_vec {

  using bloomfilter_t = dtl::bloomfilter<Tk, hash_fn, Tw>;
  bloomfilter_t& bf;

  using key_t = typename bloomfilter_t::key_t;
  using word_t = typename bloomfilter_t::word_t;

  template<u64 vector_len>
  typename vec<key_t, vector_len>::mask_t
  contains(const vec<key_t, vector_len>& keys) const {
    using key_vt = vec<key_t, vector_len>;
    using word_vt = vec<typename bloomfilter_t::word_t, vector_len>;
    const key_vt hash_vals = hash_fn<key_vt>::hash(keys);
    const key_vt bit_idxs = hash_vals & bf.length_mask;
    const key_vt word_idxs = bit_idxs >> bf.word_bitlength_log2;
    const key_vt in_word_idxs = bit_idxs & bf.word_bitlength_mask;
    const key_vt second_in_word_idxs = hash_vals >> (32 - bf.word_bitlength_log2);
    const word_vt search_masks = (word_vt::make(1) << in_word_idxs) | (word_vt::make(1) << second_in_word_idxs); // implement vec(x) constructor
    const word_vt words = dtl::gather(bf.word_array.data(), word_idxs);
    return (words & search_masks) == search_masks;
  }

};

} // namespace dtl