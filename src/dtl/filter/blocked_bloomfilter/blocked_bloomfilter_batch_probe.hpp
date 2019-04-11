#pragma once

#include <bitset>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>
#include <dtl/math.hpp>
#include <dtl/simd.hpp>
#include <dtl/filter/blocked_bloomfilter/block_addressing_logic.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_batch_dispatch.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_block_logic.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_batch_probe_base.hpp>

#include "immintrin.h"


namespace dtl {

template<typename filter_t>
struct blocked_bloomfilter_batch_probe :
    public blocked_bloomfilter_batch_probe_base {

  using key_t = typename filter_t::key_t;
  using word_t = typename filter_t::word_t;

  const filter_t filter_logic_;

  explicit
  blocked_bloomfilter_batch_probe(const filter_t& filter_logic)
      : filter_logic_(filter_logic) { }

  ~blocked_bloomfilter_batch_probe() = default;

//  template<u64 vector_len>
//  void
//  batch_contains_bitmap(const word_t* __restrict filter_data,
//      const key_t* __restrict keys, u32 key_cnt,
//      word_t* __restrict bitmap) const {
//    internal::dispatch<filter_t, vector_len>
//        ::batch_contains_bitmap(*this, filter_data, keys, key_cnt, bitmap);
//  }

  void
  batch_contains_bitmap(const word_t* __restrict filter_data,
      const key_t* __restrict keys, u32 key_cnt,
      word_t* __restrict bitmap, u32 unroll_factor) const {
//#ifdef __CUDACC__
//    batch_contains_bitmap<dtl::simd::lane_count<key_t> * 0>(
//        filter_data, keys, key_cnt, bitmap);
//#else
    switch (unroll_factor) {
      case 0: internal::dispatch<filter_t, dtl::simd::lane_count<key_t> * 0>
        ::batch_contains_bitmap(filter_logic_, filter_data, keys, key_cnt, bitmap);
        break;
      case 1: internal::dispatch<filter_t, dtl::simd::lane_count<key_t> * 1>
        ::batch_contains_bitmap(filter_logic_, filter_data, keys, key_cnt, bitmap);
        break;
      case 2: internal::dispatch<filter_t, dtl::simd::lane_count<key_t> * 2>
        ::batch_contains_bitmap(filter_logic_, filter_data, keys, key_cnt, bitmap);
        break;
     default: internal::dispatch<filter_t, dtl::simd::lane_count<key_t> * 4>
        ::batch_contains_bitmap(filter_logic_, filter_data, keys, key_cnt, bitmap);
        break;
    }
//#endif
  }

};
} // namespace dtl
