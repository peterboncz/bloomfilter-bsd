#pragma once
//===----------------------------------------------------------------------===//
#define GENERATE_EXTERN_ADDR_UNROLL(word_cnt, sector_cnt, zone_cnt, k, addr, unroll) \
namespace dtl { \
extern template \
void amsfilter::internal::bbf_type<word_cnt, sector_cnt, zone_cnt, k, addr>::type \
  ::template batch_contains_bitmap<dtl::simd::lane_count<amsfilter::key_t> * unroll>( \
    const amsfilter::word_t* __restrict /*filter_data*/, \
    const amsfilter::key_t* __restrict /*keys*/, u32 /*key_cnt*/, \
    amsfilter::word_t* __restrict /*bitmap*/) const; \
}

#define GENERATE_EXTERN_ADDR(word_cnt, sector_cnt, zone_cnt, k, addr) \
GENERATE_EXTERN_ADDR_UNROLL(word_cnt, sector_cnt, zone_cnt, k, addr, 0) \
GENERATE_EXTERN_ADDR_UNROLL(word_cnt, sector_cnt, zone_cnt, k, addr, 1) \
GENERATE_EXTERN_ADDR_UNROLL(word_cnt, sector_cnt, zone_cnt, k, addr, 2) \
GENERATE_EXTERN_ADDR_UNROLL(word_cnt, sector_cnt, zone_cnt, k, addr, 4)

// Macro to generate an extern template.
#define GENERATE_EXTERN(word_cnt, sector_cnt, zone_cnt, k) \
GENERATE_EXTERN_ADDR(word_cnt, sector_cnt, zone_cnt, k, dtl::block_addressing::POWER_OF_TWO) \
GENERATE_EXTERN_ADDR(word_cnt, sector_cnt, zone_cnt, k, dtl::block_addressing::MAGIC)
//===----------------------------------------------------------------------===//
#define GENERATE_ADDR_UNROLL(word_cnt, sector_cnt, zone_cnt, k, addr, unroll) \
namespace dtl { \
template \
void amsfilter::internal::bbf_type<word_cnt, sector_cnt, zone_cnt, k, addr>::type \
    ::template batch_contains_bitmap<dtl::simd::lane_count<amsfilter::key_t> * unroll>( \
      const amsfilter::word_t* __restrict /*filter_data*/, \
      const amsfilter::key_t* __restrict /*keys*/, u32 /*key_cnt*/, \
      amsfilter::word_t* __restrict /*bitmap*/) const; \
}

#define GENERATE_ADDR(word_cnt, sector_cnt, zone_cnt, k, addr) \
GENERATE_ADDR_UNROLL(word_cnt, sector_cnt, zone_cnt, k, addr, 0) \
GENERATE_ADDR_UNROLL(word_cnt, sector_cnt, zone_cnt, k, addr, 1) \
GENERATE_ADDR_UNROLL(word_cnt, sector_cnt, zone_cnt, k, addr, 2) \
GENERATE_ADDR_UNROLL(word_cnt, sector_cnt, zone_cnt, k, addr, 4)

// Macro to instantiate an extern template.
#define GENERATE(word_cnt, sector_cnt, zone_cnt, k) \
GENERATE_ADDR(word_cnt, sector_cnt, zone_cnt, k, dtl::block_addressing::POWER_OF_TWO) \
GENERATE_ADDR(word_cnt, sector_cnt, zone_cnt, k, dtl::block_addressing::MAGIC)
//===----------------------------------------------------------------------===//
