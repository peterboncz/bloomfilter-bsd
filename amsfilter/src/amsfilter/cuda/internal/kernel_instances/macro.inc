#pragma once
//===----------------------------------------------------------------------===//
#define GENERATE_EXTERN_ADDR(word_cnt, sector_cnt, zone_cnt, k, addr) \
extern template \
__global__ void \
contains_kernel_with_block_prefetch<typename amsfilter::internal::bbf_type<word_cnt, sector_cnt, zone_cnt, k, addr>::type>( \
    const amsfilter::internal::bbf_type<word_cnt, sector_cnt, zone_cnt, k, addr>::type /*filter_logic*/, \
    const typename amsfilter::internal::bbf_type<word_cnt, sector_cnt, zone_cnt, k, addr>::type::word_t* __restrict__ /*word_array*/, \
    u32* __restrict__ /*keys*/, u32 /*key_cnt*/, $u32* __restrict__ /*result_bitmap*/);

// Macro to generate an extern template.
#define GENERATE_EXTERN(word_cnt, sector_cnt, zone_cnt, k) \
GENERATE_EXTERN_ADDR(word_cnt, sector_cnt, zone_cnt, k, dtl::block_addressing::POWER_OF_TWO) \
GENERATE_EXTERN_ADDR(word_cnt, sector_cnt, zone_cnt, k, dtl::block_addressing::MAGIC)
//===----------------------------------------------------------------------===//
#define GENERATE_ADDR(word_cnt, sector_cnt, zone_cnt, k, addr) \
template \
__global__ void \
contains_kernel_with_block_prefetch<typename amsfilter::internal::bbf_type<word_cnt, sector_cnt, zone_cnt, k, addr>::type>( \
    const amsfilter::internal::bbf_type<word_cnt, sector_cnt, zone_cnt, k, addr>::type /*filter_logic*/, \
    const typename amsfilter::internal::bbf_type<word_cnt, sector_cnt, zone_cnt, k, addr>::type::word_t* __restrict__ /*word_array*/, \
    u32* __restrict__ /*keys*/, u32 /*key_cnt*/, $u32* __restrict__ /*result_bitmap*/);

// Macro to instantiate an extern template.
#define GENERATE(word_cnt, sector_cnt, zone_cnt, k) \
GENERATE_ADDR(word_cnt, sector_cnt, zone_cnt, k, dtl::block_addressing::POWER_OF_TWO) \
GENERATE_ADDR(word_cnt, sector_cnt, zone_cnt, k, dtl::block_addressing::MAGIC)
//===----------------------------------------------------------------------===//
