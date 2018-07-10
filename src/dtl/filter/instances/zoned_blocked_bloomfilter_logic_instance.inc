// TODO remove global namespace pollution

template<
    typename key_t,
    $u32 hash_fn_no
>
using hasher = dtl::hasher<key_t, hash_fn_no>;
using hash_value_t = $u32;

template<
    typename key_t,
    typename word_t,
    u32 word_cnt,
    u32 zone_cnt,
    u32 k,
    u1 early_out = false
>
using block_logic = typename dtl::multizone_block<key_t, word_t, word_cnt, zone_cnt, k,
                                                  hasher, hash_value_t,
                                                  1 /*block_hash_fn_idx*/, 0, zone_cnt,
                                                  early_out>;

//===----------------------------------------------------------------------===//
// Instantiate templates to parallelize builds
//===----------------------------------------------------------------------===//
// TODO fix unrolling for u64
#define GENERATE_ADDR(word_t, word_cnt, zone_cnt, k, addr) \
template struct blocked_bloomfilter_logic<$u32, hasher, block_logic<$u32, word_t, word_cnt, zone_cnt, k>, addr>; \
template $u64 blocked_bloomfilter_logic<$u32, hasher, block_logic<$u32, word_t, word_cnt, zone_cnt, k>, addr>::batch_contains<                              0>(const word_t*, u32*, u32, $u32*, u32) const; \
template $u64 blocked_bloomfilter_logic<$u32, hasher, block_logic<$u32, word_t, word_cnt, zone_cnt, k>, addr>::batch_contains<dtl::simd::lane_count<$u32> * 1>(const word_t*, u32*, u32, $u32*, u32) const; \
template $u64 blocked_bloomfilter_logic<$u32, hasher, block_logic<$u32, word_t, word_cnt, zone_cnt, k>, addr>::batch_contains<dtl::simd::lane_count<$u32> * 2>(const word_t*, u32*, u32, $u32*, u32) const; \
template $u64 blocked_bloomfilter_logic<$u32, hasher, block_logic<$u32, word_t, word_cnt, zone_cnt, k>, addr>::batch_contains<dtl::simd::lane_count<$u32> * 4>(const word_t*, u32*, u32, $u32*, u32) const; \
template $u64 blocked_bloomfilter_logic<$u32, hasher, block_logic<$u32, word_t, word_cnt, zone_cnt, k>, addr>::batch_contains<dtl::simd::lane_count<$u32> * 8>(const word_t*, u32*, u32, $u32*, u32) const; \
template struct internal::dispatch<blocked_bloomfilter_logic<$u32, hasher, block_logic<$u32, word_t, word_cnt, zone_cnt, k>, addr>,                               0>; \
template struct internal::dispatch<blocked_bloomfilter_logic<$u32, hasher, block_logic<$u32, word_t, word_cnt, zone_cnt, k>, addr>, dtl::simd::lane_count<$u32> * 1>; \
template struct internal::dispatch<blocked_bloomfilter_logic<$u32, hasher, block_logic<$u32, word_t, word_cnt, zone_cnt, k>, addr>, dtl::simd::lane_count<$u32> * 2>; \
template struct internal::dispatch<blocked_bloomfilter_logic<$u32, hasher, block_logic<$u32, word_t, word_cnt, zone_cnt, k>, addr>, dtl::simd::lane_count<$u32> * 4>; \
template struct internal::dispatch<blocked_bloomfilter_logic<$u32, hasher, block_logic<$u32, word_t, word_cnt, zone_cnt, k>, addr>, dtl::simd::lane_count<$u32> * 8>;

#define GENERATE(word_t, word_cnt, zone_cnt, k) \
GENERATE_ADDR(word_t, word_cnt, zone_cnt, k, dtl::block_addressing::POWER_OF_TWO) \
GENERATE_ADDR(word_t, word_cnt, zone_cnt, k, dtl::block_addressing::MAGIC)
