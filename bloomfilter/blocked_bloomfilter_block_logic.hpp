#pragma once

#include "blocked_bloomfilter_block_logic_sgew.hpp" // sector_cnt >= word_cnt
#include "blocked_bloomfilter_block_logic_sltw.hpp" // sector_cnt < word_cnt
#include "hash_family.hpp"

namespace dtl {


template<
    typename key_t,               // the key type
    typename word_t,              // the word type
    u32 word_cnt,                 // the number of words per block
    u32 sector_cnt,               // the numbers of sectors (must be a power of two and greater or equal to word_cnt))
    u32 k,                        // the number of bits to set/test
    template<typename Ty, u32 i> class hasher,      // the hash function family to use
    typename hash_value_t,        // the hash value type to use
    u32 hash_fn_idx               // current hash function index (used for recursion)
>
struct blocked_bloomfilter_block_logic {

  using sgew_t = multiword_block<key_t, word_t, word_cnt, sector_cnt, k,
                                 hasher, hash_value_t, hash_fn_idx, 0, word_cnt>;

  using sltw_t = multisector_block<key_t, word_t, word_cnt, sector_cnt, k,
                                   hasher, hash_value_t, hash_fn_idx, 0, sector_cnt>;

  using type = typename std::conditional<sector_cnt >= word_cnt, sgew_t, sltw_t>::type;

};

template<
    typename key_t,
    $u32 hash_fn_no
>
using hasher = dtl::hash::stat::mul32<key_t, hash_fn_no>;


//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 1, 1, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 1, 2, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 1, 3, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 1, 4, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 1, 5, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 1, 6, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 1, 7, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 1, 8, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 1, 9, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 1,10, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 1,11, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 1,12, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 1,13, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 1,14, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 1,15, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 1,16, hasher, $u32, 1>;
//
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 2, 1, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 2, 2, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 2, 3, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 2, 4, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 2, 5, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 2, 6, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 2, 7, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 2, 8, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 2, 9, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 2,10, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 2,11, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 2,12, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 2,13, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 2,14, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 2,15, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 2,16, hasher, $u32, 1>;
//
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 4, 1, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 4, 2, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 4, 3, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 4, 4, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 4, 5, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 4, 6, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 4, 7, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 4, 8, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 4, 9, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 4,10, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 4,11, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 4,12, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 4,13, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 4,14, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 4,15, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 1, 4,16, hasher, $u32, 1>;
//
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 1, 1, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 1, 2, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 1, 3, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 1, 4, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 1, 5, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 1, 6, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 1, 7, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 1, 8, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 1, 9, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 1,10, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 1,11, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 1,12, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 1,13, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 1,14, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 1,15, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 1,16, hasher, $u32, 1>;
//
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 2, 1, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 2, 2, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 2, 3, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 2, 4, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 2, 5, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 2, 6, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 2, 7, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 2, 8, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 2, 9, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 2,10, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 2,11, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 2,12, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 2,13, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 2,14, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 2,15, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 2,16, hasher, $u32, 1>;
//
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 4, 1, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 4, 2, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 4, 3, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 4, 4, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 4, 5, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 4, 6, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 4, 7, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 4, 8, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 4, 9, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 4,10, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 4,11, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 4,12, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 4,13, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 4,14, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 4,15, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 4,16, hasher, $u32, 1>;
//
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 8, 1, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 8, 2, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 8, 3, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 8, 4, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 8, 5, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 8, 6, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 8, 7, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 8, 8, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 8, 9, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 8,10, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 8,11, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 8,12, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 8,13, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 8,14, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 8,15, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 2, 8,16, hasher, $u32, 1>;
//
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 4, 1, 1, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 4, 1, 2, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 4, 1, 3, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 4, 1, 4, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 4, 1, 5, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 4, 1, 6, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 4, 1, 7, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 4, 1, 8, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 4, 1, 9, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 4, 1,10, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 4, 1,11, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 4, 1,12, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 4, 1,13, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 4, 1,14, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 4, 1,15, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 4, 1,16, hasher, $u32, 1>;
//
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 8, 1, 1, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 8, 1, 2, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 8, 1, 3, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 8, 1, 4, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 8, 1, 5, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 8, 1, 6, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 8, 1, 7, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 8, 1, 8, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 8, 1, 9, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 8, 1,10, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 8, 1,11, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 8, 1,12, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 8, 1,13, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 8, 1,14, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 8, 1,15, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32, 8, 1,16, hasher, $u32, 1>;
//
//extern template class blocked_bloomfilter_block_logic<$u32, $u32,16, 1, 1, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32,16, 1, 2, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32,16, 1, 3, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32,16, 1, 4, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32,16, 1, 5, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32,16, 1, 6, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32,16, 1, 7, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32,16, 1, 8, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32,16, 1, 9, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32,16, 1,10, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32,16, 1,11, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32,16, 1,12, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32,16, 1,13, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32,16, 1,14, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32,16, 1,15, hasher, $u32, 1>;
//extern template class blocked_bloomfilter_block_logic<$u32, $u32,16, 1,16, hasher, $u32, 1>;

} // namespace dtl