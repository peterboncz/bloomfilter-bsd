#pragma once

#include "blocked_bloomfilter_block_logic_sgew.hpp" // sector_cnt >= word_cnt
#include "blocked_bloomfilter_block_logic_sltw.hpp" // sector_cnt < word_cnt
#include "hash_family.hpp"

namespace dtl {

//===----------------------------------------------------------------------===//
// Type switch:
// 'multiword_block' is used iff the number of sectors is greater or equal
// to the number of words per block, 'multisector_block' otherwise.
//===----------------------------------------------------------------------===//
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

  // Refers to the implementation
  using type = typename std::conditional<sector_cnt >= word_cnt, sgew_t, sltw_t>::type;

};
//===----------------------------------------------------------------------===//


template<
    typename key_t,
    $u32 hash_fn_no
>
using hasher = dtl::hash::stat::mul32<key_t, hash_fn_no>;

} // namespace dtl