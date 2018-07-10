#pragma once

#include <dtl/dtl.hpp>

#include "dtl/bloomfilter/blocked_bloomfilter_block_logic.hpp"
#include "dtl/bloomfilter/hash_family.hpp"

namespace dtl {

extern template struct blocked_bloomfilter_block_logic<$u32, $u64,  1,  1,  1, hasher, $u32, 1>;
extern template struct blocked_bloomfilter_block_logic<$u32, $u64,  1,  1,  2, hasher, $u32, 1>;
extern template struct blocked_bloomfilter_block_logic<$u32, $u64,  1,  1,  3, hasher, $u32, 1>;
extern template struct blocked_bloomfilter_block_logic<$u32, $u64,  1,  1,  4, hasher, $u32, 1>;
extern template struct blocked_bloomfilter_block_logic<$u32, $u64,  1,  1,  5, hasher, $u32, 1>;
extern template struct blocked_bloomfilter_block_logic<$u32, $u64,  1,  1,  6, hasher, $u32, 1>;
extern template struct blocked_bloomfilter_block_logic<$u32, $u64,  1,  1,  7, hasher, $u32, 1>;
extern template struct blocked_bloomfilter_block_logic<$u32, $u64,  1,  1,  8, hasher, $u32, 1>;
extern template struct blocked_bloomfilter_block_logic<$u32, $u64,  1,  1,  9, hasher, $u32, 1>;
extern template struct blocked_bloomfilter_block_logic<$u32, $u64,  1,  1, 10, hasher, $u32, 1>;
extern template struct blocked_bloomfilter_block_logic<$u32, $u64,  1,  1, 11, hasher, $u32, 1>;
extern template struct blocked_bloomfilter_block_logic<$u32, $u64,  1,  1, 12, hasher, $u32, 1>;
extern template struct blocked_bloomfilter_block_logic<$u32, $u64,  1,  1, 13, hasher, $u32, 1>;
extern template struct blocked_bloomfilter_block_logic<$u32, $u64,  1,  1, 14, hasher, $u32, 1>;
extern template struct blocked_bloomfilter_block_logic<$u32, $u64,  1,  1, 15, hasher, $u32, 1>;
extern template struct blocked_bloomfilter_block_logic<$u32, $u64,  1,  1, 16, hasher, $u32, 1>;

} // namespace dtl