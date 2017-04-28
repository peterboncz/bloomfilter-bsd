#pragma once

#include <bitset>
#include <string>

#include <dtl/dtl.hpp>

namespace dtl {

namespace bits {


/// counts the number of set bits
inline u64
pop_count(u32 a) { return __builtin_popcount(a); }

/// counts the number of set bits
inline u64
pop_count(u64 a) { return __builtin_popcountll(a); }


/// counts the number of leading zeros
inline u64
lz_count(u32 a) { return __builtin_clz(a); }

/// counts the number of leading zeros
inline u64
lz_count(u64 a) { return __builtin_clzll(a); }


/// counts the number of tailing zeros
inline u64
tz_count(u32 a) { return __builtin_ctz(a); }

/// counts the number of tailing zeros
inline u64
tz_count(u64 a) { return __builtin_ctzll(a); }


} // namespace bits

} // namespace dtl