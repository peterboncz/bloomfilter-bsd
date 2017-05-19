#pragma once

#include <dtl/dtl.hpp>

#if defined(__BMI__)
#include "immintrin.h"
#endif

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


/// extract contiguous bits
inline u32
extract(u32 a, u32 start, u32 len) {
#if defined(__BMI__)
  return _bextr_u32(a, start, len);
#else
  return (a >> start) & ((u32(1) << len) - 1);
#endif
}

/// extract contiguous bits
inline u64
extract(u64 a, u32 start, u32 len) {
#if defined(__BMI__)
  return _bextr_u64(a, start, len);
#else
  return (a >> start) & ((u64(1) << len) - 1);
#endif
}


} // namespace bits

} // namespace dtl