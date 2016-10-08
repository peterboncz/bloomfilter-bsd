#pragma once

#include "adept.hpp"

namespace dtl { namespace hash {

template<typename T>
struct xorshift_64 {
  static constexpr T hash(const T& key) {
    T h = key;
    h ^= h << 13;
    h ^= h >> 7;
    h ^= h << 17;
    return h;
  }
};


template<typename T>
struct murmur1_32 {
  static constexpr T hash(const T& key) {
    const T m = 0xc6a4a793u;
    const T hi = 0x4e774912u ^(4 * m);
    T h = hi;
    h += key;
    h *= m;
    h ^= h >> 16;
    h *= m;
    h ^= h >> 10;
    h *= m;
    h ^= h >> 17;
    return h;
  }
};


template<typename T>
struct murmur64a_64 {
  static constexpr T hash(const T& key) {
    const T m = 0xc6a4a7935bd1e995ull;
    const T r = 47u;
    const T hi = 0x8445d61a4e774912ull ^ (8 * m);
    T h = hi;
    T k = key;
    k *= m;
    k ^= k >> r;
    k *= m;
    h ^= k;
    h *= m;
    h ^= h >> r;
    h *= m;
    h ^= h >> r;
    return h;
  }
};

}}
