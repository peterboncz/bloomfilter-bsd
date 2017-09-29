#pragma once

namespace dtl {


namespace {


// TODO remove macros
// inspired from
// http://www-graphics.stanford.edu/~seander/bithacks.html#ZeroInWord
// adapted from TODO
#define haszero2_u32(x) (((x)-0x55555555u) & (~(x)) & 0xAAAAAAAAu)
#define hasvalue2_u32(x, n) (haszero2_u32((x) ^ (0x55555555u * (n))))

#define haszero2_u64(x) (((x)-0x5555555555555555ull) & (~(x)) & 0xAAAAAAAAAAAAAAAAull)
#define hasvalue2_u64(x, n) (haszero2_u64((x) ^ (0x5555555555555555ull * (n))))

#define haszero3_u32(x) (((x)-0b01001001001001001001001001001001u) & (~(x)) & 0b00100100100100100100100100100100u)
#define hasvalue3_u32(x, n) (haszero3_u32((x) ^ (0b01001001001001001001001001001001u * (n))))

#define haszero3_u64(x) (((x)-0b1001001001001001001001001001001001001001001001001001001001001001ull) & (~(x)) & 0b0100100100100100100100100100100100100100100100100100100100100100ull)
#define hasvalue3_u64(x, n) (haszero3_u64((x) ^ (0b1001001001001001001001001001001001001001001001001001001001001001ull * (n))))

#define haszero4_u32(x) (((x)-0x11111111u) & (~(x)) & 0x88888888u)
#define hasvalue4_u32(x, n) (haszero4_u32((x) ^ (0x11111111u * (n))))

#define haszero4_u64(x) (((x)-0x1111111111111111ull) & (~(x)) & 0x8888888888888888ull)
#define hasvalue4_u64(x, n) (haszero4_u64((x) ^ (0x1111111111111111ull * (n))))

#define haszero5_u32(x) (((x)-0b01000010000100001000010000100001) & (~(x)) & 0b00100001000010000100001000010000)
#define hasvalue5_u32(x, n) (haszero5_u32((x) ^ (0b01000010000100001000010000100001 * (n))))

#define haszero5_u64(x) (((x)-0b0001000010000100001000010000100001000010000100001000010000100001) & (~(x)) & 0b0000100001000010000100001000010000100001000010000100001000010000)
#define hasvalue5_u64(x, n) (haszero5_u64((x) ^ (0b0001000010000100001000010000100001000010000100001000010000100001 * (n))))

#define haszero6_u32(x) (((x)-0b01000001000001000001000001000001) & (~(x)) & 0b00100000100000100000100000100000)
#define hasvalue6_u32(x, n) (haszero6_u32((x) ^ (0b01000001000001000001000001000001 * (n))))

#define haszero6_u64(x) (((x)-0b0001000001000001000001000001000001000001000001000001000001000001) & (~(x)) & 0b0000100000100000100000100000100000100000100000100000100000100000)
#define hasvalue6_u64(x, n) (haszero6_u64((x) ^ (0b0001000001000001000001000001000001000001000001000001000001000001 * (n))))

#define haszero7_u32(x) (((x)-0b00010000001000000100000010000001) & (~(x)) & 0b00001000000100000010000001000000)
#define hasvalue7_u32(x, n) (haszero7_u32((x) ^ (0b00010000001000000100000010000001 * (n))))

#define haszero7_u64(x) (((x)-0b1000000100000010000001000000100000010000001000000100000010000001) & (~(x)) & 0b0100000010000001000000100000010000001000000100000010000001000000)
#define hasvalue7_u64(x, n) (haszero7_u64((x) ^ (0b1000000100000010000001000000100000010000001000000100000010000001 * (n))))

#define haszero8_u32(x) (((x)-0x01010101u) & (~(x)) & 0x80808080u)
#define hasvalue8_u32(x, n) (haszero8_u32((x) ^ (0x01010101u * (n))))

#define haszero8_u64(x) (((x)-0x0101010101010101ull) & (~(x)) & 0x8080808080808080ull)
#define hasvalue8_u64(x, n) (haszero8_u64((x) ^ (0x0101010101010101ull * (n))))

#define haszero10_u32(x) (((x)-0b01000000000100000000010000000001) & (~(x)) & 0b00100000000010000000001000000000)
#define hasvalue10_u32(x, n) (haszero10_u32((x) ^ (0b01000000000100000000010000000001 * (n))))

#define haszero12_u32(x) (((x)-0x001001001001ull) & (~(x)) & 0x800800800800ull)
#define hasvalue12_u32(x, n) (haszero12_u32((x) ^ (0x001001001001ULL * (n))))

#define haszero16_u32(x) \
  (((x)-0x0001000100010001ULL) & (~(x)) & 0x8000800080008000ULL)
#define hasvalue16_u32(x, n) (haszero16_u32((x) ^ (0x0001000100010001ULL * (n))))


template<typename T, uint32_t bits_per_value>
struct packed_value { };

template<>
struct packed_value<uint32_t, 2> {
  __forceinline__ static bool
  contains(const uint32_t packed_value, const uint32_t search_value) { return hasvalue2_u32(packed_value, search_value); }
};

template<>
struct packed_value<uint64_t, 2> {
  __forceinline__ static bool
  contains(const uint32_t packed_value, const uint32_t search_value) { return hasvalue2_u64(packed_value, search_value); }
};

template<>
struct packed_value<uint32_t, 3> {
  __forceinline__ static bool
  contains(const uint32_t packed_value, const uint32_t search_value) { return hasvalue3_u32(packed_value, search_value); }
};

template<>
struct packed_value<uint64_t, 3> {
  __forceinline__ static bool
  contains(const uint64_t packed_value, const uint32_t search_value) { return hasvalue3_u64(packed_value, search_value); }
};

template<>
struct packed_value<uint32_t, 4> {
  __forceinline__ static bool
  contains(const uint32_t packed_value, const uint32_t search_value) { return hasvalue4_u32(packed_value, search_value); }
};

template<>
struct packed_value<uint64_t, 4> {
  __forceinline__ static bool
  contains(const uint64_t packed_value, const uint32_t search_value) { return hasvalue4_u64(packed_value, search_value); }
};

template<>
struct packed_value<uint32_t, 5> {
  __forceinline__ static bool
  contains(const uint32_t packed_value, const uint32_t search_value) { return hasvalue5_u32(packed_value, search_value); }
};

template<>
struct packed_value<uint64_t, 5> {
  __forceinline__ static bool
  contains(const uint64_t packed_value, const uint32_t search_value) { return hasvalue5_u64(packed_value, search_value); }
};

template<>
struct packed_value<uint32_t, 6> {
  __forceinline__ static bool
  contains(const uint32_t packed_value, const uint32_t search_value) { return hasvalue6_u32(packed_value, search_value); }
};

template<>
struct packed_value<uint64_t, 6> {
  __forceinline__ static bool
  contains(const uint64_t packed_value, const uint32_t search_value) { return hasvalue6_u64(packed_value, search_value); }
};

template<>
struct packed_value<uint32_t, 7> {
  __forceinline__ static bool
  contains(const uint32_t packed_value, const uint32_t search_value) { return hasvalue7_u32(packed_value, search_value); }
};

template<>
struct packed_value<uint64_t, 7> {
  __forceinline__ static bool
  contains(const uint64_t packed_value, const uint32_t search_value) { return hasvalue7_u64(packed_value, search_value); }
};

template<>
struct packed_value<uint32_t, 8> {
  __forceinline__ static bool
  contains(const uint32_t packed_value, const uint32_t search_value) { return hasvalue8_u32(packed_value, search_value); }
};

template<>
struct packed_value<uint64_t, 8> {
  __forceinline__ static bool
  contains(const uint64_t packed_value, const uint32_t search_value) { return hasvalue8_u64(packed_value, search_value); }
};



template<>
struct packed_value<uint32_t, 10> {
  __forceinline__ static bool
  contains(const uint32_t packed_value, const uint32_t search_value) { return hasvalue10_u32(packed_value, search_value); }
};
template<>
struct packed_value<uint32_t, 12> {
  __forceinline__ static bool
  contains(const uint32_t packed_value, const uint32_t search_value) { return hasvalue12_u32(packed_value, search_value); }
};
template<>
struct packed_value<uint32_t, 16> {
  __forceinline__ static bool
  contains(const uint32_t packed_value, const uint32_t search_value) { return hasvalue16_u32(packed_value, search_value); }
};


} // anonymous namespace


} // namespace dtl
