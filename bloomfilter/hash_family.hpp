#pragma once

#include <dtl/dtl.hpp>
#include <dtl/simd.hpp>

namespace dtl {
namespace hash {

namespace stat {

template<
    typename key_t,
    $u32 hash_fn_no
>
struct mul32 { };

template<typename key_t>
struct mul32<key_t, 0> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 596572387u; } // Peter 1
};

template<typename key_t>
struct mul32<key_t, 1> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 370248451u; } // Peter 2
};

template<typename key_t>
struct mul32<key_t, 2> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 2654435769u; } // Knuth 1
};

template<typename key_t>
struct mul32<key_t, 3> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 1799596469u; } // Knuth 2
};

template<typename key_t>
struct mul32<key_t, 4> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 0x9E3779B1u; } // https://lowrey.me/exploring-knuths-multiplicative-hash-2/
};

template<typename key_t>
struct mul32<key_t, 5> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 2284105051u; } // Impala 3
};

template<typename key_t>
struct mul32<key_t, 6> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 1203114875u; } // Impala 1 (odd, not prime)
};

template<typename key_t>
struct mul32<key_t, 7> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 1150766481u; } // Impala 2 (odd, not prime)
};

template<typename key_t>
struct mul32<key_t, 8> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 2729912477u; } // Impala 4 (odd, not prime)
};

template<typename key_t>
struct mul32<key_t, 9> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 1884591559u; } // Impala 5 (odd, not prime)
};

template<typename key_t>
struct mul32<key_t, 10> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 770785867u; } // Impala 6 (odd, not prime)
};

template<typename key_t>
struct mul32<key_t, 11> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 2667333959u; } // Impala 7 (odd, not prime)
};

template<typename key_t>
struct mul32<key_t, 12> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 1550580529u; } // Impala 8 (odd, not prime)
};

} // namespace stat(ic)


namespace dyn {

struct mul32 {

  __forceinline__ __host__ __device__
  static u32 hash(u32& key, u32 hash_no) {
    static constexpr u32 primes[13] {
        596572387u,   // Peter 1
        370248451u,   // Peter 2
        2654435769u,  // Knuth 1
        1799596469u,  // Knuth 2
        0x9E3779B1u,  // https://lowrey.me/exploring-knuths-multiplicative-hash-2/
        2284105051u,  // Impala 3
        1203114875u,  // Impala 1 (odd, not prime)
        1150766481u,  // Impala 2 (odd, not prime)
        2729912477u,  // Impala 4 (odd, not prime)
        1884591559u,  // Impala 5 (odd, not prime)
        770785867u,   // Impala 6 (odd, not prime)
        2667333959u,  // Impala 7 (odd, not prime)
        1550580529u,  // Impala 8 (odd, not prime)
    };
    if (hash_no > 13) {
      std::cerr << "hash_no out of bounds: " << hash_no << std::endl;
      throw "BAM";
    }
    return key * primes[hash_no];
  }

  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  static
  dtl::vec<uint32_t, dtl::vector_length<Tv>::value>
  hash(const Tv& keys,
       const uint32_t hash_no) {
    static constexpr u32 primes[13] {
        596572387u,   // Peter 1
        370248451u,   // Peter 2
        2654435769u,  // Knuth 1
        1799596469u,  // Knuth 2
        0x9E3779B1u,  // https://lowrey.me/exploring-knuths-multiplicative-hash-2/
        1203114875u,  // Impala 1 (odd, not prime)
        1150766481u,  // Impala 2 (odd, not prime)
        2284105051u,  // Impala 3
        2729912477u,  // Impala 4 (odd, not prime)
        1884591559u,  // Impala 5 (odd, not prime)
        770785867u,   // Impala 6 (odd, not prime)
        2667333959u,  // Impala 7 (odd, not prime)
        1550580529u,  // Impala 8 (odd, not prime)
    };
    return keys * primes[hash_no];
  };

};

} // namespace dyn


} // namespace hash
} // namespace dtl