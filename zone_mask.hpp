#pragma once

#include "adept.hpp"
#include "math.hpp"
#include <bitset>

namespace dtl {

  template<u64 N>
  class zone_mask {
    static_assert(is_power_of_two(N), "Template parameter 'N' must be a power of two.");

  public:

    template<u64 M>
    static std::bitset<M> compress(const std::bitset<N>& bitmask) {
      static_assert(is_power_of_two(M), "Template parameter 'M' must be a power of two.");
      u64 zone_size = N / M;
      u64 zone_cnt = N / zone_size;

      std::bitset<N> tmp_mask;
      for ($u64 i = 0; i < zone_size; i++) {
        tmp_mask |= bitmask >> i;
      }

      std::bitset<M> zone_mask;
      for ($u64 i = 0; i < zone_cnt; i++) {
        zone_mask[i] = tmp_mask[i * zone_size];
      }
      return zone_mask;
    }

    template<u64 M>
    static std::bitset<N> decode(const std::bitset<M>& compressed_bitmask) {
      static_assert(is_power_of_two(M), "Template parameter 'M' must be a power of two.");
      u64 zone_size = N / M;
      u64 zone_cnt = N / zone_size;
      std::bitset<N> bitmask;
      for ($u64 i = 0; i < zone_cnt; i++) {
        if (!compressed_bitmask[i]) continue;
        for ($u64 j = 0; j < zone_size; j++) {
          bitmask[i * zone_size + j] = true;
        }
      }
      return bitmask;
    }

    template<u64 V>
    struct it {
    private:
      friend zone_mask;
      zone_mask& zm;
      $u64 i;
    public:


    };

  };

}
