//#define AMSFILTER_NO_LUT

#include <amsfilter/amsfilter.hpp>
#include <amsfilter/config.hpp>
#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/math.hpp>
#include "fpr.hpp"
#ifndef AMSFILTER_NO_LUT
#include "fpr_lut.hpp"
#endif
using namespace dtl::bloomfilter;

namespace amsfilter {
//===----------------------------------------------------------------------===//
$f64
fpr(u64 m,
    u64 n,
    u64 k,
    u64 B,  /* block size in bits */
    $u64 S, /* sector size in bits */
    u64 z,  /* the number of zones */
    u1 self_collisions = true) {

  f64 epsilon = 0.00001;
  u64 sector_cnt = B / S;

  if (sector_cnt == 1) {
    // Blocked Bloom filter
    return fpr_blocked(m, n, k, B, self_collisions, epsilon);
  }

  if (sector_cnt > 1 && z == sector_cnt) {
    // Sectorized Bloom filter.
    // The number of zones is equal to the number of sectors. Thus, in each
    // sector, we set k/z bits (=k/s).
    return fpr_blocked_sectorized(m, n, k, B, S, self_collisions, epsilon);
  }

  if (sector_cnt > 1 && z >= 1) {
    // Cache-sectorized Bloom filter.
    return fpr_zoned(m, n, k, B, S, z, self_collisions, epsilon);
  }
  throw std::invalid_argument("Invalid filter configuration.");
}
//===----------------------------------------------------------------------===//
f64
fpr(const Config& c, u64 m, u64 n) {
  if (!amsfilter::is_config_valid(c)) {
    std::stringstream err;
    err << "Invalid filter configuration: " << c;
    throw std::invalid_argument(err.str());
  }
  try {
    const auto block_size_bits = c.word_size * 8 * c.word_cnt_per_block;
    const auto sector_size_bits = block_size_bits / c.sector_cnt;
    u1 self_collisions = true;
    return fpr(m, n, c.k, block_size_bits, sector_size_bits, c.zone_cnt,
        self_collisions);
  } catch(...) {
    std::stringstream err;
    err << "Failed to compute the FPR for filter configuration: " << c;
    throw std::invalid_argument(err.str());
  }
}
//===----------------------------------------------------------------------===//
f64
fpr(const Config& c, f64 bits_per_element) {
  static constexpr u64 m = 16ull * 1024 * 1024 * 8;
  const auto n = static_cast<u64>(m / bits_per_element);
  return fpr(c, m, n);
}
//===----------------------------------------------------------------------===//
f64
fpr_fast(const Config& c, u64 m, u64 n) {
#ifdef AMSFILTER_NO_LUT
  return fpr(c, m, n);
#else
  if (!amsfilter::is_config_valid(c)) {
    std::stringstream err;
    err << "Invalid filter configuration: " << c;
    throw std::invalid_argument(err.str());
  }
  return lookup_fpr(c, m, n);
#endif
}
//===----------------------------------------------------------------------===//
f64
fpr_fast(const Config& c, f64 bits_per_element) {
#ifdef AMSFILTER_NO_LUT
  return fpr(c, bits_per_element);
#else
  if (!amsfilter::is_config_valid(c)) {
    std::stringstream err;
    err << "Invalid filter configuration: " << c;
    throw std::invalid_argument(err.str());
  }
  static constexpr u64 m = 16ull * 1024 * 1024 * 8;
  u64 n = static_cast<u64>(m / bits_per_element);
  return lookup_fpr(c, m, n);
#endif
}
//===----------------------------------------------------------------------===//
u32
optimal_k(const Config& c, u64 m, u64 n) {
  f64 bpe = (m * 1.0) / n;
  return optimal_k(c, bpe);
}
//===----------------------------------------------------------------------===//
u32
optimal_k(const Config& c, $f64 bits_per_element) {
  Config config = c;
  $u32 opt_k = 1;
  $f64 min_fpr = 1.0;
  for ($u32 k = 1; k <= 16; ++k) {
    config.k = k;
    if (!amsfilter::is_config_valid(config)) continue;
    f64 f = amsfilter::fpr(config, bits_per_element);
    if (f < min_fpr) {
      min_fpr = f;
      opt_k = k;
    }
  }
  return opt_k;
}
//===----------------------------------------------------------------------===//
u32
optimal_k_fast(const Config& c, u64 m, u64 n) {
#ifdef AMSFILTER_NO_LUT
  return optimal_k(c, m , n);
#else
  f64 bits_per_element = (m * 1.0) / n;
  return lookup_optimal_k(c, bits_per_element);
#endif
}
//===----------------------------------------------------------------------===//
u32
optimal_k_fast(const Config& c, $f64 bits_per_element) {
#ifdef AMSFILTER_NO_LUT
  return optimal_k(c, bits_per_element);
#else
  return lookup_optimal_k(c, bits_per_element);
#endif
}
//===----------------------------------------------------------------------===//

} // namespace amsfilter
