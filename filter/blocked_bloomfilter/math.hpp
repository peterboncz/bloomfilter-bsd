#pragma once

#include <random>
#include <math.h>

#include <dtl/dtl.hpp>

#include <boost/math/distributions/poisson.hpp>

//TODO merge into fpr.hpp

namespace dtl {
namespace bloomfilter {

/// Computes an approximation of the false positive probability for standard Bloom filter.
/// Assuming independence for the probabilities of each bit being set.
static f64
fpr(u64 m,
    u64 n,
    f64 k) {
  return std::pow(1.0 - std::pow(1.0 - (1.0 / m), k * n), k);
}


/// Computes an approximation of the false positive probability for Blocked Bloom filter as
/// defined by Putze et al.
/// Note: The formula of Putze does not take self collisions into account and thus leads
///       to a significant error for small block sizes (i.e., register blocks).
///       Therefore, for small block sizes, the 'self collision' flag should be set to true.
static f64
fpr_blocked(u64 m,
            u64 n,
            f64 k,
            u64 B, /* block size in bits */
            u1 self_collisions = false,
            f64 epsilon = 0.000001) {
  $f64 f = 0;
  $f64 c = (m * 1.0) / n;
  $f64 lambda = B / c;
  boost::math::poisson_distribution<> poisson(lambda);

  $f64 k_act = k;
  if (self_collisions) {
    k_act = B * (1.0 - std::pow(1.0-1.0/B, k_act));
  }

  $f64 d_sum = 0.0;
  $u64 i = 0;
  while ((d_sum + epsilon) < 1.0) {
    auto d = boost::math::pdf(poisson, i);
    d_sum += d;
    f += d * fpr(B, i, k_act);
    i++;
  }
  return f;
}


/// Computes an approximation of the false positive probability for
/// Sectorized Blocked Bloom filter.
static f64
fpr_blocked_sectorized(u64 m,
                       u64 n,
                       f64 k,
                       u64 B, /* block size in bits */
                       u64 S, /* sector size in bits */
                       u1 self_collisions = false,
                       f64 epsilon = 0.000001) {
  $f64 f = 0;
  $f64 c = (m * 1.0) / n;
  $f64 lambda = (B * 1.0) / c;
  $f64 s = (B * 1.0) / S;
  boost::math::poisson_distribution<> poisson(lambda);

  $f64 d_sum = 0.0;
  $u64 i = 0;
  $f64 k_per_s = (k * 1.0)/s;
  if (self_collisions) {
    k_per_s = S * (1.0 - std::pow(1.0-1.0/S, k_per_s));
  }
  while ((d_sum + epsilon) < 1.0) {
    auto d = boost::math::pdf(poisson, i);
    d_sum += d;
    f += d * std::pow(fpr(S, i, k_per_s), s);
    i++;
  }
  return f;
}


static __forceinline__ f64
p_load(f64 v, u64 i) {
  f64 lambda = v;
  boost::math::poisson_distribution<> poisson(lambda);
  return boost::math::pdf(poisson, i);
}


static __forceinline__ f64
p_cache(u64 s, u64 S, u64 B, f64 k, u64 i) {
//  boost::math::binomial_distribution binomial();

  f64 k_per_s = (k * 1.0)/s;
  $f64 sum = 0.0;
  for (std::size_t j = 1; j <= i; j++) {
    auto ev = (i*s*S*1.0)/B;
    auto p_l = p_load(ev, j);
    auto f_mini = fpr(S, j, k_per_s);
    sum += p_l * f_mini;
  }
  auto r = std::pow(sum, s);
  return r;
}

/// Computes an approximation of the false positive probability for
/// Sectorized Blocked Bloom filter.
static f64
fpr_zoned(u64 m,
          u64 n,
          f64 k,
          u64 B, /* block size in bits */
          u64 S, /* sector size in bits */
          f64 z, /* the number of zones */
          u1 self_collisions = false,
          f64 epsilon = 0.000001) {
  $f64 f = 0;
  $f64 c = (m * 1.0) / n;
  $f64 lambda = (B * 1.0) / c;
  $f64 s = (B * 1.0) / S;
  boost::math::poisson_distribution<> poisson(lambda);

  $f64 d_sum = 0.0;
  $u64 i = 0;
  $f64 k_per_s = (k * 1.0) / s;
  if (self_collisions) {
    k_per_s = S * (1.0 - std::pow(1.0 - 1.0 / S, k_per_s));
  }
  while ((d_sum + epsilon) < 1.0) {
    auto d = boost::math::pdf(poisson, i);
    d_sum += d;
    f += d * p_cache(z, S, B, k, i);
    i++;
  }
  return f;
}


static __forceinline__ f64
_p_cache(u64 s, u64 S, u64 B, f64 k, u64 i) {
  f64 k_per_s = (k * 1.0)/s;
  $f64 sum = 0.0;
  for (std::size_t j = 1; j <= i; j++) {
    auto ev = (i*s*S*1.0)/B;
    auto p_l = p_load(ev, j);
    auto f_mini = fpr(S, j, k_per_s);
    sum += p_l * f_mini;
  }
  auto r = std::pow(sum, s);
  return r;
}


static f64
fpr_blocked_sectorized_zoned(u64 m,
                             u64 n,
                             u64 k,
                             u64 z, /* the number of zones */
                             u64 B, /* block size in bits */
                             u64 S, /* sector size in bits */
                             u1 self_collisions = false,
                             f64 epsilon = 0.000001) {
  $f64 f = 0;
//  f64 c = (m * 1.0) / n;
//  f64 v = (B * 1.0) / c; // aka 'Poisson lambda'
  f64 s = z;

  $f64 d_sum = 0.0;
  $u64 i = 0;
  $f64 k_per_s = (k * 1.0)/s;
  if (self_collisions) {
    k_per_s = S * (1.0 - std::pow(1.0-1.0/S, k_per_s));
  }
  while ((d_sum + epsilon) < 1.0) {
    auto ev = (n*B*1.0)/m;
    auto d = p_load(ev, i);
    d_sum += d;
    f += d * p_cache(z, S, B, k, i);
    i++;
  }
  return f;
}

} // namespace filter

namespace cuckoofilter {

//// TODO
static f64
fpr(u64 associativity,
    u64 tag_bitlength,
    f64 load_factor) {
//  return (2.0 /*k=2*/ * associativity * load_factor) / (std::pow(2, tag_bitlength) - 1); // no duplicates
  return 1 - std::pow(1.0 - 1 / (std::pow(2.0, tag_bitlength) - 1), 2.0 * associativity * load_factor); // counting - with duplicates
//  return 1 - std::pow(1.0 - 1 / (std::pow(2.0, tag_bitlength)), 2.0 * associativity * load_factor); // counting - with duplicates
}


} // namespace cuckoofilter
} // namespace dtl



//f64
//fpr_k_partitioned(u64 m,
//                  u64 n,
//                  u64 k) {
//  f64 c = (m * 1.0) / n;
//  return fpr((m * 1.0)/k, n, 1);
//}
//f64
//fpr_k_partitioned(u64 m,
//                  u64 n,
//                  u64 k) {
//  f64 c = (m * 1.0) / n;
//  return std::pow(1.0 - std::exp(-(k*1.0) / c), k);
//}

//f64
//fpr_zoned(u64 m,
//          u64 n,
//          u64 k,
//          u64 B, /* block size in bits */
//          u64 z, /* number of zones */
//          f64 epsilon = 0.000001) {
//  return std::pow(fpr_blocked(m/z,n,k/z,64), z);
//}

//f64
//fpr_blocked_k_partitioned(u64 m,
//                          u64 n,
//                          u64 k,
//                          u64 B, /* block size in bits */
//                          f64 epsilon = 0.000001) {
//  $f64 f = 0;
//  $f64 c = (m * 1.0) / n;
//  $f64 lambda = (B * 1.0) / c;
//  boost::math::poisson_distribution<> poisson(lambda);
//
//  std::random_device rd;
//  std::mt19937 gen(rd());
//
//  $f64 d_sum = 0.0;
//  $u64 i = 0;
//  while ((d_sum + epsilon) < 1.0) {
//    auto d = boost::math::pdf(poisson, i);
//    d_sum += d;
//    f += d * fpr_k_partitioned(B, i, k);
//    i++;
//  }
//  return f;
//}

//f64
//fpr_blocked_sectorized(u64 m,
//                       u64 n,
//                       u64 k,
//                       u64 B, /* block size in bits */
//                       u64 S, /* sector size in bits */
//                       f64 epsilon = 0.000001) {
//  $f64 f = 0;
//  $f64 c = (m * 1.0) / n;
//  $f64 lambda = (B * 1.0) / c;
//  $f64 s = (B * 1.0) / S;
//  boost::math::poisson_distribution<> poisson(lambda);
//
//  std::random_device rd;
//  std::mt19937 gen(rd());
//
//  $f64 d_sum = 0.0;
//  $u64 i = 0;
//  while ((d_sum + epsilon) < 1.0) {
//    auto d = boost::math::pdf(poisson, i);
//    d_sum += d;
//    f += d * std::pow(fpr(S, i, (k * 1.0)/s), s);
//    i++;
//  }
//  return f;
//}
