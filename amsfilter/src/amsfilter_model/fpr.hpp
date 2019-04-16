#pragma once

#include <dtl/dtl.hpp>
#include <amsfilter/amsfilter.hpp>

namespace amsfilter {
//===----------------------------------------------------------------------===//
/// Returns the false-positive rate (FPR).  Note that this function can be
/// very compute intense (slow) for large values of n. Consider using the
/// significantly faster fpr_fast() function.
f64
fpr(const Config& c, u64 m, u64 n);
f64
fpr(const Config& c, f64 bits_per_element);
//===----------------------------------------------------------------------===//
/// Returns an approximation of the false-positive rate (FPR).  Note that
/// this function is significantly faster than fpr().
f64
fpr_fast(const Config& c, u64 m, u64 n);
f64
fpr_fast(const Config& c, $f64 bit_per_element);
//===----------------------------------------------------------------------===//
/// Returns the optimal k, wrt. to lowest FPR. Note that the k that is set
/// in the given config is ignored.
u32
optimal_k(const Config& c, u64 m, u64 n);
u32
optimal_k(const Config& c, $f64 bits_per_element);
//===----------------------------------------------------------------------===//
/// Returns the optimal k, wrt. to lowest FPR. Note that the k that is set
/// in the given config is ignored. As above, the 'fast' variant is
/// significantly faster but may return an approximation as it relies on a
/// pre-computed lookup table.
u32
optimal_k_fast(const Config& c, u64 m, u64 n);
u32
optimal_k_fast(const Config& c, $f64 bits_per_element);
//===----------------------------------------------------------------------===//
} // namespace amsfilter