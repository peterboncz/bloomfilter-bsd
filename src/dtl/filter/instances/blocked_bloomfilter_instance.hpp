#pragma once

#include <dtl/dtl.hpp>
#include <dtl/simd.hpp>
#include <dtl/bloomfilter/hash_family.hpp>
#include <dtl/bloomfilter/blocked_bloomfilter.hpp>
#include "blocked_bloomfilter_logic_u64_instance.hpp"

namespace dtl {

//===----------------------------------------------------------------------===//
// Externalize templates to parallelize builds.
//===----------------------------------------------------------------------===//
extern template struct blocked_bloomfilter<$u32>;

} // namespace dtl
