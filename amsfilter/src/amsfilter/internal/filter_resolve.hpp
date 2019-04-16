#pragma once

#include <dtl/dtl.hpp>
#include <amsfilter/internal/blocked_bloomfilter_resolve.hpp>
#include <amsfilter/internal/cuckoofilter_resolve.hpp>
#include <amsfilter/config.hpp>

namespace amsfilter {
namespace internal {
//===----------------------------------------------------------------------===//
// Hard to explain
template<typename Fn>
static void
get_instance(const dtl::blocked_bloomfilter_config& conf, Fn& fn) {
  resolve::blocked_bloomfilter::get_instance(conf, fn);
}
//===----------------------------------------------------------------------===//
} // namespace internal
} // namespace amsfilter
