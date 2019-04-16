#pragma once

namespace amsfilter {
namespace model {
//===----------------------------------------------------------------------===//
// Min/Max filter size in bits.
static u64 m_min = u64(16) * 1024 * 8;
static u64 m_max = u64(512ull) * 1024 * 1024 * 8;
//===----------------------------------------------------------------------===//
} // namespace model
} // namespace amsfilter
