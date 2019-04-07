#pragma once

#include <memory>

#include <dtl/dtl.hpp>

#include <amsfilter/internal/buffer.hpp>

namespace amsfilter {
//===----------------------------------------------------------------------===//
using key_t = $u32;
using word_t = $u32;
using filter_data_t = internal::buffer<word_t>;
using shared_filter_data_t = std::shared_ptr<filter_data_t>;
extern std::string simd_arch;
//===----------------------------------------------------------------------===//
} // namespace amsfilter
