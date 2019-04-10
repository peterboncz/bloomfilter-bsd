#pragma once

#include "macro.inc"

//===----------------------------------------------------------------------===//
// Externalize templates to parallelize builds.
//===----------------------------------------------------------------------===//
namespace amsfilter {
namespace cuda {
namespace internal {
//===----------------------------------------------------------------------===//
#include "kernel_w1.cuh"
#include "kernel_w2.cuh"
#include "kernel_w4.cuh"
#include "kernel_w8.cuh"
#include "kernel_w16.cuh"
#include "kernel_w32.cuh"
//===----------------------------------------------------------------------===//
} // namespace internal
} // namespace cuda
} // namespace amsfilter
