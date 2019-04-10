#include <amsfilter/common.hpp>
#include <amsfilter/cuda/internal/kernel.cuh>
#include <amsfilter/internal/blocked_bloomfilter_template.hpp>
#include "macro.inc"

//===----------------------------------------------------------------------===//
// Instantiate templates to parallelize builds.
//===----------------------------------------------------------------------===//
namespace amsfilter {
namespace cuda {
namespace internal {
//===----------------------------------------------------------------------===//
GENERATE( 4, 1, 1,  1);
GENERATE( 4, 1, 1,  2);
GENERATE( 4, 1, 1,  3);
GENERATE( 4, 1, 1,  4);
GENERATE( 4, 1, 1,  5);
GENERATE( 4, 1, 1,  6);
GENERATE( 4, 1, 1,  7);
GENERATE( 4, 1, 1,  8);
GENERATE( 4, 1, 1,  9);
GENERATE( 4, 1, 1, 10);
GENERATE( 4, 1, 1, 11);
GENERATE( 4, 1, 1, 12);
GENERATE( 4, 1, 1, 13);
GENERATE( 4, 1, 1, 14);
GENERATE( 4, 1, 1, 15);
GENERATE( 4, 1, 1, 16);

GENERATE( 4, 4, 4,  4);
GENERATE( 4, 4, 4,  8);
GENERATE( 4, 4, 4, 12);
GENERATE( 4, 4, 4, 16);

GENERATE( 4, 4, 2,  2);
GENERATE( 4, 4, 2,  4);
GENERATE( 4, 4, 2,  6);
GENERATE( 4, 4, 2,  8);
GENERATE( 4, 4, 2, 10);
GENERATE( 4, 4, 2, 12);
GENERATE( 4, 4, 2, 14);
GENERATE( 4, 4, 2, 16);
//===----------------------------------------------------------------------===//
} // namespace internal
} // namespace cuda
} // namespace amsfilter
