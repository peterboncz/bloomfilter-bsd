#include "common.hpp"

std::string amsfilter::simd_arch =
#ifdef __AVX512F__
    "AVX-512";
#elif __AVX2__
    "AVX2";
#else
    "None";
#endif
