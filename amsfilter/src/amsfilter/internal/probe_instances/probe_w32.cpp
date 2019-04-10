#include <amsfilter/common.hpp>
#include <amsfilter/internal/blocked_bloomfilter_template.hpp>
#include "macro.inc"

GENERATE(32, 1, 1,  1);
GENERATE(32, 1, 1,  2);
GENERATE(32, 1, 1,  3);
GENERATE(32, 1, 1,  4);
GENERATE(32, 1, 1,  5);
GENERATE(32, 1, 1,  6);
GENERATE(32, 1, 1,  7);
GENERATE(32, 1, 1,  8);
GENERATE(32, 1, 1,  9);
GENERATE(32, 1, 1, 10);
GENERATE(32, 1, 1, 11);
GENERATE(32, 1, 1, 12);
GENERATE(32, 1, 1, 13);
GENERATE(32, 1, 1, 14);
GENERATE(32, 1, 1, 15);
GENERATE(32, 1, 1, 16);
GENERATE(32,32, 2,  2);
GENERATE(32,32, 2,  4);
GENERATE(32,32, 2,  6);
GENERATE(32,32, 2,  8);
GENERATE(32,32, 2, 10);
GENERATE(32,32, 2, 12);
GENERATE(32,32, 2, 14);
GENERATE(32,32, 2, 16);
GENERATE(32,32, 4,  4);
GENERATE(32,32, 4,  8);
GENERATE(32,32, 4, 12);
GENERATE(32,32, 4, 16);
GENERATE(32,32, 8,  8);
GENERATE(32,32, 8, 16);