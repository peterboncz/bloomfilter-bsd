#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_logic.hpp>

#include "blocked_bloomfilter_logic_instance.inc"

namespace dtl {

GENERATE($u32, 16, 1,  1)
GENERATE($u32, 16, 1,  2)
GENERATE($u32, 16, 1,  3)
GENERATE($u32, 16, 1,  4)
GENERATE($u32, 16, 1,  5)
GENERATE($u32, 16, 1,  6)
GENERATE($u32, 16, 1,  7)
GENERATE($u32, 16, 1,  8)
GENERATE($u32, 16, 1,  9)
GENERATE($u32, 16, 1, 10)
GENERATE($u32, 16, 1, 11)
GENERATE($u32, 16, 1, 12)
GENERATE($u32, 16, 1, 13)
GENERATE($u32, 16, 1, 14)
GENERATE($u32, 16, 1, 15)
GENERATE($u32, 16, 1, 16)
GENERATE($u32, 16, 2,  2)
GENERATE($u32, 16, 2,  4)
GENERATE($u32, 16, 2,  6)
GENERATE($u32, 16, 2,  8)
GENERATE($u32, 16, 2, 10)
GENERATE($u32, 16, 2, 12)
GENERATE($u32, 16, 2, 14)
GENERATE($u32, 16, 2, 16)
GENERATE($u32, 16, 4,  4)
GENERATE($u32, 16, 4,  8)
GENERATE($u32, 16, 4, 12)
GENERATE($u32, 16, 4, 16)
GENERATE($u32, 16, 8,  8)
GENERATE($u32, 16, 8, 16)
GENERATE($u32, 16,16, 16)

} // namespace dtl