#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_logic.hpp>

#include "blocked_bloomfilter_logic_instance.inc"

namespace dtl {

GENERATE($u64, 8, 1,  1)
GENERATE($u64, 8, 1,  2)
GENERATE($u64, 8, 1,  3)
GENERATE($u64, 8, 1,  4)
GENERATE($u64, 8, 1,  5)
GENERATE($u64, 8, 1,  6)
GENERATE($u64, 8, 1,  7)
GENERATE($u64, 8, 1,  8)
GENERATE($u64, 8, 1,  9)
GENERATE($u64, 8, 1, 10)
GENERATE($u64, 8, 1, 11)
GENERATE($u64, 8, 1, 12)
GENERATE($u64, 8, 1, 13)
GENERATE($u64, 8, 1, 14)
GENERATE($u64, 8, 1, 15)
GENERATE($u64, 8, 1, 16)
GENERATE($u64, 8, 2,  2)
GENERATE($u64, 8, 2,  4)
GENERATE($u64, 8, 2,  6)
GENERATE($u64, 8, 2,  8)
GENERATE($u64, 8, 2, 10)
GENERATE($u64, 8, 2, 12)
GENERATE($u64, 8, 2, 14)
GENERATE($u64, 8, 2, 16)
GENERATE($u64, 8, 4,  4)
GENERATE($u64, 8, 4,  8)
GENERATE($u64, 8, 4, 12)
GENERATE($u64, 8, 4, 16)
GENERATE($u64, 8, 8,  8)
GENERATE($u64, 8, 8, 16)
GENERATE($u64, 8,16, 16)

} // namespace dtl