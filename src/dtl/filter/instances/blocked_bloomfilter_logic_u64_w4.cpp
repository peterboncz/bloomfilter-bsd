#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_logic.hpp>

#include "blocked_bloomfilter_logic_instance.inc"

namespace dtl {

GENERATE($u64, 4, 1,  1)
GENERATE($u64, 4, 1,  2)
GENERATE($u64, 4, 1,  3)
GENERATE($u64, 4, 1,  4)
GENERATE($u64, 4, 1,  5)
GENERATE($u64, 4, 1,  6)
GENERATE($u64, 4, 1,  7)
GENERATE($u64, 4, 1,  8)
GENERATE($u64, 4, 1,  9)
GENERATE($u64, 4, 1, 10)
GENERATE($u64, 4, 1, 11)
GENERATE($u64, 4, 1, 12)
GENERATE($u64, 4, 1, 13)
GENERATE($u64, 4, 1, 14)
GENERATE($u64, 4, 1, 15)
GENERATE($u64, 4, 1, 16)
GENERATE($u64, 4, 2,  2)
GENERATE($u64, 4, 2,  4)
GENERATE($u64, 4, 2,  6)
GENERATE($u64, 4, 2,  8)
GENERATE($u64, 4, 2, 10)
GENERATE($u64, 4, 2, 12)
GENERATE($u64, 4, 2, 14)
GENERATE($u64, 4, 2, 16)
GENERATE($u64, 4, 4,  4)
GENERATE($u64, 4, 4,  8)
GENERATE($u64, 4, 4, 12)
GENERATE($u64, 4, 4, 16)
GENERATE($u64, 4, 8,  8)
GENERATE($u64, 4, 8, 16)
GENERATE($u64, 4,16, 16)

} // namespace dtl