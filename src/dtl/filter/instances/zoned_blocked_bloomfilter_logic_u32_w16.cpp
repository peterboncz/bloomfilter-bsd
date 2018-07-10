#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_logic.hpp>

#include "zoned_blocked_bloomfilter_logic_instance.inc"

namespace dtl {

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

} // namespace dtl