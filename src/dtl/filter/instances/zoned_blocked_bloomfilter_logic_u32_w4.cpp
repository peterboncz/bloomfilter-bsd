#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_logic.hpp>

#include "zoned_blocked_bloomfilter_logic_instance.inc"

namespace dtl {

GENERATE($u32, 4, 2,  2)
GENERATE($u32, 4, 2,  4)
GENERATE($u32, 4, 2,  6)
GENERATE($u32, 4, 2,  8)
GENERATE($u32, 4, 2, 10)
GENERATE($u32, 4, 2, 12)
GENERATE($u32, 4, 2, 14)
GENERATE($u32, 4, 2, 16)

} // namespace dtl