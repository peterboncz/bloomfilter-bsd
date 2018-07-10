#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_logic.hpp>

#include "zoned_blocked_bloomfilter_logic_instance.inc"

namespace dtl {

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

} // namespace dtl