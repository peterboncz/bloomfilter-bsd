#pragma once

#include <dtl/dtl.hpp>
#include <dtl/filter/model/calibration_data.hpp>

namespace dtl {
namespace filter {
namespace model {


//===----------------------------------------------------------------------===//

void
build_skyline_matrix(dtl::filter::model::calibration_data&);

//===----------------------------------------------------------------------===//


} // namespace model
} // namespace filter
} // namespace dtl
