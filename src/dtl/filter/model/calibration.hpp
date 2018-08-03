#pragma once

#include <algorithm>
#include <string>
#include <cstring>
#include <fstream>
#include <map>
#include <type_traits>

#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp>
#include <dtl/filter/cuckoofilter/cuckoofilter_config.hpp>

#include "timing.hpp"
#include "tuning_params.hpp"
#include "calibration_data.hpp"

namespace dtl {
namespace filter {
namespace model {


//===----------------------------------------------------------------------===//
//
class calibration {

  dtl::filter::model::calibration_data& data_;

  void calibrate_bbf_costs();
  void calibrate_cf_costs();

public:

  calibration(dtl::filter::model::calibration_data& data) : data_(data) { };

  void calibrate_tuning_params();
  void calibrate_cache_sizes();
  void calibrate_filter_costs();

};
//===----------------------------------------------------------------------===//


} // namespace model
} // namespace filter
} // namespace dtl
