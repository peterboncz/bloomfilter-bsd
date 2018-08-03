#include <iostream>

#include <dtl/dtl.hpp>
#include "dtl/filter/model/calibration.hpp"
#include "dtl/filter/model/calibration_data.hpp"

#include "skyline_matrix_builder.hpp"
#include "util.hpp"

using namespace dtl::filter::model;

$i32
main() {
  auto& data = dtl::filter::model::calibration_data::get_default_instance();
  dtl::filter::model::calibration calibration(data);

  std::cout << data << std::endl;

  calibration.calibrate_tuning_params();
  data.persist();

  calibration.calibrate_cache_sizes();
  data.persist();

  calibration.calibrate_filter_costs();
  data.persist();

  build_skyline_matrix(data);
  data.persist();

  std::cout << data << std::endl;

}
