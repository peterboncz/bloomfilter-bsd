#include <dtl/dtl.hpp>
#include <amsfilter/internal/blocked_bloomfilter_resolve.hpp>
#include "config.hpp"

namespace amsfilter {
//===----------------------------------------------------------------------===//
/// Validator for configuration.
class Validator {

  $u1 result;

public:

  template<u32 w, u32 s, u32 z, u32 k, dtl::block_addressing a>
  void
  operator()(const Config& conf) {
    using resolved_type =
    typename amsfilter::internal::bbf_type<w, s, z, k, a>::type;
    if (resolved_type::word_cnt_per_block != conf.word_cnt_per_block
        || resolved_type::sector_cnt != conf.sector_cnt
        || resolved_type::zone_cnt != conf.zone_cnt
        || resolved_type::k != conf.k
        || resolved_type::addr_mode != conf.addr_mode) {
      // No valid.
      return;
    }
    // Valid.
    result = true;
  }

  /// Returns true if the configuration is valid, false otherwise.
  $u1
  test(const Config& c) {
    result = false;
    // Try to resolves the filter type.
    try {
      amsfilter::internal::get_instance(c, *this);
    }
    catch (...) {}
    return result;
  }
};
//===----------------------------------------------------------------------===//
$u1
is_config_valid(const Config& c) {
  Validator validator;
  $u1 is_valid = validator.test(c);
  return is_valid;
}
//===----------------------------------------------------------------------===//
} // namespace amsfilter
