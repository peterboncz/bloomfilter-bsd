#pragma once

#include <algorithm>
#include <string>
#include <cstring>
#include <fstream>
#include <map>
#include <type_traits>

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp>
#include <dtl/filter/cuckoofilter/cuckoofilter_config.hpp>

#include "timing.hpp"

namespace dtl {
namespace filter {
namespace model {


//===----------------------------------------------------------------------===//
// Maintains the hardware related calibration data.
class calibration_data {

  // Typedefs
  using bbf_config_t = dtl::blocked_bloomfilter_config;
  using cf_config_t = dtl::cuckoofilter::config;
  using timings_t = std::vector<timing>;

  // Constants
  static constexpr u64 file_signature_size = 8; // byte
  const char file_signature[file_signature_size] = {'\211','F','I','L','\r','\n','\032','\n'}; // inspired by the PNG file signature
  static constexpr u64 file_version = 1;

  /// The file name where to store the calibration data.
  const std::string filename_;
  /// The calibrated cache sizes.
  std::vector<$u64> cache_sizes_;
  /// The benchmarked filter sizes. one for each memory level.
  std::vector<$u64> filter_sizes_;
  /// The delta t_lookup values for Bloom filters.
  std::map<bbf_config_t, timings_t> bbf_delta_tls_;
  /// The delta t_lookup values for Bloom filters.
  std::map<cf_config_t, timings_t> cf_delta_tls_;
  /// Keeps track of updates. (if true, the in-memory state has changed over the persistent state)
  $u1 changed_;
  /// The file descriptor (-1, if no file is present)
  $i32 file_descriptor_;
  /// The file size in bytes.
  $u64 file_size_;
  /// Pointer to the mapped file data (read only).
  $u8* mapped_data_;
  bbf_config_t* bbf_config_begin = nullptr;
  bbf_config_t* bbf_config_end = nullptr;
  timing* bbf_timing_begin = nullptr;
  timing* bbf_timing_end = nullptr;
  cf_config_t* cf_config_begin = nullptr;
  cf_config_t* cf_config_end = nullptr;
  timing* cf_timing_begin = nullptr;
  timing* cf_timing_end = nullptr;

public:

  /// C'tor
  explicit
  calibration_data(const std::string& filename)
    : filename_(filename),
      cache_sizes_(), filter_sizes_(),
      bbf_delta_tls_(), cf_delta_tls_(),
      changed_(false),
      file_descriptor_(-1), file_size_(0), mapped_data_(nullptr) {
    open_file();
  }

  /// D'tor
  ~calibration_data() {
    close_file();
  }

  /// Open and map the calibration data file.
  void
  open_file();

  /// Unmap and close the calibration data file.
  void
  close_file();

  /// Returns true if there changes that need to be written to the file.
  bool
  changed() const {
    return changed_;
  }

  /// Set the cache sizes [bytes];
  void
  set_cache_sizes(const std::vector<uint64_t>& cache_sizes) {
    cache_sizes_ = cache_sizes;
    changed_ = true;
  }

  /// Returns the cache size [bytes]
  uint64_t
  get_cache_size(const std::size_t level) const {
    if (level == 0 || level > cache_sizes_.size()) {
      throw std::invalid_argument("Illegal cache level: " + std::to_string(level));
    }
    return cache_sizes_[level - 1];
  }

  /// Returns height of the memory hierarchy level.
  std::size_t
  get_mem_levels() const {
    return cache_sizes_.size() + 1;
  }

  /// Set the cache sizes [bytes];
  void
  set_filter_sizes(const std::vector<uint64_t>& filter_sizes) {
    if (filter_sizes.size() != cache_sizes_.size() + 1) {
      throw std::invalid_argument("The number of filter sizes must be equal to the memory levels.");
    }
    filter_sizes_ = filter_sizes;
    changed_ = true;
  }

  /// Returns the cache size [bytes]
  uint64_t
  get_filter_size(const std::size_t mem_level) const {
    if (mem_level == 0 || mem_level > filter_sizes_.size()) {
      throw std::invalid_argument("Illegal memory level: " + std::to_string(mem_level));
    }
    return filter_sizes_[mem_level - 1];
  }

  /// Write the changes to the file.
  void
  persist();

  /// Serialize the calibration data.
  std::vector<$u8>
  serialize();

  /// Add the delta-t_l values for the given filter configuration.
  /// Note: All changes are transient, until the persist function is called.
  void
  put(const bbf_config_t& config, const timings_t& delta_timings);
  void
  put(const cf_config_t& config, const timings_t& delta_timings);

  /// Get the delta-t_l values for the given filter configuration.
  timings_t
  get(const bbf_config_t& config);
  timings_t
  get(const cf_config_t& config);

  // TODO add skyline matrix
};
//===----------------------------------------------------------------------===//


} // namespace model
} // namespace filter
} // namespace dtl
