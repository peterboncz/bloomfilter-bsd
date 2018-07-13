#pragma once

#include <algorithm>
#include <string>
#include <cstring>
#include <fstream>
#include <map>
#include <type_traits>

#include <fcntl.h>
#include <pwd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp>
#include <dtl/filter/cuckoofilter/cuckoofilter_config.hpp>

#include "timing.hpp"
#include "tuning_params.hpp"

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

  /// The calibrated tuning parameters for blocked Bloom filters.
  std::map<bbf_config_t, tuning_params> bbf_tuning_params_;
  /// The calibrated tuning parameters for Cuckoo filters.
  std::map<cf_config_t, tuning_params> cf_tuning_params_;

  /// The calibrated cache sizes.
  std::vector<$u64> cache_sizes_;
  /// The benchmarked filter sizes. one for each memory level.
  std::vector<$u64> filter_sizes_;
  /// The delta t_lookup values for Bloom filters. (one timing per memory level)
  std::map<bbf_config_t, timings_t> bbf_delta_tls_;
  /// The delta t_lookup values for Cuckoo filters. (one timing per memory level)
  std::map<cf_config_t, timings_t> cf_delta_tls_;
  /// Keeps track of updates. (if true, the in-memory state has changed over the persistent state)
  $u1 changed_;
  /// The file descriptor (-1, if no file is present)
  $i32 file_descriptor_;
  /// The file size in bytes.
  $u64 file_size_;

  /// Pointer to the mapped file data (read only).
  $u8* mapped_data_;
  bbf_config_t* bbf_config_begin_ = nullptr;
  bbf_config_t* bbf_config_end_ = nullptr;
  timing* bbf_timing_begin_ = nullptr;
  timing* bbf_timing_end_ = nullptr;
  tuning_params* bbf_tuning_params_begin_ = nullptr;
  tuning_params* bbf_tuning_params_end_ = nullptr;
  cf_config_t* cf_config_begin_ = nullptr;
  cf_config_t* cf_config_end_ = nullptr;
  timing* cf_timing_begin_ = nullptr;
  timing* cf_timing_end_ = nullptr;
  tuning_params* cf_tuning_params_begin_ = nullptr;
  tuning_params* cf_tuning_params_end_ = nullptr;

  const timing null_timing_ { 0.0, 0.0 };
  const tuning_params null_tuning_params_ { 1 };

  static std::string
  get_default_filename() { // TODO to be discussed with Peter
    const char* home_dir;
    if ((home_dir = getenv("HOME")) == NULL) {
      home_dir = getpwuid(getuid())->pw_dir;
    }
    const std::string filename = std::string(home_dir) + "/" + ".dtl-filter.dat";
    return filename;
  }

 public:

  timings_t
  get_null_timings() const {
    timings_t timings;
    u64 mem_levels = get_mem_levels();
    for (std::size_t i = 0; i < mem_levels; i++) {
      timings.push_back(null_timing_);
    }
    return timings;
  }

  tuning_params
  get_null_tuning_params() const {
    return null_tuning_params_;
  }

  static calibration_data&
  get_default_instance() {
    static calibration_data instance;
    return instance;
  }


  /// C'tor
  explicit
  calibration_data(const std::string& filename)
    : filename_(filename),
      bbf_tuning_params_(), cf_tuning_params_(),
      cache_sizes_(), filter_sizes_(),
      bbf_delta_tls_(), cf_delta_tls_(),
      changed_(false),
      file_descriptor_(-1), file_size_(0), mapped_data_(nullptr) {
    open_file();
  }

  /// C'tor (using the default filename)
  calibration_data() : calibration_data(get_default_filename()) {};


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

  /// Returns the cache sizes [bytes]
  std::vector<$u64>
  get_cache_sizes() const {
    return cache_sizes_;
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

  /// Returns the cache size [bytes]
  std::vector<$u64>
  get_filter_sizes() const {
    return filter_sizes_;
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
  put_timings(const bbf_config_t& config, const timings_t& delta_timings);
  void
  put_timings(const cf_config_t& config, const timings_t& delta_timings);

  /// Get the delta-t_l values for the given filter configuration.
  timings_t
  get_timings(const bbf_config_t& config) const;
  timings_t
  get_timings(const cf_config_t& config) const;


  /// Add the tuning parameters for the given filter configuration.
  /// Note: All changes are transient, until the persist function is called.
  void
  put_tuning_params(const bbf_config_t& config, const tuning_params& params);
  void
  put_tuning_params(const cf_config_t& config, const tuning_params& params);

  /// Get the delta-t_l values for the given filter configuration.
  tuning_params
  get_tuning_params(const bbf_config_t& config) const;
  tuning_params
  get_tuning_params(const cf_config_t& config) const;

  // TODO add skyline matrix
  // TODO write the number of threads used during calibration
};
//===----------------------------------------------------------------------===//


} // namespace model
} // namespace filter
} // namespace dtl
