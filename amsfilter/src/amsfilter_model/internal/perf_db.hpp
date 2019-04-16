#pragma once

#include <mutex>

#include <dtl/dtl.hpp>

#include <amsfilter/amsfilter.hpp>
#include <amsfilter_model/execution_env.hpp>
#include <amsfilter_model/internal/skyline_entry.hpp>
#include <amsfilter_model/internal/timing.hpp>

#include <sqlite/sqlite3.h>

namespace amsfilter {
namespace model {
//===----------------------------------------------------------------------===//
class PerfDb {

  const std::string file_;
  std::mutex mutex_;
  sqlite3* db_;
  sqlite3_stmt* insert_tuning_params_stmt_;
  sqlite3_stmt* delete_tuning_params_stmt_;
  sqlite3_stmt* select_tuning_params_stmt_;
  sqlite3_stmt* insert_tl_stmt_;
  sqlite3_stmt* delete_tl_stmt_;
  sqlite3_stmt* select_tl_stmt_;
  sqlite3_stmt* select_m_stmt_;
  sqlite3_stmt* select_tls_stmt_;

  sqlite3_stmt* insert_or_replace_bw_limit_stmt_;
  sqlite3_stmt* select_bw_limit_stmt_;

  sqlite3_stmt* insert_or_replace_skyline_entry_stmt_;
  sqlite3_stmt* find_skyline_entries_stmt_;

public:

  static std::string
  get_default_filename();

  explicit PerfDb(const std::string& file);
  virtual ~PerfDb();

  // Delete copy and move constructors and assign operators.
  PerfDb(PerfDb const&) = delete;
  PerfDb(PerfDb&&) = delete;
  PerfDb& operator=(PerfDb const&) = delete;
  PerfDb& operator=(PerfDb &&) = delete;

  // Tuning parameters.
  void
  put_tuning_params(const Config& config, const TuningParams& tuning_params);

  $u1
  has_tuning_params(const Config& config);

  TuningParams
  get_tuning_params(const Config& config);

  // Lookup costs (t_l).
  void
  put_tl(const Config& config, const std::size_t m, const Env& exec_env,
      const timing& timing);

  $u1
  has_tl(const Config& config, const std::size_t m, const Env& exec_env);

  timing
  get_tl(const Config& config, const std::size_t m, const Env& exec_env);

  /// Returns the filter sizes (in ascending order) for which reference
  /// measurements exist.
  std::vector<std::size_t>
  get_filter_sizes(const Config& config, const Env& exec_env);

  struct tl_table {
    std::vector<$u64> m;
    std::vector<$f64> nanos;
  };
  tl_table
  get_tls(const Config& config, const Env& exec_env);

  void
  set_gpu_bandwidth_limit(const Env& env, const timing& timing);

  timing
  get_gpu_bandwidth_limit(const Env& exec_env);

  void
  put_skyline_entry(const Env& exec_env, const std::size_t  n,
      const std::size_t tw, const Config& config, const std::size_t m,
      f64 overhead);

  std::vector<SkylineEntry>
  find_skyline_entries(const Env& exec_env, const std::size_t  n, f64 tw);

  void begin();
  void commit();
  void rollback();

private:

  /// Opens the database.
  void open();
  /// Initialize the database schema and prepare the SQL statements.
  void init();
  /// Close the database. Called by the destructor.
  void close();

};
//===----------------------------------------------------------------------===//
} // namespace model
} // namespace amsfilter
