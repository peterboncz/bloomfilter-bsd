#include "perf_db.hpp"

#include <pwd.h>
#include <thread>

#include <amsfilter/amsfilter.hpp>
#include <amsfilter_model/internal/util.hpp>
#include <amsfilter_model/internal/udf.hpp>

using namespace amsfilter;
using namespace amsfilter::model;
//===----------------------------------------------------------------------===//
std::string
PerfDb::get_default_filename() {
  const char* home_dir;
  if ((home_dir = getenv("HOME")) == nullptr) {
    home_dir = getpwuid(getuid())->pw_dir;
  }
  const std::string filename = std::string(home_dir) + "/" + ".amsfilter_perf.db";
  return filename;
}
//===----------------------------------------------------------------------===//
PerfDb::PerfDb(const std::string& file)
    : file_(file),
      mutex_(),
      db_(nullptr) {
  open();
  init();
}
//===----------------------------------------------------------------------===//
PerfDb::~PerfDb() {
  close();
}
//===----------------------------------------------------------------------===//
void
PerfDb::open() {
  // Register the AMS-Filter UDFs as an auto extension within SQLite.
  init_udfs();

  auto rc = sqlite3_open(file_.c_str(), &db_);
  if (rc) {
    std::stringstream err;
    err << "Can't open database: " << file_ << ". Error: "
        << sqlite3_errmsg(db_)
        << std::endl;
    sqlite3_close(db_);
    throw std::invalid_argument(err.str());
  }
}
//===----------------------------------------------------------------------===//
void
PerfDb::init() {
  auto rc = sqlite3_open(file_.c_str(), &db_);
  if (rc) {
    std::stringstream err;
    err << "Can't open database: " << file_ << ". Error: "
        << sqlite3_errmsg(db_)
        << std::endl;
    sqlite3_close(db_);
    throw std::invalid_argument(err.str());
  }

  // Create the tables if they do not already exist.
  const std::string sql_create_table =
      // Table for the tuning parameters (CPU only).
      "create table if not exists tuning_params (\n"
      "  config integer primary key,\n"
      "  unroll_factor  int not null"
      ");\n"
      // Table for the lookup costs (t_l).
      "create table if not exists tl (\n"
      "  config     int    not null,\n"
      "  m          bigint not null,\n"
      "  device     int    not null,\n"
      "  thread_cnt int    not null,\n"
      "  nanos      double not null,\n"
      "  cycles     double not null\n"
      ");\n"
      // Table for the PCIe bandwidth limits.
      "create table if not exists bw_limit (\n"
      "  device       int    not null,\n"
      "  key_location int    not null,\n" // pageable vs pinned
      "  nanos        double not null,\n"
      "  cycles       double not null,\n"
      "  primary key (device, key_location)\n"
      ");\n"
      // Insert dummy records for the CPU.
      "insert or replace into bw_limit values(-1, 0, 0.0, 0.0);\n"
      "insert or replace into bw_limit values(-1, 1, 0.0, 0.0);\n"
      "insert or replace into bw_limit values(-1, 2, 0.0, 0.0);\n"
      // Table for the skyline matrix.
      "create table if not exists skyline (\n"
      "  device         int    not null,\n"
      "  thread_cnt     int    not null,\n"
      "  key_location   int    not null,\n" // pageable vs pinned
      "  n              bigint not null,\n"
      "  tw             bigint not null,\n"
      "  config         int    not null,\n"
      "  m              bigint not null,\n"
      "  overhead_nanos double not null,\n"
      "  primary key (device, thread_cnt, n, tw, key_location)\n"
      ");\n"
  ;

  char* err_msg = 0;
  rc = sqlite3_exec(db_, sql_create_table.c_str(),
      nullptr, nullptr, &err_msg);
  if (rc) {
    std::stringstream err;
    err << "Can't create table: "
        << sqlite3_errmsg(db_)
        << std::endl;
    close();
    throw std::runtime_error(err.str());
  }

  // Prepare the SQL statements.
  auto prepare = [&](sqlite3_stmt*& prepared_stmt, const std::string sql_stmt) {
    rc = sqlite3_prepare_v2(db_, sql_stmt.c_str(), -1,
        &prepared_stmt, nullptr);
    if (rc) {
      std::stringstream err;
      err << "Can't prepare SQL statement. '" << sql_stmt << "'"
          << sqlite3_errmsg(db_)
          << std::endl;
      throw std::runtime_error(err.str());
    }
  };

  prepare(insert_tuning_params_stmt_,
      "insert into tuning_params (config, unroll_factor)\n"
      "  values (:c, :u)");
  prepare(delete_tuning_params_stmt_,
      "delete from tuning_params where config = :c");
  prepare(select_tuning_params_stmt_,
      "select unroll_factor from tuning_params where config = :c");

  prepare(insert_tl_stmt_,
      "insert into tl\n"
      "  values (:c, :m, :device, :thread_cnt, :nanos, :cycles)");
  prepare(delete_tl_stmt_,
      "delete from tl where config = :c and m = :m and device = :device"
      " and thread_cnt = :thread_cnt");
  prepare(select_tl_stmt_,
      "select nanos, cycles from tl"
      " where config = :c and m = :m and device = :device"
      "  and thread_cnt = :thread_cnt");
  prepare(select_m_stmt_,
      "select m from tl"
      " where config = :c and device = :device"
      "  and thread_cnt = :thread_cnt order by m");
  prepare(select_tls_stmt_,
      "select tl.m, max(tl.nanos, bw.nanos) as nanos"
      "  from tl, bw_limit bw"
      " where tl.device = bw.device"
      "   and tl.config = :c and tl.device = :device"
      "   and tl.thread_cnt = :thread_cnt"
      "   and key_location = :key_location"
      " order by tl.m");

  prepare(insert_or_replace_bw_limit_stmt_,
      "insert or replace into bw_limit"
      " values (:device, :key_location, :nanos, :cycles)");
  prepare(select_bw_limit_stmt_,
      "select nanos, cycles from bw_limit"
      " where device = :device and key_location = :key_location");

  prepare(insert_or_replace_skyline_entry_stmt_,
      "insert or replace into skyline"
      " values (:device, :thread_cnt, :key_location, :n, :tw, :c, :m,"
      " :overhead)");
  prepare(find_skyline_entries_stmt_,
      "select config, m, overhead_nanos"
      "  from skyline"
      " where device = :device0"
      "   and thread_cnt = :thread_cnt0"
      "   and key_location = :key_location0"
      "   and n in (select distinct n from skyline "
      "              where device = :device1"
      "                and thread_cnt = :thread_cnt1"
      "                and key_location = :key_location1"
      "              order by abs(n - :n) limit 2)"
      "   and tw in (select distinct tw from skyline "
      "               where device = :device2"
      "                 and thread_cnt = :thread_cnt2"
      "                 and key_location = :key_location2"
      "               order by abs(tw - :tw) limit 2)");
}
//===----------------------------------------------------------------------===//
void
PerfDb::close() {
  if (db_ != nullptr) {

    auto destruct = [&](sqlite3_stmt*& prepared_stmt) {
      sqlite3_reset(prepared_stmt);
      sqlite3_finalize(prepared_stmt);
    };

    destruct(insert_tuning_params_stmt_);
    destruct(delete_tuning_params_stmt_);
    destruct(select_tuning_params_stmt_);
    destruct(insert_tl_stmt_);
    destruct(delete_tl_stmt_);
    destruct(select_tl_stmt_);
    destruct(select_m_stmt_);
    destruct(select_tls_stmt_);
    destruct(insert_or_replace_bw_limit_stmt_);
    destruct(select_bw_limit_stmt_);
    destruct(insert_or_replace_skyline_entry_stmt_);
    destruct(find_skyline_entries_stmt_);

    db_ = nullptr;
    auto* db = db_;
    auto rc = sqlite3_close(db);
    if (rc) {
      std::stringstream err;
      err << "Can't close database. Error: "
          << sqlite3_errmsg(db)
          << std::endl;
      throw std::runtime_error(err.str());
    }
  }
}
//===----------------------------------------------------------------------===//
void
PerfDb::put_tuning_params(const Config& config,
    const TuningParams& tuning_params) {

  const auto enc_config = amsfilter::model::config_to_int(config);

  std::lock_guard<std::mutex> lock(mutex_);

  sqlite3_reset(delete_tuning_params_stmt_);
  sqlite3_bind_int(delete_tuning_params_stmt_,
      sqlite3_bind_parameter_index(delete_tuning_params_stmt_, ":c"),
      enc_config);

  sqlite3_reset(insert_tuning_params_stmt_);
  sqlite3_bind_int(insert_tuning_params_stmt_,
      sqlite3_bind_parameter_index(insert_tuning_params_stmt_, ":c"),
      enc_config);
  sqlite3_bind_int(insert_tuning_params_stmt_,
      sqlite3_bind_parameter_index(insert_tuning_params_stmt_, ":u"),
      tuning_params.unroll_factor);

  $i32 rc;
  rc = sqlite3_step(delete_tuning_params_stmt_);
  if (rc != SQLITE_DONE) {
    std::stringstream err;
    err << "Can't delete data. Error: " << rc << " - "
        << sqlite3_errmsg(db_)
        << std::endl;
    throw std::runtime_error(err.str());
  }

  rc = sqlite3_step(insert_tuning_params_stmt_);
  if (rc != SQLITE_DONE) {
    std::stringstream err;
    err << "Can't insert data. Error: " << rc << " - "
        << sqlite3_errmsg(db_)
        << std::endl;
    throw std::runtime_error(err.str());
  }
}
//===----------------------------------------------------------------------===//
$u1
PerfDb::has_tuning_params(const Config& config) {

  const auto enc_config = amsfilter::model::config_to_int(config);

  std::lock_guard<std::mutex> lock(mutex_);

  sqlite3_reset(select_tuning_params_stmt_);
  sqlite3_bind_int(select_tuning_params_stmt_,
      sqlite3_bind_parameter_index(select_tuning_params_stmt_, ":c"),
      enc_config);

  $i32 rc;
  TuningParams ret;
  while((rc = sqlite3_step(select_tuning_params_stmt_)) == SQLITE_ROW) {
    return true;
  }
  return false;
}
//===----------------------------------------------------------------------===//
TuningParams
PerfDb::get_tuning_params(const Config& config) {

  const auto enc_config = amsfilter::model::config_to_int(config);

  std::lock_guard<std::mutex> lock(mutex_);

  sqlite3_reset(select_tuning_params_stmt_);
  sqlite3_bind_int(select_tuning_params_stmt_,
      sqlite3_bind_parameter_index(select_tuning_params_stmt_, ":c"),
      enc_config);

  $i32 rc;
  TuningParams ret;
  while((rc = sqlite3_step(select_tuning_params_stmt_)) == SQLITE_ROW) {
    ret.unroll_factor = static_cast<u32>(
        sqlite3_column_int(select_tuning_params_stmt_, 0));
    return ret;
  }
  std::stringstream err;
  err << "Can't fetch data. Error: " << rc << " - "
      << sqlite3_errmsg(db_)
      << std::endl;
  throw std::runtime_error(err.str());
}
//===----------------------------------------------------------------------===//
void
PerfDb::put_tl(const Config& config, const std::size_t m, const Env& exec_env,
    const timing& timing) {

  const auto enc_config = amsfilter::model::config_to_int(config);

  std::lock_guard<std::mutex> lock(mutex_);

  sqlite3_reset(delete_tl_stmt_);
  sqlite3_bind_int(delete_tl_stmt_,
      sqlite3_bind_parameter_index(delete_tl_stmt_, ":c"),
      enc_config);
  sqlite3_bind_int64(delete_tl_stmt_,
      sqlite3_bind_parameter_index(delete_tl_stmt_, ":m"),
      m);
  sqlite3_bind_int(delete_tl_stmt_,
      sqlite3_bind_parameter_index(delete_tl_stmt_, ":device"),
      exec_env.get_device());
  sqlite3_bind_int(delete_tl_stmt_,
      sqlite3_bind_parameter_index(delete_tl_stmt_, ":thread_cnt"),
      exec_env.get_thread_cnt());

  sqlite3_reset(insert_tl_stmt_);
  sqlite3_bind_int(insert_tl_stmt_,
      sqlite3_bind_parameter_index(insert_tl_stmt_, ":c"),
      enc_config);
  sqlite3_bind_int64(insert_tl_stmt_,
      sqlite3_bind_parameter_index(insert_tl_stmt_, ":m"),
      m);
  sqlite3_bind_int(insert_tl_stmt_,
      sqlite3_bind_parameter_index(insert_tl_stmt_, ":device"),
      exec_env.get_device());
  sqlite3_bind_int(insert_tl_stmt_,
      sqlite3_bind_parameter_index(insert_tl_stmt_, ":thread_cnt"),
      exec_env.get_thread_cnt());
  sqlite3_bind_double(insert_tl_stmt_,
      sqlite3_bind_parameter_index(insert_tl_stmt_, ":nanos"),
      timing.nanos_per_lookup);
  sqlite3_bind_double(insert_tl_stmt_,
      sqlite3_bind_parameter_index(insert_tl_stmt_, ":cycles"),
      timing.cycles_per_lookup);

  $i32 rc;
  rc = sqlite3_step(delete_tl_stmt_);
  if (rc != SQLITE_DONE) {
    std::stringstream err;
    err << "Can't delete data. Error: " << rc << " - "
        << sqlite3_errmsg(db_)
        << std::endl;
    throw std::runtime_error(err.str());
  }

  rc = sqlite3_step(insert_tl_stmt_);
  if (rc != SQLITE_DONE) {
    std::stringstream err;
    err << "Can't insert data. Error: " << rc << " - "
        << sqlite3_errmsg(db_)
        << std::endl;
    throw std::runtime_error(err.str());
  }
}
//===----------------------------------------------------------------------===//
$u1
PerfDb::has_tl(const Config& config, const std::size_t m, const Env& exec_env) {

  const auto enc_config = amsfilter::model::config_to_int(config);

  std::lock_guard<std::mutex> lock(mutex_);

  sqlite3_reset(select_tl_stmt_);
  sqlite3_bind_int(select_tl_stmt_,
      sqlite3_bind_parameter_index(select_tl_stmt_, ":c"),
      enc_config);
  sqlite3_bind_int64(select_tl_stmt_,
      sqlite3_bind_parameter_index(select_tl_stmt_, ":m"),
      m);
  sqlite3_bind_int(select_tl_stmt_,
      sqlite3_bind_parameter_index(select_tl_stmt_, ":device"),
      exec_env.get_device());
  sqlite3_bind_int(select_tl_stmt_,
      sqlite3_bind_parameter_index(select_tl_stmt_, ":thread_cnt"),
      exec_env.get_thread_cnt());

  $i32 rc;
  while((rc = sqlite3_step(select_tl_stmt_)) == SQLITE_ROW) {
    return true;
  }
  return false;
}
//===----------------------------------------------------------------------===//
timing
PerfDb::get_tl(const Config& config, const std::size_t m, const Env& exec_env) {

  const auto enc_config = amsfilter::model::config_to_int(config);

  std::lock_guard<std::mutex> lock(mutex_);

  sqlite3_reset(select_tl_stmt_);
  sqlite3_bind_int(select_tl_stmt_,
      sqlite3_bind_parameter_index(select_tl_stmt_, ":c"),
      enc_config);
  sqlite3_bind_int64(select_tl_stmt_,
      sqlite3_bind_parameter_index(select_tl_stmt_, ":m"),
      m);
  sqlite3_bind_int(select_tl_stmt_,
      sqlite3_bind_parameter_index(select_tl_stmt_, ":device"),
      exec_env.get_device());
  sqlite3_bind_int(select_tl_stmt_,
      sqlite3_bind_parameter_index(select_tl_stmt_, ":thread_cnt"),
      exec_env.get_thread_cnt());

  $i32 rc;
  timing ret;
  while((rc = sqlite3_step(select_tl_stmt_)) == SQLITE_ROW) {
    ret.nanos_per_lookup = sqlite3_column_double(select_tl_stmt_, 0);
    ret.cycles_per_lookup = sqlite3_column_double(select_tl_stmt_, 1);
    return ret;
  }
  std::stringstream err;
  err << "Can't fetch data. Error: " << rc << " - "
      << sqlite3_errmsg(db_)
      << std::endl;
  throw std::runtime_error(err.str());
}
//===----------------------------------------------------------------------===//
std::vector<std::size_t>
PerfDb::get_filter_sizes(const Config& config, const Env& exec_env) {
  const auto enc_config = amsfilter::model::config_to_int(config);

  std::lock_guard<std::mutex> lock(mutex_);

  sqlite3_reset(select_m_stmt_);
  sqlite3_bind_int(select_m_stmt_,
      sqlite3_bind_parameter_index(select_m_stmt_, ":c"),
      enc_config);
  sqlite3_bind_int(select_m_stmt_,
      sqlite3_bind_parameter_index(select_m_stmt_, ":device"),
      exec_env.get_device());
  sqlite3_bind_int(select_m_stmt_,
      sqlite3_bind_parameter_index(select_m_stmt_, ":thread_cnt"),
      exec_env.get_thread_cnt());

  std::vector<std::size_t> ms;
  $i32 rc;
  while((rc = sqlite3_step(select_m_stmt_)) == SQLITE_ROW) {
    ms.push_back(sqlite3_column_int64(select_m_stmt_, 0));
  }
  return ms;
}
//===----------------------------------------------------------------------===//
PerfDb::tl_table
PerfDb::get_tls(const Config& config, const Env& exec_env) {
  const auto enc_config = amsfilter::model::config_to_int(config);

  std::lock_guard<std::mutex> lock(mutex_);

  sqlite3_reset(select_tls_stmt_);
  sqlite3_bind_int(select_tls_stmt_,
      sqlite3_bind_parameter_index(select_tls_stmt_, ":c"),
      enc_config);
  sqlite3_bind_int(select_tls_stmt_,
      sqlite3_bind_parameter_index(select_tls_stmt_, ":device"),
      exec_env.get_device());
  sqlite3_bind_int(select_tls_stmt_,
      sqlite3_bind_parameter_index(select_tls_stmt_, ":thread_cnt"),
      exec_env.get_thread_cnt());
  sqlite3_bind_int(select_tls_stmt_,
      sqlite3_bind_parameter_index(select_tls_stmt_, ":key_location"),
      static_cast<int>(exec_env.get_probe_key_location()));

  PerfDb::tl_table result;
  $i32 rc;
  while((rc = sqlite3_step(select_tls_stmt_)) == SQLITE_ROW) {
    result.m.push_back(sqlite3_column_int64(select_tls_stmt_, 0));
    result.nanos.push_back(sqlite3_column_double(select_tls_stmt_, 1));
  }
  return result;
}
//===----------------------------------------------------------------------===//
void
PerfDb::set_gpu_bandwidth_limit(const Env& env, const timing& timing) {

  std::lock_guard<std::mutex> lock(mutex_);

  sqlite3_reset(insert_or_replace_bw_limit_stmt_);
  sqlite3_bind_int(insert_or_replace_bw_limit_stmt_,
      sqlite3_bind_parameter_index(insert_or_replace_bw_limit_stmt_,
          ":device"),
      env.get_device());
  sqlite3_bind_int(insert_or_replace_bw_limit_stmt_,
      sqlite3_bind_parameter_index(insert_or_replace_bw_limit_stmt_,
          ":key_location"),
      static_cast<i32>(env.get_probe_key_location()));
  sqlite3_bind_double(insert_or_replace_bw_limit_stmt_,
      sqlite3_bind_parameter_index(insert_or_replace_bw_limit_stmt_, ":nanos"),
      timing.nanos_per_lookup);
  sqlite3_bind_double(insert_or_replace_bw_limit_stmt_,
      sqlite3_bind_parameter_index(insert_or_replace_bw_limit_stmt_, ":cycles"),
      timing.cycles_per_lookup);

  $i32 rc;
  rc = sqlite3_step(insert_or_replace_bw_limit_stmt_);
  if (rc != SQLITE_DONE) {
    std::stringstream err;
    err << "Can't insert data. Error: " << rc << " - "
        << sqlite3_errmsg(db_)
        << std::endl;
    throw std::runtime_error(err.str());
  }
}
//===----------------------------------------------------------------------===//
timing
PerfDb::get_gpu_bandwidth_limit(const Env& exec_env) {

  std::lock_guard<std::mutex> lock(mutex_);

  sqlite3_reset(select_bw_limit_stmt_);
  sqlite3_bind_int(select_bw_limit_stmt_,
      sqlite3_bind_parameter_index(select_bw_limit_stmt_, ":device"),
      exec_env.get_device());
  sqlite3_bind_int64(select_bw_limit_stmt_,
      sqlite3_bind_parameter_index(select_bw_limit_stmt_, ":key_location"),
      static_cast<i32>(exec_env.get_probe_key_location()));

  $i32 rc;
  timing ret;
  while((rc = sqlite3_step(select_bw_limit_stmt_)) == SQLITE_ROW) {
    ret.nanos_per_lookup = sqlite3_column_double(select_bw_limit_stmt_, 0);
    ret.cycles_per_lookup = sqlite3_column_double(select_bw_limit_stmt_, 1);
    return ret;
  }
  std::stringstream err;
  err << "Can't fetch data. Error: " << rc << " - "
      << sqlite3_errmsg(db_)
      << std::endl;
  throw std::runtime_error(err.str());
}
//===----------------------------------------------------------------------===//
void
PerfDb::put_skyline_entry(const Env& exec_env, const std::size_t  n,
    const std::size_t tw, const Config& config, const std::size_t m,
    f64 overhead) {

  const auto enc_config = amsfilter::model::config_to_int(config);

  std::lock_guard<std::mutex> lock(mutex_);

  sqlite3_reset(insert_or_replace_skyline_entry_stmt_);

  sqlite3_bind_int(insert_or_replace_skyline_entry_stmt_,
      sqlite3_bind_parameter_index(insert_or_replace_skyline_entry_stmt_,
          ":device"),
      exec_env.get_device());
  sqlite3_bind_int(insert_or_replace_skyline_entry_stmt_,
      sqlite3_bind_parameter_index(insert_or_replace_skyline_entry_stmt_,
          ":thread_cnt"),
      exec_env.get_thread_cnt());
  sqlite3_bind_int64(insert_or_replace_skyline_entry_stmt_,
      sqlite3_bind_parameter_index(insert_or_replace_skyline_entry_stmt_,
          ":key_location"),
      static_cast<i32>(exec_env.get_probe_key_location()));
  sqlite3_bind_int64(insert_or_replace_skyline_entry_stmt_,
      sqlite3_bind_parameter_index(insert_or_replace_skyline_entry_stmt_,
          ":n"),
      n);
  sqlite3_bind_int64(insert_or_replace_skyline_entry_stmt_,
      sqlite3_bind_parameter_index(insert_or_replace_skyline_entry_stmt_,
          ":tw"),
      tw);
  sqlite3_bind_int(insert_or_replace_skyline_entry_stmt_,
      sqlite3_bind_parameter_index(insert_or_replace_skyline_entry_stmt_,
          ":c"),
      enc_config);
  sqlite3_bind_int64(insert_or_replace_skyline_entry_stmt_,
      sqlite3_bind_parameter_index(insert_or_replace_skyline_entry_stmt_,
          ":m"),
      m);
  sqlite3_bind_double(insert_or_replace_skyline_entry_stmt_,
      sqlite3_bind_parameter_index(insert_or_replace_skyline_entry_stmt_,
          ":overhead"),
      overhead);

  $i32 rc;
  rc = sqlite3_step(insert_or_replace_skyline_entry_stmt_);
  if (rc != SQLITE_DONE) {
    std::stringstream err;
    err << "Can't insert data. Error: " << rc << " - "
        << sqlite3_errmsg(db_)
        << std::endl;
    throw std::runtime_error(err.str());
  }
}
//===----------------------------------------------------------------------===//
void
PerfDb::begin() {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3_exec(db_, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr);
}
//===----------------------------------------------------------------------===//
void
PerfDb::commit() {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3_exec(db_, "COMMIT TRANSACTION;", nullptr, nullptr, nullptr);
}
//===----------------------------------------------------------------------===//
void
PerfDb::rollback() {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3_exec(db_, "ROLLBACK TRANSACTION;", nullptr, nullptr, nullptr);
}
//===----------------------------------------------------------------------===//
std::vector<SkylineEntry>
PerfDb::find_skyline_entries(const Env& exec_env, const std::size_t  n, f64 tw) {
  std::lock_guard<std::mutex> lock(mutex_);

  sqlite3_reset(find_skyline_entries_stmt_);

  sqlite3_bind_int(find_skyline_entries_stmt_,
      sqlite3_bind_parameter_index(find_skyline_entries_stmt_,
          ":device0"),
      exec_env.get_device());
  sqlite3_bind_int(find_skyline_entries_stmt_,
      sqlite3_bind_parameter_index(find_skyline_entries_stmt_,
          ":device1"),
      exec_env.get_device());
  sqlite3_bind_int(find_skyline_entries_stmt_,
      sqlite3_bind_parameter_index(find_skyline_entries_stmt_,
          ":device2"),
      exec_env.get_device());
  sqlite3_bind_int(find_skyline_entries_stmt_,
      sqlite3_bind_parameter_index(find_skyline_entries_stmt_,
          ":thread_cnt0"),
      exec_env.get_thread_cnt());
  sqlite3_bind_int(find_skyline_entries_stmt_,
      sqlite3_bind_parameter_index(find_skyline_entries_stmt_,
          ":thread_cnt1"),
      exec_env.get_thread_cnt());
  sqlite3_bind_int(find_skyline_entries_stmt_,
      sqlite3_bind_parameter_index(find_skyline_entries_stmt_,
          ":thread_cnt2"),
      exec_env.get_thread_cnt());
  sqlite3_bind_int(find_skyline_entries_stmt_,
      sqlite3_bind_parameter_index(find_skyline_entries_stmt_,
          ":key_location0"),
      static_cast<i32>(exec_env.get_probe_key_location()));
  sqlite3_bind_int(find_skyline_entries_stmt_,
      sqlite3_bind_parameter_index(find_skyline_entries_stmt_,
          ":key_location1"),
      static_cast<i32>(exec_env.get_probe_key_location()));
  sqlite3_bind_int(find_skyline_entries_stmt_,
      sqlite3_bind_parameter_index(find_skyline_entries_stmt_,
          ":key_location2"),
      static_cast<i32>(exec_env.get_probe_key_location()));
  sqlite3_bind_int64(find_skyline_entries_stmt_,
      sqlite3_bind_parameter_index(find_skyline_entries_stmt_,
          ":n"),
      n);
  sqlite3_bind_int64(find_skyline_entries_stmt_,
      sqlite3_bind_parameter_index(find_skyline_entries_stmt_,
          ":tw"),
      tw);

  std::vector<SkylineEntry> results;
  $i32 rc;
  SkylineEntry entry;
  while((rc = sqlite3_step(find_skyline_entries_stmt_)) == SQLITE_ROW) {
    entry.filter_config =
        config_from_int(sqlite3_column_int(find_skyline_entries_stmt_, 0));
    entry.m = sqlite3_column_int64(find_skyline_entries_stmt_, 1);
    entry.overhead = sqlite3_column_double(find_skyline_entries_stmt_, 2);
    results.push_back(entry);
  }
  return results;
}
//===----------------------------------------------------------------------===//
