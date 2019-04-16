#pragma once

#include <dtl/dtl.hpp>
#include <dtl/math.hpp>
#include <thirdparty/sqlite/sqlite3.h>

#include <amsfilter/amsfilter.hpp>
#include <amsfilter_model/fpr.hpp>
#include <amsfilter_model/internal/util.hpp>

namespace amsfilter {
namespace model {
//===----------------------------------------------------------------------===//
/// AMS-Filter specific UDFs for SQLite.
/// The UDFs below are enabled per process using the init_udfs() function.
//===----------------------------------------------------------------------===//
extern "C" void
fpr_udf(sqlite3_context* context, int argc, sqlite3_value** argv) {
  const auto encoded_config = static_cast<u32>(sqlite3_value_int(argv[0]));
  const auto m = static_cast<u64>(sqlite3_value_int64(argv[1]));
  const auto n = static_cast<u64>(sqlite3_value_int64(argv[2]));
  const auto config = config_from_int(encoded_config);
  f64 fpr = amsfilter::fpr(config, m, n);
  sqlite3_result_double(context, fpr);
}
//===----------------------------------------------------------------------===//
extern "C" void
fpr_fast_udf(sqlite3_context* context, int argc, sqlite3_value** argv) {
  const auto encoded_config = static_cast<u32>(sqlite3_value_int(argv[0]));
  const auto m = static_cast<u64>(sqlite3_value_int64(argv[1]));
  const auto n = static_cast<u64>(sqlite3_value_int64(argv[2]));
  const auto config = config_from_int(encoded_config);
  f64 fpr = amsfilter::fpr_fast(config, m, n);
  sqlite3_result_double(context, fpr);
}
//===----------------------------------------------------------------------===//
extern "C" void
optimal_k_udf(sqlite3_context* context, int argc, sqlite3_value** argv) {
  const auto encoded_config = static_cast<u32>(sqlite3_value_int(argv[0]));
  const auto m = static_cast<u64>(sqlite3_value_int64(argv[1]));
  const auto n = static_cast<u64>(sqlite3_value_int64(argv[2]));
  const auto config = config_from_int(encoded_config);
  f64 opt_k = amsfilter::optimal_k(config, m, n);
  sqlite3_result_double(context, opt_k);
}
//===----------------------------------------------------------------------===//
extern "C" void
optimal_k_fast_udf(sqlite3_context* context, int argc, sqlite3_value** argv) {
  const auto encoded_config = static_cast<u32>(sqlite3_value_int(argv[0]));
  const auto m = static_cast<u64>(sqlite3_value_int64(argv[1]));
  const auto n = static_cast<u64>(sqlite3_value_int64(argv[2]));
  const auto config = config_from_int(encoded_config);
  f64 opt_k = amsfilter::optimal_k_fast(config, m, n);
  sqlite3_result_double(context, opt_k);
}
//===----------------------------------------------------------------------===//
extern "C" void
is_power_of_two_udf(sqlite3_context* context, int argc, sqlite3_value** argv) {
  switch (sqlite3_value_type(argv[0])) {
    case SQLITE_INTEGER: {
      const auto val = static_cast<u64>(sqlite3_value_int64(argv[0]));
      const auto is_pow2 = dtl::is_power_of_two(val);
      sqlite3_result_int(context, is_pow2 ? 1 : 0);
      return;
    }
    case SQLITE_NULL: {
      sqlite3_result_null(context);
      return;
    }
  }
  sqlite3_result_error(context, "Function expects an integer argument.", -1);
}
//===----------------------------------------------------------------------===//
extern "C" void
config_to_json_udf(sqlite3_context* context, int argc, sqlite3_value** argv) {
  switch (sqlite3_value_type(argv[0])) {
    case SQLITE_INTEGER: {
      const auto encoded_config = static_cast<u32>(sqlite3_value_int(argv[0]));
      const auto config = config_from_int(encoded_config);
      std::stringstream str;
      str << config;
      sqlite3_result_text(context, str.str().c_str(), -1, SQLITE_TRANSIENT);
      return;
    }
    case SQLITE_NULL: {
      sqlite3_result_null(context);
      return;
    }
  }
  sqlite3_result_error(context, "Function expects an integer argument.", -1);
}
//===----------------------------------------------------------------------===//
extern "C" void
config_addr_mode_is_pow2_udf(
    sqlite3_context* context, int argc, sqlite3_value** argv) {
  switch (sqlite3_value_type(argv[0])) {
    case SQLITE_INTEGER: {
      const auto encoded_config = static_cast<u32>(sqlite3_value_int(argv[0]));
      const auto config = config_from_int(encoded_config);
      const auto is_pow2 =
          config.addr_mode == dtl::block_addressing::POWER_OF_TWO;
      sqlite3_result_int(context, is_pow2 ? 1 : 0);
      return;
    }
    case SQLITE_NULL: {
      sqlite3_result_null(context);
      return;
    }
  }
  sqlite3_result_error(context, "Function expects an integer argument.", -1);
}
//===----------------------------------------------------------------------===//
extern "C" void
config_addr_mode_is_magic_udf(
    sqlite3_context* context, int argc, sqlite3_value** argv) {
  switch (sqlite3_value_type(argv[0])) {
    case SQLITE_INTEGER: {
      const auto encoded_config = static_cast<u32>(sqlite3_value_int(argv[0]));
      const auto config = config_from_int(encoded_config);
      const auto is_pow2 =
          config.addr_mode == dtl::block_addressing::MAGIC;
      sqlite3_result_int(context, is_pow2 ? 1 : 0);
      return;
    }
    case SQLITE_NULL: {
      sqlite3_result_null(context);
      return;
    }
  }
  sqlite3_result_error(context, "Function expects an integer argument.", -1);
}
//===----------------------------------------------------------------------===//
extern "C" void
config_extract_addr_mode_udf(
    sqlite3_context* context, int argc, sqlite3_value** argv) {
  switch (sqlite3_value_type(argv[0])) {
    case SQLITE_INTEGER: {
      const auto encoded_config = static_cast<u32>(sqlite3_value_int(argv[0]));
      const auto config = config_from_int(encoded_config);
      sqlite3_result_int(context, static_cast<int>(config.addr_mode));
      return;
    }
    case SQLITE_NULL: {
      sqlite3_result_null(context);
      return;
    }
  }
  sqlite3_result_error(context, "Function expects an integer argument.", -1);
}
//===----------------------------------------------------------------------===//
extern "C" void
config_extract_word_size_udf(
    sqlite3_context* context, int argc, sqlite3_value** argv) {
  switch (sqlite3_value_type(argv[0])) {
    case SQLITE_INTEGER: {
      const auto encoded_config = static_cast<u32>(sqlite3_value_int(argv[0]));
      const auto config = config_from_int(encoded_config);
      sqlite3_result_int(context, static_cast<int>(config.word_size));
      return;
    }
    case SQLITE_NULL: {
      sqlite3_result_null(context);
      return;
    }
  }
  sqlite3_result_error(context, "Function expects an integer argument.", -1);
}
//===----------------------------------------------------------------------===//
extern "C" void
config_extract_word_cnt_udf(
    sqlite3_context* context, int argc, sqlite3_value** argv) {
  switch (sqlite3_value_type(argv[0])) {
    case SQLITE_INTEGER: {
      const auto encoded_config = static_cast<u32>(sqlite3_value_int(argv[0]));
      const auto config = config_from_int(encoded_config);
      sqlite3_result_int(context, static_cast<int>(config.word_cnt_per_block));
      return;
    }
    case SQLITE_NULL: {
      sqlite3_result_null(context);
      return;
    }
  }
  sqlite3_result_error(context, "Function expects an integer argument.", -1);
}
//===----------------------------------------------------------------------===//
extern "C" void
config_extract_sector_cnt_udf(
    sqlite3_context* context, int argc, sqlite3_value** argv) {
  switch (sqlite3_value_type(argv[0])) {
    case SQLITE_INTEGER: {
      const auto encoded_config = static_cast<u32>(sqlite3_value_int(argv[0]));
      const auto config = config_from_int(encoded_config);
      sqlite3_result_int(context, static_cast<int>(config.sector_cnt));
      return;
    }
    case SQLITE_NULL: {
      sqlite3_result_null(context);
      return;
    }
  }
  sqlite3_result_error(context, "Function expects an integer argument.", -1);
}
//===----------------------------------------------------------------------===//
extern "C" void
config_extract_zone_cnt_udf(
    sqlite3_context* context, int argc, sqlite3_value** argv) {
  switch (sqlite3_value_type(argv[0])) {
    case SQLITE_INTEGER: {
      const auto encoded_config = static_cast<u32>(sqlite3_value_int(argv[0]));
      const auto config = config_from_int(encoded_config);
      sqlite3_result_int(context, static_cast<int>(config.zone_cnt));
      return;
    }
    case SQLITE_NULL: {
      sqlite3_result_null(context);
      return;
    }
  }
  sqlite3_result_error(context, "Function expects an integer argument.", -1);
}
//===----------------------------------------------------------------------===//
extern "C" void
config_extract_hashfn_cnt_udf(
    sqlite3_context* context, int argc, sqlite3_value** argv) {
  switch (sqlite3_value_type(argv[0])) {
    case SQLITE_INTEGER: {
      const auto encoded_config = static_cast<u32>(sqlite3_value_int(argv[0]));
      const auto config = config_from_int(encoded_config);
      sqlite3_result_int(context, static_cast<int>(config.k));
      return;
    }
    case SQLITE_NULL: {
      sqlite3_result_null(context);
      return;
    }
  }
  sqlite3_result_error(context, "Function expects an integer argument.", -1);
}
//===----------------------------------------------------------------------===//
/// Supposed to be called by SQLite, when sqlite3_auto_extension is triggered.
extern "C" int
register_udfs(sqlite3* db, const char**, const struct sqlite3_api_routines*) {
  // False-positive rate.
  sqlite3_create_function(
      db,
      "fpr", // the function name
      3, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::fpr_udf,
      nullptr,
      nullptr
  );
  sqlite3_create_function(
      db,
      "fpr_fast", // the function name
      3, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::fpr_fast_udf,
      nullptr,
      nullptr
  );

  // Optimal k.
  sqlite3_create_function(
      db,
      "optimal_k", // the function name
      3, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::optimal_k_udf,
      nullptr,
      nullptr
  );
  sqlite3_create_function(
      db,
      "optimal_k_fast", // the function name
      3, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::optimal_k_fast_udf,
      nullptr,
      nullptr
  );
  sqlite3_create_function(
      db,
      "opt_k", // the function name
      3, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::optimal_k_udf,
      nullptr,
      nullptr
  );
  sqlite3_create_function(
      db,
      "opt_k_fast", // the function name
      3, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::optimal_k_fast_udf,
      nullptr,
      nullptr
  );

  sqlite3_create_function(
      db,
      "config_to_json", // the function name
      1, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::config_to_json_udf,
      nullptr,
      nullptr
  );

  sqlite3_create_function(
      db,
      "config_addr_mode_is_pow2", // the function name
      1, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::config_addr_mode_is_pow2_udf,
      nullptr,
      nullptr
  );

  sqlite3_create_function(
      db,
      "config_addr_mode_is_magic", // the function name
      1, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::config_addr_mode_is_magic_udf,
      nullptr,
      nullptr
  );

  sqlite3_create_function(
      db,
      "config_extract_addr_mode", // the function name
      1, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::config_extract_addr_mode_udf,
      nullptr,
      nullptr
  );

  sqlite3_create_function(
      db,
      "config_extract_word_size", // the function name
      1, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::config_extract_word_size_udf,
      nullptr,
      nullptr
  );

  sqlite3_create_function(
      db,
      "config_extract_word_cnt", // the function name
      1, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::config_extract_word_cnt_udf,
      nullptr,
      nullptr
  );
  sqlite3_create_function(
      db,
      "config_extract_w", // the function name
      1, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::config_extract_word_cnt_udf,
      nullptr,
      nullptr
  );

  sqlite3_create_function(
      db,
      "config_extract_sector_cnt", // the function name
      1, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::config_extract_sector_cnt_udf,
      nullptr,
      nullptr
  );
  sqlite3_create_function(
      db,
      "config_extract_s", // the function name
      1, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::config_extract_sector_cnt_udf,
      nullptr,
      nullptr
  );

  sqlite3_create_function(
      db,
      "config_extract_zone_cnt", // the function name
      1, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::config_extract_zone_cnt_udf,
      nullptr,
      nullptr
  );
  sqlite3_create_function(
      db,
      "config_extract_z", // the function name
      1, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::config_extract_zone_cnt_udf,
      nullptr,
      nullptr
  );

  sqlite3_create_function(
      db,
      "config_extract_hashfn_cnt", // the function name
      1, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::config_extract_hashfn_cnt_udf,
      nullptr,
      nullptr
  );
  sqlite3_create_function(
      db,
      "config_extract_k", // the function name
      1, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::config_extract_hashfn_cnt_udf,
      nullptr,
      nullptr
  );

  // Is power of two.
  sqlite3_create_function(
      db,
      "is_power_of_two", // the function name
      1, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::is_power_of_two_udf,
      nullptr,
      nullptr
  );
  sqlite3_create_function(
      db,
      "is_pow2", // the function name
      1, // the number of arguments
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,
      nullptr, // pointer to user-data
      amsfilter::model::is_power_of_two_udf,
      nullptr,
      nullptr
  );
  return 0;
}
//===----------------------------------------------------------------------===//
/// Enables the AMS-Filter UDFs as an auto extension. Once the function is
/// called, the UDFs are available within SQLite.
void
init_udfs() {
  auto register_amsfilter_udfs = (void(*)(void)) register_udfs;
  sqlite3_auto_extension(register_amsfilter_udfs);
}
//===----------------------------------------------------------------------===//
} // namespace model
} // namespace amsfilter
