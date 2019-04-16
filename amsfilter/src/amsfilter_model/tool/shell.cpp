#include <pwd.h>
#include <dtl/env.hpp>
#include <amsfilter_model/internal/udf.hpp>
#include <thirdparty/sqlite/shell_mod.h>
//===----------------------------------------------------------------------===//
// A SQLite shell with all the AMS-Filter UDFs.
//===----------------------------------------------------------------------===//
std::string
get_default_filename() {
  const char* home_dir;
  if ((home_dir = getenv("HOME")) == nullptr) {
    home_dir = getpwuid(getuid())->pw_dir;
  }
  const std::string filename = std::string(home_dir) + "/" + ".amsfilter_perf.db";
  return filename;
}
//===----------------------------------------------------------------------===//
int
main(int argc, char** argv) {

  // Enable the AMS-Filter UDFs as an auto extension within SQLite.
  amsfilter::model::init_udfs();

  if (argc < 2) {
    // Open the default DB.
    const auto perf_db_filename = get_default_filename();
    const char* arguments[2] {argv[0], perf_db_filename.c_str()};
    main_shell(2, (char**)arguments);
  }
  else {
    // Open the SQLite shell.
    main_shell(argc, argv);
  }
}
//===----------------------------------------------------------------------===//
