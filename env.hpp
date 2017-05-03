#pragma once

#include <dtl/dtl.hpp>

namespace dtl {


template<typename T>
struct env {};

template<>
struct env<std::string> {

  static std::string
  get(const std::string name, const std::string default_value = "") {
    std::string value = default_value;
    if (const char* env = std::getenv(name.c_str())) {
      value = std::string(env);
    }
    return value;
  }

};

template<>
struct env<$i32> {

  static $i32
  get(const std::string name, const $i32 default_value = 0) {
    $i32 value = default_value;
    if (const char* env = std::getenv(name.c_str())) {
      value = std::stoi(env);
    }
    return value;
  }

};

} // namespace dtl