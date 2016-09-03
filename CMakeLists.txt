cmake_minimum_required(VERSION 3.5)
project(adept)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native")

set(SOURCE_FILES main.cpp)
add_executable(adept ${SOURCE_FILES})

add_subdirectory(test)