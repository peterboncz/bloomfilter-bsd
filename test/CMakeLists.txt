cmake_minimum_required(VERSION 3.5)
project(dtl_tester)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14 -march=core-avx2")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14 -march=knl")

# note for ubuntu users: gtest lib is no longer shipped as dep package (see http://askubuntu.com/questions/145887/why-no-library-files-installed-for-google-test)
add_subdirectory(lib/googletest)
include_directories(lib/googletest/include lib/googletest)

add_subdirectory(../ dtl)
include_directories(SYTEM ../../)

set(TEST_FILES
        bitmask_test.cpp
        bitmask_experiments.cpp
        bloomfilter_experiments.cpp
        #knl_experiments.cpp
        mem_test.cpp
#        vec_test.cpp
        )

set(STORAGE_TEST_FILES
        bitmask_test.cpp
        bitmask_experiments.cpp
        bloomfilter_experiments.cpp
        )

add_executable(dtl_tester ${TEST_FILES})
add_executable(dtl_storage_tester ${STORAGE_TEST_FILES})

target_link_libraries(dtl_tester gtest gtest_main pthread numa)
target_link_libraries(dtl_storage_tester gtest gtest_main pthread numa)

add_executable(bloomfilter bloomfilter_experiments.cpp)
target_link_libraries(bloomfilter benchmark gtest gtest_main pthread numa)
