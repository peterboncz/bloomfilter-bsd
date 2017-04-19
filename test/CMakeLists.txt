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
        #vec_test.cpp
        )

add_executable(dtl_tester ${TEST_FILES})
target_link_libraries(dtl_tester gtest gtest_main pthread numa)


set(STORAGE_TEST_FILES
        bitmask_test.cpp
        bitmask_experiments.cpp
        )

add_executable(storage_tester ${STORAGE_TEST_FILES})
target_link_libraries(storage_tester gtest gtest_main pthread numa)


set(BLOOMFILTER_EXPERIMENTS_FILES
        bloomfilter_experiments.cpp
        )

add_executable(bloomfilter_experiments ${BLOOMFILTER_EXPERIMENTS_FILES})
target_link_libraries(bloomfilter_experiments benchmark gtest gtest_main pthread numa)


set(COMPRESSION_TEST_FILES
        bitpack_test.cpp
        )

add_executable(bitpack_tester ${COMPRESSION_TEST_FILES})
target_link_libraries(bitpack_tester gtest gtest_main pthread numa)
