cmake_minimum_required(VERSION 3.5)
project(dtl_tester)

set(DEFAULT_ARCH -march=core-avx2)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14")

# build optimized executables for the following architectures
set(ARCHS
        core-avx2
        knl
        )

# replacement for 'add_executable'
function(add_multiarch_executable executable_name files)
    foreach(arch ${ARCHS})
        add_executable(${executable_name}_${arch} ${files})
        set_target_properties(${executable_name}_${arch} PROPERTIES COMPILE_FLAGS "-march=${arch}")
        target_link_libraries(${executable_name}_${arch} gtest gtest_main pthread numa)
    endforeach(arch ${ARCHS})
endfunction(add_multiarch_executable)

# note for ubuntu users: gtest lib is no longer shipped as dep package (see http://askubuntu.com/questions/145887/why-no-library-files-installed-for-google-test)
add_subdirectory(lib/googletest)
include_directories(lib/googletest/include lib/googletest)

add_subdirectory(../ dtl)
include_directories(SYTEM ../../)


set(TEST_FILES
        bitmask_test.cpp
        bitpack_test.cpp
        bloomfilter_test.cpp
        mem_test.cpp
        vec_test.cpp
        )
add_multiarch_executable(dtl_tester ${TEST_FILES})


set(VECTOR_TEST_FILES
        vec_test.cpp
        )
add_multiarch_executable(vector_tester ${VECTOR_TEST_FILES})


set(STORAGE_TEST_FILES
        bitmask_test.cpp
        bitmask_experiments.cpp
        )
add_multiarch_executable(storage_tester ${STORAGE_TEST_FILES})


set(BLOOMFILTER_EXPERIMENTS_FILES
        bloomfilter_experiments.cpp
        )
add_multiarch_executable(bloomfilter_experiments ${BLOOMFILTER_EXPERIMENTS_FILES})


set(BLOOMFILTER_TEST_FILES
        bloomfilter_test.cpp
        )
add_multiarch_executable(bloomfilter_tester ${BLOOMFILTER_TEST_FILES})


set(COMPRESSION_TEST_FILES
        bitpack_test.cpp
        )
add_multiarch_executable(bitpack_tester ${COMPRESSION_TEST_FILES})


#set(THREAD_TEST_FILES
#        thread_test.cpp
#        )
#add_multiarch_executable(thread_tester ${THREAD_TEST_FILES})


set(BITSET_TEST_FILES
        bitset_test.cpp
        )
add_multiarch_executable(bitset_tester ${BITSET_TEST_FILES})
