cmake_minimum_required(VERSION 3.5)
project(dtl_tester)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14")

# build optimized executables for the following architectures
set(ARCHS
        core-avx2
        knl
        skx
        corei7
        )


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(LIBS
        gtest
        gtest_main
        pthread
        numa
#        dtl
        )

# check for NUMA support
#find_package(NUMA)
#if (NUMA_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_NUMA")
    set(LIBS ${LIBS} numa)
#endif (NUMA_FOUND)

# check for CUDA support
find_package(CUDA)
if (CUDA_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_CUDA")
    # TODO add CUDA libs
endif (CUDA_FOUND)


# replacement for 'add_executable'
function(add_multiarch_executable executable_name files)
    foreach(arch ${ARCHS})
        add_executable(${executable_name}_${arch} ${files})
        # workaround for older compilers which don't know the 'skx' architecture.
        if (arch STREQUAL "skx")
            set_target_properties(${executable_name}_${arch} PROPERTIES COMPILE_FLAGS "-march=knl -mavx512bw")
        else()
            set_target_properties(${executable_name}_${arch} PROPERTIES COMPILE_FLAGS "-march=${arch}")
        endif()
        target_link_libraries(${executable_name}_${arch} ${LIBS} dtl_${arch})
    endforeach(arch ${ARCHS})
endfunction(add_multiarch_executable)

# note for ubuntu users: gtest lib is no longer shipped as dep package (see http://askubuntu.com/questions/145887/why-no-library-files-installed-for-google-test)
add_subdirectory(lib/googletest)
include_directories(lib/googletest/include lib/googletest)

add_subdirectory(../ dtl)
include_directories(SYTEM ../thirdparty/)
include_directories(SYTEM ../../) # TODO remove


set(TEST_FILES
        bitset_test.cpp
        div_test.cpp
        mem_test.cpp
        vec_test.cpp
        cuckoofilter_test.cpp
        ../src/dtl/filter/filter_test.cpp
        ../src/dtl/filter/platform_test.cpp
        ../src/dtl/filter/model/benchmark_test.cpp
#        ../src/dtl/filter/model/calibration_test.cpp
        ../src/dtl/filter/model/calibration_data_test.cpp
        ../src/dtl/filter/model/optimizer_test.cpp

        )
add_multiarch_executable(dtl_tester "${TEST_FILES}")
