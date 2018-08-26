cmake_minimum_required(VERSION 3.5)
project(dtl)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

#include_directories(SYSTEM ../)
include_directories(src/)
include_directories(thirdparty/)

add_subdirectory(src/dtl/simd)


### build optimized executables for the following architectures
set(ARCHS
        corei7
        core-avx2
        knl
        skx
        )

# replacement for 'add_library'
function(add_multiarch_library library_name files)
    foreach(arch ${ARCHS})
        add_library(${library_name}_${arch} ${files})
        # workaround for older compilers which don't know the 'skx' architecture.
        if (arch STREQUAL "skx")
            set_target_properties(${library_name}_${arch} PROPERTIES COMPILE_FLAGS "-march=knl -mavx512bw")
        else()
            set_target_properties(${library_name}_${arch} PROPERTIES COMPILE_FLAGS "-march=${arch}")
        endif()
        target_include_directories(${library_name}_${arch} PUBLIC src thirdparty)
    endforeach(arch ${ARCHS})
endfunction(add_multiarch_library)


set(CMAKE_VERBOSE_MAKEFILE ON)

set(CUCKOO_FILTER_SOURCE_FILES
        # cuckoo
        src/dtl/filter/cuckoofilter/bitsutil.hpp
        src/dtl/filter/cuckoofilter/cuckoofilter.hpp
        src/dtl/filter/cuckoofilter/cuckoofilter_config.hpp
        src/dtl/filter/cuckoofilter/cuckoofilter_logic.hpp
        src/dtl/filter/cuckoofilter/cuckoofilter_table.hpp
        src/dtl/filter/cuckoofilter/cuckoofilter_table_scalar.hpp
        src/dtl/filter/cuckoofilter/cuckoofilter_table_scalar_cuda.hpp
        src/dtl/filter/cuckoofilter/cuckoofilter_table_simd.hpp
        src/dtl/filter/cuckoofilter/cuckoofilter_table_simd_avx2.hpp
        src/dtl/filter/cuckoofilter/cuckoofilter_table_simd_avx512f.hpp
        src/dtl/filter/cuckoofilter/cuckoofilter_tune.hpp
        src/dtl/filter/cuckoofilter/cuckoofilter_tune_impl.hpp

        # blocked cuckoo (EXPERIMENTAL)
        src/dtl/filter/blocked_cuckoofilter/blocked_cuckoofilter.hpp
        src/dtl/filter/blocked_cuckoofilter/blocked_cuckoofilter_logic.hpp
        src/dtl/filter/blocked_cuckoofilter/blocked_cuckoofilter_block_logic.hpp
#       src/dtl/filter/blocked_cuckoofilter_concurrent_multiword_table.hpp
        src/dtl/filter/blocked_cuckoofilter/blocked_cuckoofilter_multiword_table.hpp
        src/dtl/filter/blocked_cuckoofilter/blocked_cuckoofilter_simd.hpp
        src/dtl/filter/blocked_cuckoofilter/blocked_cuckoofilter_util.hpp
        src/dtl/filter/blocked_cuckoofilter/blocked_cuckoofilter_word_table.hpp

        # blocked cuckoo filter (pimpl wrapper, EXPERIMENTAL)
        src/dtl/filter/bcf.hpp
        src/dtl/filter/bcf.cpp
        # cuckoo filter (pimpl wrapper)
        src/dtl/filter/cf.hpp
        src/dtl/filter/cf.cpp
        )


set(BLOOM_FILTER_SOURCE_FILES
        # classic bloom
        src/dtl/filter/bloomfilter/bloomfilter_logic.hpp

        # blocked bloom
        src/dtl/filter/blocked_bloomfilter/blocked_bloomfilter.hpp
        src/dtl/filter/blocked_bloomfilter/blocked_bloomfilter.hpp
        src/dtl/filter/blocked_bloomfilter/blocked_bloomfilter_batch_dispatch.hpp
        src/dtl/filter/blocked_bloomfilter/blocked_bloomfilter_logic.hpp
        src/dtl/filter/blocked_bloomfilter/blocked_bloomfilter_block_logic.hpp
        src/dtl/filter/blocked_bloomfilter/blocked_bloomfilter_block_logic_sgew.hpp
        src/dtl/filter/blocked_bloomfilter/blocked_bloomfilter_block_logic_sgew_specialization.hpp
        src/dtl/filter/blocked_bloomfilter/blocked_bloomfilter_block_logic_sltw.hpp
        src/dtl/filter/blocked_bloomfilter/blocked_bloomfilter_block_logic_zoned.hpp
        src/dtl/filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp
        src/dtl/filter/blocked_bloomfilter/blocked_bloomfilter_tune.hpp
        src/dtl/filter/blocked_bloomfilter/blocked_bloomfilter_tune_impl.hpp

        # zoned bloom (aka cache-sectorized) # TODO merge into blocked_bloom
        src/dtl/filter/blocked_bloomfilter/zoned_blocked_bloomfilter.hpp
        src/dtl/filter/blocked_bloomfilter/zoned_blocked_bloomfilter_tune_impl.hpp

        # runtime version of blocked bloom filter (EXPERIMENTAL)
        src/dtl/filter/dynamic_blocked_bloomfilter/block_addressing_logic.hpp
        src/dtl/filter/dynamic_blocked_bloomfilter/blocked_bloomfilter.hpp
        src/dtl/filter/dynamic_blocked_bloomfilter/blocked_bloomfilter_logic.hpp
        src/dtl/filter/dynamic_blocked_bloomfilter/blocked_bloomfilter_block_logic.hpp
        src/dtl/filter/dynamic_blocked_bloomfilter/hash.hpp

        # utils
        src/dtl/filter/blocked_bloomfilter/fpr.hpp
        src/dtl/filter/blocked_bloomfilter/block_addressing_logic.hpp
        src/dtl/filter/blocked_bloomfilter/vector_helper.hpp
        src/dtl/filter/blocked_bloomfilter/hash_family.hpp
        src/dtl/filter/blocked_bloomfilter/math.hpp

        # model
        src/dtl/filter/model/benchmark.hpp
        src/dtl/filter/model/benchmark.cpp
        src/dtl/filter/model/calibration.hpp
        src/dtl/filter/model/calibration.cpp
        src/dtl/filter/model/calibration_data.hpp
        src/dtl/filter/model/calibration_data.cpp
        src/dtl/filter/model/cost_fn.hpp
        src/dtl/filter/model/optimizer.hpp
        src/dtl/filter/model/optimizer.cpp
        src/dtl/filter/model/skyline_matrix.hpp
        src/dtl/filter/model/timing.hpp
        src/dtl/filter/model/tuning_params.hpp
        src/dtl/filter/model/util.hpp
        src/dtl/filter/platform.hpp
        src/dtl/filter/platform.cpp

        # blocked bloom (pimpl wrapper)
        src/dtl/filter/bbf_32.hpp
        src/dtl/filter/bbf_32.cpp
        src/dtl/filter/bbf_64.hpp
        src/dtl/filter/bbf_64.cpp
        # blocked bloom (extern templates to parallelize build)
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_instance.inc
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u32_instance.hpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u32_w1.hpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u32_w2.hpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u32_w4.hpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u32_w8.hpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u32_w16.hpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u32_w1.cpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u32_w2.cpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u32_w4.cpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u32_w8.cpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u32_w16.cpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u64_instance.hpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u64_w1.hpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u64_w2.hpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u64_w4.hpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u64_w8.hpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u64_w16.hpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u64_w1.cpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u64_w2.cpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u64_w4.cpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u64_w8.cpp
        src/dtl/filter/blocked_bloomfilter/instances/blocked_bloomfilter_logic_u64_w16.cpp

        # cache-sectorized (pimpl wrapper)
        src/dtl/filter/zbbf_32.hpp
        src/dtl/filter/zbbf_32.cpp
        src/dtl/filter/zbbf_64.hpp
        src/dtl/filter/zbbf_64.cpp
        # cache-sectorized (extern templates to parallelize build)
        src/dtl/filter/blocked_bloomfilter/instances/zoned_blocked_bloomfilter_logic_instance.inc
        src/dtl/filter/blocked_bloomfilter/instances/zoned_blocked_bloomfilter_logic_u32_instance.hpp
        src/dtl/filter/blocked_bloomfilter/instances/zoned_blocked_bloomfilter_logic_u32_w4.hpp
        src/dtl/filter/blocked_bloomfilter/instances/zoned_blocked_bloomfilter_logic_u32_w8.hpp
        src/dtl/filter/blocked_bloomfilter/instances/zoned_blocked_bloomfilter_logic_u32_w16.hpp
        src/dtl/filter/blocked_bloomfilter/instances/zoned_blocked_bloomfilter_logic_u32_w4.cpp
        src/dtl/filter/blocked_bloomfilter/instances/zoned_blocked_bloomfilter_logic_u32_w8.cpp
        src/dtl/filter/blocked_bloomfilter/instances/zoned_blocked_bloomfilter_logic_u32_w16.cpp
        src/dtl/filter/blocked_bloomfilter/instances/zoned_blocked_bloomfilter_logic_u64_instance.hpp
        src/dtl/filter/blocked_bloomfilter/instances/zoned_blocked_bloomfilter_logic_u64_w4.hpp
        src/dtl/filter/blocked_bloomfilter/instances/zoned_blocked_bloomfilter_logic_u64_w8.hpp
        src/dtl/filter/blocked_bloomfilter/instances/zoned_blocked_bloomfilter_logic_u64_w16.hpp
        src/dtl/filter/blocked_bloomfilter/instances/zoned_blocked_bloomfilter_logic_u64_w4.cpp
        src/dtl/filter/blocked_bloomfilter/instances/zoned_blocked_bloomfilter_logic_u64_w8.cpp
        src/dtl/filter/blocked_bloomfilter/instances/zoned_blocked_bloomfilter_logic_u64_w16.cpp
        )

set(FILTER_SOURCE_FILES
        ${BLOOM_FILTER_SOURCE_FILES}
        ${CUCKOO_FILTER_SOURCE_FILES}
        src/dtl/filter/filter.hpp
        src/dtl/filter/filter_base.hpp
        )

set(THIRDPARTY_SOURCE_FILES
        # third party
        thirdparty/libdivide/libdivide.h
        )


set(SOURCE_FILES ${SOURCE_FILES}
        ${FILTER_SOURCE_FILES}
        src/dtl/adept.hpp
        src/dtl/barrier.hpp
        src/dtl/batchwise.hpp
        src/dtl/bits.hpp
        src/dtl/bitset.hpp
        src/dtl/bitset_util.hpp
        src/dtl/color.hpp
        src/dtl/div.hpp
        src/dtl/dtl.hpp
        src/dtl/env.hpp
        src/dtl/hash.hpp
        src/dtl/lockstripe.hpp
        src/dtl/math.hpp
        src/dtl/mem.hpp
        src/dtl/polyfill.hpp
        src/dtl/simd.hpp
        src/dtl/thread.hpp
        src/dtl/type_traits.hpp
        )

add_multiarch_library(dtl "${SOURCE_FILES}")

set(LIBS
        ssl
        crypto
        pthread
        numa
        dl
        )

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



set(CALIBRATION_TOOL_FILES ${SOURCE_FILES}
        tool/calibration/src/filter-calibration-tool.cpp
        tool/calibration/src/skyline_matrix_builder.hpp
        tool/calibration/src/skyline_matrix_builder.cpp
        tool/calibration/src/util.hpp
        )
add_multiarch_executable(filter-calibration-tool "${CALIBRATION_TOOL_FILES}")