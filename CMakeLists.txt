cmake_minimum_required(VERSION 3.5)
project(dtl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14 -march=native")

#include_directories(SYSTEM ../)
include_directories(src/)
include_directories(thirdparty/)

add_subdirectory(src/dtl/simd)
#add_subdirectory(storage) #TODO remove

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
        src/dtl/filter/model/calibration_data.hpp
        src/dtl/filter/model/calibration_data.cpp
        src/dtl/filter/model/timing.hpp
        src/dtl/filter/model/tuning_params.hpp
        )


set(THIRDPARTY_SOURCE_FILES
        # third party
        thirdparty/libdivide/libdivide.h
        )


set(SOURCE_FILES ${SOURCE_FILES}
        ${BLOOM_FILTER_SOURCE_FILES}
        ${CUCKOO_FILTER_SOURCE_FILES}
        src/dtl/adept.hpp #TODO remove
        src/dtl/barrier.hpp
        src/dtl/batchwise.hpp
        src/dtl/bits.hpp
        src/dtl/bitmask.hpp
        src/dtl/bitpack.hpp
        src/dtl/bitset.hpp
        src/dtl/bitset_util.hpp
        src/dtl/color.hpp
        src/dtl/dict.hpp
        src/dtl/div.hpp
        src/dtl/dtl.hpp
        src/dtl/env.hpp
        src/dtl/hash.hpp
        src/dtl/index.hpp
        src/dtl/like.hpp
        src/dtl/lockstripe.hpp
        src/dtl/math.hpp
        src/dtl/mem.hpp
        src/dtl/polyfill.hpp
        src/dtl/psma.hpp
        src/dtl/simd.hpp
        src/dtl/sma.hpp
        src/dtl/storage.hpp
        src/dtl/thread.hpp
        src/dtl/tree.hpp
        src/dtl/tree_mask.hpp
        src/dtl/zone_mask.hpp
        )
add_library(dtl STATIC ${SOURCE_FILES})
target_include_directories(dtl PUBLIC src thirdparty)