cmake_minimum_required(VERSION 3.5)
project(dtl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14 -march=native")

include_directories(SYSTEM ../)

add_subdirectory(simd)
#add_subdirectory(storage) #TODO remove

set(CUCKOO_FILTER_SOURCE_FILES
        # cuckoo
        filter/cuckoofilter/bitsutil.hpp
        filter/cuckoofilter/cuckoofilter.hpp
        filter/cuckoofilter/cuckoofilter_config.hpp
        filter/cuckoofilter/cuckoofilter_logic.hpp
        filter/cuckoofilter/cuckoofilter_table.hpp
        filter/cuckoofilter/cuckoofilter_table_scalar.hpp
        filter/cuckoofilter/cuckoofilter_table_scalar_cuda.hpp
        filter/cuckoofilter/cuckoofilter_table_simd.hpp
        filter/cuckoofilter/cuckoofilter_table_simd_avx2.hpp
        filter/cuckoofilter/cuckoofilter_table_simd_avx512f.hpp
        filter/cuckoofilter/cuckoofilter_tune.hpp
        filter/cuckoofilter/cuckoofilter_tune_impl.hpp

        # blocked cuckoo (EXPERIMENTAL)
        filter/blocked_cuckoofilter/blocked_cuckoofilter.hpp
        filter/blocked_cuckoofilter/blocked_cuckoofilter_logic.hpp
        filter/blocked_cuckoofilter/blocked_cuckoofilter_block_logic.hpp
#       filter/blocked_cuckoofilter_concurrent_multiword_table.hpp
        filter/blocked_cuckoofilter/blocked_cuckoofilter_multiword_table.hpp
        filter/blocked_cuckoofilter/blocked_cuckoofilter_simd.hpp
        filter/blocked_cuckoofilter/blocked_cuckoofilter_util.hpp
        filter/blocked_cuckoofilter/blocked_cuckoofilter_word_table.hpp
        )


set(BLOOM_FILTER_SOURCE_FILES
        # classic bloom
        filter/bloomfilter/bloomfilter_logic.hpp

        # blocked bloom
        filter/blocked_bloomfilter/blocked_bloomfilter.hpp
        filter/blocked_bloomfilter/blocked_bloomfilter.hpp
        filter/blocked_bloomfilter/blocked_bloomfilter_batch_dispatch.hpp
        filter/blocked_bloomfilter/blocked_bloomfilter_logic.hpp
        filter/blocked_bloomfilter/blocked_bloomfilter_block_logic.hpp
        filter/blocked_bloomfilter/blocked_bloomfilter_block_logic_sgew.hpp
        filter/blocked_bloomfilter/blocked_bloomfilter_block_logic_sltw.hpp
        filter/blocked_bloomfilter/blocked_bloomfilter_block_logic_zoned.hpp
        filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp
        filter/blocked_bloomfilter/blocked_bloomfilter_tune.hpp
        filter/blocked_bloomfilter/blocked_bloomfilter_tune_impl.hpp

        # zoned bloom (aka cache-sectorized) # TODO merge into blocked_bloom
        filter/blocked_bloomfilter/zoned_blocked_bloomfilter.hpp
        filter/blocked_bloomfilter/zoned_blocked_bloomfilter_tune_impl.hpp

        # runtime version of blocked bloom filter (EXPERIMENTAL)
        filter/dynamic/block_addressing_logic.hpp
        filter/dynamic/blocked_bloomfilter.hpp
        filter/dynamic/blocked_bloomfilter_logic.hpp
        filter/dynamic/blocked_bloomfilter_block_logic.hpp
        filter/dynamic/hash.hpp

        # utils
        filter/blocked_bloomfilter/fpr.hpp
        filter/blocked_bloomfilter/block_addressing_logic.hpp
        filter/blocked_bloomfilter/vector_helper.hpp
        filter/blocked_bloomfilter/hash_family.hpp
        filter/blocked_bloomfilter/math.hpp
        )


set(THIRDPARTY_SOURCE_FILES
        # third party
        thirdparty/libdivide/libdivide.h
        )


set(SOURCE_FILES
        ${BLOOM_FILTER_SOURCE_FILES}
        ${CUCKOO_FILTER_SOURCE_FILES}
        main.cpp
        adept.hpp #TODO remove
        barrier.hpp
        batchwise.hpp
        bits.hpp
        bitmask.hpp
        bitpack.hpp
        bitset.hpp
        bitset_util.hpp
        color.hpp
        dict.hpp
        div.hpp
        dtl.hpp
        env.hpp
        hash.hpp
        index.hpp
        like.hpp
        lockstripe.hpp
        math.hpp
        mem.hpp
        polyfill.hpp
        psma.hpp
        simd.hpp
        sma.hpp
        storage.hpp
        #old/vec.hpp
        thread.hpp
        tree.hpp
        tree_mask.hpp
        zone_mask.hpp
        )
add_library(dtl ${SOURCE_FILES})
target_include_directories(dtl PUBLIC ..)
