cmake_minimum_required(VERSION 3.5)
project(dtl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14 -march=native")

include_directories(SYSTEM ../)

add_subdirectory(simd)
#add_subdirectory(storage)

set(SOURCE_FILES ${SOURCE_FILES}
        main.cpp
        adept.hpp
        barrier.hpp
        batchwise.hpp
        bits.hpp
        filter/dynamic/block_addressing_logic.hpp
        filter/dynamic/blocked_bloomfilter.hpp
        filter/dynamic/blocked_bloomfilter_logic.hpp
        filter/dynamic/blocked_bloomfilter_block_logic.hpp
        filter/dynamic/hash.hpp
        filter/bloomfilter_logic.hpp
        filter/blocked_bloomfilter.hpp
        filter/vector_helper.hpp
        filter/block_addressing_logic.hpp
        filter/blocked_bloomfilter.hpp
        filter/blocked_bloomfilter_batch_dispatch.hpp
        filter/fpr.hpp
        filter/blocked_bloomfilter_logic.hpp
        filter/blocked_bloomfilter_block_logic.hpp
        filter/blocked_bloomfilter_block_logic_sgew.hpp
        filter/blocked_bloomfilter_block_logic_sltw.hpp
        filter/blocked_bloomfilter_block_logic_zoned.hpp
        filter/blocked_bloomfilter_config.hpp
        filter/blocked_bloomfilter_tune.hpp
        filter/blocked_bloomfilter_tune_impl.hpp
        filter/zoned_blocked_bloomfilter.hpp
        filter/zoned_blocked_bloomfilter_tune_impl.hpp
        filter/hash_family.hpp
        filter/blocked_cuckoofilter.hpp
        filter/blocked_cuckoofilter_logic.hpp
        filter/blocked_cuckoofilter_block_logic.hpp
#        filter/blocked_cuckoofilter_concurrent_multiword_table.hpp
        filter/blocked_cuckoofilter_multiword_table.hpp
        filter/blocked_cuckoofilter_simd.hpp
        filter/blocked_cuckoofilter_util.hpp
        filter/blocked_cuckoofilter_word_table.hpp
        filter/math.hpp
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
#        compression/lz4.h
#        compression/lz4.c
#        compression/lz4hc.h
#        compression/lz4hc.c
        # third party
        thirdparty/libdivide/libdivide.h
        )

add_library(dtl ${SOURCE_FILES})
target_include_directories(dtl PUBLIC ..)
