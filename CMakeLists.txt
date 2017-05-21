cmake_minimum_required(VERSION 3.5)
project(dtl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14 -march=native")

include_directories(SYTEM ../)

add_subdirectory(simd)
add_subdirectory(storage)

set(SOURCE_FILES ${SOURCE_FILES}
        main.cpp
        adept.hpp
        bits.hpp
        bloomfilter.hpp
        bloomfilter2.hpp
        bloomfilter_vec.hpp
        bloomfilter2_vec.hpp
        bitmask.hpp
        bitpack.hpp
        bitset.hpp
        bitset_util.hpp
        color.hpp
        dict.hpp
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
        )


add_library(dtl ${SOURCE_FILES})
target_include_directories(dtl PUBLIC . )
