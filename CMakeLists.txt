cmake_minimum_required(VERSION 3.5)
project(dtl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14 -march=native")

add_subdirectory(simd)
set(SOURCE_FILES ${SOURCE_FILES}
        main.cpp
        adept.hpp
        bloomfilter.hpp
        bitmask.hpp
        color.hpp
        dict.hpp
        hash.hpp
        index.hpp
        math.hpp
        mem.hpp
        psma.hpp
        simd.hpp
        sma.hpp
        #old/vec.hpp
        thread.hpp
        tree.hpp
        tree_mask.hpp
        zone_mask.hpp
        )

add_library(dtl ${SOURCE_FILES})

target_include_directories(dtl PUBLIC . )

#add_subdirectory(test)