cmake_minimum_required(VERSION 3.5)
project(amsfilter)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
#set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fext-numeric-literals -Wall -Wno-deprecated -Wno-endif-labels -Wno-unused-function -DSQLITE_ENABLE_JSON1 -DSQLITE_OMIT_LOAD_EXTENSION")

# Include the Database Template Library (DTL) - headers only.
include_directories(../src)

include_directories(../thirdparty)

include_directories(./)

include_directories(thirdparty/)

set(LIBS
#        ssl
#        crypto
        pthread
        numa
        dl
        )


#===----------------------------------------------------------------------===#
# CUDA
#===----------------------------------------------------------------------===#
find_package(CUDA 9)

if (CUDA_FOUND)
    message("CUDA found")

    # avoid implicitly synchronization with other streams
    set(CUDA_NVCC_FLAGS --default-stream per-thread)
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --default-stream per-thread)

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --restrict --std=c++14  -Xcompiler -D__CORRECT_ISO_CPP11_MATH_H_PROTO -D_MWAITXINTRIN_H_INCLUDED)
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -march=native)

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --source-in-ptx)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xptxas -dlcm=ca)

    # enable NVCC warnings
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -Wall)

    # the target architecture(s)  - (see https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --generate-code arch=compute_52,code=sm_52) # e.g. GTX 970
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --generate-code arch=compute_60,code=sm_60) # e.g. Tesla P100
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --generate-code arch=compute_61,code=sm_61) # e.g. GTX 1080Ti

    # don't show deprecation warnings (which are a lot with CUDA 9)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-declarations)

    set(CUDA_NVCC_FLAGS_DEBUG -g --generate-line-info --keep -DDEBUG)
    set(CUDA_NVCC_FLAGS_RELEASE -O3 --use_fast_math -DNDEBUG)


    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_SEPARABLE_COMPILATION OFF)

    include_directories(/usr/local/cuda/include)

    set(LIBS ${LIBS} cudart)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_CUDA")


else()
    message("CUDA found")
endif()

find_package(CUB)
if (CUB_FOUND)
    message("including CUB: ${CUB_INCLUDE_DIR}")
    include_directories(${CUB_INCLUDE_DIR})
endif()

include_directories(src)

# For debugging purposes.
if(AMSFILTER_PARTIAL_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAMSFILTER_PARTIAL_BUILD")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DAMSFILTER_PARTIAL_BUILD")
endif()

if(AMSFILTER_NO_MAGIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAMSFILTER_NO_MAGIC")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DAMSFILTER_NO_MAGIC")
endif()


SET(SOURCE_FILES
        src/amsfilter/internal/blocked_bloomfilter_resolve.hpp
        src/amsfilter/internal/blocked_bloomfilter_template.hpp
        src/amsfilter/internal/buffer.hpp
        src/amsfilter/internal/probe_impl.hpp
        src/amsfilter/amsfilter.cpp
        src/amsfilter/amsfilter.hpp
        src/amsfilter/amsfilter_lite.hpp
        src/amsfilter/bitmap_view.hpp
        src/amsfilter/common.hpp
        src/amsfilter/probe.cpp
        src/amsfilter/probe.hpp
        src/amsfilter/probe_lite.hpp
        )

SET(SOURCE_FILES_CUDA
        src/amsfilter/cuda/probe.cu
        src/amsfilter/cuda/probe.hpp
        src/amsfilter/cuda/probe_lite.hpp
        src/amsfilter/cuda/internal/alloc.hpp
        src/amsfilter/cuda/internal/block_gather.cuh
        src/amsfilter/cuda/internal/cuda_api_helper.hpp
        src/amsfilter/cuda/internal/cuda_helper.cuh
        src/amsfilter/cuda/internal/kernel.cuh
        src/amsfilter/cuda/internal/replicas.hpp
        src/amsfilter/cuda/internal/probe_impl.cuh
        )

SET(AMSFILTER_SOURCE_FILES
        ${SOURCE_FILES})
if (CUDA_FOUND)
    SET(AMSFILTER_SOURCE_FILES
            ${AMSFILTER_SOURCE_FILES}
            ${SOURCE_FILES_CUDA})
    cuda_add_library(amsfilter STATIC ${AMSFILTER_SOURCE_FILES})
    set_target_properties(amsfilter
            PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET amsfilter PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_target_properties(amsfilter PROPERTIES COMPILE_FLAGS "-march=native")
else()
    add_library(amsfilter STATIC ${AMSFILTER_SOURCE_FILES})
    set_target_properties(amsfilter PROPERTIES COMPILE_FLAGS "-march=native")
endif()
