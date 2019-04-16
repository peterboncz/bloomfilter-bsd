cmake_minimum_required(VERSION 3.5)
project(amsfilter)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
#set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fext-numeric-literals -Wall -Wno-deprecated -Wno-endif-labels -Wno-unused-function")

# Enable ASAN in debug builds.
#set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
#set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")


# Include the Database Template Library (DTL) - headers only.
include_directories(../src)

include_directories(../thirdparty)

include_directories(./)

include_directories(thirdparty/)

set(LIBS
        pthread
        numa
        dl
        )

#===------------------------------------------------------------------------===#
# SQLite is used to store the calibration data to disk.
set(SQLITE_SOURCE_FILES
        thirdparty/sqlite/sqlite3.c
        )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSQLITE_THREADSAFE=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSQLITE_ENABLE_JSON1")
#===------------------------------------------------------------------------===#


#===------------------------------------------------------------------------===#
# CUDA
#===------------------------------------------------------------------------===#
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

#===------------------------------------------------------------------------===#
# The AMS-Filter library.
#===------------------------------------------------------------------------===#
SET(SOURCE_FILES
        # AMS-Filter
        src/amsfilter/amsfilter.cpp
        src/amsfilter/amsfilter.hpp
        src/amsfilter/amsfilter_lite.hpp
        src/amsfilter/bitmap_view.hpp
        src/amsfilter/common.cpp
        src/amsfilter/common.hpp
        src/amsfilter/config.cpp
        src/amsfilter/config.hpp
        src/amsfilter/probe.cpp
        src/amsfilter/probe.hpp
        src/amsfilter/probe_lite.hpp
        src/amsfilter/tuning_params.hpp
        src/amsfilter/internal/probe_instances/probe_w1.cpp
        src/amsfilter/internal/probe_instances/probe_w2.cpp
        src/amsfilter/internal/probe_instances/probe_w4.cpp
        src/amsfilter/internal/probe_instances/probe_w8.cpp
        src/amsfilter/internal/probe_instances/probe_w16.cpp
        src/amsfilter/internal/probe_instances/probe_w32.cpp
        src/amsfilter/internal/cuckoofilter_resolve.hpp
        src/amsfilter/internal/cuckoofilter_template.hpp
        src/amsfilter/internal/blocked_bloomfilter_resolve.hpp
        src/amsfilter/internal/blocked_bloomfilter_template.hpp
        src/amsfilter/internal/filter_resolve.hpp
        src/amsfilter/internal/filter_template.hpp
        src/amsfilter/internal/buffer.hpp
        src/amsfilter/internal/probe_impl.hpp

        # AMS-Filter Model
        src/amsfilter_model/internal/benchmark.hpp
        src/amsfilter_model/internal/benchmark_cpu.cpp
        src/amsfilter_model/internal/benchmark_cpu.hpp
        src/amsfilter_model/internal/common.hpp
        src/amsfilter_model/internal/cost_fn.hpp
        src/amsfilter_model/internal/perf_db.cpp
        src/amsfilter_model/internal/perf_db.hpp
        src/amsfilter_model/internal/platform.cpp
        src/amsfilter_model/internal/platform.hpp
        src/amsfilter_model/internal/skyline.cpp
        src/amsfilter_model/internal/skyline.hpp
        src/amsfilter_model/internal/skyline_entry.hpp
        src/amsfilter_model/internal/timing.hpp
        src/amsfilter_model/internal/tune.cpp
        src/amsfilter_model/internal/tune.hpp
        src/amsfilter_model/internal/udf.hpp
        src/amsfilter_model/internal/util.hpp
        src/amsfilter_model/execution_env.hpp
        src/amsfilter_model/fpr.cpp
        src/amsfilter_model/fpr.hpp
        src/amsfilter_model/fpr_lut.hpp
        src/amsfilter_model/model.cpp
        src/amsfilter_model/model.hpp
        )

SET(SOURCE_FILES_CUDA
        # AMS-Filter
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
        src/amsfilter/cuda/internal/kernel_instances/kernel_w1.cu
        src/amsfilter/cuda/internal/kernel_instances/kernel_w2.cu
        src/amsfilter/cuda/internal/kernel_instances/kernel_w4.cu
        src/amsfilter/cuda/internal/kernel_instances/kernel_w8.cu
        src/amsfilter/cuda/internal/kernel_instances/kernel_w16.cu
        src/amsfilter/cuda/internal/kernel_instances/kernel_w32.cu

        # AMS-Filter Model
        src/amsfilter_model/internal/benchmark_gpu.cu
        src/amsfilter_model/internal/benchmark_gpu.cuh
        src/amsfilter_model/internal/benchmark_pci_bw.cu
        src/amsfilter_model/internal/benchmark_pci_bw.cuh
        )

SET(AMSFILTER_SOURCE_FILES
        ${SOURCE_FILES}
        ${SQLITE_SOURCE_FILES}
        )

# Faster builds, for development and debugging purposes.
if(AMSFILTER_PARTIAL_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAMSFILTER_PARTIAL_BUILD")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DAMSFILTER_PARTIAL_BUILD")
endif()

if(AMSFILTER_NO_MAGIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAMSFILTER_NO_MAGIC")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DAMSFILTER_NO_MAGIC")
endif()

if (CUDA_FOUND)
    SET(AMSFILTER_SOURCE_FILES
            ${AMSFILTER_SOURCE_FILES}
            ${SOURCE_FILES_CUDA})
    cuda_add_library(amsfilter STATIC ${AMSFILTER_SOURCE_FILES})
    set_target_properties(amsfilter
            PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET amsfilter PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
else()
    add_library(amsfilter STATIC ${AMSFILTER_SOURCE_FILES})
endif()
#===------------------------------------------------------------------------===#


#===------------------------------------------------------------------------===#
# The AMS-Filter calibration tool.
SET(AMSFILTER_CALIBRATION_SOURCE_FILES
        src/amsfilter_model/tool/calibration.cpp
        )
add_executable(amsfilter_calibration
        ${AMSFILTER_CALIBRATION_SOURCE_FILES}
        )
target_link_libraries(amsfilter_calibration amsfilter)
#===------------------------------------------------------------------------===#


#===------------------------------------------------------------------------===#
# The specialized SQLite interface that provides several useful UDFs.
FIND_PACKAGE(Readline)
SET(AMSFILTER_SHELL_SOURCE_FILES
        src/amsfilter_model/tool/shell.cpp
        thirdparty/sqlite/sqlite3.c
        thirdparty/sqlite/sqlite3.h
        thirdparty/sqlite/shell_mod.c
        thirdparty/sqlite/shell_mod.h
        )
add_executable(amsfilter_shell
        ${AMSFILTER_SHELL_SOURCE_FILES}
        )
target_compile_options(amsfilter_shell PRIVATE "-DSQLITE_ENABLE_JSON1")
target_compile_options(amsfilter_shell PRIVATE "-DSQLITE_OMIT_LOAD_EXTENSION")
if (READLINE_FOUND)
    target_compile_options(amsfilter_shell PRIVATE "-DHAVE_READLINE")
    target_link_libraries(amsfilter_shell readline ncurses)
endif()
target_link_libraries(amsfilter_shell amsfilter pthread dl)
#===------------------------------------------------------------------------===#


#===------------------------------------------------------------------------===#
# A small tool to generate FPR lookup table.
add_executable(amsfilter_gen_fpr_lut
        src/amsfilter_model/tool/gen_fpr_lut.cpp
        )
target_link_libraries(amsfilter_gen_fpr_lut amsfilter)
#===------------------------------------------------------------------------===#


# Include local CMake file.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/CMakeListsLocal.cmake")
    include(CMakeListsLocal.cmake)
endif()
