#pragma once

#include <cuda_runtime.h>

//===----------------------------------------------------------------------===//
// taken from: https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
#define cuda_check_error()  __cuda_check_error( __FILE__, __LINE__ )
__forceinline__
void
__cuda_check_error(const char *file, const int line ) {
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cuda_check_error() failed at %s:%i : %s\n",
        file, line, cudaGetErrorString(err) );
    exit(-1);
  }
}
//===----------------------------------------------------------------------===//
static std::string
get_cuda_device_name(u32 cuda_device_no) {
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, cuda_device_no);
  return std::string(device_prop.name);
}
//===----------------------------------------------------------------------===//
static u32
get_cuda_device_count() {
  $i32 count;
  cudaError_t cudaGetDeviceCount(&count);
  return static_cast<u32>(count);
}
//===----------------------------------------------------------------------===//
