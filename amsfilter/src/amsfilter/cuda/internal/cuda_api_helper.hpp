#pragma once

#include <cuda_runtime.h>

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

