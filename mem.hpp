#pragma once

#include "adept.hpp"

#include <cstring>
#include <stdlib.h>
#include <sys/mman.h>
#include <numa.h>

namespace mem {

  static u64 cacheline_size = 64;

  template<typename T>
  static T* aligned_alloc(u64 alignment, u64 cnt) {
    void* ptr = ::aligned_alloc(alignment, cnt * sizeof(T));
    return reinterpret_cast<T*>(ptr);
  }

  template<typename T>
  static T* aligned_alloc(u64 alignment, u64 cnt, u32 init_value) {
    void* ptr = aligned_alloc<T>(alignment, cnt * sizeof(T));
    std::memset(ptr, init_value, cnt * sizeof(T));
    return reinterpret_cast<T*>(ptr);
  }

  template<typename T>
  static T* malloc_huge(u64 n) {
    u64 huge_page_size = 2ull * 1024 * 1024;
    u64 byte_cnt = n * sizeof(T);
    if (byte_cnt < huge_page_size) {
      void* p = malloc(byte_cnt);
      return reinterpret_cast<T*>(p);
    } else {
      byte_cnt = std::max(byte_cnt, huge_page_size);
      void* p = mmap(nullptr, byte_cnt, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      madvise(p, byte_cnt, MADV_HUGEPAGE);
      return reinterpret_cast<T*>(p);
    }
    unreachable();
  }

  template<typename T>
  static void free_huge(T* ptr, const size_t n) {
    const uint64_t huge_page_size = 2ull * 1024 * 1024;
    uint64_t byte_cnt = n * sizeof(T);
    if (byte_cnt < huge_page_size) {
      free(ptr);
    } else {
      byte_cnt = std::max(byte_cnt, huge_page_size);
      munmap(ptr, byte_cnt);
    }
  }

}