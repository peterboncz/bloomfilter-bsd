#pragma once

#include <dtl/dtl.hpp>

namespace amsfilter {
namespace model {
//===----------------------------------------------------------------------===//
/// The type of processing unit.
enum class ProcUnit {
  CPU = 0, GPU = 1
};
//===----------------------------------------------------------------------===//
/// The type of memory, where the probe keys are located in.
enum class Memory {
  HOST_PAGEABLE = 0u,
  HOST_PINNED = 1u,
  DEVICE = 2u
};
//===----------------------------------------------------------------------===//
/// Specifies the execution environment for filter probes.
class Env {
  /// The (artificial) device number of the CPU.
  static i32 CPU_DEVICE_NO = -1;

  /// The device where the probing happens.
  $i32 device_;
  /// The number of threads that probe the filter concurrently (only applicable
  /// if probing is performed by the CPU).
  $u32 thread_cnt_;
  /// Info about the memory where the probe keys a located.
  Memory probe_key_location_;

  Env(i32 device, u32 thread_cnt, Memory probe_key_location)
      : device_(device),
        thread_cnt_(thread_cnt),
        probe_key_location_(probe_key_location) {}

public:

  Env(Env&&) noexcept = default;
  Env(const Env& other) = default;
  Env& operator=(const Env& other) = default;
  Env& operator=(Env&& other) = default;
  ~Env() = default;
  
  /// Probe is executed on the CPU using the given number of threads.
  static Env cpu(u32 thread_cnt) {
    return Env(CPU_DEVICE_NO, thread_cnt, Memory::HOST_PAGEABLE);
  }

  /// Probe is executed on the GPU with the probe keys located either in
  /// host or device memory.
  static Env gpu(u32 device_no, Memory probe_key_location) {
    return Env(static_cast<i32>(device_no), 0, probe_key_location);
  }
  // Convenience functions.
  static Env gpu_keys_in_pageable_memory(u32 device_no) {
    return Env(static_cast<i32>(device_no), 0, Memory::HOST_PAGEABLE);
  }
  static Env gpu_keys_in_pinned_memory(u32 device_no) {
    return Env(static_cast<i32>(device_no), 0, Memory::HOST_PINNED);
  }
  static Env gpu_keys_in_device_memory(u32 device_no) {
    return Env(static_cast<i32>(device_no), 0, Memory::DEVICE);
  }

  /// Returns true if the execution happens on the CPU, false otherwise.
  auto is_cpu() const { return device_ == CPU_DEVICE_NO; }
  /// Returns true if the execution happens on the GPU, false otherwise.
  auto is_gpu() const { return !is_cpu(); }
  /// Returns the type of the processing unit.
  auto get_proc_unit() const { return is_cpu() ? ProcUnit::CPU : ProcUnit::GPU; }
  /// Returns the device number.
  auto get_device() const { return device_; }
  /// Returns the number of threads.
  auto get_thread_cnt() const { return thread_cnt_; }
  /// Returns the location of the probe keys.
  auto get_probe_key_location() const { return probe_key_location_; }

};
//===----------------------------------------------------------------------===//
} // namespace model
} // namespace amsfilter
