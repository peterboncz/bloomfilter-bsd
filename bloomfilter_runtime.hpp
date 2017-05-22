#pragma once

#include <stdexcept>

#include <dtl/dtl.hpp>
#include <dtl/bloomfilter.hpp>
#include <dtl/bloomfilter_vec.hpp>
#include <dtl/bloomfilter2_vec.hpp>
#include <dtl/hash.hpp>
#include <dtl/mem.hpp>


namespace dtl {

/// a runtime wrapper for a bloomfilter instance.
/// the actual bloomfilter type is determined by the parameter 'k'.
/// note: this should only be used if the parameter is NOT know at compile time.
struct bloomfilter_runtime_t {

  using key_t = $u32;
  using word_t = $u32;

  /// the number of hash functions
  $u32 k;
  /// pointer to the bloomfilter instance
  void* instance;
  /// pointer to the bloomfilter vector extension
  void* instance_vec;

  std::function<void(const key_t /*key*/)>
  insert;

  std::function<$u1(const key_t /*key*/)>
  contains;

  std::function<$u64(const key_t* /*keys*/, u32 /*key_cnt*/, $u32* /*match_positions*/, u32 /*match_offset*/)>
  batch_contains;

  std::function<f64()>
  load_factor;

  // the supported bloomfilter implementations
  using bf_k1_t = dtl::bloomfilter<key_t, dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 1, false>;
  using bf_k2_t = dtl::bloomfilter2<key_t, dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, 2, false>;
  using bf_k3_t = dtl::bloomfilter2<key_t, dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, 3, false>;
  using bf_k4_t = dtl::bloomfilter2<key_t, dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, 4, false>;
  using bf_k5_t = dtl::bloomfilter2<key_t, dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, 5, false>;
  using bf_k6_t = dtl::bloomfilter2<key_t, dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, 6, false>;
  using bf_k7_t = dtl::bloomfilter2<key_t, dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, 7, false>;
  using bf_k8_t = dtl::bloomfilter2<key_t, dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, 8, false>;

  // the supported bloomfilter vectorization extensions
  using bf_k1_vt = dtl::bloomfilter_vec<key_t, dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 1, false>;
  using bf_k2_vt = dtl::bloomfilter2_vec<key_t, dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, 2, false>;
  using bf_k3_vt = dtl::bloomfilter2_vec<key_t, dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, 3, false>;
  using bf_k4_vt = dtl::bloomfilter2_vec<key_t, dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, 4, false>;
  using bf_k5_vt = dtl::bloomfilter2_vec<key_t, dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, 5, false>;
  using bf_k6_vt = dtl::bloomfilter2_vec<key_t, dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, 6, false>;
  using bf_k7_vt = dtl::bloomfilter2_vec<key_t, dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, 7, false>;
  using bf_k8_vt = dtl::bloomfilter2_vec<key_t, dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, 8, false>;


  static constexpr u64 unroll_factor = 4;
  static constexpr u64 vector_length = dtl::simd::lane_count<key_t> * unroll_factor;


  template<
      typename bf_t, // the scalar bloomfilter type
      typename bf_vt // the vector extension for the bloomfilter
  >
  void
  _construct_and_bind(u64 m) {
    using namespace std::placeholders;
    bf_t* bf = new bf_t(m);
    bf_vt* bf_v = new bf_vt { *bf };
    instance = bf;
    instance_vec = bf_v;
    insert = std::bind(&bf_t::insert, bf, _1);
    contains = std::bind(&bf_t::contains, bf, _1);
    batch_contains = std::bind(&bf_vt::template batch_contains<vector_length>, bf_v, _1, _2, _3, _4);
    load_factor = std::bind(&bf_t::load_factor, bf);
  }

  template<
      typename bf_t, // the scalar bloomfilter type
      typename bf_vt // the vector extension for the bloomfilter
  >
  void
  _destruct() {
    bf_vt* bf_v = static_cast<bf_vt*>(instance_vec);
    delete bf_v;
    instance_vec = nullptr;
    bf_t* bf = static_cast<bf_t*>(instance);
    delete bf;
    instance = nullptr;
  }


  static
  bloomfilter_runtime_t
  construct(u32 k, u64 m) {
    bloomfilter_runtime_t wrapper;
    wrapper.k = k;
    switch (k) {
      case 1: wrapper._construct_and_bind<bf_k1_t, bf_k1_vt>(m); break;
      case 2: wrapper._construct_and_bind<bf_k2_t, bf_k2_vt>(m); break;
      case 3: wrapper._construct_and_bind<bf_k3_t, bf_k3_vt>(m); break;
      case 4: wrapper._construct_and_bind<bf_k4_t, bf_k4_vt>(m); break;
      case 5: wrapper._construct_and_bind<bf_k5_t, bf_k5_vt>(m); break;
      case 6: wrapper._construct_and_bind<bf_k6_t, bf_k6_vt>(m); break;
      case 7: wrapper._construct_and_bind<bf_k7_t, bf_k7_vt>(m); break;
      case 8: wrapper._construct_and_bind<bf_k8_t, bf_k8_vt>(m); break;
      default:
        throw std::invalid_argument("The given 'k' is not supported.");
    }
    return wrapper;
  }


  void
  destruct() {
    if (!is_initialized()) return;
    switch (k) {
      case 1: _destruct<bf_k1_t, bf_k1_vt>(); break;
      case 2: _destruct<bf_k2_t, bf_k2_vt>(); break;
      case 3: _destruct<bf_k3_t, bf_k3_vt>(); break;
      case 4: _destruct<bf_k4_t, bf_k4_vt>(); break;
      case 5: _destruct<bf_k5_t, bf_k5_vt>(); break;
      case 6: _destruct<bf_k6_t, bf_k6_vt>(); break;
      case 7: _destruct<bf_k7_t, bf_k7_vt>(); break;
      case 8: _destruct<bf_k8_t, bf_k8_vt>(); break;
      default:
        throw std::invalid_argument("The given 'k' is not supported.");
    }
  }

  forceinline u1
  is_initialized() {
    return instance != nullptr;
  }

};

} // namespace dtl
