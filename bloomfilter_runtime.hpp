#pragma once

#include <stdexcept>

#include <dtl/dtl.hpp>
#include <dtl/bloomfilter.hpp>
#include <dtl/bloomfilter_vec.hpp>
#include <dtl/bloomfilter2_vec.hpp>
#include <dtl/hash.hpp>
#include <dtl/mem.hpp>


namespace dtl {

/// A runtime wrapper for a Bloom filter instance.
/// The actual Bloom filter type is determined by the parameters 'm' and 'k'.
/// Note: This should only be used if the parameter is NOT known at compile time.
struct bloomfilter_runtime {

  using key_t = $u32;
  using word_t = $u32;

  /// The bit length of the Bloom filter.
  $u32 m;
  /// The number of hash functions.
  $u32 h;
  /// The number of bits set per entry.
  $u32 k;
  /// Pointer to the Bloom filter instance.
  void* instance;
  /// Pointer to the Bloom filter vector extension.
  void* instance_vec;

  // ---- the API functions ----
  std::function<void(const key_t /*key*/)>
  insert;

  std::function<$u1(const key_t /*key*/)>
  contains;

  std::function<$u64(const key_t* /*keys*/, u32 /*key_cnt*/, $u32* /*match_positions*/, u32 /*match_offset*/)>
  batch_contains;

  std::function<f64()>
  load_factor;

  std::function<u64()>
  pop_count;

  std::function<void()>
  print_info;

  std::function<void()>
  print;
  // ---- ----

  template<typename T>
  using hash_fn_0 = dtl::hash::knuth<T>;
//  using hash_fn_0 = dtl::hash::murmur_32<T>;
//  using hash_fn_0 = dtl::hash::identity<T>;

  template<typename T>
  using hash_fn_1 = dtl::hash::knuth_alt<T>;
//  using hash_fn_1 = dtl::hash::knuth<T>;
//  using hash_fn_0 = dtl::hash::identity<T>;

  // The supported Bloom filter implementations. (Note: Sectorization is not supported via the runtime API.)
  using bf1_k1_t = dtl::bloomfilter<key_t, hash_fn_0, word_t, dtl::mem::numa_allocator<word_t>, 1, false>;
  using bf1_k2_t = dtl::bloomfilter<key_t, hash_fn_0, word_t, dtl::mem::numa_allocator<word_t>, 2, false>;
  using bf1_k3_t = dtl::bloomfilter<key_t, hash_fn_0, word_t, dtl::mem::numa_allocator<word_t>, 3, false>;
  using bf1_k4_t = dtl::bloomfilter<key_t, hash_fn_0, word_t, dtl::mem::numa_allocator<word_t>, 4, false>;
  using bf1_k5_t = dtl::bloomfilter<key_t, hash_fn_0, word_t, dtl::mem::numa_allocator<word_t>, 5, false>;
  using bf1_k6_t = dtl::bloomfilter<key_t, hash_fn_0, word_t, dtl::mem::numa_allocator<word_t>, 6, false>;

  using bf2_k2_t = dtl::bloomfilter2<key_t, hash_fn_0, hash_fn_1, word_t, dtl::mem::numa_allocator<word_t>, 2, false>;
  using bf2_k3_t = dtl::bloomfilter2<key_t, hash_fn_0, hash_fn_1, word_t, dtl::mem::numa_allocator<word_t>, 3, false>;
  using bf2_k4_t = dtl::bloomfilter2<key_t, hash_fn_0, hash_fn_1, word_t, dtl::mem::numa_allocator<word_t>, 4, false>;
  using bf2_k5_t = dtl::bloomfilter2<key_t, hash_fn_0, hash_fn_1, word_t, dtl::mem::numa_allocator<word_t>, 5, false>;
  using bf2_k6_t = dtl::bloomfilter2<key_t, hash_fn_0, hash_fn_1, word_t, dtl::mem::numa_allocator<word_t>, 6, false>;
  using bf2_k7_t = dtl::bloomfilter2<key_t, hash_fn_0, hash_fn_1, word_t, dtl::mem::numa_allocator<word_t>, 7, false>;

  // The supported Bloom filter vectorization extensions.
  using bf1_k1_vt = dtl::bloomfilter_vec<key_t, hash_fn_0, word_t, dtl::mem::numa_allocator<word_t>, 1, false>;
  using bf1_k2_vt = dtl::bloomfilter_vec<key_t, hash_fn_0, word_t, dtl::mem::numa_allocator<word_t>, 2, false>;
  using bf1_k3_vt = dtl::bloomfilter_vec<key_t, hash_fn_0, word_t, dtl::mem::numa_allocator<word_t>, 3, false>;
  using bf1_k4_vt = dtl::bloomfilter_vec<key_t, hash_fn_0, word_t, dtl::mem::numa_allocator<word_t>, 4, false>;
  using bf1_k5_vt = dtl::bloomfilter_vec<key_t, hash_fn_0, word_t, dtl::mem::numa_allocator<word_t>, 5, false>;
  using bf1_k6_vt = dtl::bloomfilter_vec<key_t, hash_fn_0, word_t, dtl::mem::numa_allocator<word_t>, 6, false>;

  using bf2_k2_vt = dtl::bloomfilter2_vec<key_t, hash_fn_0, hash_fn_1, word_t, dtl::mem::numa_allocator<word_t>, 2, false>;
  using bf2_k3_vt = dtl::bloomfilter2_vec<key_t, hash_fn_0, hash_fn_1, word_t, dtl::mem::numa_allocator<word_t>, 3, false>;
  using bf2_k4_vt = dtl::bloomfilter2_vec<key_t, hash_fn_0, hash_fn_1, word_t, dtl::mem::numa_allocator<word_t>, 4, false>;
  using bf2_k5_vt = dtl::bloomfilter2_vec<key_t, hash_fn_0, hash_fn_1, word_t, dtl::mem::numa_allocator<word_t>, 5, false>;
  using bf2_k6_vt = dtl::bloomfilter2_vec<key_t, hash_fn_0, hash_fn_1, word_t, dtl::mem::numa_allocator<word_t>, 6, false>;
  using bf2_k7_vt = dtl::bloomfilter2_vec<key_t, hash_fn_0, hash_fn_1, word_t, dtl::mem::numa_allocator<word_t>, 7, false>;

  // Vectorization related compile time constants.
  static constexpr u64 unroll_factor = 4;
  static constexpr u64 vector_length = dtl::simd::lane_count<key_t> * unroll_factor;


  template<
      typename bf_t, // the scalar bloomfilter type
      typename bf_vt // the vector extension for the bloomfilter
  >
  void
  _construct_and_bind(u64 m) {
    using namespace std::placeholders;

    // instantiate a Bloom filter
    bf_t* bf = new bf_t(m);
    instance = bf;

    // instantiate the vectorized extension
    bf_vt* bf_v = new bf_vt { *bf };
    instance_vec = bf_v;

    // bind functions
    insert = std::bind(&bf_t::insert, bf, _1);
    contains = std::bind(&bf_t::contains, bf, _1);
    batch_contains = std::bind(&bf_vt::template batch_contains<vector_length>, bf_v, _1, _2, _3, _4);
    load_factor = std::bind(&bf_t::load_factor, bf);
    pop_count = std::bind(&bf_t::popcnt, bf);
    print_info = std::bind(&bf_t::print_info, bf);
    print = std::bind(&bf_t::print, bf);
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


  /// Instantiate a Bloom filter based on the given parameters 'k' and 'm'.
  static
  bloomfilter_runtime
  construct(u32 k, u64 m) {

    // Determine the number of bits required to identify the individual words of the Bloom filter.
    using bf_t = bf1_k1_t;
    u64 actual_m = bf_t::determine_actual_length(m);
    u64 word_bit_cnt = dtl::log_2(actual_m / bf_t::word_bitlength);

    // Determine the number of hash functions needed.
    // Note: Currently only 1 or 2 hash functions are supported.
    u64 hash_fn_cnt = ((bf_t::hash_value_bitlength - word_bit_cnt) / (bf_t::bit_cnt_per_k * k)) > 0 ? 1 : 2;

    bloomfilter_runtime wrapper;
    wrapper.m = actual_m;
    wrapper.h = hash_fn_cnt;
    wrapper.k = k;
    switch (wrapper.h) {
      case 1:
        // Instantiate a Bloom filter with one hash function.
        switch (wrapper.k) {
          case 1: wrapper._construct_and_bind<bf1_k1_t, bf1_k1_vt>(m); break;
          case 2: wrapper._construct_and_bind<bf1_k2_t, bf1_k2_vt>(m); break;
          case 3: wrapper._construct_and_bind<bf1_k3_t, bf1_k3_vt>(m); break;
          case 4: wrapper._construct_and_bind<bf1_k4_t, bf1_k4_vt>(m); break;
          case 5: wrapper._construct_and_bind<bf1_k5_t, bf1_k5_vt>(m); break;
          case 6: wrapper._construct_and_bind<bf1_k6_t, bf1_k6_vt>(m); break;
          default:
            throw std::invalid_argument("The given 'k' is not supported.");
        }
        break;
      case 2:
        // Instantiate a Bloom filter with two hash functions.
        switch (wrapper.k) {
          // k must be > 1, otherwise bf1 should be used.
          case 2: wrapper._construct_and_bind<bf2_k2_t, bf2_k2_vt>(m); break;
          case 3: wrapper._construct_and_bind<bf2_k3_t, bf2_k3_vt>(m); break;
          case 4: wrapper._construct_and_bind<bf2_k4_t, bf2_k4_vt>(m); break;
          case 5: wrapper._construct_and_bind<bf2_k5_t, bf2_k5_vt>(m); break;
          case 6: wrapper._construct_and_bind<bf2_k6_t, bf2_k6_vt>(m); break;
          case 7: wrapper._construct_and_bind<bf2_k7_t, bf2_k7_vt>(m); break;
          default:
            throw std::invalid_argument("The given 'k' is not supported.");
        }
        break;
      default:
        unreachable();
    }
    return wrapper;
  }


  /// Destruct the Bloom filter instance.
  void
  destruct() {
    if (!is_initialized()) return;
    switch (h) {
      case 1:
        switch (k) {
          case 1: _destruct<bf1_k1_t, bf1_k1_vt>(); break;
          case 2: _destruct<bf1_k2_t, bf2_k2_vt>(); break;
          case 3: _destruct<bf1_k3_t, bf2_k3_vt>(); break;
          case 4: _destruct<bf1_k4_t, bf2_k4_vt>(); break;
          case 5: _destruct<bf1_k5_t, bf2_k5_vt>(); break;
          case 6: _destruct<bf1_k6_t, bf2_k6_vt>(); break;
          default:
            throw std::invalid_argument("The given 'k' is not supported.");
        }
        break;
      case 2:
        switch (k) {
          case 2: _destruct<bf2_k2_t, bf2_k2_vt>(); break;
          case 3: _destruct<bf2_k3_t, bf2_k3_vt>(); break;
          case 4: _destruct<bf2_k4_t, bf2_k4_vt>(); break;
          case 5: _destruct<bf2_k5_t, bf2_k5_vt>(); break;
          case 6: _destruct<bf2_k6_t, bf2_k6_vt>(); break;
          case 7: _destruct<bf2_k7_t, bf2_k7_vt>(); break;
          default:
            throw std::invalid_argument("The given 'k' is not supported.");
        }
        break;
      default:
        throw std::invalid_argument("The given 'h' is not supported.");
    }

  }


  /// Returns 'true' if the Bloom filter is initialized, 'false' otherwise.
  forceinline
  u1
  is_initialized() {
    return instance != nullptr;
  }


  /// Computes an approximation of the false positive probability.
  /// Assuming independence for the probabilities of each bit being set,
  /// which is not the case in the current implementation.
  f64
  false_positive_probability(u64 element_cnt) {
    auto n = element_cnt;
    return std::pow(1.0 - std::pow(1.0 - (1.0 / m), k * n), k);
  }

};

} // namespace dtl
