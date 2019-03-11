#pragma once

#include <cmath>
#include <chrono>
#include <iomanip>
#include <random>
#include <stdexcept>

#include <dtl/dtl.hpp>
#include <dtl/batchwise.hpp>
#include <dtl/math.hpp>
#include <dtl/mem.hpp>
#include <dtl/simd.hpp>

#include <dtl/filter/blocked_bloomfilter/block_addressing_logic.hpp>
#include <dtl/filter/blocked_bloomfilter/hash_family.hpp>

#include "cuckoofilter_logic.hpp"
#include "cuckoofilter_table.hpp"
#include "cuckoofilter_tune.hpp"

namespace dtl {
namespace cuckoofilter {


//===----------------------------------------------------------------------===//
// The global tuning instance.
static cuckoofilter_tune* cuckoofilter_default_tuning = nullptr;
//    new cuckoofilter_tune_impl;
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
template<u1 has_victim_cache = true>
struct cuckoofilter {

  using key_t = $u32;
  using hash_value_t = $u32;
  using word_t = $u32;

  template<
      typename key_t,
      $u32 hash_fn_no
  >
  using hasher = dtl::hash::stat::mul32<key_t, hash_fn_no>;

  // The operations used for dynamic dispatching.
  enum class op_t {
    CONSTRUCT,  // constructs the filter logic
    BIND,       // assigns the function pointers of the filter API
    DESTRUCT,   // destructs the filter logic
  };

  static constexpr dtl::block_addressing power = dtl::block_addressing::POWER_OF_TWO;
  static constexpr dtl::block_addressing magic = dtl::block_addressing::MAGIC;

  template <std::size_t bits_per_tag, std::size_t tags_per_bucket>
  using table_t = cuckoofilter_table<bits_per_tag, tags_per_bucket>;


  template<
      std::size_t bits_per_tag,
      std::size_t tags_per_bucket,
      dtl::block_addressing block_addressing = dtl::block_addressing::POWER_OF_TWO>
  using cf = cuckoofilter_logic<
      bits_per_tag,
      tags_per_bucket,
      table_t,
      block_addressing,
      has_victim_cache>;


  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  /// The (desired) bit length of the filter.
  $u64 m;
  /// The (actual) bit length of the filter.
  $u64 m_actual;
  /// The number of bits per tag.
  $u32 bits_per_tag;
  /// The number of tags per bucket.
  $u32 tags_per_bucket;
  /// Pointer to the cuckoo filter logic instance.
  void* instance = nullptr;
  /// A container for hardware dependent tuning parameters.
  cuckoofilter_tune& tune;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // The API functions. (Function pointers to the actual implementation.)
  //===----------------------------------------------------------------------===//
  std::function<$u1(__restrict word_t* /*filter data*/, const key_t /*key*/)>
  insert;

  std::function<$u1(__restrict word_t* /*filter data*/, const key_t* /*keys*/, u32 /*key_cnt*/)>
  batch_insert;

  std::function<$u1(const __restrict word_t* /*filter data*/, const key_t /*key*/)>
  contains;

  std::function<$u64(const __restrict word_t* /*filter data*/,
                     const key_t* /*keys*/, u32 /*key_cnt*/,
                     $u32* /*match_positions*/, u32 /*match_offset*/)>
  batch_contains;

  std::function<$u64(const __restrict word_t* /*filter data*/)>
  count_occupied_slots;

  std::function<$u64()>
  get_bucket_count;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  cuckoofilter(const std::size_t m, u32 bits_per_tag, u32 tags_per_bucket) :
    cuckoofilter(m, bits_per_tag, tags_per_bucket, *cuckoofilter_default_tuning) {}; // delegate using default tuning instance

  cuckoofilter(const std::size_t m, u32 bits_per_tag, u32 tags_per_bucket,
                     cuckoofilter_tune& tune)
      : m(m), bits_per_tag(bits_per_tag), tags_per_bucket(tags_per_bucket), tune(tune) {

    // Construct the filter logic instance.
    dispatch(*this, op_t::CONSTRUCT);

    // Bind the API functions.
    dispatch(*this, op_t::BIND);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  cuckoofilter(cuckoofilter&& src)
      : m(src.m), m_actual(src.m_actual),
        bits_per_tag(src.bits_per_tag), tags_per_bucket(src.tags_per_bucket),
        instance(src.instance),
        insert(std::move(src.insert)),
        batch_insert(std::move(src.batch_insert)),
        contains(std::move(src.contains)),
        batch_contains(std::move(src.batch_contains)),
        count_occupied_slots(std::move(src.count_occupied_slots)),
        get_bucket_count(std::move(src.get_bucket_count)),
        tune(src.tune) {
    // Invalidate pointer in src
    src.instance = nullptr;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  ~cuckoofilter() {
    // Destruct logic instance (if any).
    if (instance != nullptr) dispatch(*this, op_t::DESTRUCT);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  cuckoofilter&
  operator=(cuckoofilter&& src) {
    m = src.m;
    m_actual = src.m_actual;
    bits_per_tag = src.bits_per_tag;
    tags_per_bucket = src.tags_per_bucket;
    instance = src.instance;
    insert = std::move(src.insert);
    batch_insert = std::move(src.batch_insert);
    contains = std::move(src.contains);
    batch_contains = std::move(src.batch_contains);
    count_occupied_slots = std::move(src.count_occupied_slots);
    get_bucket_count = std::move(src.get_bucket_count);

    // invalidate pointers
    src.instance = nullptr;
    return *this;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Dynamic Dispatching
  //===----------------------------------------------------------------------===//
  //TODO make private
  static void
  dispatch(cuckoofilter& instance, op_t op) {
      switch (instance.bits_per_tag) {
        case  4: _b< 4>(instance, op); break;
        case  8: _b< 8>(instance, op); break;
        case 12: _b<12>(instance, op); break;
        case 16: _b<16>(instance, op); break;
        case 32: _b<32>(instance, op); break;
        default:
          throw std::invalid_argument("The given tag size is not supported.");
      }
  };

  template<u32 t>
  static void
  _b(cuckoofilter& instance, op_t op) {
    switch (instance.tags_per_bucket) {
      case  1: _a<t, 1>(instance, op); break;
      case  2: _a<t, 2>(instance, op); break;
      case  4: _a<t, 4>(instance, op); break;
      default:
        throw std::invalid_argument("The given bucket size (associativity) is not supported.");
    }
  }

  template<u32 t, u32 b>
  static void
  _a(cuckoofilter& instance, op_t op) {
    dtl::block_addressing addr = dtl::is_power_of_two(instance.m)
                                 ? dtl::block_addressing::POWER_OF_TWO
                                 : dtl::block_addressing::MAGIC;
    switch (addr) {
      case dtl::block_addressing::POWER_OF_TWO: _u<t, b, dtl::block_addressing::POWER_OF_TWO>(instance, op); break;
      case dtl::block_addressing::MAGIC:        _u<t, b, dtl::block_addressing::MAGIC>(instance, op);        break;
      case dtl::block_addressing::DYNAMIC:      /* must not happen */                                        break;
    }
  }

  template<u32 t, u32 b, dtl::block_addressing a>
  static void
  _u(cuckoofilter& instance, op_t op) {
    switch (instance.tune.get_unroll_factor(t, b, a)) {
      case  0: _o<t, b, a, 0>(instance, op); break;
      case  1: _o<t, b, a, 1>(instance, op); break;
      case  2: _o<t, b, a, 2>(instance, op); break;
      case  4: _o<t, b, a, 4>(instance, op); break;
      case  8: _o<t, b, a, 8>(instance, op); break;
      default:
        throw std::invalid_argument("The given 'unroll_factor' is not supported.");
    }
  }

  template<u32 t, u32 b, dtl::block_addressing a, u32 unroll_factor>
  static void
  _o(cuckoofilter& instance, op_t op) {
    using _t = cf<t, b, a>;
    switch (op) {
      case op_t::CONSTRUCT: instance._construct_logic<_t>();           break;
      case op_t::BIND:      instance._bind_logic<_t, unroll_factor>(); break;
      case op_t::DESTRUCT:  instance._destruct_logic<_t>();            break;
    }
  };
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Constructs a filter logic instance.
  template<
      typename filter_t
  >
  void
  _construct_logic() {
    // Instantiate a filter logic.
    filter_t* f = new filter_t(m);
    instance = f;
    // Get the actual size of the filter.
    m_actual = f->size_in_bytes() * 8;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Bind API functions to the (previously constructed) filter logic.
  template<
      typename filter_t,
      u32 unroll_factor = 1
  >
  void
  _bind_logic() {
    using namespace std::placeholders;
    auto* f = static_cast<filter_t*>(instance);

    // Bind the API functions.
    insert = std::bind(&filter_t::insert, f, _1, _2);
    batch_insert = std::bind(&filter_t::batch_insert, f, _1, _2, _3);
    contains = std::bind(&filter_t::contains, f, _1, _2);

    // SIMD vector length (0 = run scalar code)
    static constexpr u64 vector_len = dtl::simd::lane_count<key_t> * unroll_factor;
    batch_contains = std::bind(&filter_t::template batch_contains<vector_len>, f, _1, _2, _3, _4, _5);

    count_occupied_slots = std::bind(&filter_t::count_occupied_slots, f, _1);
    get_bucket_count = std::bind(&filter_t::get_bucket_count, f);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Destructs the filter logic.
  template<typename filter_t>
  void
  _destruct_logic() {
    filter_t* f = static_cast<filter_t*>(instance);
    delete f;
    instance = nullptr;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the block addressing mode.
  dtl::block_addressing get_addressing_mode() const {
    return dtl::is_power_of_two(m)
           ? dtl::block_addressing::POWER_OF_TWO
           : dtl::block_addressing::MAGIC;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the (actual) size in bytes.
  std::size_t
  size_in_bytes() const noexcept {
    return (m_actual + 7) / 8;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the (total) number of words.
  std::size_t
  size() const noexcept {
    constexpr u32 word_bitlength = sizeof(word_t) * 8;
    return (m_actual + (word_bitlength - 1)) / word_bitlength;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the name of the filter instance including the most
  /// important parameters (in JSON).
  std::string
  name() {
    return "{\"name\":\"cuckoo\",\"size\":" + std::to_string(size_in_bytes())
         + ",\"tag_bits\":" + std::to_string(bits_per_tag)
         + ",\"associativity\":" + std::to_string(tags_per_bucket)
         + ",\"delete_support\":" + "false"
         + ",\"u\":" + std::to_string(tune.get_unroll_factor(bits_per_tag, tags_per_bucket, get_addressing_mode()))
         + ",\"addr\":" + (get_addressing_mode() == dtl::block_addressing::POWER_OF_TWO ? "\"pow2\"" : "\"magic\"")
         + ",\"has_victim_cache\":" + (has_victim_cache ? "\"true\"" : "\"false\"")
         + "}";
  }
  //===----------------------------------------------------------------------===//

};

} // namespace cuckoofilter
} // namespace dtl
