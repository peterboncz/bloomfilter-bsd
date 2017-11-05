#pragma once

#include <cmath>
#include <stdexcept>

#include <dtl/dtl.hpp>
#include <dtl/batchwise.hpp>
#include <dtl/mem.hpp>

#include <boost/math/common_factor.hpp>

#include <dtl/bloomfilter/blocked_bloomfilter_logic.hpp>
#include <dtl/bloomfilter/hash_family.hpp>
#include <dtl/hash.hpp>
#include "bloomfilter_addressing_logic.hpp"
#include "bloomfilter_h2_mod.hpp"



namespace dtl {

// taken from https://stackoverflow.com/questions/36279198/calculate-lcm-of-2-numbers-using-templates
template <int Op>
struct abs {
  static constexpr unsigned const value{Op >= 0 ? Op : -Op};
};

template <unsigned A, unsigned B>
struct gcd : gcd<B, A % B> {};

template <unsigned A>
struct gcd<A, 0> {
  static constexpr unsigned const value{A};
};

template <int A, int B>
struct lcm {
  static constexpr unsigned const value{
      abs<A * B>::value / gcd<abs<A>::value, abs<B>::value>::value
  };
};


static std::array<$u32, 16> unroll_factors_32 = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
static std::array<$u32, 16> unroll_factors_64 = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

template<typename Tw = $u32>
struct blocked_bloomfilter {


  using key_t = $u32;
  using hash_value_t = $u32;
  using word_t = Tw;


  template<
      typename key_t,
      $u32 hash_fn_no
  >
  using hasher = dtl::hash::stat::mul32<key_t, hash_fn_no>;

  enum class op_t {
    CONSTRUCT,
    DESTRUCT
  };


  /// The bit length of the Bloom filter.
  $u64 m;
  /// The number of bits set per entry.
  $u32 k;
  ///
  $u32 word_cnt_per_block;
  ///
  $u32 sector_cnt;
  /// Pointer to the Bloom filter instance.
  void* instance = nullptr;

  // ---- The API functions. ----
  std::function<void(const key_t /*key*/)>
  insert;

  std::function<$u1(const key_t /*key*/)>
  contains;

  std::function<$u64(const key_t* /*keys*/, u32 /*key_cnt*/, $u32* /*match_positions*/, u32 /*match_offset*/)>
  batch_contains;

  // ---- ----

  blocked_bloomfilter() = default;

  blocked_bloomfilter(blocked_bloomfilter&& src)
      : m(src.m), k(src.k), word_cnt_per_block(src.word_cnt_per_block), sector_cnt(src.sector_cnt),
        instance(src.instance),
        insert(std::move(src.insert)),
        contains(std::move(src.contains)),
        batch_contains(std::move(src.batch_contains)) {
    // invalidate pointers
    src.instance = nullptr;
  }

  ~blocked_bloomfilter() {
    if (instance != nullptr) dispatch(*this, word_cnt_per_block, sector_cnt, k, m, op_t::DESTRUCT);
  }

  blocked_bloomfilter&
  operator=(blocked_bloomfilter&& src) {
    m = src.m;
    k = src.k;
    word_cnt_per_block = src.word_cnt_per_block;
    sector_cnt = src.sector_cnt;
    instance = src.instance;
    insert = std::move(src.insert);
    contains = std::move(src.contains);
    batch_contains = std::move(src.batch_contains);
    // invalidate pointers
    src.instance = nullptr;
    return *this;
  }


  static constexpr auto power = dtl::block_addressing::POWER_OF_TWO;
  static constexpr auto magic = dtl::block_addressing::MAGIC;

  template<u32 word_cnt, u32 sector_cnt, u32 k, dtl::block_addressing addr = power>
  using bbf = dtl::blocked_bloomfilter_logic<key_t, hasher, word_t, word_cnt, sector_cnt, k, addr, dtl::mem::numa_allocator<word_t>>;


  template<
      typename bf_t,
      u32 unroll_factor = 1
  >
  void
  _construct_and_bind(u64 m) {
    using namespace std::placeholders;

    // Instantiate a Bloom filter.
    bf_t* bf = new bf_t(m);
    instance = bf;
    k = bf_t::k;
    this->m = m;
    word_cnt_per_block = bf_t::word_cnt;
    sector_cnt = bf_t::sector_cnt;

    // Bind the API functions.
    insert = std::bind(&bf_t::insert, bf, _1);
    contains = std::bind(&bf_t::contains, bf, _1);

    static constexpr u64 vector_len = dtl::simd::lane_count<key_t> * unroll_factor;
    batch_contains = std::bind(&bf_t::template batch_contains<vector_len>, bf, _1, _2, _3, _4);
  }



//  template<
//      typename bf_t
//  >
//  void
//  _copy_and_bind(blocked_bloomfilter& copy,
//                 const dtl::mem::allocator_config allocator_config) const {
//    using namespace std::placeholders;
//
//    // Copy the Bloom filter.
//    const bf_t* bf_src = static_cast<bf_t*>(instance);
//    auto allocator = dtl::mem::numa_allocator<word_t>(allocator_config);
//    bf_t* bf_dst = bf_src->make_heap_copy(allocator);
//    copy.instance = bf_dst;
//
//    // Bind the API functions.
//    copy.insert = std::bind(&bf_t::insert, bf_dst, _1);
//    copy.contains = std::bind(&bf_t::contains, bf_dst, _1);
//    copy.batch_contains = std::bind(&bf_t::template batch_contains<>, bf_dst, _1, _2, _3, _4);
//  }


  template<
      typename bf_t // the scalar bloomfilter_h1 type
  >
  void
  _destruct() {
    bf_t* bf = static_cast<bf_t*>(instance);
    delete bf;
    instance = nullptr;
  }


  // helper // fIXME
  template<typename T>
  using hash_fn_0 = dtl::hash::knuth<T>;

  /// Instantiate a Bloom filter based on the given parameters 'k' and 'm'.
  static
  blocked_bloomfilter
  construct(u32 word_cnt_per_block, u32 sector_cnt, u32 k, u64 m) {

    const u1 only_pow_of_two = false;

    // Determine the number of bits required to identify the individual words/blocks of the Bloom filter.
    using bf_helper_t = dtl::bloomfilter_h1<key_t, hash_fn_0, word_t, dtl::mem::numa_allocator<word_t>, 2, true>;
    $u64 actual_m = bf_helper_t::determine_actual_length(m);
    $u64 word_bit_cnt = dtl::log_2(actual_m / bf_helper_t::word_bitlength);
    if (!dtl::is_power_of_two(m) && !only_pow_of_two) {
      using bf_mod_t = dtl::bloomfilter_h2_mod<key_t, hash_fn_0, hash_fn_0, word_t, dtl::mem::numa_allocator<word_t>, 2, true>;
      actual_m = std::min(actual_m, bf_mod_t::determine_word_cnt(m) * bf_mod_t::word_bitlength);
    }

    if (actual_m > (1ull << 32)) {
      throw "m must not exceed 2^32 bits.";
    }

    blocked_bloomfilter wrapper;

    dispatch(wrapper, word_cnt_per_block, sector_cnt, k, m, op_t::CONSTRUCT);

    if (wrapper.k != k
        || wrapper.word_cnt_per_block != word_cnt_per_block
        || wrapper.sector_cnt != sector_cnt) {
      dispatch(wrapper, word_cnt_per_block, sector_cnt, k, m, op_t::DESTRUCT);
      throw std::invalid_argument("Invalid configuration: w=" + std::to_string(word_cnt_per_block)
                                  + ", s=" + std::to_string(sector_cnt)
                                  + ", k=" + std::to_string(k));
    }
    return wrapper;
  }

  static void dispatch(blocked_bloomfilter& wrapper, u32 word_cnt_per_block, u32 sector_cnt, u32 k, size_t m, op_t op) {
      switch (word_cnt_per_block) {
        case 1: _w<1>(wrapper, sector_cnt, k, m, op); break;
        case 2: _w<2>(wrapper, sector_cnt, k, m, op); break;
        case 4: _w<4>(wrapper, sector_cnt, k, m, op); break;
        case 8: _w<8>(wrapper, sector_cnt, k, m, op); break;
        default:
          throw std::invalid_argument("The given 'word_cnt_per_block' is not supported.");
      }
  };


  template<u32 w>
  static void _w(blocked_bloomfilter& wrapper, u32 sector_cnt, u32 k, size_t m, op_t op) {
    switch (sector_cnt) {
      case 1: _s<w, boost::static_unsigned_max<1, w>::value>(wrapper, k, m, op); break;
      case 2: _s<w, boost::static_unsigned_max<2, w>::value>(wrapper, k, m, op); break;
      case 4: _s<w, boost::static_unsigned_max<4, w>::value>(wrapper, k, m, op); break;
      case 8: _s<w, boost::static_unsigned_max<8, w>::value>(wrapper, k, m, op); break;
      default:
        throw std::invalid_argument("The given 'sector_cnt' is not supported.");
    }
  }


  template<u32 w, u32 s>
  static void _s(blocked_bloomfilter& wrapper, u32 k, size_t m, op_t op) {
    switch (k) {
      case  1: _k<w, s, boost::static_unsigned_max<1, s>::value>(wrapper, m, op); break;
      case  2: _k<w, s, boost::static_unsigned_max<( 2 % s == 0 ?  2 : 1 /*invalid*/), s>::value>(wrapper, m, op); break;
      case  3: _k<w, s, boost::static_unsigned_max<( 3 % s == 0 ?  3 : 1 /*invalid*/), s>::value>(wrapper, m, op); break;
      case  4: _k<w, s, boost::static_unsigned_max<( 4 % s == 0 ?  4 : 1 /*invalid*/), s>::value>(wrapper, m, op); break;
      case  5: _k<w, s, boost::static_unsigned_max<( 5 % s == 0 ?  5 : 1 /*invalid*/), s>::value>(wrapper, m, op); break;
      case  6: _k<w, s, boost::static_unsigned_max<( 6 % s == 0 ?  6 : 1 /*invalid*/), s>::value>(wrapper, m, op); break;
      case  7: _k<w, s, boost::static_unsigned_max<( 7 % s == 0 ?  7 : 1 /*invalid*/), s>::value>(wrapper, m, op); break;
      case  8: _k<w, s, boost::static_unsigned_max<( 8 % s == 0 ?  8 : 1 /*invalid*/), s>::value>(wrapper, m, op); break;
      case  9: _k<w, s, boost::static_unsigned_max<( 9 % s == 0 ?  9 : 1 /*invalid*/), s>::value>(wrapper, m, op); break;
      case 10: _k<w, s, boost::static_unsigned_max<(10 % s == 0 ? 10 : 1 /*invalid*/), s>::value>(wrapper, m, op); break;
      case 11: _k<w, s, boost::static_unsigned_max<(11 % s == 0 ? 11 : 1 /*invalid*/), s>::value>(wrapper, m, op); break;
      case 12: _k<w, s, boost::static_unsigned_max<(12 % s == 0 ? 12 : 1 /*invalid*/), s>::value>(wrapper, m, op); break;
      case 13: _k<w, s, boost::static_unsigned_max<(13 % s == 0 ? 13 : 1 /*invalid*/), s>::value>(wrapper, m, op); break;
      case 14: _k<w, s, boost::static_unsigned_max<(14 % s == 0 ? 14 : 1 /*invalid*/), s>::value>(wrapper, m, op); break;
      case 15: _k<w, s, boost::static_unsigned_max<(15 % s == 0 ? 15 : 1 /*invalid*/), s>::value>(wrapper, m, op); break;
      case 16: _k<w, s, boost::static_unsigned_max<16 , s>::value>(wrapper, m, op); break;
    }
  }


  template<u32 w, u32 s, u32 k>
  static void _k(blocked_bloomfilter& wrapper, size_t m, op_t op) {
    dtl::block_addressing addr = dtl::is_power_of_two(m)
                                 ? dtl::block_addressing::POWER_OF_TWO
                                 : dtl::block_addressing::MAGIC;
    switch (addr) {
      case dtl::block_addressing::POWER_OF_TWO: _u<w, s, k, dtl::block_addressing::POWER_OF_TWO>(wrapper, m, op); break;
      case dtl::block_addressing::MAGIC:        _u<w, s, k, dtl::block_addressing::MAGIC>(wrapper, m, op);        break;
    }
  }

  template<u32 w, u32 s, u32 k, dtl::block_addressing a>
  static void _u(blocked_bloomfilter& wrapper, size_t m, op_t op) {
    auto& unroll_factors = sizeof(word_t) == 8 ? unroll_factors_64 : unroll_factors_32;
    switch (unroll_factors[k-1]) {
      case  0: _o<w, s, k, a,  0>(wrapper, m, op); break;
      case  1: _o<w, s, k, a,  1>(wrapper, m, op); break;
      case  2: _o<w, s, k, a,  2>(wrapper, m, op); break;
      case  4: _o<w, s, k, a,  4>(wrapper, m, op); break;
      case  8: _o<w, s, k, a,  8>(wrapper, m, op); break;
      case 16: _o<w, s, k, a, 16>(wrapper, m, op); break;
      default:
        throw std::invalid_argument("The given 'unroll_factor' is not supported.");
    }
  }

  template<u32 w, u32 s, u32 k, dtl::block_addressing a, u32 unroll_factor>
  static void _o(blocked_bloomfilter& wrapper, size_t m, op_t op) {
    using _t = bbf<w, s, k, a>;
    switch (op) {
      case op_t::CONSTRUCT: wrapper._construct_and_bind<_t, unroll_factor>(m); break;
      case op_t::DESTRUCT:  wrapper._destruct<_t>();                           break;
    }
  };


//  /// Destruct the Bloom filter instance.
//  void
//  destruct() {
//    if (!is_initialized()) return;
//    if (dtl::is_power_of_two(m)) {
//      switch (h) {
//        case 1:
//          switch (k) {
//            case 1: _destruct<bf1_k1_t, bf1_k1_vt>(); break;
//            case 2: _destruct<bf1_k2_t, bf1_k2_vt>(); break;
//            case 3: _destruct<bf1_k3_t, bf1_k3_vt>(); break;
//            case 4: _destruct<bf1_k4_t, bf1_k4_vt>(); break;
//            case 5: _destruct<bf1_k5_t, bf1_k5_vt>(); break;
//            case 6: _destruct<bf1_k6_t, bf1_k6_vt>(); break;
//            default:
//              throw std::invalid_argument("The given 'k' is not supported.");
//          }
//          break;
//        case 2:
//          switch (k) {
//            case 2: _destruct<bf2_k2_t, bf2_k2_vt>(); break;
//            case 3: _destruct<bf2_k3_t, bf2_k3_vt>(); break;
//            case 4: _destruct<bf2_k4_t, bf2_k4_vt>(); break;
//            case 5: _destruct<bf2_k5_t, bf2_k5_vt>(); break;
//            case 6: _destruct<bf2_k6_t, bf2_k6_vt>(); break;
//            case 7: _destruct<bf2_k7_t, bf2_k7_vt>(); break;
//            default:
//              throw std::invalid_argument("The given 'k' is not supported.");
//          }
//          break;
//        case 3:
//          switch (k) {
//            case 6: _destruct<bf3_k6_t, bf3_k6_vt>(); break;
//            case 7: _destruct<bf3_k7_t, bf3_k7_vt>(); break;
//            case 8: _destruct<bf3_k8_t, bf3_k8_vt>(); break;
//            case 9: _destruct<bf3_k9_t, bf3_k9_vt>(); break;
//            case 10: _destruct<bf3_k10_t, bf3_k10_vt>(); break;
//            case 11: _destruct<bf3_k11_t, bf3_k11_vt>(); break;
//            default:
//              throw std::invalid_argument("The given 'k' is not supported.");
//          }
//          break;
//        default:
//          throw std::invalid_argument("The given 'h' is not supported.");
//      }
//    }
//    else {
//      // m is not a power of two. pick a (slightly) slower implementation
//      switch (h) {
//        case 1:
//          switch (k) {
//            case 1: _destruct<bf1_k1_mod_t, bf1_k1_mod_vt>(); break;
//            case 2: _destruct<bf1_k2_mod_t, bf1_k2_mod_vt>(); break;
//            case 3: _destruct<bf1_k3_mod_t, bf1_k3_mod_vt>(); break;
//            case 4: _destruct<bf1_k4_mod_t, bf1_k4_mod_vt>(); break;
//            case 5: _destruct<bf1_k5_mod_t, bf1_k5_mod_vt>(); break;
//            case 6: _destruct<bf1_k6_mod_t, bf1_k6_mod_vt>(); break;
//            default:
//              throw std::invalid_argument("The given 'k' is not supported.");
//          }
//          break;
//        case 2:
//          switch (k) {
//            case 2: _destruct<bf2_k2_mod_t, bf2_k2_mod_vt>(); break;
//            case 3: _destruct<bf2_k3_mod_t, bf2_k3_mod_vt>(); break;
//            case 4: _destruct<bf2_k4_mod_t, bf2_k4_mod_vt>(); break;
//            case 5: _destruct<bf2_k5_mod_t, bf2_k5_mod_vt>(); break;
//            case 6: _destruct<bf2_k6_mod_t, bf2_k6_mod_vt>(); break;
//            case 7: _destruct<bf2_k7_mod_t, bf2_k7_mod_vt>(); break;
//            default:
//              throw std::invalid_argument("The given 'k' is not supported.");
//          }
//          break;
//        case 3:
//          switch (k) {
//            case 6: _destruct<bf3_k6_mod_t, bf3_k6_mod_vt>(); break;
//            case 7: _destruct<bf3_k7_mod_t, bf3_k7_mod_vt>(); break;
//            case 8: _destruct<bf3_k8_mod_t, bf3_k8_mod_vt>(); break;
//            case 9: _destruct<bf3_k9_mod_t, bf3_k9_mod_vt>(); break;
//            case 10: _destruct<bf3_k10_mod_t, bf3_k10_mod_vt>(); break;
//            case 11: _destruct<bf3_k11_mod_t, bf3_k11_mod_vt>(); break;
//            default:
//              throw std::invalid_argument("The given 'k' is not supported.");
//          }
//          break;
//        default:
//          throw std::invalid_argument("The given 'h' is not supported.");
//      }
//    }
//
//  }


  /// Returns 'true' if the Bloom filter is initialized, 'false' otherwise.
  forceinline
  u1
  is_initialized() const {
    return instance != nullptr;
  }


  block_addressing get_addressing_mode() const {
    return dtl::is_power_of_two(m)
           ? block_addressing::POWER_OF_TWO
           : block_addressing::MAGIC;
  }

  static void
  calibrate() {
    std::cout << "Running calibration..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis;

    static constexpr u32 data_size = 4*1024;
    std::vector<key_t> random_data;
    random_data.reserve(data_size);
    for (std::size_t i = 0; i < data_size; i++) {
      random_data.push_back(dis(gen));
    }

    auto& unroll_factors = sizeof(word_t) == 8 ? unroll_factors_64 : unroll_factors_32;
    for ($u32 k = 1; k <= 16; k++) {
      std::cout <<"k = " <<  std::setw(2) << k << ": " << std::flush;

      $f64 cycles_per_lookup_min = std::numeric_limits<$f64>::max();
      $u32 u_min = 1;

      std::size_t match_count = 0;
      uint32_t match_pos[dtl::BATCH_SIZE];

      for ($u32 u = 0; u <= 16; u = (u == 0) ? 1 : u*2) {
        std::cout <<  std::setw(4) << "u(" << std::setw(2) << std::to_string(u) + ") = "<< std::flush;
        unroll_factors[k-1] = u;
        auto bbf = blocked_bloomfilter::construct(1, 1, k, 4u * 1024);
        $u64 rep_cntr = 0;
        auto start = std::chrono::high_resolution_clock::now();
        auto tsc_start = _rdtsc();
        while (true) {
          std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - start;
          if (diff.count() > 0.25) break;
          dtl::batch_wise(random_data.begin(), random_data.end(), [&](const auto batch_begin, const auto batch_end) {
            match_count += bbf.batch_contains(&batch_begin[0], batch_end - batch_begin, match_pos, 0);
          });
          rep_cntr++;
        }
        auto tsc_end = _rdtsc();
        auto cycles_per_lookup = (tsc_end - tsc_start) / (data_size * rep_cntr * 1.0);
        std::cout << std::setprecision(2) << cycles_per_lookup << ", ";
        if (cycles_per_lookup < cycles_per_lookup_min) {
          cycles_per_lookup_min = cycles_per_lookup;
          u_min = u;
        }
      }
      unroll_factors[k-1] = u_min;
      std::cout << " picked u=" << u_min << " (chksum: " << match_count << ")" << std::endl;
    }
  }



};

} // namespace dtl
