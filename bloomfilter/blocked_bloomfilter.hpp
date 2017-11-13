#pragma once

#include <cmath>
#include <stdexcept>

#include <dtl/dtl.hpp>
#include <dtl/batchwise.hpp>
#include <dtl/math.hpp>
#include <dtl/mem.hpp>

#include <boost/math/common_factor.hpp>

#include <dtl/bloomfilter/block_addressing_logic.hpp>
#include <dtl/bloomfilter/blocked_bloomfilter_logic.hpp>
#include <dtl/bloomfilter/hash_family.hpp>
#include <dtl/hash.hpp>

#include <random>
#include <iomanip>
#include <chrono>


namespace dtl {

namespace internal {

//===----------------------------------------------------------------------===//
/// @see $u32& unroll_factor(u32, dtl::block_addressing, u32)
static constexpr u32 max_k = 16;

static
std::array<$u32, max_k * 5 /* different block sizes */ * 2 /* addressing modes*/>
    unroll_factors_32 = {
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  1, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  2, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  4, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  8, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w = 16, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  1, a = magic
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  2, a = magic
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  4, a = magic
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  8, a = magic
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w = 16, a = magic
  };

static
std::array<$u32, max_k * 5 /* different block sizes */ * 2 /* addressing modes*/>
    unroll_factors_64 = {
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  1, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  2, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  4, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  8, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w = 16, a = pow2
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  1, a = magic
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  2, a = magic
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  4, a = magic
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w =  8, a = magic
    1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, // w = 16, a = magic
  };
//===----------------------------------------------------------------------===//

} // namespace internal


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
    BIND,
    DESTRUCT
  };


  static constexpr dtl::block_addressing power = dtl::block_addressing::POWER_OF_TWO;
  static constexpr dtl::block_addressing magic = dtl::block_addressing::MAGIC;

  template<u32 word_cnt, u32 sector_cnt, u32 k, dtl::block_addressing addr = power>
  using bbf = dtl::blocked_bloomfilter_logic<key_t, hasher, word_t, word_cnt, sector_cnt, k, addr, dtl::mem::numa_allocator<word_t>>;


  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  /// The (desired) bit length of the Bloom filter.
  $u64 m;
  /// The (actual) bit length of the Bloom filter.
  $u64 m_actual;
  /// The number of bits set per entry.
  $u32 k;
  /// The number of words per block.
  $u32 word_cnt_per_block;
  /// The number of sectors.
  $u32 sector_cnt;
  /// Pointer to the Bloom filter logic instance.
  void* instance = nullptr;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // The API functions.
  //===----------------------------------------------------------------------===//
  std::function<void(const key_t /*key*/)>
  insert;

  std::function<void(const key_t* /*keys*/, u32 /*key_cnt*/)>
  batch_insert;

  std::function<$u1(const key_t /*key*/)>
  contains;

  std::function<$u64(const key_t* /*keys*/, u32 /*key_cnt*/, $u32* /*match_positions*/, u32 /*match_offset*/)>
  batch_contains;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Bit vector data
  //===----------------------------------------------------------------------===//
  using allocator_t = std::allocator<word_t>;
  const allocator_t allocator;
  std::vector<word_t, allocator_t> filter_data;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  blocked_bloomfilter(const size_t m, u32 k, u32 word_cnt_per_block = 1, u32 sector_cnt = 1)
      : m(m), k(k), word_cnt_per_block(word_cnt_per_block), sector_cnt(sector_cnt) {

    // Construct the Bloom filter logic instance.
    dispatch(*this, op_t::CONSTRUCT);

    // Create and init the bit vector.
    filter_data.clear();
    filter_data.resize(m_actual, 0);

    // Bind the API functions.
    dispatch(*this, op_t::BIND);

    // Check whether the constructed filter matches the given arguments.
    if (this->k != k
        || this->word_cnt_per_block != word_cnt_per_block
        || this->sector_cnt != sector_cnt) {
      dispatch(*this, op_t::DESTRUCT);
      throw std::invalid_argument("Invalid configuration: k=" + std::to_string(k)
                                  + ", w=" + std::to_string(word_cnt_per_block)
                                  + ", s=" + std::to_string(sector_cnt));
    }

  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  blocked_bloomfilter(blocked_bloomfilter&& src)
      : m(src.m), m_actual(src.m_actual), k(src.k),
        word_cnt_per_block(src.word_cnt_per_block), sector_cnt(src.sector_cnt),
        instance(src.instance),
        insert(std::move(src.insert)),
        batch_insert(std::move(src.batch_insert)),
        contains(std::move(src.contains)),
        batch_contains(std::move(src.batch_contains)),
        allocator(std::move(src.allocator)),
        filter_data(std::move(src.filter_data)) {
    // Invalidate pointer in src
    src.instance = nullptr;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  ~blocked_bloomfilter() {
    // Destruct logic instance (if any).
    if (instance != nullptr) dispatch(*this, op_t::DESTRUCT);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  blocked_bloomfilter&
  operator=(blocked_bloomfilter&& src) {
    m = src.m;
    m_actual = src.m_actual;
    k = src.k;
    word_cnt_per_block = src.word_cnt_per_block;
    sector_cnt = src.sector_cnt;
    instance = src.instance;
    insert = std::move(src.insert);
    batch_insert = std::move(src.batch_insert);
    contains = std::move(src.contains);
    batch_contains = std::move(src.batch_contains);
    // invalidate pointers
    src.instance = nullptr;
    return *this;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Dynamic Dispatching
  //===----------------------------------------------------------------------===//
  static void dispatch(blocked_bloomfilter& instance, op_t op) {
      switch (instance.word_cnt_per_block) {
        case  1: _s< 1>(instance, op); break;
        case  2: _s< 2>(instance, op); break;
        case  4: _s< 4>(instance, op); break;
        case  8: _s< 8>(instance, op); break;
        case 16: _s<16>(instance, op); break;
        default:
          throw std::invalid_argument("The given 'word_cnt_per_block' is not supported.");
      }
  };


  template<u32 w>
  static void _s(blocked_bloomfilter& instance, op_t op) {
    switch (instance.sector_cnt) {
      case  1: _k<w,  1>(instance, op); break;
      case  2: _k<w,  2>(instance, op); break;
      case  4: _k<w,  4>(instance, op); break;
      case  8: _k<w,  8>(instance, op); break;
      case 16: _k<w, 16>(instance, op); break;
      default:
        throw std::invalid_argument("The given 'sector_cnt' is not supported.");
    }
  }


  template<u32 w, u32 s>
  static void _k(blocked_bloomfilter& instance, op_t op) {
    switch (instance.k) {
      case  1: _a<w, s, boost::static_unsigned_max<1, s>::value>(instance, op); break;
      case  2: _a<w, s, boost::static_unsigned_max<( 2 % s == 0 ?  2 : 1 /*invalid*/), s>::value>(instance, op); break;
      case  3: _a<w, s, boost::static_unsigned_max<( 3 % s == 0 ?  3 : 1 /*invalid*/), s>::value>(instance, op); break;
      case  4: _a<w, s, boost::static_unsigned_max<( 4 % s == 0 ?  4 : 1 /*invalid*/), s>::value>(instance, op); break;
      case  5: _a<w, s, boost::static_unsigned_max<( 5 % s == 0 ?  5 : 1 /*invalid*/), s>::value>(instance, op); break;
      case  6: _a<w, s, boost::static_unsigned_max<( 6 % s == 0 ?  6 : 1 /*invalid*/), s>::value>(instance, op); break;
      case  7: _a<w, s, boost::static_unsigned_max<( 7 % s == 0 ?  7 : 1 /*invalid*/), s>::value>(instance, op); break;
      case  8: _a<w, s, boost::static_unsigned_max<( 8 % s == 0 ?  8 : 1 /*invalid*/), s>::value>(instance, op); break;
      case  9: _a<w, s, boost::static_unsigned_max<( 9 % s == 0 ?  9 : 1 /*invalid*/), s>::value>(instance, op); break;
      case 10: _a<w, s, boost::static_unsigned_max<(10 % s == 0 ? 10 : 1 /*invalid*/), s>::value>(instance, op); break;
      case 11: _a<w, s, boost::static_unsigned_max<(11 % s == 0 ? 11 : 1 /*invalid*/), s>::value>(instance, op); break;
      case 12: _a<w, s, boost::static_unsigned_max<(12 % s == 0 ? 12 : 1 /*invalid*/), s>::value>(instance, op); break;
      case 13: _a<w, s, boost::static_unsigned_max<(13 % s == 0 ? 13 : 1 /*invalid*/), s>::value>(instance, op); break;
      case 14: _a<w, s, boost::static_unsigned_max<(14 % s == 0 ? 14 : 1 /*invalid*/), s>::value>(instance, op); break;
      case 15: _a<w, s, boost::static_unsigned_max<(15 % s == 0 ? 15 : 1 /*invalid*/), s>::value>(instance, op); break;
      case 16: _a<w, s, boost::static_unsigned_max<16 , s>::value>(instance, op); break;
    }
  }


  template<u32 w, u32 s, u32 k>
  static void _a(blocked_bloomfilter& instance, op_t op) {
    dtl::block_addressing addr = dtl::is_power_of_two(instance.m)
                                 ? dtl::block_addressing::POWER_OF_TWO
                                 : dtl::block_addressing::MAGIC;
    switch (addr) {
      case dtl::block_addressing::POWER_OF_TWO: _u<w, s, k, dtl::block_addressing::POWER_OF_TWO>(instance, op); break;
      case dtl::block_addressing::MAGIC:        _u<w, s, k, dtl::block_addressing::MAGIC>(instance, op);        break;
    }
  }


  template<u32 w, u32 s, u32 k, dtl::block_addressing a>
  static void _u(blocked_bloomfilter& instance, op_t op) {
    switch (unroll_factor(k, a, w)) {
      case  0: _o<w, s, k, a,  0>(instance, op); break;
      case  1: _o<w, s, k, a,  1>(instance, op); break;
      case  2: _o<w, s, k, a,  2>(instance, op); break;
      case  4: _o<w, s, k, a,  4>(instance, op); break;
      case  8: _o<w, s, k, a,  8>(instance, op); break;
      default:
        throw std::invalid_argument("The given 'unroll_factor' is not supported.");
    }
  }


  template<u32 w, u32 s, u32 k, dtl::block_addressing a, u32 unroll_factor>
  static void _o(blocked_bloomfilter& instance, op_t op) {
    using _t = bbf<w, s, k, a>;
    switch (op) {
      case op_t::CONSTRUCT: instance._construct_logic<_t>();           break;
      case op_t::BIND:      instance._bind_logic<_t, unroll_factor>(); break;
      case op_t::DESTRUCT:  instance._destruct_logic<_t>();            break;
    }
  };
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Constructs a blocked Bloom filter logic instance.
  template<
      typename bf_t
  >
  void
  _construct_logic() {
    using namespace std::placeholders;

    // Instantiate a Bloom filter.
    bf_t* bf = new bf_t(m);
    instance = bf;
    k = bf_t::k;
    word_cnt_per_block = bf_t::word_cnt_per_block;
    sector_cnt = bf_t::sector_cnt;

    // Get the actual size of the filter.
    m_actual = bf->length();
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Bind API functions to the (previously constructed) Bloom filter logic.
  template<
      typename bf_t,
      u32 unroll_factor = 1
  >
  void
  _bind_logic() {
    using namespace std::placeholders;
    auto* bf = static_cast<bf_t*>(instance);

    // Bind the API functions.
    insert = std::bind(&bf_t::insert, bf, &filter_data[0], _1);
    batch_insert = std::bind(&bf_t::batch_insert, bf, &filter_data[0], _1, _2);
    contains = std::bind(&bf_t::contains, bf, &filter_data[0], _1);

    // SIMD vector length (0 = run scalar code)
    static constexpr u64 vector_len = dtl::simd::lane_count<key_t> * unroll_factor;
    batch_contains = std::bind(&bf_t::template batch_contains<vector_len>, bf, &filter_data[0], _1, _2, _3, _4);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Destructs the Bloom filter logic.
  template<
      typename bf_t // the scalar bloomfilter_h1 type
  >
  void
  _destruct_logic() {
    bf_t* bf = static_cast<bf_t*>(instance);
    delete bf;
    instance = nullptr;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the block addressing mode.
  block_addressing get_addressing_mode() const {
    return dtl::is_power_of_two(m)
           ? block_addressing::POWER_OF_TWO
           : block_addressing::MAGIC;
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
  /// Returns the name of the Bloom filter instance including the most
  /// important parameters.
  std::string
  name() {
    return "blocked_bloom_multiword[size=" + std::to_string(size_in_bytes())
        + ",word_size=" + std::to_string(sizeof(word_t))
           + ",k=" + std::to_string(k)
           + ",w=" + std::to_string(word_cnt_per_block)
           + ",s=" + std::to_string(sector_cnt)
           + ",u=" + std::to_string(unroll_factor(k, get_addressing_mode(), word_cnt_per_block))
           + ",addr=" + (get_addressing_mode() == dtl::block_addressing::POWER_OF_TWO ? "pow2" : "magic")
        + "]";
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Runs the calibration code. Results are memorized in global variables.
  static void
  calibrate() __attribute__ ((noinline)) {
    std::cout << "Running calibration..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis;

    static constexpr u32 data_size = 4u*1024*8;
    std::vector<key_t> random_data;
    random_data.reserve(data_size);
    for (std::size_t i = 0; i < data_size; i++) {
      random_data.push_back(dis(gen));
    }

    static const u32 max_unroll_factor = 8;
    for ($u32 w = 1; w <= 16; w *= 2) {
      for (auto addr_mode : {dtl::block_addressing::POWER_OF_TWO, dtl::block_addressing::MAGIC}) {
        for ($u32 k = 1; k <= 16; k++) {
          try {
            std::cout << "w = " << std::setw(2) << w << ", "
                      << "addr = " << std::setw(5) << (addr_mode == block_addressing::POWER_OF_TWO ? "pow2" : "magic") << ", "
                      << "k = " <<  std::setw(2) << k << ": " << std::flush;

            $f64 cycles_per_lookup_min = std::numeric_limits<$f64>::max();
            $u32 u_min = 1;

            std::size_t match_count = 0;
            uint32_t match_pos[dtl::BATCH_SIZE];

            // baselines
            $f64 cycles_per_lookup_u0 = 0.0;
            $f64 cycles_per_lookup_u1 = 0.0;
            for ($u32 u = 0; u <= max_unroll_factor; u = (u == 0) ? 1 : u*2) {
              std::cout << std::setw(2) << "u(" << std::to_string(u) + ") = "<< std::flush;
              unroll_factor(k, addr_mode, w) = u;
              blocked_bloomfilter bbf(data_size + 128 * static_cast<u32>(addr_mode), k, w, w); // word_cnt = sector_cnt
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
              if (u == 0) cycles_per_lookup_u0 = cycles_per_lookup;
              if (u == 1) cycles_per_lookup_u1 = cycles_per_lookup;
              std::cout << std::setprecision(3) << std::setw(4) << std::right << cycles_per_lookup << ", ";
              if (cycles_per_lookup < cycles_per_lookup_min) {
                cycles_per_lookup_min = cycles_per_lookup;
                u_min = u;
              }
            }
            unroll_factor(k, addr_mode, w) = u_min;
            std::cout << " picked u = " << unroll_factor(k, addr_mode, w)
                      << ", speedup over u(0) = " << std::setprecision(3) << std::setw(4) << std::right << (cycles_per_lookup_u0 / cycles_per_lookup_min)
                      << ", speedup over u(1) = " << std::setprecision(3) << std::setw(4) << std::right << (cycles_per_lookup_u1 / cycles_per_lookup_min)
                      << " (chksum: " << match_count << ")" << std::endl;

          } catch (...) {
            std::cout<< " -> Failed to calibrate for k = " << k << "." << std::endl;
          }
        }
      }
    }
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the SIMD unrolling factor for the given k and addressing mode.
  /// Note: unrolling by 0 means -> scalar code (no SIMD)
  static $u32&
  unroll_factor(u32 k, dtl::block_addressing addr_mode, u32 word_cnt_per_block) {
    auto& unroll_factors = sizeof(word_t) == 8
                           ? internal::unroll_factors_64
                           : internal::unroll_factors_32;
    return unroll_factors[
        internal::max_k * dtl::log_2(word_cnt_per_block)
        + (k - 1)
        + (static_cast<u32>(addr_mode) * internal::max_k * 5)
    ];
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Force unroll factor for all implementations (used for benchmarking)
  static $u32&
  force_unroll_factor(u32 u) {
    for ($u32 w = 1; w <= 16; w *= 2) {
      for (auto addr_mode : {dtl::block_addressing::POWER_OF_TWO, dtl::block_addressing::MAGIC}) {
        for ($u32 k = 1; k <= 16; k++) {
          unroll_factor(k, addr_mode, w) = u;
        }
      }
    }
  }
  //===----------------------------------------------------------------------===//

};

} // namespace dtl
