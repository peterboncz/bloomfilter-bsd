#pragma once

#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp>
#include <amsfilter/internal/blocked_bloomfilter_template.hpp>

namespace amsfilter {
namespace internal {
namespace resolve {
//===----------------------------------------------------------------------===//
struct valid_t {};
struct invalid_t {};

static void
_fail(const dtl::blocked_bloomfilter_config& conf) {
  throw std::invalid_argument(
      "Failed to construct a blocked Bloom filter with the parameters w="
          + std::to_string(conf.word_cnt_per_block)
          + ", s=" + std::to_string(conf.sector_cnt)
          + ", z=" + std::to_string(conf.zone_cnt)
          + ", k=" + std::to_string(conf.k)
          + ", and, a=" + std::to_string(static_cast<i32>(conf.addr_mode))
          + ".");
}

template<typename Fn, u32 w, u32 s, u32 z, u32 k, dtl::block_addressing a>
static void
_term(const dtl::blocked_bloomfilter_config& conf, Fn& fn, const invalid_t& /*selector*/) {
  _fail(conf);
}
template<typename Fn, u32 w, u32 s, u32 z, u32 k, dtl::block_addressing a>
static void
_term(const dtl::blocked_bloomfilter_config& conf, Fn& fn, const valid_t& /*selector*/) {
  fn.template operator()<w, s, z, k, a>(conf);
}

template<typename Fn, u32 w, u32 s, u32 z, u32 k, dtl::block_addressing a>
static void
_v(const dtl::blocked_bloomfilter_config& conf, Fn& fn) {
  constexpr u1 config_is_valid = bbf_config_is_valid<w, s, z, k, a>::value;
  using selector_t =
      typename std::conditional<config_is_valid, valid_t, invalid_t>::type;
  _term<Fn, w, s, z, k, a>(conf, fn, selector_t());
}

// Block addressing mode.
template<typename Fn, u32 w, u32 s, u32 z, u32 k>
static void
_a(const dtl::blocked_bloomfilter_config& conf, Fn& fn, const valid_t& /*selector*/) {
#if defined(AMSFILTER_NO_MAGIC)
  _v<Fn, w, s, z, k, dtl::block_addressing::POWER_OF_TWO>(conf, fn);
#else
  switch (conf.addr_mode) {
    case  dtl::block_addressing::POWER_OF_TWO:
      _v<Fn, w, s, z, k, dtl::block_addressing::POWER_OF_TWO>(conf, fn);
      break;
    case  dtl::block_addressing::MAGIC:
      _v<Fn, w, s, z, k, dtl::block_addressing::MAGIC>(conf, fn);
      break;
    default:
      _fail(conf);
  }
#endif
}
template<typename Fn, u32 w, u32 s, u32 z, u32 k>
void
_a(const dtl::blocked_bloomfilter_config& conf, Fn& fn, const invalid_t& /*selector*/) {
  _fail(conf);
}

// Number of hash functions (k).
template<typename Fn, u32 w, u32 s, u32 z, u32 k>
static void
_k_post(const dtl::blocked_bloomfilter_config& conf, Fn& fn) {
  using selector_t =
      typename std::conditional<(k % z == 0), valid_t, invalid_t>::type;
  const selector_t selector;
  _a<Fn, w, s, z, k>(conf, fn, selector);
};
template<typename Fn, u32 w, u32 s, u32 z>
static void
_k(const dtl::blocked_bloomfilter_config& conf, Fn& fn, const valid_t& /*selector*/) {
  switch (conf.k) {
    case  1: _k_post<Fn, w, s, z,  1>(conf, fn); break;
    case  2: _k_post<Fn, w, s, z,  2>(conf, fn); break;
    case  3: _k_post<Fn, w, s, z,  3>(conf, fn); break;
    case  4: _k_post<Fn, w, s, z,  4>(conf, fn); break;
    case  5: _k_post<Fn, w, s, z,  5>(conf, fn); break;
    case  6: _k_post<Fn, w, s, z,  6>(conf, fn); break;
    case  7: _k_post<Fn, w, s, z,  7>(conf, fn); break;
    case  8: _k_post<Fn, w, s, z,  8>(conf, fn); break;
#ifndef AMSFILTER_PARTIAL_BUILD
    case  9: _k_post<Fn, w, s, z,  9>(conf, fn); break;
    case 10: _k_post<Fn, w, s, z, 10>(conf, fn); break;
    case 11: _k_post<Fn, w, s, z, 11>(conf, fn); break;
    case 12: _k_post<Fn, w, s, z, 12>(conf, fn); break;
    case 13: _k_post<Fn, w, s, z, 13>(conf, fn); break;
    case 14: _k_post<Fn, w, s, z, 14>(conf, fn); break;
    case 15: _k_post<Fn, w, s, z, 15>(conf, fn); break;
    case 16: _k_post<Fn, w, s, z, 16>(conf, fn); break;
#endif
    default: _fail(conf);
  }
}
template<typename Fn, u32 w, u32 s, u32 z>
static void
_k(const dtl::blocked_bloomfilter_config& conf, Fn& fn, const invalid_t& /*selector*/) {
  _fail(conf);
}

// Zone count
template<typename Fn, u32 w, u32 s, u32 z>
static void
_z_post(const dtl::blocked_bloomfilter_config& conf, Fn& fn) {
  // Zone count must not exceed the sector count.
  using selector_t =
      typename std::conditional<(z <= s), valid_t, invalid_t>::type;
  const selector_t selector;
  _k<Fn, w, s, z>(conf, fn, selector);
}
template<typename Fn, u32 w, u32 s>
static void
_z(const dtl::blocked_bloomfilter_config& conf, Fn& fn, const valid_t& /*selector*/) {
  switch (conf.zone_cnt) {
    case   1: _z_post<Fn, w, s,   1>(conf, fn); break;
    case   2: _z_post<Fn, w, s,   2>(conf, fn); break;
    case   4: _z_post<Fn, w, s,   4>(conf, fn); break;
    case   8: _z_post<Fn, w, s,   8>(conf, fn); break;
    default:  _fail(conf);
  }
}
template<typename Fn, u32 w, u32 s>
static void
_z(const dtl::blocked_bloomfilter_config& conf, Fn& fn, const invalid_t& /*selector*/) {
  _fail(conf);
}

// Sector count
template<typename Fn, u32 w, u32 s>
static void
_s_post(const dtl::blocked_bloomfilter_config& conf, Fn& fn) {
  // The sector count must not exceed the word count.
  using selector_t =
  typename std::conditional<(w >= s), valid_t, invalid_t>::type;
  const selector_t selector;
  _z<Fn, w, s>(conf, fn, selector);
};
template<typename Fn, u32 w>
static void
_s(const dtl::blocked_bloomfilter_config& conf, Fn& fn) {
  switch (conf.sector_cnt) {
    case   1: _s_post<Fn, w,   1>(conf, fn); break;
    case   2: _s_post<Fn, w,   2>(conf, fn); break;
    case   4: _s_post<Fn, w,   4>(conf, fn); break;
    case   8: _s_post<Fn, w,   8>(conf, fn); break;
#ifndef AMSFILTER_PARTIAL_BUILD
    case  16: _s_post<Fn, w,  16>(conf, fn); break;
    case  32: _s_post<Fn, w,  32>(conf, fn); break;
//    case  64: _s_post<Fn, w,  64>(conf, fn); break;
//    case 128: _s_post<Fn, w, 128>(conf, fn); break;
#endif
    default:  _fail(conf);
  }
}

// Word count
template<typename Fn>
static void
_w(const dtl::blocked_bloomfilter_config& conf, Fn& fn) {
  switch (conf.word_cnt_per_block) {
    case   1: _s<Fn,   1>(conf, fn); break;
    case   2: _s<Fn,   2>(conf, fn); break;
    case   4: _s<Fn,   4>(conf, fn); break;
    case   8: _s<Fn,   8>(conf, fn); break;
#ifndef AMSFILTER_PARTIAL_BUILD
    case  16: _s<Fn,  16>(conf, fn); break;
    case  32: _s<Fn,  32>(conf, fn); break;
//    case  64: _s<Fn,  64>(conf, fn); break;
//    case 128: _s<Fn, 128>(conf, fn); break;
#endif
    default:  _fail(conf);
  }
}
//===----------------------------------------------------------------------===//
} // namespace resolve
//===----------------------------------------------------------------------===//
// Hard to explain
template<typename Fn>
static void
get_instance(const dtl::blocked_bloomfilter_config& conf, Fn& fn) {
  if (conf.word_size != 4) resolve::_fail(conf);
  resolve::_w<Fn>(conf,fn);
}
//===----------------------------------------------------------------------===//
} // namespace internal
} // namespace amsfilter
