#pragma once

#include <dtl/dtl.hpp>
#include <dtl/filter/cuckoofilter/cuckoofilter_config.hpp>
#include <amsfilter/internal/cuckoofilter_template.hpp>
#include <amsfilter/config.hpp>

namespace amsfilter {
namespace internal {
namespace resolve {
namespace cuckoofilter {
//===----------------------------------------------------------------------===//
struct valid_t {};
struct invalid_t {};

static void
_fail(const  dtl::cuckoofilter::config& conf) {
  throw std::invalid_argument(
      "Failed to construct a Cuckoo filter with the parameters "
            "bits_per_tag=" + std::to_string(conf.bits_per_tag)
          + ", tags_per_bucket=" + std::to_string(conf.tags_per_bucket)
          + ", and, a=" + std::to_string(static_cast<i32>(conf.addr_mode))
          + ".");
}

template<typename Fn,
    u32 bits_per_tag, u32 tags_per_bucket, dtl::block_addressing addr>
static void
_term(const  dtl::cuckoofilter::config& conf, Fn& fn, const invalid_t& /*selector*/) {
  _fail(conf);
}
template<typename Fn,
    u32 bits_per_tag, u32 tags_per_bucket, dtl::block_addressing addr>
static void
_term(const  dtl::cuckoofilter::config& conf, Fn& fn, const valid_t& /*selector*/) {
  fn.template operator()<bits_per_tag, tags_per_bucket, addr>(conf);
}

template<typename Fn,
    u32 bits_per_tag, u32 tags_per_bucket, dtl::block_addressing addr>
static void
_v(const  dtl::cuckoofilter::config& conf, Fn& fn) {
  constexpr u1 config_is_valid =
      cf_config_is_valid<bits_per_tag, tags_per_bucket, addr>::value;
  using selector_t =
      typename std::conditional<config_is_valid, valid_t, invalid_t>::type;
  _term<Fn, bits_per_tag, tags_per_bucket, addr>(conf, fn, selector_t());
}

// Block addressing mode.
template<typename Fn, u32 bits_per_tag, u32 tags_per_bucket>
static void
_a(const  dtl::cuckoofilter::config& conf, Fn& fn) {
#if defined(AMSFILTER_NO_MAGIC)
  _v<Fn, bits_per_tag, tags_per_bucket, dtl::block_addressing::POWER_OF_TWO>(
      conf, fn);
#else
  switch (conf.addr_mode) {
    case  dtl::block_addressing::POWER_OF_TWO:
      _v<Fn, bits_per_tag, tags_per_bucket, dtl::block_addressing::POWER_OF_TWO>(
          conf, fn);
      break;
    case  dtl::block_addressing::MAGIC:
      _v<Fn, bits_per_tag, tags_per_bucket, dtl::block_addressing::MAGIC>(
          conf, fn);
      break;
    default:
      _fail(conf);
  }
#endif
}

// Tags per bucket (associativity).
template<typename Fn, u32 bits_per_tag>
static void
_t(const  dtl::cuckoofilter::config& conf, Fn& fn) {
  switch (conf.tags_per_bucket) {
    case  1: _a<Fn, bits_per_tag, 1>(conf, fn); break;
    case  2: _a<Fn, bits_per_tag, 2>(conf, fn); break;
    case  4: _a<Fn, bits_per_tag, 4>(conf, fn); break;
    default:  _fail(conf);
  }
}

// Bits per tag.
template<typename Fn>
static void
_b(const dtl::cuckoofilter::config& conf, Fn& fn) {
  switch (conf.bits_per_tag) {
    case   4: _t<Fn,  4>(conf, fn); break;
    case   8: _t<Fn,  8>(conf, fn); break;
    case  12: _t<Fn, 12>(conf, fn); break;
    case  16: _t<Fn, 16>(conf, fn); break;
    case  32: _t<Fn, 32>(conf, fn); break;
    default:  _fail(conf);
  }
}

// Hard to explain
template<typename Fn>
static void
get_instance(const dtl::cuckoofilter::config& conf, Fn& fn) {
  resolve::cuckoofilter::_b<Fn>(conf,fn);
}
//===----------------------------------------------------------------------===//
} // namespace cuckoofilter
} // namespace resolve
} // namespace internal
} // namespace amsfilter
