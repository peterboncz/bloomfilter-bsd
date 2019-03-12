#pragma once

#include <dtl/dtl.hpp>
#include <amsfilter/internal/blocked_bloomfilter_template.hpp>

#define AMSFILTER_PARTIAL_BUILD

namespace amsfilter {
namespace internal {
namespace resolve {
//===----------------------------------------------------------------------===//
template<typename Fn, u32 w, u32 s, u32 z, u32 k, dtl::block_addressing a>
void
_term(const dtl::blocked_bloomfilter_config& conf, Fn& fn) {
  fn.template operator()<w, s, z, k, a>(conf);
}

// Block addressing mode.
template<typename Fn, u32 w, u32 s, u32 z, u32 k>
void
_a(const dtl::blocked_bloomfilter_config& conf, Fn& fn) {
#ifdef AMSFILTER_PARTIAL_BUILD
  _term<Fn, w, s, z, k, dtl::block_addressing::POWER_OF_TWO>(conf, fn);
#else
  switch (conf.addr_mode) {
    case  dtl::block_addressing::POWER_OF_TWO:
      _term<Fn, w, s, z, k, dtl::block_addressing::POWER_OF_TWO>(conf, fn);
      break
    case  dtl::block_addressing::MAGIC:
      _term<Fn, w, s, z, k, dtl::block_addressing::MAGIC>(conf, fn);
      break;
    default:
      _term<Fn, w, s, z, k, dtl::block_addressing::DYNAMIC>(conf, fn);
  }
#endif
}

// No. of hash functions (k).
template<typename Fn, u32 w, u32 s, u32 z>
void
_k(const dtl::blocked_bloomfilter_config& conf, Fn& fn) {
  switch (conf.k) {
    case  1: _a<Fn, w, s, z,  1>(conf, fn); break;
    case  2: _a<Fn, w, s, z,  2>(conf, fn); break;
    case  3: _a<Fn, w, s, z,  3>(conf, fn); break;
    case  4: _a<Fn, w, s, z,  4>(conf, fn); break;
    case  5: _a<Fn, w, s, z,  5>(conf, fn); break;
    case  6: _a<Fn, w, s, z,  6>(conf, fn); break;
    case  7: _a<Fn, w, s, z,  7>(conf, fn); break;
    case  8: _a<Fn, w, s, z,  8>(conf, fn); break;
#ifndef AMSFILTER_PARTIAL_BUILD
    case  9: _a<Fn, w, s, z,  9>(conf, fn); break;
    case 10: _a<Fn, w, s, z, 10>(conf, fn); break;
    case 11: _a<Fn, w, s, z, 11>(conf, fn); break;
    case 12: _a<Fn, w, s, z, 12>(conf, fn); break;
    case 13: _a<Fn, w, s, z, 13>(conf, fn); break;
    case 14: _a<Fn, w, s, z, 14>(conf, fn); break;
    case 15: _a<Fn, w, s, z, 15>(conf, fn); break;
    case 16: _a<Fn, w, s, z, 16>(conf, fn); break;
#endif
    default: _a<Fn, w, s, z,  0>(conf, fn);
  }
}

// Zone count
template<typename Fn, u32 w, u32 s>
void
_z(const dtl::blocked_bloomfilter_config& conf, Fn& fn) {
  switch (conf.zone_cnt) {
    case   1: _k<Fn, w, s,   1>(conf, fn); break;
    case   2: _k<Fn, w, s,   2>(conf, fn); break;
    case   4: _k<Fn, w, s,   4>(conf, fn); break;
    case   8: _k<Fn, w, s,   8>(conf, fn); break;
#ifndef AMSFILTER_PARTIAL_BUILD
    case  16: _k<Fn, w, s,  16>(conf, fn); break;
    case  32: _k<Fn, w, s,  32>(conf, fn); break;
    case  64: _k<Fn, w, s,  64>(conf, fn); break;
    case 128: _k<Fn, w, s, 128>(conf, fn); break;
#endif
    default:  _k<Fn, w, s,   0>(conf, fn);
  }
}

// Sector count
template<typename Fn, u32 w>
void
_s(const dtl::blocked_bloomfilter_config& conf, Fn& fn) {
  switch (conf.sector_cnt) {
    case   1: _z<Fn, w,   1>(conf, fn); break;
    case   2: _z<Fn, w,   2>(conf, fn); break;
    case   4: _z<Fn, w,   4>(conf, fn); break;
    case   8: _z<Fn, w,   8>(conf, fn); break;
#ifndef AMSFILTER_PARTIAL_BUILD
    case  16: _z<Fn, w,  16>(conf, fn); break;
    case  32: _z<Fn, w,  32>(conf, fn); break;
    case  64: _z<Fn, w,  64>(conf, fn); break;
    case 128: _z<Fn, w, 128>(conf, fn); break;
#endif
    default:  _z<Fn, w,   0>(conf, fn);
  }
}

// Word count
template<typename Fn>
void
_w(const dtl::blocked_bloomfilter_config& conf, Fn& fn) {
  switch (conf.word_cnt_per_block) {
    case   1: _s<Fn,   1>(conf, fn); break;
    case   2: _s<Fn,   2>(conf, fn); break;
    case   4: _s<Fn,   4>(conf, fn); break;
    case   8: _s<Fn,   8>(conf, fn); break;
#ifndef AMSFILTER_PARTIAL_BUILD
    case  16: _s<Fn,  16>(conf, fn); break;
    case  32: _s<Fn,  32>(conf, fn); break;
    case  64: _s<Fn,  64>(conf, fn); break;
    case 128: _s<Fn, 128>(conf, fn); break;
#endif
    default:  _s<Fn,   0>(conf, fn);
  }
}
//===----------------------------------------------------------------------===//
} // namespace resolve
//===----------------------------------------------------------------------===//
// Hard to explain
template<typename Fn>
void
get_instance(const dtl::blocked_bloomfilter_config& conf, Fn& fn) {
  resolve::_w<Fn>(conf,fn);
}
//===----------------------------------------------------------------------===//
} // namespace internal
} // namespace amsfilter

