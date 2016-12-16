#pragma once

#ifndef _DTL_SIMD_INCLUDED
#error "Never use <dtl/simd/intrin_x64.hpp> directly; include <dtl/simd.hpp> instead."
#endif

#include "../adept.hpp"
#include <bitset>

namespace dtl {
namespace simd {

/// ---
/// Implements a scalar fall back if no SIMD implementation is available
/// for the underlying hardware.  The fall back can also be useful for debugging
/// algorithms.
///
/// Note: primitve type Tp =  vector type Tv
/// ---

namespace {

/// The mask is a single boolean in that case.
struct mask {
  $u1 data;
  inline u1 all() const { return data; };
  inline u1 any() const { return data; };
  inline u1 none() const { return !data; };
  inline void set(u1 value) {
    data = value;
  }
  inline void set(u64 /* idx */, u1 value) {
    data = value;
  };
  inline void set(u64 bits) {
    data = bits != 0;
  };
  inline u1 get(u64 /* idx */) const {
    return data;
  };
  inline mask bit_and(const mask& o) const { return mask { data & o.data }; };
  inline mask bit_or(const mask& o) const { return mask { data | o.data }; };
  inline mask bit_xor(const mask& o) const { return mask { data ^ o.data }; };
  inline mask bit_not() const { return mask { !data }; };
};

} // anonymous namespace

/// Wraps a scalar value into the vector interface.
template<typename Tp>
struct vs<Tp, 1> : base<Tp, 1> {
  using type = Tp;
  using mask_type = mask;
  type data;
};

namespace {
using native_mask_t = mask;
}

template<typename Tp>
struct broadcast<Tp, Tp, Tp> : vector_fn<Tp, Tp, Tp> {
  using fn = vector_fn<Tp, Tp, Tp>;
  inline typename fn::vector_type
  operator()(const typename fn::value_type& a) const noexcept {
    return a;
  }
  inline typename fn::vector_type
  operator()(const typename fn::value_type& a,
             // merge masking
             const typename fn::vector_type& src,
             const native_mask_t mask) const noexcept {
    return mask.data ? a : src;
  }
};

template<typename Tp>
struct set<Tp, Tp, Tp> : vector_fn<Tp, Tp, Tp> {
  using fn = vector_fn<Tp, Tp, Tp>;
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& a) const noexcept {
    return a;
  }
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& a,
             // merge masking
             const typename fn::vector_type& src,
             const native_mask_t mask) const noexcept {
    return mask.data ? a : src;
  }
};


// Load
template<typename Tp, typename Ta>
struct gather<Tp, Tp, Ta> : vector_fn<Tp, Tp, Ta> {
  using fn = vector_fn<Tp, Tp, Ta>;
  inline typename fn::vector_type
  operator()(u8* const base_addr,
             const typename fn::argument_type& idxs) const noexcept {
    return *reinterpret_cast<typename fn::vector_type*>(idxs);
  }
};

// Store
template<typename Tp, typename Ti>
struct scatter<Tp, Tp, Ti> : vector_fn<Tp, Tp, Ti> {
  using fn = vector_fn<Tp, Tp, Ti>;
  inline typename fn::vector_type
  operator()(u8* const base_addr,
             const typename fn::argument_type& idxs,
             const typename fn::vector_type& what) const noexcept {
    base_addr[idxs] = what;
  }
};


// Arithmetic
template<typename Tp>
struct plus<Tp, Tp> : vector_fn<Tp, Tp> {
  using fn = vector_fn<Tp>;
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs) const noexcept {
    return lhs + rhs;
  }
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs,
             // merge masking
             const typename fn::vector_type& src,
             const native_mask_t mask) const noexcept {
    return mask.data ? lhs + rhs : src;
  }
};

template<typename Tp>
struct minus<Tp, Tp> : vector_fn<Tp, Tp> {
  using fn = vector_fn<Tp, Tp>;
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs) const noexcept {
    return lhs - rhs;
  }
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs,
             // merge masking
             const typename fn::vector_type& src,
             const native_mask_t mask) const noexcept {
    return mask.data ? lhs - rhs : src;
  }
};

template<typename Tp>
struct multiplies<Tp, Tp> : vector_fn<Tp, Tp> {
  using fn = vector_fn<Tp, Tp>;
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs) const noexcept {
    return lhs * rhs;
  }
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs,
             // merge masking
             const typename fn::vector_type& src,
             const native_mask_t mask) const noexcept {
    return mask.data ? lhs * rhs : src;
  }
};


// Shift
template<typename Tp, typename Ta>
struct shift_left<Tp, Tp, Ta> : vector_fn<Tp, Tp, Ta> {
  using fn = vector_fn<Tp, Tp, Ta>;
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::argument_type& count) const noexcept {
    return lhs << count;
  }
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs,
             // merge masking
             const typename fn::vector_type& src,
             const native_mask_t mask) const noexcept {
    return mask.data ? lhs << rhs : src;
  }
};

template<typename Tp, typename Ta>
struct shift_left_var<Tp, Tp, Ta> : vector_fn<Tp, Tp, Ta> {
  using fn = vector_fn<Tp, Tp, Ta>;
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::argument_type& count) const noexcept {
    return lhs << count;
  }
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs,
             // merge masking
             const typename fn::vector_type& src,
             const native_mask_t mask) const noexcept {
    return mask.data ? lhs << rhs : src;
  }
};

template<typename Tp, typename Ta>
struct shift_right<Tp, Tp, Ta> : vector_fn<Tp, Tp, Ta> {
  using fn = vector_fn<Tp, Tp, Ta>;
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::argument_type& count) const noexcept {
    return lhs >> count;
  }
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs,
             // merge masking
             const typename fn::vector_type& src,
             const native_mask_t mask) const noexcept {
    return mask.data ? lhs >> rhs : src;
  }
};

template<typename Tp, typename Ta>
struct shift_right_var<Tp, Tp, Ta> : vector_fn<Tp, Tp, Ta> {
  using fn = vector_fn<Tp, Tp, Ta>;
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::argument_type& count) const noexcept {
    return lhs >> count;
  }
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs,
             // merge masking
             const typename fn::vector_type& src,
             const native_mask_t mask) const noexcept {
    return mask.data ? lhs >> rhs : src;
  }
};


// Bitwise operators
template<typename Tp, typename Tv>
struct bit_and<Tp, Tv> : vector_fn<Tp, Tv> {
  using fn = vector_fn<Tp, Tv>;
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs) const noexcept {
    return lhs & rhs;
  }
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs,
             // merge masking
             const typename fn::vector_type& src,
             const native_mask_t mask) const noexcept {
    return mask.data ? lhs & rhs : src;
  }
};

template<typename Tp, typename Tv>
struct bit_or<Tp, Tv> : vector_fn<Tp, Tv> {
  using fn = vector_fn<Tp, Tv>;
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs) const noexcept {
    return lhs | rhs;
  }
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs,
             // merge masking
             const typename fn::vector_type& src,
             const native_mask_t mask) const noexcept {
    return mask.data ? lhs | rhs : src;
  }
};

template<typename Tp, typename Tv>
struct bit_xor<Tp, Tv> : vector_fn<Tp, Tv> {
  using fn = vector_fn<Tp, Tv>;
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs) const noexcept {
    return lhs ^ rhs;
  }
  inline typename fn::vector_type
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs,
             // merge masking
             const typename fn::vector_type& src,
             const native_mask_t mask) const noexcept {
    return mask.data ? lhs ^ rhs : src;
  }
};


// Comparison
template<typename Tp>
struct less<Tp, Tp, Tp, mask> : vector_fn<Tp, Tp, Tp, mask> {
  using fn = vector_fn<Tp, Tp, Tp, mask>;
  inline mask
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs) const noexcept {
    return mask { lhs < rhs };
  }
};

template<typename Tp>
struct greater<Tp, Tp, Tp, mask> : vector_fn<Tp, Tp, Tp, mask> {
  using fn = vector_fn<Tp, Tp, Tp, mask>;
  inline mask
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs) const noexcept {
    return mask { lhs > rhs };
  }
};

template<typename Tp>
struct equal<Tp, Tp, Tp, mask> : vector_fn<Tp, Tp, Tp, mask> {
  using fn = vector_fn<Tp, Tp, Tp, mask>;
  inline mask
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs) const noexcept {
    return mask { lhs == rhs };
  }
};

template<typename Tp>
struct not_equal<Tp, Tp, Tp, mask> : vector_fn<Tp, Tp, Tp, mask> {
  using fn = vector_fn<Tp, Tp, Tp, mask>;
  inline mask
  operator()(const typename fn::vector_type& lhs,
             const typename fn::vector_type& rhs) const noexcept {
    return mask { lhs != rhs };
  }
};


} // namespace simd
} // namespace dtl