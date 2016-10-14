#pragma once

#ifndef _DTL_SIMD_INCLUDED
#error "Never use <dtl/simd/vec.hpp> directly; include <dtl/simd.hpp> instead."
#endif

#include "../adept.hpp"
#include "../math.hpp"

#include <array>
#include <bitset>
#include <functional>

namespace dtl {
namespace simd {

/// Base class for a vector consisting of N primitive values of type Tp
template<typename Tp, u64 N>
struct base {
  using type = Tp;
  static constexpr u64 value = N;
};

/// Recursive template to find the largest possible implementation.
template<typename Tp, u64 N>
struct vs : vs<Tp, N / 2> {
  static_assert(is_power_of_two(N), "Template parameter 'N' must be a power of two.");
};

} // namespace simd
} // namespace dtl

// include architecture dependent implementations...
#include "intrinsics.hpp"

namespace dtl {
namespace simd {

/*
/// smallest vector type, consisting of 1 component (used when no specialization is found)
template<typename T>
struct vs<T, 1> {
  static constexpr u64 value = 1;
  using type = vec<T, 1>;
  type data;
};
*/

/*
template<typename T>
struct vs<T, 32> {
  static constexpr u64 value = 32;
  using type = vec<T, 32>;
  type data;
};
*/

/// The general vector class with N components of the (primitive) type T.
///
/// If there exists a native vector type that can hold N values of type T, e.g. __m256i,
/// then an instance makes direct use of it. If the N exceeds the size of the largest
/// available native vector type an instance will be a composition of multiple (smaller)
/// native vectors.
template<typename Tp, u64 N>
struct v {
  static_assert(is_power_of_two(N), "Template parameter 'N' must be a power of two.");
  // TODO assert fundamental type

  /// The overall length of the vector, in terms of number of elements.
  static constexpr u64 length = N;

  /// The native vector wrapper that is used under the hood.
  /// Note: The wrapper determines the largest available native vector type.
  using nested_vector = vs<Tp, N>; // TODO make it a template parameter. give the user the possibility to specify the native vector type

  /// The length of the native vector, in terms of number of elements.
  static constexpr u64 nested_vector_length = nested_vector::value;

  /// The number of nested native vectors, if the vector is a composition of multiple smaller vectors, 1 otherwise.
  static constexpr u64 nested_vector_cnt = N / nested_vector::value;

  /// True, if the vector is a composition of multiple native vectors, false otherwise.
  static constexpr u1 is_compound = (nested_vector_cnt != 1);

  /// The native vector type (e.g., __m256i).
  using nested_type = typename nested_vector::type;

  /// Helper to typedef a compound type.
  template<typename T_inner, u64 Cnt>
  using make_compound = typename std::array<T_inner, Cnt>;

  /// The compound vector type. Note: Is the same as nested_type, if not compound.
  using compound_type = typename std::conditional<is_compound,
      make_compound<nested_type, nested_vector_cnt>,
      nested_type>::type;

  /// The actual vector data. (the one and only non-static member variable of this class).
  compound_type data;

  /// The native 'mask' type of the surrounding vector.
  using nested_mask_type = typename nested_vector::mask_type;

  /// The 'mask' type is a composition if the surrounding vector is also a composition.
  using compound_mask_type = typename std::conditional<is_compound,
      make_compound<nested_mask_type, nested_vector_cnt>,
      nested_mask_type>::type;

  /// The mask type (template) of the surrounding vector.
  ///
  /// As the vector can be a composition of multiple (smaller) native
  /// vectors, the same applies for the mask type.
  /// Note, that the mask implementations are architecture dependent.
  ///
  /// General rules for working with masks:
  ///  1) Mask are preferably created by comparison operations and used
  ///     with masked vector operations. Manual construction should be avoided.
  ///  2) Avoid materialization. Instances should have a very short lifetime
  ///     and are not supposed to be stored in main-memory. Use the 'to_int'
  ///     function to obtain a bitmask represented as an integer.
  ///  3) Avoid (costly) direct access through the set/get functions. On pre-KNL
  ///     architectures this has a severe performance impact.
  ///  4) The special functions 'all', 'any' and 'none' are supposed to be
  ///     fast and efficient on all architectures. - Semantics are equal to
  ///     the std::bitset implementations.
  ///  5) Bitwise operations are supposed to be fast and efficient on all
  ///     architectures.
  struct m {

    /// The actual mask data. (the one and only non-static member variable of this class)
    compound_mask_type data;

    struct all_fn {
      constexpr u1 operator()(const nested_mask_type& mask) const {
        return mask.all();
      }
      constexpr u1 aggr(u1 a, u1 b) const { return a & b; };
    };

    struct any_fn {
      constexpr u1 operator()(const nested_mask_type& mask) const {
        return mask.any();
      }
      constexpr u1 aggr(u1 a, u1 b) const { return a | b; };
    };

    struct none_fn {
      constexpr u1 operator()(const nested_mask_type& mask) const {
        return mask.none();
      }
      constexpr u1 aggr(u1 a, u1 b) const { return a & b; };
    };

    template<u1 Compound = false, typename Fn>
    static inline u1
    op(Fn fn, const nested_mask_type& mask) {
      return fn(mask);
    }

    template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
    static inline u1
    op(Fn fn, const compound_mask_type& masks) {
      $u1 result = op<!Compound>(fn, masks[0]);
      for ($u64 i = 0; i < nested_vector_cnt; i++) {
        result = fn.aggr(result, op<!Compound>(fn, masks[i]));
      }
      return result;
    }

    /// Returns true if all boolean values in the mask are true, false otherwise.
    u1 all() const { return op<is_compound>(all_fn(), data); }

    /// Returns true if at least one boolean value in the mask is true, false otherwise.
    u1 any() const { return op<is_compound>(any_fn(), data); }

    /// Returns true if all boolean values in the mask are false, false otherwise.
    u1 none() const { return op<is_compound>(none_fn(), data); }

    /// Sets the bit a position 'idx' to the given 'value'.
    template<u1 Compound = false>
    static inline void
    set(nested_mask_type& mask, u64 idx, u1 value) {
      return mask.set(idx, value);
    }

    /// Sets the bit a position 'idx' to the given 'value'.
    template<u1 Compound, typename = std::enable_if_t<Compound>>
    static inline void
    set(compound_mask_type& masks, u64 idx, u1 value) {
      u64 m_idx = idx / nested_vector_length;
      u64 n_idx = idx % nested_vector_length;
      return set<!Compound>(masks[m_idx], n_idx, value);
    }

    /// Sets ALL bits to the given 'value'.
    template<u1 Compound = false>
    static inline void
    set(nested_mask_type& mask, u1 value) {
      mask.set(value);
    }

    /// Sets ALL bits to the given 'value'.
    template<u1 Compound, typename = std::enable_if_t<Compound>>
    static inline void
    set(compound_mask_type& masks, u1 value) {
      for ($u64 i = 0; i < nested_vector_cnt; i++) {
        set<!Compound>(masks[i], value);
      }
    }

    template<u1 Compound = false>
    static inline u1
    get(const nested_mask_type& mask, u64 idx) {
      return mask.get(idx);
    }

    template<u1 Compound, typename = std::enable_if_t<Compound>>
    static inline u1
    get(const compound_mask_type& masks, u64 idx) {
      u64 m_idx = idx / nested_vector_length;
      u64 n_idx = idx % nested_vector_length;
      return get<!Compound>(masks[m_idx], n_idx);
    }

    /// Sets the mask at position 'idx' to 'true'. Use with caution as this operation might be very expensive.
    void set(u64 idx, u1 value) { set<is_compound>(data, idx, value); }

    /// Gets the boolean value from the mask at position 'idx'. Use with caution as this operation might be very expensive.
    u1 get(u64 idx) const { return get<is_compound>(data, idx); }


    struct bit_and_fn {
      constexpr nested_mask_type operator()(const nested_mask_type& a, const nested_mask_type& b) const {
        return a.bit_and(b);
      }
    };

    struct bit_or_fn {
      constexpr nested_mask_type operator()(const nested_mask_type& a, const nested_mask_type& b) const {
        return a.bit_or(b);
      }
    };

    struct bit_xor_fn {
      constexpr nested_mask_type operator()(const nested_mask_type& a, const nested_mask_type& b) const {
        return a.bit_xor(b);
      }
    };

    struct bit_not_fn {
      constexpr nested_mask_type operator()(const nested_mask_type& a) const {
        return a.bit_not();
      }
    };

    // binary functions
    template<u1 Compound = false, typename Fn>
    static inline nested_mask_type
    bit_op(Fn fn, const nested_mask_type& a, const nested_mask_type& b) {
      return fn(a, b);
    }
    // binary functions
    template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
    static inline compound_mask_type
    bit_op(Fn fn, const compound_mask_type& a, const compound_mask_type& b) {
      compound_mask_type result;
      for ($u64 i = 0; i < nested_vector_cnt; i++) {
        result[i] = bit_op<!Compound>(fn, a[i], b[i]);
      }
      return result;
    }

    // unary functions
    template<u1 Compound = false, typename Fn>
    static inline nested_mask_type
    bit_op(Fn fn, const nested_mask_type& a) {
      return fn(a);
    }
    // unary functions
    template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
    static inline compound_mask_type
    bit_op(Fn fn, const compound_mask_type& a) {
      compound_mask_type result;
      for ($u64 i = 0; i < nested_vector_cnt; i++) {
        result[i] = bit_op<!Compound>(fn, a[i]);
      }
      return result;
    }


    /// Performs a bitwise AND.
    m operator&(const m& o) const { return m { bit_op<is_compound>(bit_and_fn(), data, o.data) }; }
    m& operator&=(const m& o) { data = bit_op<is_compound>(bit_and_fn(), data, o.data); return (*this); }

    /// Performs a bitwise OR.
    m operator|(const m& o) const { return m { bit_op<is_compound>(bit_or_fn(), data, o.data) }; }
    m& operator|=(const m& o) { data = bit_op<is_compound>(bit_or_fn(), data, o.data); return (*this); }

    /// Performs a bitwise XOR.
    m operator^(const m& o) const { return m { bit_op<is_compound>(bit_xor_fn(), data, o.data) }; }
    m& operator^=(const m& o) { data = bit_op<is_compound>(bit_xor_fn(), data, o.data); return (*this); }

    /// Performs a bitwise negation.
    m operator!() const { return m { bit_op<is_compound>(bit_not_fn(), data) }; }


    /// Returns a mask instance where all components are set to 'true'.
    static m make_all_mask() {
      m result;
      set<is_compound>(result.data, true);
      return result;
    };

    /// Returns a mask instance where all components are set to 'false'.
    static m make_none_mask() {
      m result;
      set<is_compound>(result.data, false);
      return result;
    };

  };

  // Public API

  /// The mask type of the surrounding vector.
  using mask_t = m;
  using mask = m; // alias for those who don't like '_t's

  /// Returns a mask instance where all components are set to 'true'.
  static mask_t make_all_mask() { return mask_t::make_all_mask(); };

  /// Returns a mask instance where all components are set to 'false'.
  static mask_t make_none_mask() { return mask_t::make_none_mask(); };

  // --- end of mask




  // Specialize function objects for the current native vector type. (reduces verboseness later on)
  struct op {

    // template parameters are
    //   1) primitive type
    //   2) native vector type
    //   3) argument type
    //   4) return type (defaults to vector type)

    using broadcast = dtl::simd::broadcast<Tp, nested_type, Tp>;
    using set = dtl::simd::set<Tp, nested_type, nested_type>;

    using plus = dtl::simd::plus<Tp, nested_type>;
    using minus = dtl::simd::minus<Tp, nested_type>;
    using multiplies = dtl::simd::multiplies<Tp, nested_type>;

    using shift_left = dtl::simd::shift_left<Tp, nested_type, i32>;
    using shift_left_var = dtl::simd::shift_left_var<Tp, nested_type, nested_type>;
    using shift_right = dtl::simd::shift_left<Tp, nested_type, i32>;
    using shift_right_var = dtl::simd::shift_right_var<Tp, nested_type, nested_type>;

    using bit_and = dtl::simd::bit_and<Tp, nested_type>;
    using bit_or = dtl::simd::bit_or<Tp, nested_type>;
    using bit_xor = dtl::simd::bit_xor<Tp, nested_type>;
    using bit_not = dtl::simd::bit_not<Tp, nested_type>;

    using less = dtl::simd::less<Tp, nested_type, nested_type, nested_mask_type>;
    using equal = dtl::simd::equal<Tp, nested_type, nested_type, nested_mask_type>;
    using not_equal = dtl::simd::not_equal<Tp, nested_type, nested_type, nested_mask_type>;
    using greater = dtl::simd::greater<Tp, nested_type, nested_type, nested_mask_type>; // TODO remove

  };


  // --- C'tors

//  v() = default;

//  template<typename Tp_other>
//  explicit
//  v(const v<Tp_other, N>& other) {
//    for (auto )
//  }

    // brace-initializer list c'tor
//  template<typename ...T>
//  explicit
//  v(T&&... t) : data{ std::forward<T>(t)... } { }

//  explicit
//  v(v&& other) : data(std::move(other.data)) { }


  /// Assignment
  inline v&
  operator=(const v& other) = default;

//  inline v&
//  operator=(v&& other) {
//    data = std::move(other.data);
//  }


  /// Assigns the given scalar value to all vector components.
  inline v&
  operator=(const Tp& scalar_value) noexcept {
    data = unary_op<is_compound>(typename op::set(), data, make_nested(scalar_value));
    return *this;
  }

  /// Assigns the given scalar value to the vector components specified by the mask.
  inline v&
  mask_assign(const Tp& scalar_value, const m& mask) noexcept {
    data = unary_op<is_compound>(typename op::set(), data, make_nested(scalar_value), data, mask);
    return *this;
  }

  inline v&
  mask_assign(const v& other, const m& mask) noexcept {
    data = unary_op<is_compound>(typename op::set(), data, other.data, data, mask);
    return *this;
  }


  /// Creates a vector where all components are set to the given scalar value.
  static inline v
  make(const Tp& scalar_value) {
    v result;
    result = scalar_value;
    return std::move(result);
  }

  /// Creates a copy of the given vector.
  static inline v
  make(const v& other) {
    v result;
    result.data = other.data;
    return result;
  }

  /// Creates a nested vector with all components set to the given scalar value.
  /// In other words, the given value is broadcasted to all vector components.
  static inline nested_type
  make_nested(const Tp& scalar_value) {
    auto fn = typename op::broadcast();
    return fn(scalar_value);
  }


  // --- Unary functions ---

  /// Unary operation: op(native vector)
  template<u1 Compound = false, typename Fn>
  static inline nested_type
  unary_op(Fn op, const nested_type& /* type_selector */,
           const nested_type& a) noexcept {
    return op(a);
  }

  /// Unary operation (merge masked): op(native vector)
  template<u1 Compound = false, typename Fn>
  static inline nested_type
  unary_op(Fn op, const nested_type& /* type_selector */,
           const nested_type& a,
           // merge masking
           const nested_type& src,
           const m& mask) noexcept {
    return op(a, mask);
  }

  /// Unary operation: op(compound vector)
  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
  static inline compound_type
  unary_op(Fn op, const compound_type& /* type_selector */,
           const compound_type& a) noexcept {
    compound_type result;
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      result[i] = op(a);
    }
    return result;
  }

  /// Unary operation (merge masked): op(compound vector)
  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
  static inline compound_type
  unary_op(Fn op, const compound_type& /* type_selector */,
           const compound_type& a,
           // merge masking
           const compound_type& src,
           const m& mask) noexcept {
    compound_type result;
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      result[i] = op(a, src, mask);
    }
    return result;
  }

  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
  static inline compound_type
  unary_op(Fn op, const compound_type& /* type_selector */,
           const nested_type& a) noexcept {
    compound_type result;
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      result[i] = op(a);
    }
    return result;
  }

  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
  static inline compound_type
  unary_op(Fn op, const compound_type& /* type_selector */,
           const nested_type& a,
           // merge masking
           const compound_type& src,
           const m& mask) noexcept {
    compound_type result;
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      result[i] = op(a, src[i], mask.data[i]);
    }
    return result;
  }


  // --- Binary functions ---

  /// Applies a binary operation to a NON-compound (native) vector type.
  template<u1 Compound = false, typename Fn>
  static inline typename Fn::result_type
  binary_op(Fn op, const typename Fn::vector_type& lhs,
                   const typename Fn::vector_type& rhs) noexcept {
    return op(lhs, rhs);
  }
  template<u1 Compound = false, typename Fn>
  static inline typename Fn::result_type
  binary_op(Fn op, const typename Fn::vector_type& lhs,
                   const typename Fn::vector_type& rhs,
                   // merge masking
                   const typename Fn::vector_type& src,
                   const nested_mask_type& mask) noexcept {
    return op(lhs, rhs, src, mask);
  }


  /// Applies a binary operation to a compound vector type.
  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
  static inline make_compound<typename Fn::result_type, nested_vector_cnt>
  binary_op(Fn op, const compound_type& lhs,
                   const compound_type& rhs) noexcept {
    make_compound<typename Fn::result_type, nested_vector_cnt> result;
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      result[i] = binary_op(op, lhs[i], rhs[i]);
    }
    return result;
  }
  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
  static inline make_compound<typename Fn::result_type, nested_vector_cnt>
  binary_op(Fn op, const compound_type& lhs,
                   const compound_type& rhs,
                   // merge masking
                   const compound_type& src,
                   const compound_mask_type& mask) noexcept {
    make_compound<typename Fn::result_type, nested_vector_cnt> result;
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      result[i] = binary_op(op, lhs[i], rhs[i], src[i], mask[i]);
    }
    return result;
  }

  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
  static inline make_compound<typename Fn::result_type, nested_vector_cnt>
  binary_op(Fn op, const nested_type& lhs,
                   const compound_type& rhs) noexcept {
    make_compound<typename Fn::result_type, nested_vector_cnt> result;
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      result[i] = binary_op(op, lhs, rhs[i]);
    }
    return result;
  }
  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
  static inline make_compound<typename Fn::result_type, nested_vector_cnt>
  binary_op(Fn op, const nested_type& lhs,
                   const compound_type& rhs,
                   // merge masking
                   const compound_type& src,
                   const compound_mask_type& mask) noexcept {
    make_compound<typename Fn::result_type, nested_vector_cnt> result;
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      result[i] = binary_op(op, lhs, rhs[i], src[i], mask[i]);
    }
    return result;
  }

  /// Applies an operation of type: vector op scalar
  /// The scalar value needs to be broadcasted to all SIMD lanes first.
  /// Note: This is an optimization for compound vectors.
  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>, typename = std::enable_if_t<Compound>> // TODO why twice?
  static inline make_compound<typename Fn::result_type, nested_vector_cnt>
  binary_op(Fn op, const compound_type& lhs,
                   const nested_type& rhs) noexcept {
    make_compound<typename Fn::result_type, nested_vector_cnt> result;
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      result[i] = binary_op(op, lhs[i], rhs);
    }
    return result;
  }
  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>, typename = std::enable_if_t<Compound>> // TODO why twice?
  static inline make_compound<typename Fn::result_type, nested_vector_cnt>
  binary_op(Fn op, const compound_type& lhs,
                   const nested_type& rhs,
                   // merge masking
                   const compound_type& src,
                   const compound_mask_type& mask) noexcept {
    make_compound<typename Fn::result_type, nested_vector_cnt> result;
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      result[i] = binary_op(op, lhs[i], rhs, src[i], mask[i]);
    }
    return result;
  }

//  template<typename Fn>
//  static inline nested_type
//  binary_op(Fn op, const nested_type& lhs, i32& rhs) noexcept {
//    return op(lhs, rhs);
//  }

//  /// Applies an operation of type: vector op scalar (w/o broadcasting the value to all SIMD lanes)
//  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
//  static inline compound_type
//  binary_op(Fn op, const compound_type& lhs, i32& rhs) noexcept {
//    compound_type result;
//    for ($u64 i = 0; i < nested_vector_cnt; i++) {
//      result[i] = binary_op(op, lhs[i], rhs);
//    }
//    return result;
//  }



  v operator+(const v& o) const noexcept { return v { binary_op<is_compound>(typename op::plus(), data, o.data) }; }
  v operator+(const Tp& s) const noexcept { return v { binary_op<is_compound>(typename op::plus(), data, make_nested(s)) }; }
  v operator+() const noexcept { return v { binary_op<is_compound>(typename op::plus(), data, make_nested(0)) }; }
  v& operator+=(const v& o) noexcept { data = binary_op<is_compound>(typename op::plus(), data, o.data); return *this; }
  v& operator+=(const Tp& s) noexcept  { data = binary_op<is_compound>(typename op::plus(), data, make_nested(s)); return *this; }

  v mask_plus(const v& o, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::plus(), data, o.data, data, op_mask.data) }; }
  v mask_plus(const Tp& s, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::plus(), data, make_nested(s), data, op_mask.data) }; }
  v mask_plus(const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::plus(), data, make_nested(0), data, op_mask.data) }; }
  v& mask_assign_plus(const v& o, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::plus(), data, o.data, data, op_mask.data ); return *this; }
  v& mask_assign_plus(const Tp& s, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::plus(), data, make_nested(s), data, op_mask.data ); return *this; }

  v operator-(const v& o) const noexcept { return v { binary_op<is_compound>(typename op::minus(), data, o.data) }; }
  v operator-(const Tp& s) const noexcept { return v { binary_op<is_compound>(typename op::minus(), data, make_nested(s)) }; }
  v operator-() const noexcept { return v { binary_op<is_compound>(typename op::minus(), data, make_nested(0)) }; }
  v& operator-=(const v& o) noexcept { data = binary_op<is_compound>(typename op::minus(), data, o.data); return (*this); }
  v& operator-=(const Tp& s) noexcept  { data = binary_op<is_compound>(typename op::minus(), make_nested(s)); return (*this); }

  v mask_minus(const v& o, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::minus(), data, o.data, data, op_mask.data) }; }
  v mask_minus(const Tp& s, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::minus(), data, make_nested(s), data, op_mask.data) }; }
  v mask_minus(const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::minus(), make_nested(0), data, data, op_mask.data) }; }
  v& mask_assign_minus(const v& o, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::minus(), data, o.data, data, op_mask.data ); return *this; }
  v& mask_assign_minus(const Tp& s, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::minus(), data, make_nested(s), data, op_mask.data ); return *this; }

  v operator*(const v& o) const noexcept { return v { binary_op<is_compound>(typename op::multiplies(), data, o.data) }; }
  v operator*(const Tp& s) const noexcept { return v { binary_op<is_compound>(typename op::multiplies(), data, make_nested(s)) }; }
  v& operator*=(const v& o) noexcept { data = binary_op<is_compound>(typename op::multiplies(), data, o.data); return (*this); }
  v& operator*=(const Tp& s) noexcept  { data = binary_op<is_compound>(typename op::multiplies(), make_nested(s)); return (*this); }

  v mask_multiplies(const v& o, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::multiplies(), data, o.data, data, op_mask.data) }; }
  v mask_multiplies(const Tp& s, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::multiplies(), data, make_nested(s), data, op_mask.data) }; }
  v& mask_assign_multiplies(const v& o, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::multiplies(), data, o.data, data, op_mask.data ); return *this; }
  v& mask_assign_multiplies(const Tp& s, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::multiplies(), data, make_nested(s), data, op_mask.data ); return *this; }

  v operator<<(const v& o) const noexcept { return v { binary_op<is_compound>(typename op::shift_left_var(), data, o.data) }; }
  v operator<<(const i32& s) const noexcept { return v { binary_op<is_compound>(typename op::shift_left(), data, s) }; }
  v& operator<<=(const v& o) noexcept { data = binary_op<is_compound>(typename op::shift_left_var(), data, o.data); return (*this); }
  v& operator<<=(const i32& s) noexcept  { data = binary_op<is_compound>(typename op::shift_left(), s); return (*this); }

  v mask_shift_left(const v& o, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::shift_left_var(), data, o.data, data, op_mask.data) }; }
  v mask_shift_left(const i32& s, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::shift_left(), data, make_nested(s), data, op_mask.data) }; }
  v& mask_assign_shift_left(const v& o, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::shift_left_var(), data, o.data, data, op_mask.data ); return *this; }
  v& mask_assign_shift_left(const i32& s, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::shift_left(), data, make_nested(s), data, op_mask.data ); return *this; }

  v operator>>(const v& o) const noexcept { return v { binary_op<is_compound>(typename op::shift_right_var(), data, o.data) }; }
  v operator>>(const i32& s) const noexcept { return v { binary_op<is_compound>(typename op::shift_right(), data, s) }; }
  v& operator>>=(const v& o) noexcept { data = binary_op<is_compound>(typename op::shift_right_var(), data, o.data); return (*this); }
  v& operator>>=(const i32& s) noexcept  { data = binary_op<is_compound>(typename op::shift_right(), make_nested(s)); return (*this); }

  v mask_shift_right(const v& o, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::shift_right_var(), data, o.data, data, op_mask.data) }; }
  v mask_shift_right(const i32& s, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::shift_right(), data, make_nested(s), data, op_mask.data) }; }
  v& mask_assign_shift_right(const v& o, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::shift_right_var(), data, o.data, data, op_mask.data ); return *this; }
  v& mask_assign_shift_right(const i32& s, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::shift_right(), data, make_nested(s), data, op_mask.data ); return *this; }

  v operator&(const v& o) const noexcept { return v { binary_op<is_compound>(typename op::bit_and(), data, o.data) }; }
  v operator&(const Tp& s) const noexcept { return v { binary_op<is_compound>(typename op::bit_and(), data, make_nested(s)) }; }
  v& operator&=(const v& o) noexcept { data = binary_op<is_compound>(typename op::bit_and(), data, o.data); return (*this); }
  v& operator&=(const Tp& s) noexcept  { data = binary_op<is_compound>(typename op::bit_and(), make_nested(s)); return (*this); }

  v mask_bit_and(const v& o, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::bit_and(), data, o.data, data, op_mask.data) }; }
  v mask_bit_and(const Tp& s, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::bit_and(), data, make_nested(s), data, op_mask.data) }; }
  v& mask_assign_bit_and(const v& o, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::bit_and(), data, o.data, data, op_mask.data ); return *this; }
  v& mask_assign_bit_and(const Tp& s, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::bit_and(), data, make_nested(s), data, op_mask.data ); return *this; }

  v operator|(const v& o) const noexcept { return v { binary_op<is_compound>(typename op::bit_or(), data, o.data) }; }
  v operator|(const Tp& s) const noexcept { return v { binary_op<is_compound>(typename op::bit_or(), data, make_nested(s)) }; }
  v& operator|=(const v& o) noexcept { data = binary_op<is_compound>(typename op::bit_or(), data, o.data); return (*this); }
  v& operator|=(const i32& s) noexcept  { data = binary_op<is_compound>(typename op::bit_or(), make_nested(s)); return (*this); }

  v mask_bit_or(const v& o, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::bit_or(), data, o.data, data, op_mask.data) }; }
  v mask_bit_or(const Tp& s, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::bit_or(), data, make_nested(s), data, op_mask.data) }; }
  v& mask_assign_bit_or(const v& o, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::bit_or(), data, o.data, data, op_mask.data ); return *this; }
  v& mask_assign_bit_or(const Tp& s, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::bit_or(), data, make_nested(s), data, op_mask.data ); return *this; }

  v operator^(const v& o) const noexcept { return v { binary_op<is_compound>(typename op::bit_xor(), data, o.data) }; }
  v operator^(const Tp& s) const noexcept { return v { binary_op<is_compound>(typename op::bit_xor(), make_nested(s)) }; }
  v& operator^=(const v& o) noexcept { data = binary_op<is_compound>(typename op::bit_xor(), data, o.data); return (*this); }
  v& operator^=(const Tp& s) noexcept { data = binary_op<is_compound>(typename op::bit_xor(), data, make_nested(s)); return (*this); }

  v mask_bit_xor(const v& o, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::bit_xor(), data, o.data, data, op_mask.data) }; }
  v mask_bit_xor(const Tp& s, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::bit_xor(), data, make_nested(s), data, op_mask.data) }; }
  v& mask_assign_bit_xor(const v& o, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::bit_xor(), data, o.data, data, op_mask.data ); return *this; }
  v& mask_assign_bit_xor(const Tp& s, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::bit_xor(), data, make_nested(s), data, op_mask.data ); return *this; }


  static inline Tp
  extract(const nested_type& native_vector, u64 idx) noexcept {
    return reinterpret_cast<const Tp*>(&native_vector)[idx]; // TODO improve performance
  }

  template<typename T, typename = std::enable_if_t<(sizeof(T), is_compound)>>
  static inline Tp
  extract(const T& compound_vector, u64 idx) noexcept {
    return extract(compound_vector[idx / nested_vector_length], idx % nested_vector_length);
  }

  template<u1 Compound = false>
  static inline void
  insert(nested_type& native_vector, const Tp& value, u64 idx) noexcept {
    reinterpret_cast<Tp*>(&native_vector)[idx] = value; // TODO improve performance
  }

  template<u1 Compound, typename = std::enable_if_t<Compound>>
  static inline void
  insert(compound_type& compound_vector, const Tp& value, u64 idx) noexcept {
    insert<!Compound>(compound_vector[idx / nested_vector_length], value, idx % nested_vector_length);
  }

  inline void
  insert(const Tp& value, u64 idx) noexcept {
    insert<is_compound>(data, value, idx);
  }

  /// Read-only access to the individual vector components
  Tp operator[](u64 idx) const noexcept {
    return extract(data, idx);
  }
  // ---


  // Comparisons
  inline m
  operator<(const v& o) const noexcept {
    return m { binary_op<is_compound>(typename op::less(), data, o.data) };
  }

  inline m
  operator>(const v& o) const noexcept {
    return m { binary_op<is_compound>(typename op::less(), o.data, data) };
  }

  inline m
  operator==(const v& o) const noexcept {
    return m { binary_op<is_compound>(typename op::equal(), o.data, data) };
  }

  inline m
  operator==(const Tp& s) const noexcept {
    return m { binary_op<is_compound>(typename op::equal(), data, make_nested(s)) };
  }

  inline m
  operator!=(const v& o) const noexcept {
    return m { binary_op<is_compound>(typename op::not_equal(), o.data, data) };
  }

  inline m
  operator!=(const Tp& s) const noexcept {
    return m { binary_op<is_compound>(typename op::not_equal(), data, make_nested(s)) };
  }
  // ---


  // load
  template<u1 Compound = false, typename T>
  static inline typename v<T, N>::nested_type
  load(const T* const base_addr, const nested_type& idxs) {
    return gather<Tp, nested_type, T>()(base_addr, idxs);
  }

  template<u1 Compound, typename T, typename = std::enable_if_t<Compound>>
  static inline typename v<T, N>::compound_type
  load(const T* const base_addr, const compound_type& idxs) {
    using result_t = typename v<T, N>::compound_type;
    result_t result;
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      result[i] = load<is_compound>(base_addr, idxs[i]);
    }
    return result;
  }

  template<typename T>
  v<T, N> load(const T* const base_address) const {
    using result_t = v<T, N>;
    return result_t { load<is_compound>(base_address, data) };
  }
  // ---


  // store
  template<u1 Compound = false, typename T>
  static inline typename v<T, N>::nested_type
  store(T* const base_addr,
        const nested_type& where_idxs,
        const typename v<T, N>::nested_type what) {
    return scatter<Tp, nested_type, T>()(base_addr, where_idxs, what);
  }

  template<u1 Compound = false, typename T>
  static inline typename v<T, N>::nested_type
  store(T* const base_addr,
        const nested_type& where_idxs,
        const typename v<T, N>::nested_type what,
        const nested_mask_type& mask) {
    return scatter<Tp, nested_type, T>()(base_addr, where_idxs, what, mask);
  }

  template<u1 Compound, typename T, typename = std::enable_if_t<Compound>>
  static inline void
  store(T* const base_addr,
        const compound_type& where_idxs,
        const typename v<T, N>::compound_type& what) {
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      store<is_compound>(base_addr, where_idxs[i], what[i]);
    }
  }

  template<u1 Compound, typename T, typename = std::enable_if_t<Compound>>
  static inline void
  store(T* const base_addr,
        const compound_type& where_idxs,
        const typename v<T, N>::compound_type& what,
        const compound_mask_type& mask) {
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      store<is_compound>(base_addr, where_idxs[i], what[i], mask[i]);
    }
  }

  template<typename T>
  inline void
  store(T* const base_address, const v<T, N>& what) {
    store<is_compound>(base_address, data, what.data);
  }

  template<typename T>
  inline void
  store(T* const base_address, const v<T, N>& what, const m& mask) {
    store<is_compound>(base_address, data, what.data, mask.data);
  }
  // ---


  // helper
  void
  print(std::ostream& os) const {
    os << "[" << (*this)[0];
    for ($u64 i = 1; i < length; i++) {
      os << ", ";
      os << (*this)[i];
    }
    os << "]";
  }

  template<u64... Idxs>
  static constexpr v
  make_index_vector(std::array<Tp, N>* const arr, integer_sequence<Idxs...>) {
    *arr = { Idxs... };
  }

  static constexpr v
  make_index_vector() {
    v result = make(0);
    std::array<Tp, N>* const arr = reinterpret_cast<std::array<Tp, N>*>(&result.data);
    make_index_vector(arr, make_integer_sequence<N>());
    return result;
  };





  template<typename Tp_target>
  inline v<Tp_target, N>
  cast() const {
    v<Tp_target, N> result;
    for ($u64 i = 0; i < N; i++) {
      result.data[i] = data[i];
    }
    return result;
  };



  // --- syntactic sugar for masked operations

  struct masked_reference {
    v& vector;
    m& mask;

    inline v& operator=(const v& o) { vector.mask_assign(o, mask); return vector; }
    inline v& operator=(const Tp& s) { vector.mask_assign(s, mask); return vector; }
    inline v operator+(const v& o) const noexcept { return vector.mask_plus(o, mask); }
    inline v operator+(const Tp& s) const noexcept { return vector.mask_plus(s, mask); }
    inline v operator+() const noexcept { return vector.mask_plus(mask); }
    inline v& operator+=(const v& o) noexcept { vector.mask_assign_plus(o,mask); return vector; }
    inline v& operator+=(const Tp& s) noexcept { vector.mask_assign_plus(s,mask); return vector; }
    inline v operator-(const v& o) const noexcept { return vector.mask_minus(o, mask); }
    inline v operator-(const Tp& s) const noexcept { return vector.mask_minus(s, mask); }
    inline v operator-() const noexcept { return vector.mask_minus(mask); }
    inline v& operator-=(const v& o) noexcept { vector.mask_assign_minus(o, mask); return vector; }
    inline v& operator-=(const Tp& s) noexcept { vector.mask_assign_minus(s, mask); return vector; }
    inline v operator*(const v& o) const noexcept { return vector.mask_multiplies(o, mask); }
    inline v operator*(const Tp& s) const noexcept { return vector.mask_multiplies(s, mask); }
    inline v& operator*=(const v& o) noexcept { vector.mask_assign_multiplies(o, mask); return vector; }
    inline v& operator*=(const Tp& s) noexcept { vector.mask_assign_multiplies(s, mask); return vector; }
    inline v operator<<(const v& o) const noexcept { return vector.mask_shift_left(o, mask); }
    inline v operator<<(const Tp& s) const noexcept { return vector.mask_shift_left(s, mask); }
    inline v& operator<<=(const v& o) noexcept { vector.mask_assign_shift_left(o, mask); return vector; }
    inline v& operator<<=(const Tp& s) noexcept { vector.mask_assign_shift_left(s, mask); return vector; }
    inline v operator>>(const v& o) const noexcept { return vector.mask_shift_right(o, mask); }
    inline v operator>>(const Tp& s) const noexcept { return vector.mask_shift_right(s, mask); }
    inline v& operator>>=(const v& o) noexcept { vector.mask_assign_shift_right(o, mask); return vector; }
    inline v& operator>>=(const Tp& s) noexcept { vector.mask_assign_shift_right(s, mask); return vector; }
    inline v operator&(const v& o) const noexcept { return vector.mask_bit_and(o, mask); }
    inline v operator&(const Tp& s) const noexcept { return vector.mask_bit_and(s, mask); }
    inline v& operator&=(const v& o) noexcept { vector.mask_assign_bit_and(o, mask); return vector; }
    inline v& operator&=(const Tp& s) noexcept { vector.mask_assign_bit_and(s, mask); return vector; }
    inline v operator|(const v& o) const noexcept { return vector.mask_bit_or(o, mask); }
    inline v operator|(const Tp& s) const noexcept { return vector.mask_bit_or(s, mask); }
    inline v& operator|=(const v& o) noexcept { vector.mask_assign_bit_or(o, mask); return vector; }
    inline v& operator|=(const Tp& s) noexcept { vector.mask_assign_bit_or(s, mask); return vector; }
    inline v operator^(const v& o) const noexcept { return vector.mask_bit_xor(o, mask); }
    inline v operator^(const Tp& s) const noexcept { return vector.mask_bit_xor(s, mask); }
    inline v& operator^=(const v& o) noexcept { vector.mask_assign_bit_xor(o, mask); return vector; }
    inline v& operator^=(const Tp& s) noexcept { vector.mask_assign_bit_xor(s, mask); return vector; }
  };

  inline masked_reference
  operator[](m& op_mask) noexcept {
    return masked_reference{ *this, op_mask };
  }

  // ---

};

/// left shift of form: scalar << vector
template<typename T, u64 N>
v<T, N> operator<<(const T& lhs, const v<T, N>& rhs) {
  v<T, N> lhs_vec = v<T, N>::make(lhs);
  return lhs_vec << rhs;
}
/// not sure if this is causing problems...
template<typename Tl, typename T, u64 N>
v<T, N> operator<<(const Tl& lhs, const v<T, N>& rhs) {
  v<T, N> lhs_vec = v<T, N>::make(Tl(lhs));
  return lhs_vec << rhs;
}

} // namespace simd
} // namespace dtl