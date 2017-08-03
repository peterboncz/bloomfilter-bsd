#include "gtest/gtest.h"
#include <dtl/dtl.hpp>
#include <dtl/div.hpp>
#include <dtl/simd.hpp>

using namespace dtl;

template<typename T, u64 N>
using v = typename simd::v<T, N>;

TEST(vec, ensure_native_implementation) {
  using vec_t = v<i32, simd::lane_count<i32>>;
  if (simd::lane_count<i32> > 1) {
    ASSERT_FALSE(vec_t::is_compound) << "Missing implementation of native vector type.";
  }
}

TEST(vec, make_from_scalar_value) {
  using vec_t = v<i32, simd::lane_count<i32>>;
  vec_t a = vec_t::make(42);
  for ($u64 i = 0; i < vec_t::length; i++) {
    ASSERT_EQ(42, a[i]);
  }
}

TEST(vec, make_from_vector) {
  using vec_t = v<i32, simd::lane_count<i32>>;
  vec_t a = vec_t::make(42);
  vec_t b = vec_t::make(a);
  for ($u64 i = 0; i < vec_t::length; i++) {
    ASSERT_EQ(a[i], b[i]);
  }
}

TEST(vec, make_from_integer_sequence) {
  using vec_t = v<i32, simd::lane_count<i32> * 4>;
  vec_t act = vec_t::make_index_vector();
  for ($i32 i = 0; i < vec_t::length; i++) {
    ASSERT_EQ(i, act[i]);
  }
}

TEST(vec, make_compound_vector) {
  using vec_t = v<i32, 4 * simd::lane_count<i32>>;
  vec_t a = vec_t::make(42);
  for ($u64 i = 0; i < vec_t::length; i++) {
    ASSERT_EQ(42, a[i]);
  }
}

TEST(vec, move_assignment) {
  using vec_t = v<i32, 32>;
  vec_t a = vec_t::make(41);
  vec_t b = a + 1;
  vec_t exp = vec_t::make(42);
  for ($u64 i = 0; i < 8; i++) {
    ASSERT_EQ(exp[i], b[i]);
  }
}

TEST(vec, comparison_and_mask) {
  using vec_t = v<i32, 32>;
  const vec_t a = vec_t::make(41);
  vec_t b = a + 1;
  auto m = a < b;
  ASSERT_TRUE(m.all());
  ASSERT_TRUE(m.any());
  ASSERT_FALSE(m.none());

  m = b < a;
  ASSERT_FALSE(m.all());
  ASSERT_FALSE(m.any());
  ASSERT_TRUE(m.none());

  m.set(0, true);
  ASSERT_TRUE(m.get(0));
  ASSERT_FALSE(m.all());
  ASSERT_FALSE(m.none());
  ASSERT_TRUE(m.any());

  m = m ^ m;
  ASSERT_TRUE(m.none());
}

TEST(vec, var_shift) {
  u64 vec_len = simd::lane_count<i32>;
  using vec_t = v<i32, vec_len>;
  vec_t rhs;
  for ($u64 i = 0; i < vec_len; i++) {
    rhs.insert(i, i);
  }
  vec_t act = 1 << rhs;
  for ($u64 i = 0; i < vec_len; i++) {
    ASSERT_EQ(1u << i, act[i]);
  }
}

TEST(vec, make_mask) {
  u64 vec_len = simd::lane_count<i32> * 4;
  using vec_t = v<i32, vec_len>;

  auto all_ones = vec_t::mask_t::make_all_mask();
  for ($u64 i = 0; i < vec_len; i++) {
    ASSERT_TRUE(all_ones.get(i));
  }

  auto all_zeros = vec_t::mask_t::make_none_mask();
  for ($u64 i = 0; i < vec_len; i++) {
    ASSERT_FALSE(all_zeros.get(i));
  }
}

TEST(vec, bitwise) {
  u64 vec_len = simd::lane_count<i32> * 4;
  using vec_t = v<i32, vec_len>;
  vec_t a = vec_t::make(42);
  vec_t act = a & 2;

  for ($u64 i = 0; i < vec_len; i++) {
    ASSERT_EQ(2, act[i]);
  }
}


// --- custom SIMD function ---
namespace custom {
namespace simd {

template<typename primitive_t, typename vector_t, typename argument_t>
struct mulhi_u32 : dtl::simd::vector_fn<primitive_t, vector_t, argument_t> {};

template<>
struct mulhi_u32<$u32, __m256i, __m256i> : dtl::simd::vector_fn<$u32, __m256i, __m256i> {
  inline __m256i operator()(const __m256i& a, const __m256i& b) const noexcept {
    return dtl::mulhi_u32(a, b);
  }
};

#if defined(__AVX512F__)
template<>
struct mulhi_u32<$u32, __m512i, __m512i> : dtl::simd::vector_fn<$u32, __m512i, __m512i> {
  inline __m512i operator()(const __m512i& a, const __m512i& b) const noexcept {
    return dtl::mulhi_u32(a, b);
  }
};
#endif // defined(__AVX512F__)

} // namespace simd
} // namespace custom

namespace dtl {

template<typename Tv>
Tv
mulhi_u32(const Tv& a, const Tv& b) {
  using mulhi = custom::simd::mulhi_u32<typename Tv::scalar_type, typename Tv::nested_type, typename Tv::nested_type>;
  const Tv c = a.template map<mulhi>(b);
  return c;
}

} // namespace dtl
// --- ---

TEST(vec, custom_vector_function) {
  u64 vec_len = simd::lane_count<u32> * 4;

  using vec_t = v<u32, vec_len>;
  const vec_t a = vec_t::make_index_vector();
  const vec_t b = vec_t::make_index_vector() << 10;

  const vec_t c = dtl::mulhi_u32(a, b);

  for ($u64 i = 0; i < vec_len; i++) {
    ASSERT_EQ(dtl::mulhi_u32(a[i], b[i]), c[i]);
  }

}


// TODO implement scatter
//TEST(vec, gather) {
//  u64 vec_len = simd::lane_count<$i32> * 2;
//  using vec_t = v<$i32, vec_len>;
//
//  u64 arr_len = 128;
//  std::array<$i32, arr_len> arr;
//  for ($i32 i = 0; i < arr_len; i++) {
//    arr[i] = i;
//  }
//
//  vec_t exp = vec_t::make_index_vector() * 4;
//  vec_t act = dtl::gather(&arr[0], exp);
//  for ($u64 i = 0; i < vec_len; i++) {
//    ASSERT_EQ(exp[i], act[i]);
//  }
//
//  act = act + 1;
//  dtl::scatter(act, &arr[0], exp);
//  for ($u64 i = 0; i < vec_len; i++) {
//    ASSERT_EQ(exp[i] + 1, act[i]);
//  }
//}


//// TODO implement gather of non 64-bit types
//TEST(vec, gather_from_absolute_addresses) {
//  u64 vec_len = simd::lane_count<$i64> * 2;
//  using vec_t = v<$u64, vec_len>;
//  using ptr_vt = v<$u64, vec_len>;
//
//  u64 arr_len = 128;
//  std::array<$u64, arr_len> arr;
//  for ($u64 i = 0; i < arr_len; i++) {
//    arr[i] = i;
//  }
//
//  vec_t exp = vec_t::make_index_vector() * 4;
//
//  ptr_vt addrs = ptr_vt::make(reinterpret_cast<$u64>(&arr[0]));
//  addrs += ptr_vt::make_index_vector() * 4 * sizeof(i32);
//  vec_t act = dtl::gather<$u64>(addrs);
//  for ($u64 i = 0; i < vec_len; i++) {
//    ASSERT_EQ(exp[i], act[i]);
//  }
//
////  act = act + 1;
////  dtl::scatter(act, &arr[0], exp);
////  for ($u64 i = 0; i < vec_len; i++) {
////    ASSERT_EQ(exp[i] + 1, act[i]);
////  }
//}

TEST(vec, masked_operation_assign) {
  u64 vec_len = simd::lane_count<i32> * 2;
  using vec_t = v<i32, vec_len>;
  vec_t a = vec_t::make_index_vector();
  vec_t b = vec_t::make(2);

  vec_t::mask op_mask = a > b;

//  a.mask_assign(0, op_mask);
  a[op_mask] = 0;

  for ($u64 i = 0; i < vec_len; i++) {
    auto exp_val = i > 2 ? 0 : i;
    ASSERT_EQ(exp_val, a[i]);
  }
}

TEST(vec, masked_operation_assign_arithmetic) {
  u64 vec_len = simd::lane_count<i32> * 2;
  using vec_t = v<i32, vec_len>;
  vec_t a = vec_t::make_index_vector();
  vec_t b = vec_t::make(2);

  vec_t::mask op_mask = a > b;

  a[op_mask] += 42;

  for ($u64 i = 0; i < vec_len; i++) {
    auto exp_val = i > 2 ? i + 42 : i;
    ASSERT_EQ(exp_val, a[i]);
  }
}

TEST(vec, masked_operation_arithmetic) {
  u64 vec_len = simd::lane_count<i32> * 2;
  using vec_t = v<i32, vec_len>;
  vec_t a = vec_t::make_index_vector();
  vec_t b = vec_t::make(2);

  vec_t::mask op_mask = a > b;
  vec_t r = a[op_mask] + 42;

  for ($u64 i = 0; i < vec_len; i++) {
    auto exp_val = i > 2 ? i + 42 : i;
    ASSERT_EQ(exp_val, r[i]);
  }
}

TEST(vec, masked_unary_operation) {
  u64 vec_len = simd::lane_count<i32> * 2;
  using vec_t = v<i32, vec_len>;
  vec_t a = vec_t::make_index_vector();
  vec_t b = vec_t::make(2);

  vec_t::mask op_mask = a > b;
  vec_t r = -a[op_mask];

  for ($i32 i = 0; i < vec_len; i++) {
    auto exp_val = i > 2 ? -i : i;
    ASSERT_EQ(exp_val, r[i]);
  }
}

TEST(vec, mask_to_32bit_positions) {
  constexpr u64 vec_len = simd::lane_count<i32> * 2;

  $u32 positions[vec_len];
  alignas(64) $i32 input[vec_len];
  for ($i32 i = 0; i < vec_len; i++) {
    input[i] = i;
  }

  using vec_t = v<i32, vec_len>;
  vec_t& a = reinterpret_cast<vec_t&>(input);
  vec_t b = vec_t::make(2);

  vec_t::mask mask = a > b;

  u64 match_cnt = mask.to_positions(positions);

  u64 expected_match_cnt = vec_len - 3;
  ASSERT_EQ(expected_match_cnt, match_cnt);

  $u32* reader = positions;
  for ($i32 i = 0; i < vec_len; i++) {
    if (i > 2) {
      ASSERT_EQ(i, *reader) << " reader pos = " << (reader - positions);
      reader++;
    }
  }
}
