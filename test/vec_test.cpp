#include "gtest/gtest.h"
#include "../adept.hpp"
#include "../simd.hpp"

using namespace dtl;

template<typename T, u64 N>
using v = typename simd::v<T, N>;

TEST(vec, ensure_native_implementation) {
  using vec_t = v<$i32, simd::lane_count<$i32>>;
  ASSERT_FALSE(vec_t::is_compound) << "Missing implementation of native vector type.";
}

TEST(vec, make_from_scalar_value) {
  using vec_t = v<$i32, simd::lane_count<$i32>>;
  vec_t a = vec_t::make(42);
  for ($u64 i = 0; i < vec_t::length; i++) {
    ASSERT_EQ(42, a[i]);
  }
}

TEST(vec, make_from_vector) {
  using vec_t = v<$i32, simd::lane_count<$i32>>;
  vec_t a = vec_t::make(42);
  vec_t b = vec_t::make(a);
  for ($u64 i = 0; i < vec_t::length; i++) {
    ASSERT_EQ(a[i], b[i]);
  }
}

TEST(vec, make_from_integer_sequence) {
  using vec_t = v<$i32, simd::lane_count<$i32> * 4>;
  vec_t act = vec_t::make_index_vector();
  for ($i32 i = 0; i < vec_t::length; i++) {
    ASSERT_EQ(i, act[i]);
  }
}

TEST(vec, make_compound_vector) {
  using vec_t = v<$i32, 4 * simd::lane_count<$i32>>;
  vec_t a = vec_t::make(42);
  for ($u64 i = 0; i < vec_t::length; i++) {
    ASSERT_EQ(42, a[i]);
  }
}

TEST(vec, move_assignment) {
  using vec_t = v<$i32, 32>;
  vec_t a = vec_t::make(41);
  vec_t b = a + 1;
  vec_t exp = vec_t::make(42);
  for ($u64 i = 0; i < 8; i++) {
    ASSERT_EQ(exp[i], b[i]);
  }
}

TEST(vec, comparison_and_mask) {
  using vec_t = v<$i32, 32>;
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
  u64 vec_len = simd::lane_count<$i32>;
  using vec_t = v<$i32, vec_len>;
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
  u64 vec_len = simd::lane_count<$i32> * 4;
  using vec_t = v<$i32, vec_len>;

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
  u64 vec_len = simd::lane_count<$i32> * 4;
  using vec_t = v<$i32, vec_len>;
  vec_t a = vec_t::make(42);
  vec_t act = a & 2;

  for ($u64 i = 0; i < vec_len; i++) {
    ASSERT_EQ(2, act[i]);
  }
}

TEST(vec, gather) {
  u64 vec_len = simd::lane_count<$i32> * 2;
  using vec_t = v<$i32, vec_len>;

  u64 arr_len = 128;
  std::array<$i32, arr_len> arr;
  for ($i32 i = 0; i < arr_len; i++) {
    arr[i] = i;
  }

  vec_t exp = vec_t::make_index_vector() * 4;
  vec_t act = exp.load(&arr[0]);
  for ($u64 i = 0; i < vec_len; i++) {
    ASSERT_EQ(exp[i], act[i]);
  }

  act = act + 1;
  exp.store(&arr[0], act);
  for ($u64 i = 0; i < vec_len; i++) {
    ASSERT_EQ(exp[i] + 1, act[i]);
  }
}

TEST(vec, masked_operation_assign) {
  u64 vec_len = simd::lane_count<$i32> * 2;
  using vec_t = v<$i32, vec_len>;
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
  u64 vec_len = simd::lane_count<$i32> * 2;
  using vec_t = v<$i32, vec_len>;
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
  u64 vec_len = simd::lane_count<$i32> * 2;
  using vec_t = v<$i32, vec_len>;
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
  u64 vec_len = simd::lane_count<$i32> * 2;
  using vec_t = v<$i32, vec_len>;
  vec_t a = vec_t::make_index_vector();
  vec_t b = vec_t::make(2);

  vec_t::mask op_mask = a > b;
  vec_t r = -a[op_mask];

  for ($i32 i = 0; i < vec_len; i++) {
    auto exp_val = i > 2 ? -i : i;
    ASSERT_EQ(exp_val, r[i]);
  }
}
