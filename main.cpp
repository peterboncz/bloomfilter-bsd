
#include <immintrin.h>
#include <iostream>
#include <typeinfo>
#include <functional>

#include "sma.hpp"
#include "dict.hpp"
#include "psma.hpp"
#include "vec.hpp"
#include "math.hpp"
#include "simd.hpp"



template<typename T, size_t N>
class indexed_vector {
  static_assert(is_power_of_two(N), "Template parameter N must be a power of two.");
  static_assert(std::is_unsigned<T>::value, "Template parameter T must be an unsigned integer.");

public:
  psma<T> idx;
  T data[N];

  explicit indexed_vector(const std::vector<T> in) {
    const size_t n = std::min(N, in.size());
    std::memcpy(data, in.data(), n * sizeof(T));
    idx.update(data, N);
  }

};

void code_gen() {
  const uint32_t N = 1 << 17;

  std::vector<int8_t> input1;
  for (size_t i = 0; i < N; i++) {
    input1.push_back(1);
  }
  std::random_shuffle(input1.begin(), input1.end());

  std::vector<int16_t> input2;
  for (size_t i = 0; i < N; i++) {
    input2.push_back(2);
  }
  std::random_shuffle(input2.begin(), input2.end());


  using T = int8_t;
  using S = int16_t;
  const size_t W = 64;

  vec<S, W> output(0);

  for (size_t i = 0; i < N; i += W) {
    const vec<T, W>* reader1 = reinterpret_cast<const vec<T, W>*>(&input1[i]);
    const vec<T, W> c = *reader1 + vec<T, W>(2) - 1;
    const vec<int16_t, W>* reader2 = reinterpret_cast<const vec<int16_t, W>*>(&input2[i]);
    output += c.cast<S>() + (*reader2).cast<S>();
  }
  for (int i = 0; i < W; i++) {
    std::cout << static_cast<int64_t>(output.data[i]) << ", ";
  }
  std::cout << std::endl;
}

template<typename L, typename R, size_t N, typename OP, typename S = typename super<L, R>::type>
constexpr vec<S, N> binary_expr(const vec<L, N>& lhs, const vec<R, N>& rhs) {
  const size_t n_result_elements_per_register = simd::lane<S>::count;
  const size_t n_left_input_elements_per_register = simd::lane<L>::count;
  const size_t n_right_input_elements_per_register = simd::lane<R>::count;
  vec<S, N> result;
  vec<S, n_result_elements_per_register>* result_reg_writer = result.template begin<n_result_elements_per_register>();

};



template<typename S, typename T, size_t N, typename OP>
constexpr vec<S, N> unary_expr_cast(const vec<S, N>& in) {
  const size_t n_result_elements_per_register = simd::lane<T>::count;
  const size_t n_input_elements_per_register = simd::lane<S>::count;
  vec<S, N> result;
  vec<S, n_result_elements_per_register>* result_reg_writer = result.template begin<n_result_elements_per_register>();

};


template<typename L, typename R>
struct plus {
  using lhs_type = L;
  using rhs_type = R;
  using T = typename super<L, R>::type;
  using result_type = T;
  static constexpr T apply(const L &lhs, const R &rhs) {
    return lhs + rhs;
  }
};


template<typename L, typename R>
struct simd_plus {
    using lhs_type = L;
    using rhs_type = R;
    using T = typename super<L, R>::type;
    using result_type = T;
    constexpr T apply(const L &lhs, const R &rhs) {
        return lhs + rhs;
    }
};

template<typename T, size_t N = simd::lane<T>::count, size_t W>
constexpr vec<T, N>& reg(vec<T, W> v, size_t i) {
  return v.template begin<N>()[i];
};

void code_gen2() {
  const uint32_t N = 1 << 17;
  const size_t W = simd::lane<int8_t>::count;
  std::cout << "vector size " << W << std::endl;

  int16_t a = 1;
  int16_t b = 2;
  std::cout << typeid(super<int8_t, int16_t>::type).name() << " " << plus<int16_t,int16_t>::apply(a, b) << std::endl;


  std::vector<int16_t> input1;
  for (size_t i = 0; i < N; i++) {
    input1.push_back(i);
  }

  const vec<int16_t, W>* const t = reinterpret_cast<const vec<int16_t, W>* const>(&input1[0]);
  for (int i = 0; i < W; i++) {
    std::cout << static_cast<int64_t>(t->data[i]) << ", ";
  }
  std::cout << std::endl;

  auto r = reg(*t, 0);
  for (int i = 0; i < simd::lane<int16_t>::count; i++) {
    std::cout << static_cast<int64_t>(r.data[i]) << ", ";
  }
  std::cout << std::endl;
return;
  std::random_shuffle(input1.begin(), input1.end());

  std::vector<int16_t> input2;
  for (size_t i = 0; i < N; i++) {
    input2.push_back(2);
  }
  std::random_shuffle(input2.begin(), input2.end());


  vec<int32_t, W> output(0);

  for (size_t i = 0; i < N; i += W) {
    const vec<int8_t, W>* const reader1 = reinterpret_cast<const vec<int8_t, W>* const>(&input1[i]);
    const vec<int16_t, W>* const reader2 = reinterpret_cast<const vec<int16_t, W>* const>(&input2[i]);

    const vec<int8_t, W> t0 = *reader1 + vec<int8_t, W>(2) - 1;
    const vec<int32_t, W> t1 = t0.cast<int32_t>();
    const vec<int32_t, W> t2 = (*reader2).cast<int32_t>();
    output += t1 + t2;
  }
  for (int i = 0; i < W; i++) {
    std::cout << static_cast<int64_t>(output.data[i]) << ", ";
  }
  std::cout << std::endl;
}

void string_dict() {
  std::vector<std::string> strings {"hallo", "c++", "welt", "!", "!", "!"};
  dict<std::string> d(strings);
  std::cout << d.size << " " << d.lookup("welt") << std::endl;

  std::vector<uint64_t> ints = d.map(strings);
  for (uint32_t i = 0; i < ints.size(); i++) {
    std::cout << strings[i] << "->" << ints[i] << ", ";
  }
  std::cout << std::endl;
}

void int_dict() {
  std::vector<uint32_t> values {1, 42, 1337, 42, 4711};
  dict<uint32_t> d(values);
  std::cout << d.size << " " << d.lookup(42) << std::endl;

  std::vector<uint64_t> ints = d.map(values);
  for (uint32_t i = 0; i < ints.size(); i++) {
    std::cout << values[i] << "->" << ints[i] << ", ";
  }
  std::cout << std::endl;
}

void idx_vector() {
  const uint32_t N = 1 << 17;
  std::vector<uint32_t> v;
  for (uint32_t i = 0; i < N; i++) {
    v.push_back(i);
  }

  indexed_vector<uint32_t, N> iv(v);
  auto range = iv.idx.lookup(pred::GT, 10u, 10);
  std::cout << range.begin << "," << range.end << std::endl;

  for (uint32_t i = 0; i < N; i++) {
    if (v[i] != iv.data[i]) std::cout << i << std::endl;
  }
}

int main() {

  const size_t N = 4;
  std::cout << "SIMD " << simd::bitwidth::value << std::endl;

  //code_gen2();
  //uint64_t r = rank<uint64_t, 4>({1u,2u,3u,4u});

  for (size_t a = 0 ; a < ct::factorial<N>::value; a++) {
    std::cout << a << " -> ";
    auto pi = unrank<uint64_t, N>(a);
    for (size_t i = 0; i < N; i++) {
      std::cout << pi[i] ;
    }
    std::cout << " -> ";
    std::cout << rank(pi) << std::endl;
  }
  std::cout << ct::n_choose_k<0, 3>::value << std::endl;
  std::cout << ct::catalan_number<33>::value << std::endl;
  std::cout << ct::ballot_number<1, 3>::value << std::endl;
  std::cout << ct::number_of_paths<8, 0, 0>::value << std::endl;
  for (size_t a = 0 ; a < ct::catalan_number<N>::value; a++) {
    std::cout << a << " -> ";
    auto tree = unrank_tree<N>(a);
    for (size_t i = 0; i < tree.size(); i++) {
      std::cout << (tree[i] ? "[" : "]") ;
    }
    std::cout << " -> ";
    std::cout << rank_tree<N>(tree) << std::endl;
  }

  return 0;
}
