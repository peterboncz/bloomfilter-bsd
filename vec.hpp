#pragma once

template<typename T, size_t N>
struct vec {
  alignas(64) T data[N];

  using type = T;

  vec() {
  }

  vec(const T scalar_value) {
    for (size_t i = 0; i < N; i++) {
      data[i] = scalar_value;
    }
  }

  vec(const vec &o) {
    for (size_t i = 0; i < N; i++) {
      data[i] = o.data[i];
    }
  }

//  vec(std::initializer_list<T> l) {
//    for (size_t i = 0; i < N; i++) {
//      data[i] = l[i];
//    }
//  }

  vec operator+(const vec &o) const noexcept {
    vec<T, N> d(*this);
    for (size_t i = 0; i < N; i++) {
      d.data[i] = data[i] + o.data[i];
    }
    return d;
  }

  vec operator+(const T &scalar_value) const noexcept {
    vec<T, N> d(*this);
    for (size_t i = 0; i < N; i++) {
      d.data[i] += scalar_value;
    }
    return d;
  }

  vec operator-(const vec &o) const noexcept {
    vec<T, N> d(*this);
    for (size_t i = 0; i < N; i++) {
      d.data[i] = data[i] - o.data[i];
    }
    return d;
  }

  vec operator-(const T &scalar_value) const noexcept {
    vec<T, N> d(*this);
    for (size_t i = 0; i < N; i++) {
      d.data[i] -= scalar_value;
    }
    return d;
  }

  void operator++() noexcept {
    for (size_t i = 0; i < N; i++) {
      data[i] += 1;
    }
  }

  void operator--() noexcept {
    for (size_t i = 0; i < N; i++) {
      data[i] -= 1;
    }
  }

  void operator+=(const vec &o) noexcept {
    for (size_t i = 0; i < N; i++) {
      data[i] += o.data[i];
    }
  }

  void operator-=(const vec &o) noexcept {
    for (size_t i = 0; i < N; i++) {
      data[i] -= o.data[i];
    }
  }

  void operator^=(const vec &o) noexcept {
    for (size_t i = 0; i < N; i++) {
      data[i] ^= o.data[i];
    }
  }

  void operator|=(const vec &o) noexcept {
    for (size_t i = 0; i < N; i++) {
      data[i] |= o.data[i];
    }
  }

  void operator&=(const vec &o) noexcept {
    for (size_t i = 0; i < N; i++) {
      data[i] &= o.data[i];
    }
  }

  template<typename S>
  vec<S, N> cast() const noexcept {
    vec<S, N> d;
    for (size_t i = 0; i < N; i++) {
      d.data[i] = data[i];
    }
    return d;
  }

  vec operator<(const vec &r) const noexcept {
    vec<T, N> d(0);
    for (size_t i = 0; i < N; i++) {
      d.data[i] = data[i] < r.data[i] ? -1 : 0;
    }
    return d;
  }

  template<size_t W>
  vec<T, W>* begin()  {
    return reinterpret_cast<vec<T, W>*>(data);
  }

  template<size_t W>
  vec<T, W>* end() const {
    return begin() + (N / W);
  }

};
