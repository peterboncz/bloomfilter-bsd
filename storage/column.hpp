#pragma once

#include <bitset>
#include <memory>

#include <dtl/dtl.hpp>

#include "types.hpp"

namespace dtl {

const u16 file_signature = 0x484c;

enum class null { value };

/// A naive column-block implementation which does not support deletions
template<u64 N>
struct column_block_base {
  $u64 write_pos;
  alignas(64) std::bitset<N> null;

  virtual ~column_block_base() {};

  inline u64
  size() const {
    return write_pos;
  }

  inline void
  push_back(const dtl::null /* val */) {
    null[write_pos] = true;
    write_pos++;
  }

  inline u1
  is_null(u64 n) const {
    return null[n];
  }

};


template<typename T, u64 N>
struct column_block : public column_block_base<N> {
  using column_block_base<N>::write_pos;
  using column_block_base<N>::is_null;
  using column_block_base<N>::push_back;

  alignas(64) std::array<T, N> data;

  inline void
  push_back(const T& val) {
    data[write_pos++] = val;
  }

  inline void
  push_back(T&& val) {
    data[write_pos++] = std::move(val);
  }

  inline T&
  operator[](u64 n) {
    return data[n];
  }

  inline const T&
  operator[](u64 n) const {
    return data[n];
  }

};


/// A naive column implementation which does not support NULL values and deletions
template<typename T>
struct column {

  static u64 block_size_bits = 7;
  static u64 block_size = 1ull << block_size_bits;

  using block = column_block<T, block_size>;

  /// references to all blocks of this column
  std::vector<std::unique_ptr<block>> blocks;

  /// the very last block (where new data is to be inserted)
  block* tail_block;

  /// c'tor
  column() {
    allocate_block();
  }

  inline void
  allocate_block() {
    blocks.push_back(std::make_unique<block>());
    tail_block = blocks[blocks.size() - 1].get();
  }

  inline void
  push_back(const T& val) {
    // painful branch + indirection !!!
    if (tail_block->size() == block_size) {
      allocate_block();
    }
    tail_block->push_back(val);
  }

  inline void
  push_back(T&& val) {
    // painful branch + indirection !!!
    if (tail_block->size() == block_size) {
      allocate_block();
    }
    tail_block->push_back(std::move(val));
  }

  inline T&
  operator[](u64 n) {
    block& b = *blocks[n >> block_size_bits].get();
    return b[n & ((1ull << block_size_bits) - 1)];
  }

  inline const T&
  operator[](u64 n) const {
    const block& b = *blocks[n >> block_size_bits].get();
    return b[n & ((1ull << block_size_bits) - 1)];
  }


  /// returns the total number of elements (linear runtime complexity; rarely used function)
  inline u64
  size() const {
    $u64 s = 0;
    for (auto& block_ptr : blocks) {
      s += block_ptr->size();
    }
    return s;
  }

  void
  print(std::ostream& os) {
    std::cout << "[";
    for (auto& block_ptr : blocks) {
      print(block_ptr.get());
    }
    std::cout << "]";
  }

};


template<u64 N>
dtl::column_block_base<N>*
make_column_block(const dtl::rtt type) {
  dtl::column_block_base<N>* ptr;
  switch (type) {
#define DTL_GENERATE(T) \
    case dtl::rtt::T:  \
      ptr = new dtl::column_block<dtl::map<dtl::rtt::T>::type, N>(); \
      break;
    DTL_GENERATE(u8)
    DTL_GENERATE(i8)
    DTL_GENERATE(u16)
    DTL_GENERATE(i16)
    DTL_GENERATE(u32)
    DTL_GENERATE(i32)
    DTL_GENERATE(u64)
    DTL_GENERATE(i64)
    DTL_GENERATE(str)
#undef DTL_GENERATE
  }
  return ptr;
}

template<u64 N>
inline void
column_block_insert(dtl::column_block_base<N>* block_ptr,
                    const dtl::rtt type,
                    const std::string& value,
                    const std::string& null_indicator) {
  switch (type) {
#define DTL_GENERATE(T) \
    case dtl::rtt::T: {                                                                \
      using block_type = dtl::column_block<dtl::map<dtl::rtt::T>::type, N>;            \
      block_type* b = static_cast<block_type*>(block_ptr);                             \
      if (value == null_indicator) {                                                   \
        b->push_back(dtl::null::value);                                                \
      }                                                                                \
      else {                                                                           \
        const auto parse = dtl::parse<dtl::rtt::T>();                                  \
        b->push_back(parse(value));                                                    \
      }                                                                                \
      break;                                                                           \
    }
    DTL_GENERATE(u8)
    DTL_GENERATE(i8)
    DTL_GENERATE(u16)
    DTL_GENERATE(i16)
    DTL_GENERATE(u32)
    DTL_GENERATE(i32)
    DTL_GENERATE(u64)
    DTL_GENERATE(i64)
    DTL_GENERATE(str)
#undef DTL_GENERATE
  }
}

} // namespace