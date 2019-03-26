#pragma once

#include <memory>
#include <dtl/dtl.hpp>

namespace amsfilter {
namespace internal {
//===----------------------------------------------------------------------===//
template<typename T, typename Alloc = boost::alignment::aligned_allocator<T, 64>>
class buffer {
  Alloc allocator_;
  T* begin_;
  std::size_t size_;

public:

  explicit
  buffer(std::size_t size, Alloc allocator = Alloc())
      : allocator_(allocator),
        begin_(allocator_.allocate(size)),
        size_(size) {}
  buffer(const buffer& other) = delete;
  buffer(buffer&& other) noexcept
      : allocator_(other.allocator_),
        begin_(nullptr),
        size_(other.size_){
    std::swap(begin_, other.begin_);
  };
  buffer& operator=(const buffer& other) = delete;
  buffer& operator=(buffer&& other) noexcept = delete;
  ~buffer() {
    if (begin_) allocator_.deallocate(begin_, size_);
  };

  T* begin() { return begin_; }
  const T* begin() const { return begin_; }

  T* end() { return begin_ + size_; }
  const T* end() const { return begin_ + size_; }

  T* data() { return begin_; }
  const T* data() const { return begin_; }

  std::size_t
  size() const noexcept {
    return size_;
  }
};
//===----------------------------------------------------------------------===//
} // namespace internal
} // namespace amsfilter
