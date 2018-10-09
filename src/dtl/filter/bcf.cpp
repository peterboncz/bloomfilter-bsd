#include <memory>

#include "bcf.hpp"

#include "blocked_cuckoofilter/blocked_cuckoofilter.hpp"

namespace dtl {

struct bcf::impl {
  using bcf_t = dtl::blocked_cuckoofilter;
  bcf_t instance;

  impl(size_t m, u32 block_size_bytes = 64, u32 tag_size_bits = 16, u32 associativity = 4)
      : instance(m, block_size_bytes, tag_size_bits, associativity) { };
  ~impl() = default;
  impl(impl&&) = default;
  impl(const impl&) = delete;
  impl& operator=(impl&&) = default;
  impl& operator=(const impl&) = delete;
};

bcf::bcf(const size_t m, u32 block_size_bytes, u32 tag_size_bits, u32 associativity)
    : pimpl{ std::make_unique<impl>(m, block_size_bytes, tag_size_bits, associativity) } {}
bcf::bcf(bcf&&) noexcept = default;
bcf::~bcf() = default;
bcf& bcf::operator=(bcf&&) = default;

$u1
bcf::insert(bcf::word_t* __restrict filter_data, u32 key) {
  pimpl->instance.insert(filter_data, key);
  return true; // inserts never fail
}

$u1
bcf::batch_insert(bcf::word_t* __restrict filter_data, u32* __restrict keys, u32 key_cnt) {
  pimpl->instance.batch_insert(filter_data, keys, key_cnt);
  return true; // inserts never fail
}

$u1
bcf::contains(const bcf::word_t* __restrict filter_data, u32 key) const {
  return pimpl->instance.contains(filter_data, key);
}

$u64
bcf::batch_contains(const bcf::word_t* __restrict filter_data,
                    u32* __restrict keys, u32 key_cnt,
                    $u32* __restrict match_positions, u32 match_offset) const {
  return pimpl->instance.batch_contains(filter_data, keys, key_cnt, match_positions, match_offset);
}

void
bcf::calibrate() {
  impl::bcf_t::calibrate();
}

void
bcf::force_unroll_factor(u32 u) {
  impl::bcf_t::force_unroll_factor(u);
}

std::string
bcf::name() const {
  return pimpl->instance.name();
}

std::size_t
bcf::size_in_bytes() const {
  return pimpl->instance.size_in_bytes();
}

std::size_t
bcf::size() const {
  return pimpl->instance.size();
}

} // namespace dtl
