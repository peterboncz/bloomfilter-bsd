#include <memory>

#include "cf2.hpp"

#include <dtl/filter/cuckoofilter/cuckoofilter.hpp>
#include <dtl/filter/cuckoofilter/cuckoofilter_tune_impl.hpp>

namespace dtl {

namespace {
static dtl::cuckoofilter::cuckoofilter_tune_impl tuner;
} // anonymous namespace

struct cf2::impl {
  static constexpr u1 has_victim_cache = true;
  using cf_t = dtl::cuckoofilter::cuckoofilter<has_victim_cache>;
  cf_t instance;

  impl(std::size_t m, u32 tag_size_bits = 16, u32 associativity = 4)
      : instance(m, tag_size_bits, associativity, tuner) { };
  ~impl() = default;
  impl(impl&&) = default;
  impl(const impl&) = delete;
  impl& operator=(impl&&) = default;
  impl& operator=(const impl&) = delete;
};

cf2::cf2(const std::size_t m, u32 bits_per_tag, u32 tags_per_bucket)
    : pimpl{ std::make_unique<impl>(m, bits_per_tag, tags_per_bucket) },
      bits_per_tag(bits_per_tag), tags_per_bucket(tags_per_bucket) {}
cf2::cf2(cf2&&) noexcept = default;
cf2::~cf2() = default;
cf2& cf2::operator=(cf2&&) noexcept = default;

$u1
cf2::insert(cf2::word_t* __restrict filter_data, u32 key) {
  return pimpl->instance.insert(
      reinterpret_cast<impl::cf_t::word_t*>(filter_data), key);
}

$u1
cf2::batch_insert(cf2::word_t* __restrict filter_data, u32* __restrict keys,
    u32 key_cnt) {
  return pimpl->instance.batch_insert(
      reinterpret_cast<impl::cf_t::word_t*>(filter_data), keys, key_cnt);
}

$u1
cf2::contains(const cf2::word_t* __restrict filter_data, u32 key) const {
  return pimpl->instance.contains(
      reinterpret_cast<const impl::cf_t::word_t*>(filter_data), key);
}

$u64
cf2::batch_contains(const cf2::word_t* __restrict filter_data,
                   u32* __restrict keys, u32 key_cnt,
                   $u32* __restrict match_positions, u32 match_offset) const {
  return pimpl->instance.batch_contains(
      reinterpret_cast<const impl::cf_t::word_t*>(filter_data), keys, key_cnt,
      match_positions, match_offset);
}

void
cf2::calibrate() {
  tuner.tune_unroll_factor();
}

void
cf2::force_unroll_factor(u32 u) {
  tuner.set_unroll_factor(u);
}

std::string
cf2::name() const {
  return pimpl->instance.name();
}

std::size_t
cf2::size_in_bytes() const {
  return pimpl->instance.size_in_bytes();
}

std::size_t
cf2::size() const {
  return (pimpl->instance.size() + 1) / 2; // convert from the number of 32-bit words to the number of 64-bit words.
}

std::size_t
cf2::count_occupied_slots(const cf2::word_t* __restrict filter_data) const {
  return pimpl->instance.count_occupied_slots(
      reinterpret_cast<const impl::cf_t::word_t*>(filter_data));
}

std::size_t
cf2::get_bucket_count() const {
  return pimpl->instance.get_bucket_count();
};

} // namespace dtl
