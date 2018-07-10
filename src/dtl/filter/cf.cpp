#include "cf.hpp"

#include <dtl/filter/cuckoofilter/cuckoofilter.hpp>
#include <dtl/filter/cuckoofilter/cuckoofilter_tune_impl.hpp>

namespace dtl {

namespace {
static dtl::cuckoofilter::cuckoofilter_tune_impl tuner;
} // anonymous namespace

struct cf::impl {
  using cf_t = dtl::cuckoofilter::cuckoofilter;
  cf_t instance;

  impl(std::size_t m, u32 tag_size_bits = 16, u32 associativity = 4)
      : instance(m, tag_size_bits, associativity, tuner) { };
  ~impl() = default;
  impl(impl&&) = default;
  impl(const impl&) = delete;
  impl& operator=(impl&&) = default;
  impl& operator=(const impl&) = delete;
};

cf::cf(const std::size_t m, u32 bits_per_tag, u32 tags_per_bucket)
    : pimpl{ std::make_unique<impl>(m, bits_per_tag, tags_per_bucket) },
      bits_per_tag(bits_per_tag), tags_per_bucket(tags_per_bucket) {}
cf::cf(cf&&) = default;
cf::~cf() = default;
cf& cf::operator=(cf&&) = default;

void
cf::insert(cf::word_t* __restrict filter_data, u32 key) {
  if (!pimpl->instance.insert(filter_data, key)) {
    throw std::logic_error("Failed to insert element. Filter has not enough space.");
  }
}

void
cf::batch_insert(cf::word_t* __restrict filter_data, u32* keys, u32 key_cnt) {
  pimpl->instance.batch_insert(filter_data, keys, key_cnt);
}

$u1
cf::contains(const cf::word_t* __restrict filter_data, u32 key) const {
  return pimpl->instance.contains(filter_data, key);
}

$u64
cf::batch_contains(const cf::word_t* __restrict filter_data,
                   u32* keys, u32 key_cnt, $u32* match_positions, u32 match_offset) const {
  return pimpl->instance.batch_contains(filter_data, keys, key_cnt, match_positions, match_offset);
}

void
cf::calibrate() {
  tuner.tune_unroll_factor();
}

void
cf::force_unroll_factor(u32 u) {
  throw "not yet implemented";
}

std::string
cf::name() const {
  return pimpl->instance.name();
}

std::size_t
cf::size_in_bytes() const {
  return pimpl->instance.size_in_bytes();
}

std::size_t
cf::size() const {
  return pimpl->instance.size();
}

std::size_t
cf::count_occupied_slots(const cf::word_t* __restrict filter_data) const {
  return pimpl->instance.count_occupied_slots(filter_data);
}

std::size_t
cf::get_bucket_count() const {
  return pimpl->instance.get_bucket_count();
};

} // namespace dtl
