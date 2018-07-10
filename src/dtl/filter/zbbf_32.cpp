#include "zbbf_32.hpp"

#include "instances/zoned_blocked_bloomfilter_logic_u32_instance.hpp" // extern templates to parallelize builds
#include "blocked_bloomfilter/zoned_blocked_bloomfilter.hpp"
#include "blocked_bloomfilter/zoned_blocked_bloomfilter_tune_impl.hpp"

namespace dtl {


namespace {
static dtl::zoned_blocked_bloomfilter_tune_impl<$u32> tuner;
} // anonymous namespace


struct zbbf_32::impl {
  using bbf_t = dtl::zoned_blocked_bloomfilter<$u32>;
  bbf_t instance;

  impl(const size_t m, u32 k, u32 word_cnt_per_block = 4, u32 zone_cnt = 2)
      : instance(m, k, word_cnt_per_block, zone_cnt, tuner) { };
  ~impl() = default;
  impl(impl&&) = default;
  impl(const impl&) = delete;
  impl& operator=(impl&&) = default;
  impl& operator=(const impl&) = delete;
};

zbbf_32::zbbf_32(const size_t m, u32 k, u32 word_cnt_per_block, u32 zone_cnt)
    : pimpl{ std::make_unique<impl>(m, k, word_cnt_per_block, zone_cnt) } {}
zbbf_32::zbbf_32(zbbf_32&&) noexcept = default;
zbbf_32::~zbbf_32() = default;
zbbf_32& zbbf_32::operator=(zbbf_32&&) = default;

void
zbbf_32::insert(zbbf_32::word_t* __restrict filter_data, u32 key) {
  pimpl->instance.insert(filter_data, key);
}

void
zbbf_32::batch_insert(zbbf_32::word_t* __restrict filter_data, u32* keys, u32 key_cnt) {
  pimpl->instance.batch_insert(filter_data, keys, key_cnt);
}

$u1
zbbf_32::contains(const zbbf_32::word_t* __restrict filter_data, u32 key) const {
  return pimpl->instance.contains(filter_data, key);
}

$u64
zbbf_32::batch_contains(const zbbf_32::word_t* __restrict filter_data,
                       u32* keys, u32 key_cnt, $u32* match_positions, u32 match_offset) const {
  return pimpl->instance.batch_contains(filter_data, keys, key_cnt, match_positions, match_offset);
}

void
zbbf_32::calibrate() {
  tuner.tune_unroll_factor();
}

void
zbbf_32::force_unroll_factor(u32 u) {
  throw "not yet implemented";
}

std::string
zbbf_32::name() const {
  return pimpl->instance.name();
}

std::size_t
zbbf_32::size_in_bytes() const {
  return pimpl->instance.size_in_bytes();
}

std::size_t
zbbf_32::size() const {
  return pimpl->instance.size();
}

} // namespace dtl
