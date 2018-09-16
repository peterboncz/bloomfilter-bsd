#include "zbbf_64.hpp"

#include "blocked_bloomfilter/instances/zoned_blocked_bloomfilter_logic_u64_instance.hpp" // extern templates to parallelize builds
#include "blocked_bloomfilter/zoned_blocked_bloomfilter.hpp"
#include "blocked_bloomfilter/zoned_blocked_bloomfilter_tune_impl.hpp"

namespace dtl {


namespace {
static dtl::zoned_blocked_bloomfilter_tune_impl<$u64> tuner;
} // anonymous namespace


struct zbbf_64::impl {
  using bbf_t = dtl::zoned_blocked_bloomfilter<$u64>;
  bbf_t instance;

  impl(const size_t m, u32 k, u32 word_cnt_per_block, u32 zone_cnt)
      : instance(m, k, word_cnt_per_block, zone_cnt, tuner) { };
  ~impl() = default;
  impl(impl&&) = default;
  impl(const impl&) = delete;
  impl& operator=(impl&&) = default;
  impl& operator=(const impl&) = delete;
};

zbbf_64::zbbf_64(const size_t m, u32 k, u32 word_cnt_per_block, u32 zone_cnt)
    : pimpl{ std::make_unique<impl>(m, k, word_cnt_per_block, zone_cnt) } {}
zbbf_64::zbbf_64(zbbf_64&&) noexcept = default;
zbbf_64::~zbbf_64() = default;
zbbf_64& zbbf_64::operator=(zbbf_64&&) = default;

$u1
zbbf_64::insert(zbbf_64::word_t* __restrict filter_data, u32 key) {
  pimpl->instance.insert(filter_data, key);
  return true; // inserts never fail
}

$u1
zbbf_64::batch_insert(zbbf_64::word_t* __restrict filter_data, u32* keys, u32 key_cnt) {
  pimpl->instance.batch_insert(filter_data, keys, key_cnt);
  return true; // inserts never fail
}

$u1
zbbf_64::contains(const zbbf_64::word_t* __restrict filter_data, u32 key) const {
  return pimpl->instance.contains(filter_data, key);
}

$u64
zbbf_64::batch_contains(const zbbf_64::word_t* __restrict filter_data,
                        u32* __restrict keys, u32 key_cnt,
                        $u32* __restrict match_positions, u32 match_offset) const {
  return pimpl->instance.batch_contains(filter_data, keys, key_cnt, match_positions, match_offset);
}

void
zbbf_64::calibrate() {
  tuner.tune_unroll_factor();
}

void
zbbf_64::force_unroll_factor(u32 u) {
  tuner.set_unroll_factor(u);
}

std::string
zbbf_64::name() const {
  return pimpl->instance.name();
}

std::size_t
zbbf_64::size_in_bytes() const {
  return pimpl->instance.size_in_bytes();
}

std::size_t
zbbf_64::size() const {
  return pimpl->instance.size();
}

} // namespace dtl
