#include <memory>

#include "bbf_64.hpp"

#include "blocked_bloomfilter/instances/blocked_bloomfilter_logic_u64_instance.hpp" // extern templates to parallelize builds
#include "blocked_bloomfilter/blocked_bloomfilter.hpp"
#include "blocked_bloomfilter/blocked_bloomfilter_tune_impl.hpp"

namespace dtl {


namespace {
static dtl::blocked_bloomfilter_tune_impl<$u64> tuner;
} // anonymous namespace


struct bbf_64::impl {
  using bbf_t = dtl::blocked_bloomfilter<$u64>;
  bbf_t instance;

  impl(const size_t m, u32 k, u32 word_cnt_per_block, u32 sector_cnt)
      : instance(m, k, word_cnt_per_block, sector_cnt, tuner) { };
  ~impl() = default;
  impl(impl&&) = default;
  impl(const impl&) = delete;
  impl& operator=(impl&&) = default;
  impl& operator=(const impl&) = delete;
};

bbf_64::bbf_64(const size_t m, u32 k, u32 word_cnt_per_block, u32 sector_cnt)
    : pimpl{ std::make_unique<impl>(m, k, word_cnt_per_block, sector_cnt) } {}
bbf_64::bbf_64(bbf_64&&) noexcept = default;
bbf_64::~bbf_64() = default;
bbf_64& bbf_64::operator=(bbf_64&&) = default;

$u1
bbf_64::insert(bbf_64::word_t* __restrict filter_data, u32 key) {
  pimpl->instance.insert(filter_data, key);
  return true; // inserts never fail
}

$u1
bbf_64::batch_insert(bbf_64::word_t* __restrict filter_data, u32* __restrict keys, u32 key_cnt) {
  pimpl->instance.batch_insert(filter_data, keys, key_cnt);
  return true; // inserts never fail
}

$u1
bbf_64::contains(const bbf_64::word_t* __restrict filter_data, u32 key) const {
  return pimpl->instance.contains(filter_data, key);
}

$u64
bbf_64::batch_contains(const bbf_64::word_t* __restrict filter_data,
                       u32* __restrict keys, u32 key_cnt,
                       $u32* __restrict match_positions, u32 match_offset) const {
  return pimpl->instance.batch_contains(filter_data, keys, key_cnt, match_positions, match_offset);
}

void
bbf_64::calibrate(u64 filter_size_bits) {
  tuner.tune_unroll_factor(filter_size_bits);
}

void
bbf_64::force_unroll_factor(u32 u) {
  tuner.set_unroll_factor(u);
}

std::string
bbf_64::name() const {
  return pimpl->instance.name();
}

std::size_t
bbf_64::size_in_bytes() const {
  return pimpl->instance.size_in_bytes();
}

std::size_t
bbf_64::size() const {
  return pimpl->instance.size();
}

} // namespace dtl
