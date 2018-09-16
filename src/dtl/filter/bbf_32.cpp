#include "bbf_32.hpp"

#include "blocked_bloomfilter/instances/blocked_bloomfilter_logic_u32_instance.hpp" // extern templates to parallelize builds
#include "blocked_bloomfilter/blocked_bloomfilter.hpp"
#include "blocked_bloomfilter/blocked_bloomfilter_tune_impl.hpp"

namespace dtl {

namespace {
static dtl::blocked_bloomfilter_tune_impl<$u32> tuner;
} // anonymous namespace


struct bbf_32::impl {
  using bbf_t = dtl::blocked_bloomfilter<$u32>;
  bbf_t instance;

  impl(const size_t m, u32 k, u32 word_cnt_per_block = 1, u32 sector_cnt = 1)
      : instance(m, k, word_cnt_per_block, sector_cnt, tuner) { };
  ~impl() = default;
  impl(impl&&) = default;
  impl(const impl&) = delete;
  impl& operator=(impl&&) = default;
  impl& operator=(const impl&) = delete;
};

bbf_32::bbf_32(const size_t m, u32 k, u32 word_cnt_per_block, u32 sector_cnt)
    : pimpl{ std::make_unique<impl>(m, k, word_cnt_per_block, sector_cnt) } {}
bbf_32::bbf_32(bbf_32&&) noexcept = default;
bbf_32::~bbf_32() = default;
bbf_32& bbf_32::operator=(bbf_32&&) = default;

$u1
bbf_32::insert(bbf_32::word_t* __restrict filter_data, u32 key) {
  pimpl->instance.insert(reinterpret_cast<impl::bbf_t::word_t*>(filter_data), key);
  return true; // inserts never fail
}

$u1
bbf_32::batch_insert(bbf_32::word_t* __restrict filter_data, u32* __restrict keys, u32 key_cnt) {
  pimpl->instance.batch_insert(reinterpret_cast<impl::bbf_t::word_t*>(filter_data), keys, key_cnt);
  return true; // inserts never fail
}

$u1
bbf_32::contains(const bbf_32::word_t* __restrict filter_data, u32 key) const {
  return pimpl->instance.contains(reinterpret_cast<const impl::bbf_t::word_t*>(filter_data), key);
}

$u64
bbf_32::batch_contains(const bbf_32::word_t* __restrict filter_data,
                       u32* __restrict keys, u32 key_cnt,
                       $u32* __restrict match_positions, u32 match_offset) const {
  return pimpl->instance.batch_contains(reinterpret_cast<const impl::bbf_t::word_t*>(filter_data), keys, key_cnt, match_positions, match_offset);
}

void
bbf_32::calibrate() {
  tuner.tune_unroll_factor();
}

void
bbf_32::force_unroll_factor(u32 u) {
  tuner.set_unroll_factor(u);
}

std::string
bbf_32::name() const {
  return pimpl->instance.name();
}

std::size_t
bbf_32::size_in_bytes() const {
  return pimpl->instance.size_in_bytes();
}

std::size_t
bbf_32::size() const {
  return (pimpl->instance.size() + 1) / 2; // convert from the number of 32-bit words to the number of 64-bit words.
}

} // namespace dtl
