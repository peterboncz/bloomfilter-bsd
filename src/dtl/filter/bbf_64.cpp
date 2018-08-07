#include "bbf_64.hpp"

#include "blocked_bloomfilter/instances/blocked_bloomfilter_logic_u64_instance.hpp" // extern templates to parallelize builds
#include "blocked_bloomfilter/blocked_bloomfilter.hpp"
#include "blocked_bloomfilter/blocked_bloomfilter_tune_impl.hpp"

namespace dtl {

//GENERATE_EXTERN($u64, 1, 1,  1)
//GENERATE_EXTERN($u64, 1, 1,  2)
//GENERATE_EXTERN($u64, 1, 1,  3)
//GENERATE_EXTERN($u64, 1, 1,  4)
//GENERATE_EXTERN($u64, 1, 1,  5)
//GENERATE_EXTERN($u64, 1, 1,  6)
//GENERATE_EXTERN($u64, 1, 1,  7)
//GENERATE_EXTERN($u64, 1, 1,  8)
//GENERATE_EXTERN($u64, 1, 1,  9)
//GENERATE_EXTERN($u64, 1, 1, 10)
//GENERATE_EXTERN($u64, 1, 1, 11)
//GENERATE_EXTERN($u64, 1, 1, 12)
//GENERATE_EXTERN($u64, 1, 1, 13)
//GENERATE_EXTERN($u64, 1, 1, 14)
//GENERATE_EXTERN($u64, 1, 1, 15)
//GENERATE_EXTERN($u64, 1, 1, 16)
//GENERATE_EXTERN($u64, 1, 2,  2)
//GENERATE_EXTERN($u64, 1, 2,  4)
//GENERATE_EXTERN($u64, 1, 2,  6)
//GENERATE_EXTERN($u64, 1, 2,  8)
//GENERATE_EXTERN($u64, 1, 2, 10)
//GENERATE_EXTERN($u64, 1, 2, 12)
//GENERATE_EXTERN($u64, 1, 2, 14)
//GENERATE_EXTERN($u64, 1, 2, 16)
//GENERATE_EXTERN($u64, 1, 4,  4)
//GENERATE_EXTERN($u64, 1, 4,  8)
//GENERATE_EXTERN($u64, 1, 4, 12)
//GENERATE_EXTERN($u64, 1, 4, 16)
//GENERATE_EXTERN($u64, 1, 8,  8)
//GENERATE_EXTERN($u64, 1, 8, 16)
//
//GENERATE_EXTERN($u64, 2, 1,  1)
//GENERATE_EXTERN($u64, 2, 1,  2)
//GENERATE_EXTERN($u64, 2, 1,  3)
//GENERATE_EXTERN($u64, 2, 1,  4)
//GENERATE_EXTERN($u64, 2, 1,  5)
//GENERATE_EXTERN($u64, 2, 1,  6)
//GENERATE_EXTERN($u64, 2, 1,  7)
//GENERATE_EXTERN($u64, 2, 1,  8)
//GENERATE_EXTERN($u64, 2, 1,  9)
//GENERATE_EXTERN($u64, 2, 1, 10)
//GENERATE_EXTERN($u64, 2, 1, 11)
//GENERATE_EXTERN($u64, 2, 1, 12)
//GENERATE_EXTERN($u64, 2, 1, 13)
//GENERATE_EXTERN($u64, 2, 1, 14)
//GENERATE_EXTERN($u64, 2, 1, 15)
//GENERATE_EXTERN($u64, 2, 1, 16)
//GENERATE_EXTERN($u64, 2, 2,  2)
//GENERATE_EXTERN($u64, 2, 2,  4)
//GENERATE_EXTERN($u64, 2, 2,  6)
//GENERATE_EXTERN($u64, 2, 2,  8)
//GENERATE_EXTERN($u64, 2, 2, 10)
//GENERATE_EXTERN($u64, 2, 2, 12)
//GENERATE_EXTERN($u64, 2, 2, 14)
//GENERATE_EXTERN($u64, 2, 2, 16)
//GENERATE_EXTERN($u64, 2, 4,  4)
//GENERATE_EXTERN($u64, 2, 4,  8)
//GENERATE_EXTERN($u64, 2, 4, 12)
//GENERATE_EXTERN($u64, 2, 4, 16)
//GENERATE_EXTERN($u64, 2, 8,  8)
//GENERATE_EXTERN($u64, 2, 8, 16)
//GENERATE_EXTERN($u64, 2,16, 16)
//
//GENERATE_EXTERN($u64, 4, 1,  1)
//GENERATE_EXTERN($u64, 4, 1,  2)
//GENERATE_EXTERN($u64, 4, 1,  3)
//GENERATE_EXTERN($u64, 4, 1,  4)
//GENERATE_EXTERN($u64, 4, 1,  5)
//GENERATE_EXTERN($u64, 4, 1,  6)
//GENERATE_EXTERN($u64, 4, 1,  7)
//GENERATE_EXTERN($u64, 4, 1,  8)
//GENERATE_EXTERN($u64, 4, 1,  9)
//GENERATE_EXTERN($u64, 4, 1, 10)
//GENERATE_EXTERN($u64, 4, 1, 11)
//GENERATE_EXTERN($u64, 4, 1, 12)
//GENERATE_EXTERN($u64, 4, 1, 13)
//GENERATE_EXTERN($u64, 4, 1, 14)
//GENERATE_EXTERN($u64, 4, 1, 15)
//GENERATE_EXTERN($u64, 4, 1, 16)
//GENERATE_EXTERN($u64, 4, 2,  2)
//GENERATE_EXTERN($u64, 4, 2,  4)
//GENERATE_EXTERN($u64, 4, 2,  6)
//GENERATE_EXTERN($u64, 4, 2,  8)
//GENERATE_EXTERN($u64, 4, 2, 10)
//GENERATE_EXTERN($u64, 4, 2, 12)
//GENERATE_EXTERN($u64, 4, 2, 14)
//GENERATE_EXTERN($u64, 4, 2, 16)
//GENERATE_EXTERN($u64, 4, 4,  4)
//GENERATE_EXTERN($u64, 4, 4,  8)
//GENERATE_EXTERN($u64, 4, 4, 12)
//GENERATE_EXTERN($u64, 4, 4, 16)
//GENERATE_EXTERN($u64, 4, 8,  8)
//GENERATE_EXTERN($u64, 4, 8, 16)
//GENERATE_EXTERN($u64, 4,16, 16)
//
//GENERATE_EXTERN($u64, 8, 1,  1)
//GENERATE_EXTERN($u64, 8, 1,  2)
//GENERATE_EXTERN($u64, 8, 1,  3)
//GENERATE_EXTERN($u64, 8, 1,  4)
//GENERATE_EXTERN($u64, 8, 1,  5)
//GENERATE_EXTERN($u64, 8, 1,  6)
//GENERATE_EXTERN($u64, 8, 1,  7)
//GENERATE_EXTERN($u64, 8, 1,  8)
//GENERATE_EXTERN($u64, 8, 1,  9)
//GENERATE_EXTERN($u64, 8, 1, 10)
//GENERATE_EXTERN($u64, 8, 1, 11)
//GENERATE_EXTERN($u64, 8, 1, 12)
//GENERATE_EXTERN($u64, 8, 1, 13)
//GENERATE_EXTERN($u64, 8, 1, 14)
//GENERATE_EXTERN($u64, 8, 1, 15)
//GENERATE_EXTERN($u64, 8, 1, 16)
//GENERATE_EXTERN($u64, 8, 2,  2)
//GENERATE_EXTERN($u64, 8, 2,  4)
//GENERATE_EXTERN($u64, 8, 2,  6)
//GENERATE_EXTERN($u64, 8, 2,  8)
//GENERATE_EXTERN($u64, 8, 2, 10)
//GENERATE_EXTERN($u64, 8, 2, 12)
//GENERATE_EXTERN($u64, 8, 2, 14)
//GENERATE_EXTERN($u64, 8, 2, 16)
//GENERATE_EXTERN($u64, 8, 4,  4)
//GENERATE_EXTERN($u64, 8, 4,  8)
//GENERATE_EXTERN($u64, 8, 4, 12)
//GENERATE_EXTERN($u64, 8, 4, 16)
//GENERATE_EXTERN($u64, 8, 8,  8)
//GENERATE_EXTERN($u64, 8, 8, 16)
//GENERATE_EXTERN($u64, 8,16, 16)
//
//GENERATE_EXTERN($u64, 16, 1,  1)
//GENERATE_EXTERN($u64, 16, 1,  2)
//GENERATE_EXTERN($u64, 16, 1,  3)
//GENERATE_EXTERN($u64, 16, 1,  4)
//GENERATE_EXTERN($u64, 16, 1,  5)
//GENERATE_EXTERN($u64, 16, 1,  6)
//GENERATE_EXTERN($u64, 16, 1,  7)
//GENERATE_EXTERN($u64, 16, 1,  8)
//GENERATE_EXTERN($u64, 16, 1,  9)
//GENERATE_EXTERN($u64, 16, 1, 10)
//GENERATE_EXTERN($u64, 16, 1, 11)
//GENERATE_EXTERN($u64, 16, 1, 12)
//GENERATE_EXTERN($u64, 16, 1, 13)
//GENERATE_EXTERN($u64, 16, 1, 14)
//GENERATE_EXTERN($u64, 16, 1, 15)
//GENERATE_EXTERN($u64, 16, 1, 16)
//GENERATE_EXTERN($u64, 16, 2,  2)
//GENERATE_EXTERN($u64, 16, 2,  4)
//GENERATE_EXTERN($u64, 16, 2,  6)
//GENERATE_EXTERN($u64, 16, 2,  8)
//GENERATE_EXTERN($u64, 16, 2, 10)
//GENERATE_EXTERN($u64, 16, 2, 12)
//GENERATE_EXTERN($u64, 16, 2, 14)
//GENERATE_EXTERN($u64, 16, 2, 16)
//GENERATE_EXTERN($u64, 16, 4,  4)
//GENERATE_EXTERN($u64, 16, 4,  8)
//GENERATE_EXTERN($u64, 16, 4, 12)
//GENERATE_EXTERN($u64, 16, 4, 16)
//GENERATE_EXTERN($u64, 16, 8,  8)
//GENERATE_EXTERN($u64, 16, 8, 16)
//GENERATE_EXTERN($u64, 16,16, 16)


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
bbf_64::calibrate() {
  tuner.tune_unroll_factor();
}

void
bbf_64::force_unroll_factor(u32 u) {
  throw "not yet implemented";
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
