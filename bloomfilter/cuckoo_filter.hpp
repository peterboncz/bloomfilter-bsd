#pragma once

#include <cstdlib>

#include <dtl/dtl.hpp>
#include <dtl/hash.hpp>
#include <dtl/math.hpp>
#include <dtl/mem.hpp>

#include <dtl/bloomfilter/block_addressing_logic.hpp>
#include <dtl/bloomfilter/cuckoo_filter_word_table.hpp>
#include <dtl/bloomfilter/cuckoo_filter_multiword_table.hpp>
#include <dtl/bloomfilter/cuckoo_filter_helper.hpp>

#include <dtl/bloomfilter/bloomfilter_h1_vec.hpp>


namespace dtl {
namespace cuckoo_filter {


//template<block_addressing addressing>
template<typename _filter_t>
__forceinline__ __unroll_loops__ __host__
static std::size_t
//batch_contains(const dtl::blocked_cuckoo_filter<16, 4, addressing>& filter,
simd_batch_contains_8_4(const _filter_t& filter,
                        u32* __restrict keys, u32 key_cnt,
                        $u32* __restrict match_positions, u32 match_offset) {
  using namespace dtl;
  using filter_t = _filter_t;
  using key_t = $u32;
  using hash_value_t = $u32;

  const key_t* reader = keys;
  $u32* match_writer = match_positions;

  // determine the number of keys that need to be probed sequentially, due to alignment
  u64 required_alignment_bytes = 64;
  u64 t = dtl::mem::is_aligned(reader)  // should always be true
          ? (required_alignment_bytes - (reinterpret_cast<uintptr_t>(reader) % required_alignment_bytes)) / sizeof(key_t) // FIXME first elements are processed sequentially even if aligned
          : key_cnt;
  u64 unaligned_key_cnt = std::min(static_cast<$u64>(key_cnt), t);
  // process the unaligned keys sequentially
  $u64 read_pos = 0;
  for (; read_pos < unaligned_key_cnt; read_pos++) {
    u1 is_match = filter.contains(*reader);
    *match_writer = static_cast<$u32>(read_pos) + match_offset;
    match_writer += is_match;
    reader++;
  }

  // process the aligned keys vectorized
  constexpr std::size_t vector_len = 64;
  using key_vt = vec<key_t, vector_len>;
  using ptr_vt = vec<$u64, vector_len>;

  constexpr u32 block_size_log2 = dtl::ct::log_2_u64<sizeof(typename filter_t::block_type)>::value;
  constexpr u32 word_size_log2 = dtl::ct::log_2_u64<sizeof(typename filter_t::table_type::word_t)>::value;

  r256 offset_vec = {.i = _mm256_set1_epi32(match_offset + read_pos) };
  const r256 overflow_tag = {.i = _mm256_set1_epi64x(-1) };
  using mask_t = typename vec<key_t, vector_len>::mask;
  u64 aligned_key_cnt = ((key_cnt - unaligned_key_cnt) / vector_len) * vector_len;

  if ((filter.filter.addr.get_required_addressing_bits() + filter_t::block_type::required_hash_bits) <= (sizeof(hash_value_t) * 8)) {

    // --- contains hash ---
    for (; read_pos < (unaligned_key_cnt + aligned_key_cnt); read_pos += vector_len) {

      const key_vt& key_v = *reinterpret_cast<const key_vt*>(reader);
      const auto block_hash_v = dtl::hash::knuth_32_alt<key_vt>::hash(key_v);
      auto block_idx_v = filter.filter.addr.get_block_idxs(block_hash_v);

      // compute block address
      ptr_vt ptr_v = dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(block_idx_v);
      ptr_v <<= block_size_log2;
      ptr_v += reinterpret_cast<std::uintptr_t>(&filter.filter.blocks[0]);

      auto bucket_hash_v = block_hash_v << filter.filter.addr.get_required_addressing_bits();
      auto bucket_idx_v = filter_t::block_type::get_bucket_idxs(bucket_hash_v);
      auto tag_v = (bucket_hash_v >> (32 - filter_t::table_type::bucket_addressing_bits - filter_t::table_type::tag_size_bits))
                   & static_cast<uint32_t>(filter_t::table_type::tag_mask);
      tag_v[tag_v == 0] += 1; // tag must not be zero
      auto alternative_bucket_idx_v = filter_t::block_type::get_alternative_bucket_idxs(bucket_idx_v, tag_v);

      const auto word_idx_v = bucket_idx_v & ((1u << filter_t::table_type::word_cnt_log2) - 1);
      const auto alternative_word_idx_v = alternative_bucket_idx_v & ((1u << filter_t::table_type::word_cnt_log2) - 1);


//      const auto bucket = word >> (bucket_size_bits * in_word_bucket_idx);

//      const auto in_word_bucket_idx = bucket_idx_v >> filter_t::table_type::word_cnt_log2; // can either be 0 or 1
      const auto bucket_ptr_v = ptr_v + dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(
          word_idx_v << word_size_log2);

//      const auto in_word_alternative_bucket_idx = alternative_bucket_idx_v >> filter_t::table_type::word_cnt_log2; // can either be 0 or 1
      const auto alternative_bucket_ptr_v = ptr_v + dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(
          alternative_word_idx_v << word_size_log2);

      // load the buckets
      const auto bucket_v = dtl::internal::vector_gather<$u32, $u64, vector_len>::gather(bucket_ptr_v);
      const auto alternative_bucket_v = dtl::internal::vector_gather<$u32, $u64, vector_len>::gather(alternative_bucket_ptr_v);

      auto dup_tag_v = (tag_v | (tag_v << 8)) ;
      dup_tag_v |= dup_tag_v << 16;

      const auto b = reinterpret_cast<const r256*>(&bucket_v.data);
      const auto ba = reinterpret_cast<const r256*>(&alternative_bucket_v.data);
      auto t = reinterpret_cast<r256*>(&dup_tag_v.data);
      for (std::size_t i = 0; i < bucket_v.nested_vector_cnt; i++) {
        const r256 tags = t[i];
        const r256 bucket_content0 = b[i];
        const r256 bucket_content1 = ba[i];
        const r256 t0 = {.i = _mm256_cmpeq_epi8(bucket_content0.i,tags.i) };
        const r256 o0 = {.i = _mm256_cmpeq_epi8(bucket_content0.i,overflow_tag.i) };
        const r256 t1 = {.i = _mm256_cmpeq_epi8(bucket_content1.i,tags.i) };
        const r256 o1 = {.i = _mm256_cmpeq_epi8(bucket_content1.i,overflow_tag.i) };
        const r256 t2 = {.i = _mm256_or_si256(_mm256_or_si256(t0.i,o0.i), _mm256_or_si256(t1.i,o1.i)) };
        const r256 t3 = {.i = _mm256_cmpeq_epi32(t2.i, _mm256_setzero_si256()) };
        const auto mt = _mm256_movemask_ps(t3.s) ^ 0b11111111u;
//        std::cout << std::bitset<4>(mt) << " ";
        const r256 match_pos_vec = { .i = { _mm256_cvtepi16_epi32(dtl::simd::lut_match_pos[mt].i) } };
        const r256 pos_vec = {.i = _mm256_add_epi32(offset_vec.i, match_pos_vec.i) };
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(match_writer), pos_vec.i);
        match_writer += _popcnt32(mt);
        offset_vec.i = _mm256_add_epi32(offset_vec.i, _mm256_set1_epi32(8));
      }

      reader += vector_len;
    }
  }
  else {
    // --- contains key ---
    for (; read_pos < (unaligned_key_cnt + aligned_key_cnt); read_pos += vector_len) {

      const key_vt& key_v = *reinterpret_cast<const key_vt*>(reader);
      const auto block_hash_v = dtl::hash::knuth_32_alt<key_vt>::hash(key_v);
      auto block_idx_v = filter.filter.addr.get_block_idxs(block_hash_v);

      // compute block address
      ptr_vt ptr_v = dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(block_idx_v);
      ptr_v <<= block_size_log2;
      ptr_v += reinterpret_cast<std::uintptr_t>(&filter.filter.blocks[0]);

      auto bucket_hash_v = dtl::hash::knuth_32<key_vt>::hash(key_v);
      auto bucket_idx_v = filter_t::block_type::get_bucket_idxs(bucket_hash_v);
      auto tag_v = (bucket_hash_v >> (32 - filter_t::table_type::bucket_addressing_bits - filter_t::table_type::tag_size_bits))
                   & static_cast<uint32_t>(filter_t::table_type::tag_mask);
      tag_v[tag_v == 0] += 1; // tag must not be zero
      auto alternative_bucket_idx_v = filter_t::block_type::get_alternative_bucket_idxs(bucket_idx_v, tag_v);

      const auto word_idx_v = bucket_idx_v & ((1u << filter_t::table_type::word_cnt_log2) - 1);
      const auto alternative_word_idx_v = alternative_bucket_idx_v & ((1u << filter_t::table_type::word_cnt_log2) - 1);


//      const auto bucket = word >> (bucket_size_bits * in_word_bucket_idx);

//      const auto in_word_bucket_idx = bucket_idx_v >> filter_t::table_type::word_cnt_log2; // can either be 0 or 1
      const auto bucket_ptr_v = ptr_v + dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(
          word_idx_v << word_size_log2);

//      const auto in_word_alternative_bucket_idx = alternative_bucket_idx_v >> filter_t::table_type::word_cnt_log2; // can either be 0 or 1
      const auto alternative_bucket_ptr_v = ptr_v + dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(
          alternative_word_idx_v << word_size_log2);

      // load the buckets
      const auto bucket_v = dtl::internal::vector_gather<$u32, $u64, vector_len>::gather(bucket_ptr_v);
      const auto alternative_bucket_v = dtl::internal::vector_gather<$u32, $u64, vector_len>::gather(alternative_bucket_ptr_v);

      auto dup_tag_v = (tag_v | (tag_v << 8)) ;
      dup_tag_v |= dup_tag_v << 16;

      const auto b = reinterpret_cast<const r256*>(&bucket_v.data);
      const auto ba = reinterpret_cast<const r256*>(&alternative_bucket_v.data);
      auto t = reinterpret_cast<r256*>(&dup_tag_v.data);
      for (std::size_t i = 0; i < bucket_v.nested_vector_cnt; i++) {
        const r256 tags = t[i];
        const r256 bucket_content0 = b[i];
        const r256 bucket_content1 = ba[i];
        const r256 t0 = {.i = _mm256_cmpeq_epi8(bucket_content0.i,tags.i) };
        const r256 o0 = {.i = _mm256_cmpeq_epi8(bucket_content0.i,overflow_tag.i) };
        const r256 t1 = {.i = _mm256_cmpeq_epi8(bucket_content1.i,tags.i) };
        const r256 o1 = {.i = _mm256_cmpeq_epi8(bucket_content1.i,overflow_tag.i) };
        const r256 t2 = {.i = _mm256_or_si256(_mm256_or_si256(t0.i,o0.i), _mm256_or_si256(t1.i,o1.i)) };
        const r256 t3 = {.i = _mm256_cmpeq_epi32(t2.i, _mm256_setzero_si256()) };
        const auto mt = _mm256_movemask_ps(t3.s) ^ 0b11111111u;
//        std::cout << std::bitset<4>(mt) << " ";
        const r256 match_pos_vec = { .i = { _mm256_cvtepi16_epi32(dtl::simd::lut_match_pos[mt].i) } };
        const r256 pos_vec = {.i = _mm256_add_epi32(offset_vec.i, match_pos_vec.i) };
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(match_writer), pos_vec.i);
        match_writer += _popcnt32(mt);
        offset_vec.i = _mm256_add_epi32(offset_vec.i, _mm256_set1_epi32(8));
      }

      reader += vector_len;
    }

  }


  // process remaining keys sequentially
  for (; read_pos < key_cnt; read_pos++) {
    u1 is_match = filter.contains(*reader);
    *match_writer = static_cast<$u32>(read_pos) + match_offset;
    match_writer += is_match;
    reader++;
  }
  return match_writer - match_positions;
}


template<typename _filter_t>
__forceinline__ __unroll_loops__ __host__
static std::size_t
//batch_contains(const dtl::blocked_cuckoo_filter<16, 4, addressing>& filter,
simd_batch_contains_16_4(const _filter_t& filter,
                         u32* __restrict keys, u32 key_cnt,
                         $u32* __restrict match_positions, u32 match_offset) {
  using namespace dtl;
  using filter_t = _filter_t;
  using key_t = $u32;
  using hash_value_t = $u32;

  const key_t* reader = keys;
  $u32* match_writer = match_positions;

  // determine the number of keys that need to be probed sequentially, due to alignment
  u64 required_alignment_bytes = 64;
  u64 t = dtl::mem::is_aligned(reader)  // should always be true
          ? (required_alignment_bytes - (reinterpret_cast<uintptr_t>(reader) % required_alignment_bytes)) / sizeof(key_t) // FIXME first elements are processed sequentially even if aligned
          : key_cnt;
  u64 unaligned_key_cnt = std::min(static_cast<$u64>(key_cnt), t);
  // process the unaligned keys sequentially
  $u64 read_pos = 0;
  for (; read_pos < unaligned_key_cnt; read_pos++) {
    u1 is_match = filter.contains(*reader);
    *match_writer = static_cast<$u32>(read_pos) + match_offset;
    match_writer += is_match;
    reader++;
  }

  // process the aligned keys vectorized
  constexpr std::size_t vector_len = 32;
  using key_vt = vec<key_t, vector_len>;
  using ptr_vt = vec<$u64, vector_len>;

  constexpr u32 block_size_log2 = dtl::ct::log_2_u64<sizeof(typename filter_t::block_type)>::value;
  constexpr u32 word_size_log2 = dtl::ct::log_2_u64<sizeof(typename filter_t::table_type::word_t)>::value;

  r128 offset_vec = {.i = _mm_set1_epi32(match_offset + read_pos) };
  const r256 overflow_tag = {.i = _mm256_set1_epi64x(-1) };
  using mask_t = typename vec<key_t, vector_len>::mask;
  u64 aligned_key_cnt = ((key_cnt - unaligned_key_cnt) / vector_len) * vector_len;

  if ((filter.filter.addr.get_required_addressing_bits() + filter_t::block_type::required_hash_bits) <= (sizeof(hash_value_t) * 8)) {
    // --- contains hash ---
    for (; read_pos < (unaligned_key_cnt + aligned_key_cnt); read_pos += vector_len) {

      const key_vt& key_v = *reinterpret_cast<const key_vt*>(reader);
      const auto block_hash_v = dtl::hash::knuth_32_alt<key_vt>::hash(key_v);
      auto block_idx_v = filter.filter.addr.get_block_idxs(block_hash_v);

      // compute block address
      ptr_vt ptr_v = dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(block_idx_v);
      ptr_v <<= block_size_log2;
      ptr_v += reinterpret_cast<std::uintptr_t>(&filter.filter.blocks[0]);

      auto bucket_hash_v = block_hash_v << filter.filter.addr.get_required_addressing_bits();
      auto bucket_idx_v = filter_t::block_type::get_bucket_idxs(bucket_hash_v);
      auto tag_v = (bucket_hash_v >> (32 - filter_t::table_type::bucket_addressing_bits - filter_t::table_type::tag_size_bits))
                   & static_cast<uint32_t>(filter_t::table_type::tag_mask);
      tag_v[tag_v == 0] += 1; // tag must not be zero
      auto alternative_bucket_idx_v = filter_t::block_type::get_alternative_bucket_idxs(bucket_idx_v, tag_v);

      const auto word_idx_v = bucket_idx_v & ((1u << filter_t::table_type::word_cnt_log2) - 1);
      const auto alternative_word_idx_v = alternative_bucket_idx_v & ((1u << filter_t::table_type::word_cnt_log2) - 1);
//      const auto in_word_bucket_idx = bucket_idx >> word_cnt_log2;
//      const auto bucket = word >> (bucket_size_bits * in_word_bucket_idx);

      const auto bucket_ptr_v = ptr_v + dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(word_idx_v << word_size_log2);
      const auto alternative_bucket_ptr_v = ptr_v + dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(alternative_word_idx_v << word_size_log2);

      // load the buckets
      const auto bucket_v = dtl::internal::vector_gather<$u64, $u64, vector_len>::gather(bucket_ptr_v);
      const auto alternative_bucket_v = dtl::internal::vector_gather<$u64, $u64, vector_len>::gather(alternative_bucket_ptr_v);

      auto dup_tag_v = dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(tag_v | (tag_v << 16)) ;
      dup_tag_v |= dup_tag_v << 32;

      const auto b = reinterpret_cast<const r256*>(&bucket_v.data);
      const auto ba = reinterpret_cast<const r256*>(&alternative_bucket_v.data);
      auto t = reinterpret_cast<r256*>(&dup_tag_v.data);
      for (std::size_t i = 0; i < bucket_v.nested_vector_cnt; i++) {
        const r256 tags = t[i];
        const r256 bucket_content0 = b[i];
        const r256 bucket_content1 = ba[i];
        const r256 t0 = {.i = _mm256_cmpeq_epi16(bucket_content0.i,tags.i) };
        const r256 o0 = {.i = _mm256_cmpeq_epi16(bucket_content0.i,overflow_tag.i) };
        const r256 t1 = {.i = _mm256_cmpeq_epi16(bucket_content1.i,tags.i) };
        const r256 o1 = {.i = _mm256_cmpeq_epi16(bucket_content1.i,overflow_tag.i) };
        const r256 t2 = {.i = _mm256_or_si256(_mm256_or_si256(t0.i,o0.i), _mm256_or_si256(t1.i,o1.i)) };
        const r256 t3 = {.i = _mm256_cmpeq_epi64(t2.i, _mm256_setzero_si256()) };
        const auto mt = _mm256_movemask_pd(t3.d) ^ 0b1111;
//        std::cout << std::bitset<4>(mt) << " ";
        const r128 match_pos_vec = { .i = dtl::simd::lut_match_pos_4bit[mt].i };
        const r128 pos_vec = {.i = _mm_add_epi32(offset_vec.i, match_pos_vec.i) };
        _mm_storeu_si128(reinterpret_cast<__m128i*>(match_writer), pos_vec.i);
        match_writer += _popcnt32(mt);
        offset_vec.i = _mm_add_epi32(offset_vec.i, _mm_set1_epi32(4));
      }

      reader += vector_len;
    }
  }
  else {
    // --- contains key ---
    for (; read_pos < (unaligned_key_cnt + aligned_key_cnt); read_pos += vector_len) {

      const key_vt& key_v = *reinterpret_cast<const key_vt*>(reader);
      const auto block_hash_v = dtl::hash::knuth_32_alt<key_vt>::hash(key_v);
      auto block_idx_v = filter.filter.addr.get_block_idxs(block_hash_v);

      // compute block address
      ptr_vt ptr_v = dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(block_idx_v);
      ptr_v <<= block_size_log2;
      ptr_v += reinterpret_cast<std::uintptr_t>(&filter.filter.blocks[0]);

      // contains hash
      auto bucket_hash_v = dtl::hash::knuth_32<key_vt>::hash(key_v);
      auto bucket_idx_v = filter_t::block_type::get_bucket_idxs(bucket_hash_v);
      auto tag_v = (bucket_hash_v >> (32 - filter_t::table_type::bucket_addressing_bits - filter_t::table_type::tag_size_bits))
                   & static_cast<uint32_t>(filter_t::table_type::tag_mask);
      tag_v[tag_v == 0] += 1; // tag must not be zero
      auto alternative_bucket_idx_v = filter_t::block_type::get_alternative_bucket_idxs(bucket_idx_v, tag_v);

      const auto word_idx_v = bucket_idx_v & ((1u << filter_t::table_type::word_cnt_log2) - 1);
      const auto alternative_word_idx_v = alternative_bucket_idx_v & ((1u << filter_t::table_type::word_cnt_log2) - 1);
//      const auto in_word_bucket_idx = bucket_idx >> word_cnt_log2;
//      const auto bucket = word >> (bucket_size_bits * in_word_bucket_idx);

      const auto bucket_ptr_v = ptr_v + dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(word_idx_v << word_size_log2);
      const auto alternative_bucket_ptr_v = ptr_v + dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(alternative_word_idx_v << word_size_log2);

      // load the buckets
      const auto bucket_v = dtl::internal::vector_gather<$u64, $u64, vector_len>::gather(bucket_ptr_v);
      const auto alternative_bucket_v = dtl::internal::vector_gather<$u64, $u64, vector_len>::gather(alternative_bucket_ptr_v);

      auto dup_tag_v = dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(tag_v | (tag_v << 16)) ;
      dup_tag_v |= dup_tag_v << 32;

      const auto b = reinterpret_cast<const r256*>(&bucket_v.data);
      const auto ba = reinterpret_cast<const r256*>(&alternative_bucket_v.data);
      auto t = reinterpret_cast<r256*>(&dup_tag_v.data);
      for (std::size_t i = 0; i < bucket_v.nested_vector_cnt; i++) {
        const r256 tags = t[i];
        const r256 bucket_content0 = b[i];
        const r256 bucket_content1 = ba[i];
        const r256 t0 = {.i = _mm256_cmpeq_epi16(bucket_content0.i,tags.i) };
        const r256 o0 = {.i = _mm256_cmpeq_epi16(bucket_content0.i,overflow_tag.i) };
        const r256 t1 = {.i = _mm256_cmpeq_epi16(bucket_content1.i,tags.i) };
        const r256 o1 = {.i = _mm256_cmpeq_epi16(bucket_content1.i,overflow_tag.i) };
        const r256 t2 = {.i = _mm256_or_si256(_mm256_or_si256(t0.i,o0.i), _mm256_or_si256(t1.i,o1.i)) };
        const r256 t3 = {.i = _mm256_cmpeq_epi64(t2.i, _mm256_setzero_si256()) };
        const auto mt = _mm256_movemask_pd(t3.d) ^ 0b1111;
//        std::cout << std::bitset<4>(mt) << " ";
        const r128 match_pos_vec = { .i = dtl::simd::lut_match_pos_4bit[mt].i };
        const r128 pos_vec = {.i = _mm_add_epi32(offset_vec.i, match_pos_vec.i) };
        _mm_storeu_si128(reinterpret_cast<__m128i*>(match_writer), pos_vec.i);
        match_writer += _popcnt32(mt);
        offset_vec.i = _mm_add_epi32(offset_vec.i, _mm_set1_epi32(4));
      }

      reader += vector_len;
    }
  }


  // process remaining keys sequentially
  for (; read_pos < key_cnt; read_pos++) {
    u1 is_match = filter.contains(*reader);
    *match_writer = static_cast<$u32>(read_pos) + match_offset;
    match_writer += is_match;
    reader++;
  }
  return match_writer - match_positions;
}






namespace internal {


// A statically sized cuckoo filter (used for blocking).
template<
    typename __key_t = uint32_t,
    typename __table_t = cuckoo_filter_multiword_table<uint64_t, 64, 16, 4>
>
struct cuckoo_filter {

  using key_t = __key_t;
  using table_t = __table_t;
  using hash_value_t = uint32_t;
  using hasher = dtl::hash::knuth_32<hash_value_t>;

  static constexpr uint32_t capacity = table_t::capacity;
  static constexpr uint32_t required_hash_bits = table_t::required_hash_bits;

  // for compatibility with block addressing logic
  static constexpr uint32_t block_bitlength = sizeof(table_t) * 8;
  // ---

//private:

  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  table_t table;
  //===----------------------------------------------------------------------===//

  __forceinline__
  static uint32_t
  get_bucket_idx(const hash_value_t hash_value) {
    return (hash_value >> (32 - table_t::bucket_addressing_bits)); //% table_t::bucket_count;
  }


  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  static dtl::vec<hash_value_t, dtl::vector_length<Tv>::value>
  get_bucket_idxs(const Tv& hash_values) {
    return (hash_values >> (32 - table_t::bucket_addressing_bits)); //% table_t::bucket_count;
  }


  __forceinline__
  static uint32_t
  get_alternative_bucket_idx(const uint32_t bucket_idx, const uint32_t tag) {
    return (bucket_idx ^ ((tag * 0x5bd1e995u) >> (32 - table_t::bucket_addressing_bits)));// & ((1u << table_t::bucket_addressing_bits) - 1);
  }


  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  static dtl::vec<hash_value_t, dtl::vector_length<Tv>::value>
  get_alternative_bucket_idxs(const Tv& bucket_idxs, const Tv& tags) {
    return (bucket_idxs ^ ((tags * 0x5bd1e995u) >> (32 - table_t::bucket_addressing_bits)));// & ((1u << table_t::bucket_addressing_bits) - 1);
  }


  __forceinline__
  void
  insert(const uint32_t bucket_idx, const uint32_t tag) {
    uint32_t current_idx = bucket_idx;
    uint32_t current_tag = tag;
    uint32_t old_tag;

    // Try to insert without kicking other tags out.
    old_tag = table.insert_tag(current_idx, current_tag);
    if (old_tag == table_t::null_tag) { return; } // successfully inserted
    if (old_tag == table_t::overflow_tag) { return; } // hit an overflowed bucket (always return true)

    // Re-try at the alternative bucket.
    current_idx = get_alternative_bucket_idx(current_idx, current_tag);

    for (uint32_t count = 0; count < 50; count++) {
      old_tag = table.insert_tag_relocate(current_idx, current_tag);
      if (old_tag == table_t::null_tag) { return; } // successfully inserted
      if (old_tag == table_t::overflow_tag) { return; } // hit an overflowed bucket (always return true)
//      std::cout << ".";
      current_tag = old_tag;
      current_idx = get_alternative_bucket_idx(current_idx, current_tag);
    }
    // Failed to find a place for the current tag through partial-key cuckoo hashing.
    // Introduce an overflow bucket.
//    std::cout << "!";
    table.mark_overflow(current_idx);
  }


public:

  __forceinline__
  void
  insert_hash(const hash_value_t& hash_value) {
    auto bucket_idx = get_bucket_idx(hash_value);
    auto tag = (hash_value >> (32 - table_t::bucket_addressing_bits - table_t::tag_size_bits)) & table_t::tag_mask;
    tag += (tag == 0); // tag must not be zero
    insert(bucket_idx, tag);
  }


  __forceinline__
  void
  insert_key(const key_t& key) {
    auto hash_value = hasher::hash(key);
    insert_hash(hash_value);
  }


  __forceinline__
  bool
  contains_hash(const hash_value_t& hash_value) const {
    auto bucket_idx = get_bucket_idx(hash_value);
    auto tag = (hash_value >> (32 - table_t::bucket_addressing_bits - table_t::tag_size_bits)) & table_t::tag_mask;
    tag += (tag == 0); // tag must not be zero
    const auto alt_bucket_idx = get_alternative_bucket_idx(bucket_idx, tag);
    return table.find_tag_in_buckets(bucket_idx, alt_bucket_idx, tag);
  }


  __forceinline__
  bool
  contains_key(const key_t& key) const {
    const auto hash_value = hasher::hash(key);
    return contains_hash(hash_value);
  }


};


//===----------------------------------------------------------------------===//


// A blocked cuckoo filter.
template<
    typename __key_t = uint32_t,
    typename __block_t = cuckoo_filter<__key_t>,
    block_addressing __block_addressing = block_addressing::POWER_OF_TWO
>
struct blocked_cuckoo_filter {

  using key_t = __key_t;
  using block_t = __block_t;
  using hash_value_t = uint32_t;
  using hasher = dtl::hash::knuth_32_alt<hash_value_t>;
  using addr_t = block_addressing_logic<__block_addressing>;


  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  const addr_t addr;
  std::vector<block_t, dtl::mem::numa_allocator<block_t>> blocks;
  //===----------------------------------------------------------------------===//


  explicit
  blocked_cuckoo_filter(const std::size_t length) : addr(length, block_t::block_bitlength), blocks(addr.get_block_cnt()) { }

  blocked_cuckoo_filter(const blocked_cuckoo_filter&) noexcept = default;

  blocked_cuckoo_filter(blocked_cuckoo_filter&&) noexcept = default;

  ~blocked_cuckoo_filter() noexcept = default;


  __forceinline__
  void
  insert(const key_t& key) {
    auto h = hasher::hash(key);
    auto i = addr.get_block_idx(h);
    if ((addr.get_required_addressing_bits() + block_t::required_hash_bits) <= (sizeof(hash_value_t) * 8)) {
      blocks[i].insert_hash(h << addr.get_required_addressing_bits());
    }
    else {
      blocks[i].insert_key(key);
    }
  }


  __forceinline__
  uint64_t
  batch_insert(const key_t* keys, const uint32_t key_cnt) {
    for (uint32_t i = 0; i < key_cnt; i++) {
      insert(keys[i]);
    }
  }


  __forceinline__
  bool
  contains(const key_t& key) const {
    auto h = hasher::hash(key);
    auto i = addr.get_block_idx(h);
    if ((addr.get_required_addressing_bits() + block_t::required_hash_bits) <= (sizeof(hash_value_t) * 8)) {
      return blocks[i].contains_hash(h << addr.get_required_addressing_bits());
    }
    else {
      return blocks[i].contains_key(key);
    }
  }


  /// Performs a batch-probe
  __forceinline__ __unroll_loops__ __host__
  std::size_t
  batch_contains(const key_t* __restrict keys, u32 key_cnt,
                 $u32* __restrict match_positions, u32 match_offset) const {
    constexpr u32 mini_batch_size = 16;
    const u32 mini_batch_cnt = key_cnt / mini_batch_size;

    $u32* match_writer = match_positions;
    if ((addr.get_required_addressing_bits() + block_t::required_hash_bits) <= (sizeof(hash_value_t) * 8)) {

      for ($u32 mb = 0; mb < mini_batch_cnt; mb++) {
        for (uint32_t j = mb * mini_batch_size; j < ((mb + 1) * mini_batch_size); j++) {
          auto h = hasher::hash(keys[j]);
          auto i = addr.get_block_idx(h);
          auto is_contained = blocks[i].contains_hash(h << addr.get_required_addressing_bits());
          *match_writer = j + match_offset;
          match_writer += is_contained;
        }
      }
      for (uint32_t j = (mini_batch_cnt * mini_batch_size); j < key_cnt; j++) {
        auto h = hasher::hash(keys[j]);
        auto i = addr.get_block_idx(h);
        auto is_contained = blocks[i].contains_hash(h << addr.get_required_addressing_bits());
        *match_writer = j + match_offset;
        match_writer += is_contained;
      }
    }

    else {
      for ($u32 mb = 0; mb < mini_batch_cnt; mb++) {
        for (uint32_t j = mb * mini_batch_size; j < ((mb + 1) * mini_batch_size); j++) {
          auto k = keys[j];
          auto h = hasher::hash(k);
          auto i = addr.get_block_idx(h);
          auto is_contained = blocks[i].contains_key(k);
          *match_writer = j + match_offset;
          match_writer += is_contained;
        }
      }
      for (uint32_t j = (mini_batch_cnt * mini_batch_size); j < key_cnt; j++) {
        auto k = keys[j];
        auto h = hasher::hash(k);
        auto i = addr.get_block_idx(h);
        auto is_contained = blocks[i].contains_key(k);
        *match_writer = j + match_offset;
        match_writer += is_contained;
      }
    }
    return match_writer - match_positions;
  }


};


static constexpr uint64_t cache_line_size = 64;


} // namespace internal
} // namespace cuckoo_filter


//===----------------------------------------------------------------------===//
// Instantiations of some reasonable cuckoo filters
//===----------------------------------------------------------------------===//
template<typename __key_t, typename __derived>
struct blocked_cuckoo_filter_base {

  using key_t = __key_t;


  __forceinline__ void
  insert(const key_t& key) {
    return static_cast<__derived*>(this)->filter.insert(key);
  }


  __forceinline__ uint64_t
  batch_insert(const key_t* keys, const uint32_t key_cnt) {
    return static_cast<__derived*>(this)->filter.batch_insert(keys, key_cnt);
  }


  __forceinline__ bool
  contains(const key_t& key) const {
    return static_cast<const __derived*>(this)->filter.contains(key);
  }


  __forceinline__ uint64_t
  batch_contains(const key_t* __restrict keys, const uint32_t key_cnt,
                 uint32_t* __restrict match_positions, const uint32_t match_offset) const {
    return static_cast<const __derived*>(this)->filter.batch_contains(keys, key_cnt, match_positions, match_offset);
  };

};


template<uint32_t block_size_bytes, uint32_t bits_per_element, uint32_t associativity, block_addressing addressing>
struct blocked_cuckoo_filter {};


template<uint32_t block_size_bytes, block_addressing addressing>
struct blocked_cuckoo_filter<block_size_bytes, 16, 4, addressing> : blocked_cuckoo_filter_base<uint32_t, blocked_cuckoo_filter<block_size_bytes, 16, 4, addressing>> {
  using key_type = uint32_t;
  using table_type = cuckoo_filter::cuckoo_filter_multiword_table<uint64_t, block_size_bytes, 16, 4>;
  using block_type = cuckoo_filter::internal::cuckoo_filter<key_type, table_type>;
  using filter_type = cuckoo_filter::internal::blocked_cuckoo_filter<uint32_t, block_type, addressing>;

  filter_type filter; // the actual filter instance

  explicit blocked_cuckoo_filter(const std::size_t length) : filter(length) { }

  __forceinline__ uint64_t
  batch_contains(const key_type* __restrict keys, const uint32_t key_cnt,
                 uint32_t* __restrict match_positions, const uint32_t match_offset) const {
    return dtl::cuckoo_filter::simd_batch_contains_16_4(*this, keys, key_cnt, match_positions, match_offset);
  };


};


template<uint32_t block_size_bytes, block_addressing addressing>
struct blocked_cuckoo_filter<block_size_bytes, 16, 2, addressing> : blocked_cuckoo_filter_base<uint32_t, blocked_cuckoo_filter<block_size_bytes, 16, 2, addressing>> {
  using key_type = uint32_t;
  using table_type = cuckoo_filter::cuckoo_filter_multiword_table<uint64_t, block_size_bytes, 16, 2>;
  using block_type = cuckoo_filter::internal::cuckoo_filter<key_type, table_type>;
  using filter_type = cuckoo_filter::internal::blocked_cuckoo_filter<uint32_t, block_type, addressing>;

  filter_type filter; // the actual filter instance

  explicit blocked_cuckoo_filter(const std::size_t length) : filter(length) { }

};


template<uint32_t block_size_bytes, block_addressing addressing>
struct blocked_cuckoo_filter<block_size_bytes, 12, 4, addressing> : blocked_cuckoo_filter_base<uint32_t, blocked_cuckoo_filter<block_size_bytes, 12, 4, addressing>> {
  using key_type = uint32_t;
  using table_type = cuckoo_filter::cuckoo_filter_multiword_table<uint64_t, block_size_bytes, 12, 4>;
  using block_type = cuckoo_filter::internal::cuckoo_filter<key_type, table_type>;
  using filter_type = cuckoo_filter::internal::blocked_cuckoo_filter<uint32_t, block_type, addressing>;

  filter_type filter; // the actual filter instance

  explicit blocked_cuckoo_filter(const std::size_t length) : filter(length) { }

};


template<uint32_t block_size_bytes, block_addressing addressing>
struct blocked_cuckoo_filter<block_size_bytes, 10, 6, addressing> : blocked_cuckoo_filter_base<uint32_t, blocked_cuckoo_filter<block_size_bytes, 10, 6, addressing>> {
  using key_type = uint32_t;
  using table_type = cuckoo_filter::cuckoo_filter_multiword_table<uint64_t, block_size_bytes, 10, 6>;
  using block_type = cuckoo_filter::internal::cuckoo_filter<key_type, table_type>;
  using filter_type = cuckoo_filter::internal::blocked_cuckoo_filter<uint32_t, block_type, addressing>;

  filter_type filter; // the actual filter instance

  explicit blocked_cuckoo_filter(const std::size_t length) : filter(length) { }

};


template<uint32_t block_size_bytes, block_addressing addressing>
struct blocked_cuckoo_filter<block_size_bytes, 8, 8, addressing> : blocked_cuckoo_filter_base<uint32_t, blocked_cuckoo_filter<block_size_bytes, 8, 8, addressing>> {
  using key_type = uint32_t;
  using table_type = cuckoo_filter::cuckoo_filter_multiword_table<uint64_t, block_size_bytes, 8, 8>;
  using block_type = cuckoo_filter::internal::cuckoo_filter<key_type, table_type>;
  using filter_type = cuckoo_filter::internal::blocked_cuckoo_filter<uint32_t, block_type, addressing>;

  filter_type filter; // the actual filter instance

  explicit blocked_cuckoo_filter(const std::size_t length) : filter(length) { }

};


template<uint32_t block_size_bytes, block_addressing addressing>
struct blocked_cuckoo_filter<block_size_bytes, 8, 4, addressing> : blocked_cuckoo_filter_base<uint32_t, blocked_cuckoo_filter<block_size_bytes, 8, 4, addressing>> {
  using key_type = uint32_t;
  using table_type = cuckoo_filter::cuckoo_filter_multiword_table<uint32_t, block_size_bytes, 8, 4>;
  using block_type = cuckoo_filter::internal::cuckoo_filter<key_type, table_type>;
  using filter_type = cuckoo_filter::internal::blocked_cuckoo_filter<uint32_t, block_type, addressing>;

  filter_type filter; // the actual filter instance

  explicit blocked_cuckoo_filter(const std::size_t length) : filter(length) { }

  __forceinline__ uint64_t
  batch_contains(const key_type* __restrict keys, const uint32_t key_cnt,
                 uint32_t* __restrict match_positions, const uint32_t match_offset) const {
    return dtl::cuckoo_filter::simd_batch_contains_8_4(*this, keys, key_cnt, match_positions, match_offset);
  };

};



} // namespace dtl