#pragma once

#include <dtl/dtl.hpp>
#include <dtl/thread.hpp>
#include <dtl/simd.hpp>
#include <dtl/filter/blocked_bloomfilter/vector_helper.hpp>

#include <xmmintrin.h>


namespace dtl {
namespace cuckoofilter {
namespace internal {

//===----------------------------------------------------------------------===//
// SIMD implementations to find tags in buckets.
//
// Note: Not all cuckoo filter configurations are SIMD-friendly.
//       Currently supported tag sizes are 8, 16 and 32,
//       with associativities 1, 2 and 4.
//===----------------------------------------------------------------------===//

template<>
struct find_tag_in_buckets_simd<8, 1> {

  static constexpr u1 vectorized = true;

  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__ static
  typename dtl::vec<uint32_t, dtl::vector_length<Tv>::value>::mask
  dispatch(const uint32_t* __restrict filter_data,
           const Tv& bucket_idx1,
           const Tv& bucket_idx2,
           const Tv& search_tag) {

    constexpr std::size_t n = dtl::vector_length<Tv>::value;
    constexpr std::size_t bits_per_tag = 8;
    constexpr uint32_t tag_mask = static_cast<uint32_t>((1ull << bits_per_tag) - 1);

    // Load tags using 32-bit gathers.
    const auto* d = reinterpret_cast<const uint32_t*>(filter_data);

    const auto u32_word_idx1 = bucket_idx1 >> 2;
    const auto word1 = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx1);
    const auto in_word_idx1 = (bucket_idx1 & (4 - 1)) << 3;
    const auto tag1 = (word1 >> in_word_idx1) & tag_mask;

    const auto u32_word_idx2 = bucket_idx2 >> 2;
    const auto word2 = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx2);
    const auto in_word_idx2 = (bucket_idx2 & (4 - 1)) << 3;
    const auto tag2 = (word2 >> in_word_idx2) & tag_mask;

    const auto found_mask = (tag1 == search_tag) | (tag2 == search_tag);
    return found_mask;
  }

};


template<>
struct find_tag_in_buckets_simd<8, 2> {

  static constexpr u1 vectorized = true;

  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __unroll_loops__ __host__ static
  typename dtl::vec<uint32_t, dtl::vector_length<Tv>::value>::mask
  dispatch(const uint32_t* __restrict filter_data,
           const Tv& bucket_idx1,
           const Tv& bucket_idx2,
           const Tv& search_tag) {
    using namespace dtl;
    constexpr std::size_t n = dtl::vector_length<Tv>::value;
    using vec_t = vec<uint32_t, n>;
    using mask_t = typename vec<uint32_t, n>::mask;
    constexpr std::size_t bits_per_tag = 8;
    constexpr std::size_t tags_per_bucket = 2;
    constexpr uint32_t tag_mask_x2 = static_cast<uint32_t>((1ull << (bits_per_tag * tags_per_bucket)) - 1); // mask for one bucket = two consecutive tags

    // Load tags using 32-bit gathers.
    const auto* d = reinterpret_cast<const uint32_t*>(filter_data);

    const auto u32_word_idx1 = bucket_idx1 >> 1; // two buckets per word
    const auto word1 = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx1);
    const auto in_word_idx1 = (bucket_idx1 & (2 - 1)) << 4; // offset is either 0 or 16 bit
    const auto tag1 = (word1 >> in_word_idx1) & tag_mask_x2;

    const auto u32_word_idx2 = bucket_idx2 >> 1;
    const auto in_word_idx2 = (bucket_idx2 & (2 - 1)) << 4;
    const auto word2 = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx2);
    const auto tag2 = (word2 >> in_word_idx2) & tag_mask_x2;

    const vec_t search_tag_u8 = search_tag * 0x01010101;
    mask_t found_mask;

    const auto t1 = reinterpret_cast<const r512*>(&tag1.data);
    const auto t2 = reinterpret_cast<const r512*>(&tag2.data);
    const auto st_u8 = reinterpret_cast<const r512*>(&search_tag_u8.data);
    auto fm = reinterpret_cast<__mmask16*>(&found_mask.data);

#if !defined(__AVX512BW__)
    const r512 msb_off = {.i = _mm512_set1_epi32( uint32_t(-1) >> 1 )};
    const auto zero = _mm512_setzero_si512();
#endif

    using vector_type = decltype(tag1);
    u64 vector_cnt = sizeof(vector_type) / sizeof(r512);
    for (std::size_t i = 0; i < vector_cnt; i++) {
#if defined(__AVX512BW__)
      const __mmask64 found1_m64 = _mm512_cmpeq_epi8_mask(t1[i].i, st_u8[i].i);
      const __mmask64 found2_m64 = _mm512_cmpeq_epi8_mask(t2[i].i, st_u8[i].i);

      const __mmask64 found1_m32 = found1_m64 | (found1_m64 >> 1);
      const __mmask16 found1_m = _pext_u64(found1_m32 | (found1_m32 >> 2), 0x1111111111111111ull);
      const __mmask64 found2_m32 = found2_m64 | (found2_m64 >> 1);
      const __mmask16 found2_m = _pext_u64(found2_m32 | (found2_m32 >> 2), 0x1111111111111111ull);
#else
      // fall back to AVX2 instructions for 8-bit compare
      r512 t1_m;
      t1_m.r256[0].i = _mm256_cmpeq_epi8(t1[i].r256[0].i, st_u8[i].r256[0].i);
      t1_m.r256[1].i = _mm256_cmpeq_epi8(t1[i].r256[1].i, st_u8[i].r256[1].i);

      r512 t2_m;
      t2_m.r256[0].i = _mm256_cmpeq_epi8(t2[i].r256[0].i, st_u8[i].r256[0].i);
      t2_m.r256[1].i = _mm256_cmpeq_epi8(t2[i].r256[1].i, st_u8[i].r256[1].i);

      const __mmask16 found1_m = _mm512_cmpgt_epi32_mask(_mm512_and_si512(t1_m.i, msb_off.i), zero); // turn off msb of mask for signed comparison
      const __mmask16 found2_m = _mm512_cmpgt_epi32_mask(_mm512_and_si512(t2_m.i, msb_off.i), zero);
#endif

      fm[i] = found1_m | found2_m;
    }
    return found_mask;
  }

};

template<>
struct find_tag_in_buckets_simd<8, 4> {

  static constexpr u1 vectorized = true;

  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __unroll_loops__ __host__ static
  typename dtl::vec<uint32_t, dtl::vector_length<Tv>::value>::mask
  dispatch(const uint32_t* __restrict filter_data,
           const Tv& bucket_idx1,
           const Tv& bucket_idx2,
           const Tv& search_tag) {
    using namespace dtl;
    constexpr std::size_t n = dtl::vector_length<Tv>::value;
    using vec_t = vec<uint32_t, n>;
    using mask_t = typename vec<uint32_t, n>::mask;
    const vec_t search_tag_u8 = search_tag * 0x01010101;
    mask_t found_mask;

    // Load tags using 32-bit gathers.
    const auto* d = reinterpret_cast<const uint32_t*>(filter_data);

    // one buckets per word
    const auto tag1 = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, bucket_idx1);
    const auto tag2 = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, bucket_idx2);


    const auto t1 = reinterpret_cast<const r512*>(&tag1.data);
    const auto t2 = reinterpret_cast<const r512*>(&tag2.data);
    const auto st_u8 = reinterpret_cast<const r512*>(&search_tag_u8.data);
    auto fm = reinterpret_cast<__mmask16*>(&found_mask.data);

#if !defined(__AVX512BW__)
    const r512 msb_off = {.i = _mm512_set1_epi32( uint32_t(-1) >> 1 )};
    const auto zero = _mm512_setzero_si512();
#endif

    using vector_type = decltype(tag1);
    u64 vector_cnt = sizeof(vector_type) / sizeof(r512);
    for (std::size_t i = 0; i < vector_cnt; i++) {
#if defined(__AVX512BW__)
      const __mmask64 found1_m64 = _mm512_cmpeq_epi8_mask(t1[i].i, st_u8[i].i);
      const __mmask64 found2_m64 = _mm512_cmpeq_epi8_mask(t2[i].i, st_u8[i].i);

      const __mmask64 found1_m32 = found1_m64 | (found1_m64 >> 1);
      const __mmask16 found1_m = _pext_u64(found1_m32 | (found1_m32 >> 2), 0x1111111111111111ull);
      const __mmask64 found2_m32 = found2_m64 | (found2_m64 >> 1);
      const __mmask16 found2_m = _pext_u64(found2_m32 | (found2_m32 >> 2), 0x1111111111111111ull);
#else
      // fall back to AVX2 instructions for 8-bit compare
      r512 t1_m;
      t1_m.r256[0].i = _mm256_cmpeq_epi8(t1[i].r256[0].i, st_u8[i].r256[0].i);
      t1_m.r256[1].i = _mm256_cmpeq_epi8(t1[i].r256[1].i, st_u8[i].r256[1].i);

      r512 t2_m;
      t2_m.r256[0].i = _mm256_cmpeq_epi8(t2[i].r256[0].i, st_u8[i].r256[0].i);
      t2_m.r256[1].i = _mm256_cmpeq_epi8(t2[i].r256[1].i, st_u8[i].r256[1].i);

      const __mmask16 found1_m = _mm512_cmpgt_epi32_mask(_mm512_and_si512(t1_m.i, msb_off.i), zero); // turn off msb of mask for signed comparison
      const __mmask16 found2_m = _mm512_cmpgt_epi32_mask(_mm512_and_si512(t2_m.i, msb_off.i), zero);
#endif
      fm[i] = found1_m | found2_m;
    }
    return found_mask;
  }

};


template<>
struct find_tag_in_buckets_simd<16, 1> {

  static constexpr u1 vectorized = true;

  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __unroll_loops__ __host__ static
  typename dtl::vec<uint32_t, dtl::vector_length<Tv>::value>::mask
  dispatch(const uint32_t* __restrict filter_data,
           const Tv& bucket_idx1,
           const Tv& bucket_idx2,
           const Tv& search_tag) {
    using namespace dtl;
    constexpr std::size_t n = dtl::vector_length<Tv>::value;
    using mask_t = typename vec<uint32_t, n>::mask;
    constexpr std::size_t bits_per_tag = 16;
    constexpr std::size_t tags_per_bucket = 1;
    constexpr uint32_t tag_mask = static_cast<uint32_t>((1ull << (bits_per_tag * tags_per_bucket)) - 1);

    // Load tags using 32-bit gathers.
    const auto* d = reinterpret_cast<const uint32_t*>(filter_data);

    const auto u32_word_idx1 = bucket_idx1 >> 1; // two buckets per word
    const auto word1 = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx1);
    const auto in_word_idx1 = (bucket_idx1 & (2 - 1)) << 4; // offset is either 0 or 16 bit
    const auto tag1 = (word1 >> in_word_idx1) & tag_mask;

    const auto u32_word_idx2 = bucket_idx2 >> 1;
    const auto word2 = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx2);
    const auto in_word_idx2 = (bucket_idx2 & (2 - 1)) << 4;
    const auto tag2 = (word2 >> in_word_idx2) & tag_mask;

    mask_t found_mask = (tag1 == search_tag) | (tag2 == search_tag);

    return found_mask;
  }

};


template<>
struct find_tag_in_buckets_simd<16, 2> {

  static constexpr u1 vectorized = true;

  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __unroll_loops__ __host__ static
  typename dtl::vec<uint32_t, dtl::vector_length<Tv>::value>::mask
  dispatch(const uint32_t* __restrict filter_data,
           const Tv& bucket_idx1,
           const Tv& bucket_idx2,
           const Tv& search_tag) {
    using namespace dtl;
    constexpr std::size_t n = dtl::vector_length<Tv>::value;
    using vec_t = vec<uint32_t, n>;
    using mask_t = typename vec<uint32_t, n>::mask;
    const vec_t search_tag_u16 = search_tag * 0x00010001;
    mask_t found_mask;

    // Load tags using 32-bit gathers.
    const auto* d = reinterpret_cast<const uint32_t*>(filter_data);

    // one bucket per word
    const auto tag1 = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, bucket_idx1);
    const auto tag2 = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, bucket_idx2);

    const auto t1 = reinterpret_cast<const r512*>(&tag1.data);
    const auto t2 = reinterpret_cast<const r512*>(&tag2.data);
    const auto st_u16 = reinterpret_cast<const r512*>(&search_tag_u16.data);
    auto fm = reinterpret_cast<__mmask16*>(&found_mask.data);

#if !defined(__AVX512BW__)
    const r512 msb_off = {.i = _mm512_set1_epi32( uint32_t(-1) >> 1 )};
    const auto zero = _mm512_setzero_si512();
#endif

    using vector_type = decltype(tag1);
    u64 vector_cnt = sizeof(vector_type) / sizeof(r512);
    for (std::size_t i = 0; i < vector_cnt; i++) {
#if defined(__AVX512BW__)
      const __mmask32 found1_m32 = _mm512_cmpeq_epi16_mask(t1[i].i, st_u16[i].i);
      const __mmask32 found2_m32 = _mm512_cmpeq_epi16_mask(t2[i].i, st_u16[i].i);

      const __mmask16 found1_m = _pext_u32(found1_m32 | (found1_m32 >> 1), 0x55555555u);
      const __mmask16 found2_m = _pext_u32(found2_m32 | (found2_m32 >> 1), 0x55555555u);
#else
      // fall back to AVX2 instructions for 16-bit compare
      r512 t1_m;
      t1_m.r256[0].i = _mm256_cmpeq_epi16(t1[i].r256[0].i, st_u16[i].r256[0].i);
      t1_m.r256[1].i = _mm256_cmpeq_epi16(t1[i].r256[1].i, st_u16[i].r256[1].i);

      r512 t2_m;
      t2_m.r256[0].i = _mm256_cmpeq_epi16(t2[i].r256[0].i, st_u16[i].r256[0].i);
      t2_m.r256[1].i = _mm256_cmpeq_epi16(t2[i].r256[1].i, st_u16[i].r256[1].i);

      const __mmask16 found1_m = _mm512_cmpgt_epi32_mask(_mm512_and_si512(t1_m.i, msb_off.i), zero); // turn off msb of mask for signed comparison
      const __mmask16 found2_m = _mm512_cmpgt_epi32_mask(_mm512_and_si512(t2_m.i, msb_off.i), zero);
#endif
      fm[i] = found1_m | found2_m;
    }
    return found_mask;
  }

};


template<>
struct find_tag_in_buckets_simd<16, 4> {

  static constexpr u1 vectorized = true;

  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __unroll_loops__ __host__ static
  typename dtl::vec<uint32_t, dtl::vector_length<Tv>::value>::mask
  dispatch(const uint32_t* __restrict filter_data,
           const Tv& bucket_idx1,
           const Tv& bucket_idx2,
           const Tv& search_tag) {
    using namespace dtl;
    constexpr std::size_t n = dtl::vector_length<Tv>::value;
    using vec_t = vec<uint32_t, n>;
    using mask_t = typename vec<uint32_t, n>::mask;
    const vec_t search_tag_u16 = search_tag * 0x00010001;
    mask_t found_mask;

    // Load tags using 32-bit gathers.
    const auto* d = reinterpret_cast<const uint32_t*>(filter_data);

    // One bucket = two words
    const auto u32_word_idx1a = bucket_idx1 << 1;
    const auto tag1a = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx1a);
    const auto u32_word_idx1b = u32_word_idx1a + 1;
    const auto tag1b = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx1b);

    const auto u32_word_idx2a = bucket_idx2 << 1;
    const auto tag2a = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx2a);
    const auto u32_word_idx2b = u32_word_idx2a + 1;
    const auto tag2b = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx2b);

    const auto t1a = reinterpret_cast<const r512*>(&tag1a.data);
    const auto t1b = reinterpret_cast<const r512*>(&tag1b.data);
    const auto t2a = reinterpret_cast<const r512*>(&tag2a.data);
    const auto t2b = reinterpret_cast<const r512*>(&tag2b.data);
    const auto st_u16 = reinterpret_cast<const r512*>(&search_tag_u16.data);
    auto fm = reinterpret_cast<__mmask16*>(&found_mask.data);

#if !defined(__AVX512BW__)
    const r512 msb_off = {.i = _mm512_set1_epi32( uint32_t(-1) >> 1 )};
    const auto zero = _mm512_setzero_si512();
#endif

    using vector_type = decltype(tag1a);
    u64 vector_cnt = sizeof(vector_type) / sizeof(r512);
    for (std::size_t i = 0; i < vector_cnt; i++) {
#if defined(__AVX512BW__)
      const __mmask32 found1a_m32 = _mm512_cmpeq_epi16_mask(t1a[i].i, st_u16[i].i);
      const __mmask32 found1b_m32 = _mm512_cmpeq_epi16_mask(t1b[i].i, st_u16[i].i);
      const __mmask32 found1_m32 = found1a_m32 | found1b_m32;
      const __mmask32 found2a_m32 = _mm512_cmpeq_epi16_mask(t2a[i].i, st_u16[i].i);
      const __mmask32 found2b_m32 = _mm512_cmpeq_epi16_mask(t2b[i].i, st_u16[i].i);
      const __mmask32 found2_m32 = found2a_m32 | found2b_m32;

      const __mmask16 found1_m = _pext_u32(found1_m32 | (found1_m32 >> 1), 0x55555555u);
      const __mmask16 found2_m = _pext_u32(found2_m32 | (found2_m32 >> 1), 0x55555555u);
#else
      // fall back to AVX2 instructions for 16-bit compare
      r512 t1a_m;
      t1a_m.r256[0].i = _mm256_cmpeq_epi16(t1a[i].r256[0].i, st_u16[i].r256[0].i);
      t1a_m.r256[1].i = _mm256_cmpeq_epi16(t1a[i].r256[1].i, st_u16[i].r256[1].i);
      r512 t1b_m;
      t1b_m.r256[0].i = _mm256_cmpeq_epi16(t1b[i].r256[0].i, st_u16[i].r256[0].i);
      t1b_m.r256[1].i = _mm256_cmpeq_epi16(t1b[i].r256[1].i, st_u16[i].r256[1].i);

      r512 t2a_m;
      t2a_m.r256[0].i = _mm256_cmpeq_epi16(t2a[i].r256[0].i, st_u16[i].r256[0].i);
      t2a_m.r256[1].i = _mm256_cmpeq_epi16(t2a[i].r256[1].i, st_u16[i].r256[1].i);
      r512 t2b_m;
      t2b_m.r256[0].i = _mm256_cmpeq_epi16(t2b[i].r256[0].i, st_u16[i].r256[0].i);
      t2b_m.r256[1].i = _mm256_cmpeq_epi16(t2b[i].r256[1].i, st_u16[i].r256[1].i);

      const __mmask16 found1a_m = _mm512_cmpgt_epi32_mask(_mm512_and_si512(t1a_m.i, msb_off.i), zero); // turn off msb of mask for signed comparison
      const __mmask16 found1b_m = _mm512_cmpgt_epi32_mask(_mm512_and_si512(t1b_m.i, msb_off.i), zero);
      const __mmask16 found1_m = found1a_m | found1b_m;
      const __mmask16 found2a_m = _mm512_cmpgt_epi32_mask(_mm512_and_si512(t2a_m.i, msb_off.i), zero);
      const __mmask16 found2b_m = _mm512_cmpgt_epi32_mask(_mm512_and_si512(t2b_m.i, msb_off.i), zero);
      const __mmask16 found2_m = found2a_m | found2b_m;
#endif
      fm[i] = found1_m | found2_m;
    }
    return found_mask;
  }

};


template<>
struct find_tag_in_buckets_simd<32, 1> {

  static constexpr u1 vectorized = true;

  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __unroll_loops__ __host__ static
  typename dtl::vec<uint32_t, dtl::vector_length<Tv>::value>::mask
  dispatch(const uint32_t* __restrict filter_data,
           const Tv& bucket_idx1,
           const Tv& bucket_idx2,
           const Tv& search_tag) {
    using namespace dtl;
    constexpr std::size_t n = dtl::vector_length<Tv>::value;

    // Load tags using 32-bit gathers.
    const auto* d = reinterpret_cast<const uint32_t*>(filter_data);

    // one buckets per word
    const auto tag1 = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, bucket_idx1);
    const auto tag2 = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, bucket_idx2);

    return (tag1 == search_tag) | (tag2 == search_tag);
  }

};


template<>
struct find_tag_in_buckets_simd<32, 2> {

  static constexpr u1 vectorized = true;

  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __unroll_loops__ __host__ static
  typename dtl::vec<uint32_t, dtl::vector_length<Tv>::value>::mask
  dispatch(const uint32_t* __restrict filter_data,
           const Tv& bucket_idx1,
           const Tv& bucket_idx2,
           const Tv& search_tag) {
    using namespace dtl;
    constexpr std::size_t n = dtl::vector_length<Tv>::value;
    using mask_t = typename vec<uint32_t, n>::mask;
    mask_t found_mask;

    // Load tags using 32-bit gathers.
    const auto* d = reinterpret_cast<const uint32_t*>(filter_data);

    // One bucket = two words
    const auto u32_word_idx1a = bucket_idx1 << 1;
    const auto tag1a = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx1a);
    const auto u32_word_idx1b = u32_word_idx1a + 1;
    const auto tag1b = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx1b);

    const auto u32_word_idx2a = bucket_idx2 << 1;
    const auto tag2a = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx2a);
    const auto u32_word_idx2b = u32_word_idx2a + 1;
    const auto tag2b = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx2b);

    return (tag1a == search_tag) | (tag1b == search_tag) | (tag2a == search_tag) | (tag2b == search_tag);
  }

};


template<>
struct find_tag_in_buckets_simd<32, 4> {

  static constexpr u1 vectorized = true;

  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __unroll_loops__ __host__ static
  typename dtl::vec<uint32_t, dtl::vector_length<Tv>::value>::mask
  dispatch(const uint32_t* __restrict filter_data,
           const Tv& bucket_idx1,
           const Tv& bucket_idx2,
           const Tv& search_tag) {
    using namespace dtl;
    constexpr std::size_t n = dtl::vector_length<Tv>::value;
    using mask_t = typename vec<uint32_t, n>::mask;
    mask_t found_mask;

    // Load tags using 32-bit gathers.
    const auto* d = reinterpret_cast<const uint32_t*>(filter_data);

    // One bucket = four words
    {
      const auto u32_word_idx1a = bucket_idx1 << 2;
      const auto tag1a = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx1a);
      const auto u32_word_idx1b = u32_word_idx1a + 1;
      const auto tag1b = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx1b);

      const auto u32_word_idx2a = bucket_idx2 << 2;
      const auto tag2a = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx2a);
      const auto u32_word_idx2b = u32_word_idx2a + 1;
      const auto tag2b = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx2b);

      found_mask = (tag1a == search_tag) | (tag1b == search_tag) | (tag2a == search_tag) | (tag2b == search_tag);
    }

    {
      // hopefully both buckets are now cached
      const auto u32_word_idx1c = (bucket_idx1 << 2) + 2;
      const auto tag1c = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx1c);
      const auto u32_word_idx1d = u32_word_idx1c + 1;
      const auto tag1d = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx1d);

      const auto u32_word_idx2c = (bucket_idx2 << 2) + 2;
      const auto tag2c = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx2c);
      const auto u32_word_idx2d = u32_word_idx2c + 1;
      const auto tag2d = dtl::internal::vector_gather<uint32_t, uint32_t, n>::gather(d, u32_word_idx2d);

      found_mask |= (tag1c == search_tag) | (tag1d == search_tag) | (tag2c == search_tag) | (tag2d == search_tag);
    }
    return found_mask;
  }

};

} // namespace internal
} // namespace cuckoofilter
} // namespace dtl
