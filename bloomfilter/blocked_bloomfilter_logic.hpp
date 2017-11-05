#pragma once

#include <bitset>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>
#include <dtl/math.hpp>

#include "immintrin.h"
#include "dtl/bloomfilter/bloomfilter_h1.hpp"

#include <boost/integer/static_min_max.hpp>


namespace dtl {


//===----------------------------------------------------------------------===//
// Recursive template to compute a search mask with k bits set.
//===----------------------------------------------------------------------===//
template<
    typename key_t,               // the key type
    typename word_t,              // the word type
    u32 s,                        // the numbers of sectors (must be a power of two)
    u32 k,                        // the number of bits to set
    template<typename Ty, u32 i> class hasher,      // the hash function family to use
    typename hash_value_t,        // the hash value type

    u32 hash_fn_idx,              // current hash function index (used for recursion)
    u32 remaining_hash_bit_cnt,   // the number of remaining hash bits (used for recursion)
    u32 remaining_k_cnt           // current k (used for recursion)
>
struct word_block {

  //===----------------------------------------------------------------------===//
  // Static part
  static_assert(dtl::is_power_of_two(s), "Parameter 's' must be a power of two.");

  static constexpr u32 word_bitlength = sizeof(word_t) * 8;
  static constexpr u32 word_bitlength_mask = word_bitlength - 1;

  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;

  static constexpr u32 sector_bitlength = word_bitlength / s;
  static constexpr u32 sector_bitlength_log2 = dtl::ct::log_2_u32<sector_bitlength>::value;
  static constexpr word_t sector_mask() { return static_cast<word_t>(sector_bitlength) - 1; }

  static constexpr u32 hash_bits_per_k = sector_bitlength_log2;
  static constexpr u32 k_cnt_per_hash_value = ((sizeof(hash_value_t) * 8) / hash_bits_per_k) ; // consider -1 to respect hash fn weakness in the low order bits
  static constexpr u32 k_cnt_per_sector = k / s;

  static constexpr u32 current_k = k - remaining_k_cnt;

  static constexpr u1 rehash = remaining_hash_bit_cnt < hash_bits_per_k;
  //===----------------------------------------------------------------------===//


  __forceinline__ __unroll_loops__ __host__ __device__
  static void
  which_bits(const key_t key, hash_value_t& hash_val, word_t& word) noexcept {

    hash_val = rehash ? hasher<key_t, hash_fn_idx>::hash(key) : hash_val;
    const hash_value_t remaining_hash_bit_cnt_after_rehash = rehash ? hash_value_bitlength : remaining_hash_bit_cnt;

    // Set one bit in the given word; rehash if necessary
    constexpr u32 sector_idx = current_k / k_cnt_per_sector;
    constexpr u32 shift = remaining_hash_bit_cnt_after_rehash - hash_bits_per_k;
    $u32 bit_idx = ((hash_val >> shift) & sector_mask()) + (sector_idx * sector_bitlength);
    word |= word_t(1) << bit_idx;

    // Recurse
    word_block<key_t, word_t, s, k,
        hasher, hash_value_t,
        (rehash ? hash_fn_idx + 1 : hash_fn_idx), // increment the hash function index
        (rehash ? hash_value_bitlength - hash_bits_per_k : remaining_hash_bit_cnt - hash_bits_per_k), // the number of remaining hash bits
        remaining_k_cnt - 1> // decrement the remaining k counter
        ::which_bits(key, hash_val, word);
  }


  template<u64 n>
  __forceinline__ __unroll_loops__ __host__ __device__
  static void
  which_bits(const vec<key_t, n>& keys,
             vec<hash_value_t, n>& hash_vals,
             vec<word_t, n>& words) noexcept {

    // Typedef vector types
    using key_vt = vec<key_t, n>;
    using hash_value_vt = vec<hash_value_t, n>;
    using word_vt = vec<word_t, n>;

    hash_vals = rehash ? hasher<key_vt, hash_fn_idx>::hash(keys) : hash_vals;
    const hash_value_t remaining_hash_bit_cnt_after_rehash = rehash ? hash_value_bitlength : remaining_hash_bit_cnt;

    // Set one bit in the given word; rehash if necessary
    constexpr u32 sector_idx = current_k / k_cnt_per_sector;
    constexpr u32 shift = remaining_hash_bit_cnt_after_rehash - hash_bits_per_k;
    hash_value_vt bit_idxs = ((hash_vals >> shift) & sector_mask()) + (sector_idx * sector_bitlength);
    words |= word_vt(1) << internal::vector_convert<hash_value_t, word_t, n>::convert(bit_idxs);

    // Recurse
    word_block<key_t, word_t, s, k,
        hasher, hash_value_t,
        (rehash ? hash_fn_idx + 1 : hash_fn_idx), // increment the hash function index
        (rehash ? hash_value_bitlength - hash_bits_per_k : remaining_hash_bit_cnt - hash_bits_per_k), // the number of remaining hash bits
        remaining_k_cnt - 1> // decrement the remaining k counter
        ::which_bits(keys, hash_vals, words);
  }


  // The number of required hash functions.
  static constexpr u32 hash_fn_cnt =
  word_block<key_t, word_t, s, k,
      hasher, hash_value_t,
      (rehash ? hash_fn_idx + 1 : hash_fn_idx), // increment the hash function index
      (rehash ? hash_value_bitlength - hash_bits_per_k : remaining_hash_bit_cnt - hash_bits_per_k), // the number of remaining hash bits
      remaining_k_cnt - 1> // decrement the remaining k counter
      ::hash_fn_cnt;

  // The number of required hash bits.
  static constexpr u32 remaining_hash_bits =
  word_block<key_t, word_t, s, k,
      hasher, hash_value_t,
      (rehash ? hash_fn_idx + 1 : hash_fn_idx), // increment the hash function index
      (rehash ? hash_value_bitlength - hash_bits_per_k : remaining_hash_bit_cnt - hash_bits_per_k), // the number of remaining hash bits
      remaining_k_cnt - 1> // decrement the remaining k counter
      ::remaining_hash_bits;

};


template<
    typename key_t,               // the key type
    typename word_t,              // the word type
    u32 s,                        // the numbers of sectors (must be a power of two)
    u32 k,                        // the number of bits to set
    template<typename Ty, u32 i> class hasher,      // the hash function family to use
    typename hash_value_t,        // the hash value type to use

    u32 hash_fn_idx,              // current hash function index (used for recursion)
    u32 remaining_hash_bit_cnt    // the number of remaining hash bits (used for recursion)
>
struct word_block<key_t, word_t, s, k, hasher, hash_value_t, hash_fn_idx, remaining_hash_bit_cnt, 0 /* no remaining k's */> {

  __forceinline__ __unroll_loops__ __host__ __device__
  static word_t
  which_bits(const key_t key, hash_value_t& hash_value, word_t& word) noexcept {
    // End of recursion
  }


  template<u64 n>
  __forceinline__ __unroll_loops__ __host__ __device__
  static void
  which_bits(const vec<key_t, n>& keys,
             vec<hash_value_t, n>& hash_values,
             vec<word_t, n>& words) noexcept {
    // End of recursion
  }

  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;

  static constexpr u32 hash_fn_cnt = hash_fn_idx;

  static constexpr u32 remaining_hash_bits = remaining_hash_bit_cnt;

};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Recursive template to work with multi-word blocks.
//===----------------------------------------------------------------------===//
template<
    typename key_t,               // the key type
    typename word_t,              // the word type
    u32 word_cnt,                 // the number of words per block
    u32 s,                        // the numbers of sectors (must be a power of two and greater or equal to word_cnt))
    u32 k,                        // the number of bits to set/test
    template<typename Ty, u32 i> class hasher,      // the hash function family to use
    typename hash_value_t,        // the hash value type to use

    u32 hash_fn_idx,              // current hash function index (used for recursion)
    u32 remaining_hash_bit_cnt,   // the number of remaining hash bits (used for recursion)
    u32 remaining_word_cnt        // the remaining number of words to process in the block (used for recursion)
>
struct multiword_block {

  //===----------------------------------------------------------------------===//
  // Static part
  static constexpr u32 word_cnt_log2 = dtl::ct::log_2<word_cnt>::value;
  static_assert(dtl::is_power_of_two(word_cnt), "Parameter 'word_cnt' must be a power of two.");
  static_assert(k >= word_cnt, "Parameter 'k' must be greater or equal to 'word_cnt'.");
  static_assert(k % word_cnt == 0, "Parameter 'k' must be dividable by 'word_cnt'.");

  static constexpr u32 sector_cnt = s;
  static_assert(dtl::is_power_of_two(sector_cnt), "Parameter 'sector_cnt' must be a power of two.");

  static constexpr u32 sector_cnt_per_word = s / word_cnt;
  static_assert(sector_cnt_per_word > 0, "The number of sectors must be at least 'word_cnt'.");

  static constexpr u32 k_cnt_per_word = k / word_cnt;
  static constexpr u32 hash_fn_per_word = word_block<key_t, word_t, s, k, hasher, hash_value_t, 0, 0, k_cnt_per_word>::hash_fn_cnt;

  static constexpr u32 current_word_idx() { return word_cnt - remaining_word_cnt; }

  // for compatibility with block addressing logic // FIXME
  static constexpr u32 block_bitlength = word_cnt * sizeof(word_t) * 8;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Insert
  //===----------------------------------------------------------------------===//
  __forceinline__ __unroll_loops__
  static void
  insert(const key_t key, word_t* __restrict block_ptr) noexcept {

    hash_value_t hash_val = 0;

    // Call the recursive function
    static constexpr u32 remaining_hash_bits = 0;
    multiword_block<key_t, word_t, word_cnt, sector_cnt, k,
        hasher, hash_value_t, hash_fn_idx + hash_fn_per_word, remaining_hash_bits, remaining_word_cnt>
        ::insert(key, hash_val, block_ptr);
  }


  __forceinline__ __unroll_loops__
  static void
  insert(const key_t key, hash_value_t& hash_val, word_t* __restrict block_ptr) noexcept {
    // Load the word of interest
    word_t word = block_ptr[current_word_idx()];

    word_t bit_mask = 0;

    using word_block_t =
      word_block<key_t, word_t, sector_cnt_per_word, k_cnt_per_word,
        hasher, hash_value_t, hash_fn_idx, remaining_hash_bit_cnt,
        k_cnt_per_word>;
    word_block_t::which_bits(key, hash_val, bit_mask);

    // Update the bit vector
    word |= bit_mask;
    block_ptr[current_word_idx()] = word;

    // Process remaining words recursively, if any
    multiword_block<key_t, word_t, word_cnt, sector_cnt, k,
        hasher, hash_value_t, hash_fn_idx + hash_fn_per_word, word_block_t::remaining_hash_bits, remaining_word_cnt - 1>
        ::insert(key, hash_val, block_ptr);
  }


  //===----------------------------------------------------------------------===//
  // Contains
  //===----------------------------------------------------------------------===//
  __forceinline__ __unroll_loops__
  static u1
  contains(const key_t key, const word_t* __restrict block_ptr) noexcept {

    hash_value_t hash_val = 0;

    // Call the recursive function
    static constexpr u32 remaining_hash_bits = 0;
    return multiword_block<key_t, word_t, word_cnt, sector_cnt, k,
        hasher, hash_value_t, hash_fn_idx + hash_fn_per_word, remaining_hash_bits, remaining_word_cnt>
        ::contains(key, hash_val, block_ptr, false);
  }


  __forceinline__ __unroll_loops__
  static u1
  contains(const key_t key, hash_value_t& hash_val, const word_t* __restrict block_ptr, u1 is_contained) noexcept {
    // Load the word of interest
    word_t word = block_ptr[current_word_idx()];

    word_t bit_mask = 0;

    // Compute the search mask
    static constexpr u32 k_cnt_per_word = k / word_cnt;
    using word_block_t =
      word_block<key_t, word_t, sector_cnt_per_word, k_cnt_per_word,
        hasher, hash_value_t, hash_fn_idx, remaining_hash_bit_cnt,
        k_cnt_per_word>;
    word_block_t::which_bits(key, hash_val, bit_mask);

    // Update the bit vector
    u1 found = (word & bit_mask) == bit_mask;

    // Process remaining words recursively, if any
    return multiword_block<key_t, word_t, word_cnt, sector_cnt, k,
        hasher, hash_value_t, hash_fn_idx + hash_fn_per_word, word_block_t::remaining_hash_bits, remaining_word_cnt - 1>
        ::contains(key, hash_val, block_ptr, is_contained | found);
  }


  template<u64 n>
  __forceinline__ __unroll_loops__
  static auto
  contains(const vec<key_t,n>& keys,
           const word_t* __restrict bitvector_base_address,
           const vec<key_t,n>& block_start_word_idxs) noexcept {

    vec<hash_value_t, n> hash_vals(0);
    typename vec<word_t,n>::mask is_contained_mask(0);

    // Call recursive function
    static constexpr u32 remaining_hash_bits = 0;
    return multiword_block<key_t, word_t, word_cnt, sector_cnt, k,
        hasher, hash_value_t, hash_fn_idx + hash_fn_per_word, remaining_hash_bits, remaining_word_cnt>
        ::contains(keys, hash_vals, bitvector_base_address, block_start_word_idxs, is_contained_mask);
  }


  template<u64 n>
  __forceinline__ __unroll_loops__
  static auto
  contains(const vec<key_t,n>& keys,
           vec<hash_value_t,n>& hash_vals,
           const word_t* __restrict bitvector_base_address,
           const vec<key_t,n>& block_start_word_idxs,
           const typename vec<word_t,n>::mask is_contained_mask) noexcept {

    // Typedef the vector types
    using key_vt = vec<key_t, n>;
    using hash_value_vt = vec<hash_value_t, n>;
    using word_vt = vec<word_t, n>;

    // Load the words of interest
    auto word_idxs = block_start_word_idxs + current_word_idx();
    const word_vt words = internal::vector_gather<word_t, hash_value_t, n>::gather(bitvector_base_address, word_idxs);

    // Compute the search mask
    word_vt bit_masks = 0;
    static constexpr u32 k_cnt_per_word = k / word_cnt;
    using word_block_t =
      word_block<key_t, word_t, sector_cnt_per_word, k_cnt_per_word,
        hasher, hash_value_t, hash_fn_idx, remaining_hash_bit_cnt,
        k_cnt_per_word>;
    word_block_t::which_bits(keys, hash_vals, bit_masks);

    // Update the bit vector
    auto found = (words & bit_masks) == bit_masks;

    // Process remaining words recursively, if any
    return multiword_block<key_t, word_t, word_cnt, sector_cnt, k,
        hasher, hash_value_t, hash_fn_idx + hash_fn_per_word, word_block_t::remaining_hash_bits, remaining_word_cnt - 1>
        ::contains(keys, hash_vals, bitvector_base_address, block_start_word_idxs, is_contained_mask | found);
  }

};


template<
    typename key_t,               // the word type
    typename word_t,              // the word type
    u32 word_cnt,                 // the number of words per block
    u32 s,                        // the numbers of sectors (must be a power of two and greater or equal to word_cnt))
    u32 k,                        // the number of bits to set/test
    template<typename Ty, u32 i> class hasher,      // the hash function family to use
    typename hash_value_t,        // the hash value type to use

    u32 hash_fn_idx,              // current hash function index (used for recursion)
    u32 remaining_hash_bit_cnt    // the number of remaining hash bits (used for recursion)
>
struct multiword_block<key_t, word_t, word_cnt, s, k,
    hasher, hash_value_t, hash_fn_idx, remaining_hash_bit_cnt,
    0 /* no more words remaining */> {

  //===----------------------------------------------------------------------===//
  // Insert
  //===----------------------------------------------------------------------===//
  __forceinline__ __unroll_loops__
  static void
  insert(const key_t key, hash_value_t& hash_val, word_t* __restrict block_ptr) noexcept {
    // End of recursion
  }


  //===----------------------------------------------------------------------===//
  // Contains
  //===----------------------------------------------------------------------===//
  __forceinline__ __unroll_loops__
  static u1
  contains(const key_t key, hash_value_t& hash_val, const word_t* __restrict block_ptr, u1 is_contained) noexcept {
    // End of recursion
    return is_contained;
  }


  //===----------------------------------------------------------------------===//
  // Contains (SIMD)
  //===----------------------------------------------------------------------===//
  template<u64 n>
  __forceinline__ __unroll_loops__
  static auto
  contains(const vec<key_t,n>& keys,
           vec<hash_value_t,n>& hash_vals,
           const word_t* __restrict bitvector_base_address,
           const vec<key_t,n>& block_idxs,
           const typename vec<word_t,n>::mask is_contained_mask) noexcept {
    // End of recursion
    return is_contained_mask;
  }

};
//===----------------------------------------------------------------------===//



//===----------------------------------------------------------------------===//
// A high-performance blocked Bloom filter template.
//===----------------------------------------------------------------------===//
namespace internal {

//===----------------------------------------------------------------------===//
// Batch-wise Contains
//===----------------------------------------------------------------------===//
template<
    typename filter_t,
    u64 vector_len
>
struct dispatch {

  __forceinline__
  static $u64
  batch_contains(const filter_t& filter,
                 const typename filter_t::key_t* keys, u32 key_cnt,
                 $u32* match_positions, u32 match_offset) {
    // Typedefs
    using key_t = typename filter_t::key_t;
    using vec_t = vec<key_t, vector_len>;
    using mask_t = typename vec<key_t, vector_len>::mask;

    const key_t* reader = keys;
    $u32* match_writer = match_positions;

    // Determine the number of keys that need to be probed sequentially, due to alignment
    u64 required_alignment_bytes = vec_t::byte_alignment;
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
    // Process the aligned keys vectorized
    u64 aligned_key_cnt = ((key_cnt - unaligned_key_cnt) / vector_len) * vector_len;
    for (; read_pos < (unaligned_key_cnt + aligned_key_cnt); read_pos += vector_len) {
      const auto mask = filter.template contains_vec<vector_len>(*reinterpret_cast<const vec_t*>(reader));
      u64 match_cnt = mask.to_positions(match_writer, read_pos + match_offset);
      match_writer += match_cnt;
      reader += vector_len;
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

};


//===----------------------------------------------------------------------===//
// Batch-wise Contains (no SIMD)
//===----------------------------------------------------------------------===//
template<
    typename filter_t
>
struct dispatch<filter_t, 0> {

  __forceinline__
  static $u64
  batch_contains(const filter_t& filter,
                 const typename filter_t::key_t* keys, u32 key_cnt,
                 $u32* match_positions, u32 match_offset) {
    $u32* match_writer = match_positions;
    $u32 i = 0;
    for (; i < key_cnt; i += 4) {
      u1 is_match_0 = filter.contains(keys[i]);
      u1 is_match_1 = filter.contains(keys[i + 1]);
      u1 is_match_2 = filter.contains(keys[i + 2]);
      u1 is_match_3 = filter.contains(keys[i + 3]);
      *match_writer = i + match_offset;
      match_writer += is_match_0;
      *match_writer = (i + 1) + match_offset;
      match_writer += is_match_1;
      *match_writer = (i + 2) + match_offset;
      match_writer += is_match_2;
      *match_writer = (i + 3) + match_offset;
      match_writer += is_match_3;
    }
    for (; i < key_cnt; i++) {
      u1 is_match = filter.contains(keys[i]);
      *match_writer = i + match_offset;
      match_writer += is_match;
    }
    return match_writer - match_positions;
  }

};


} // namespace internal


template<
    typename Tk,                  // the key type
    template<typename Ty, u32 i> class Hasher,      // the hash function family to use
    typename Tw,                  // the word type to use for the bitset
    u32 Wc = 2,                   // the number of words per block
    u32 s = Wc,                   // the word type to use for the bitset
    u32 K = 8,                    // the number of hash functions to use
    dtl::block_addressing block_addressing = dtl::block_addressing::POWER_OF_TWO,
    typename Alloc = std::allocator<Tw>
>
struct blocked_bloomfilter_logic {

  using key_t = typename std::remove_cv<Tk>::type;
  using word_t = typename std::remove_cv<Tw>::type;
  static_assert(std::is_integral<key_t>::value, "The key type must be an integral type.");
  static_assert(std::is_integral<word_t>::value, "The word type must be an integral type.");

  using allocator_t = Alloc;
  using size_t = $u64;

  static constexpr u32 word_cnt = Wc;
  static constexpr u32 word_cnt_log2 = dtl::ct::log_2<Wc>::value;
  static_assert(dtl::is_power_of_two(Wc), "Parameter 'Wc' must be a power of two.");

  static constexpr u32 sector_cnt = s;

  static constexpr u32 word_bitlength = sizeof(word_t) * 8;
  static constexpr u32 word_bitlength_log2 = dtl::ct::log_2_u32<word_bitlength>::value;
  static constexpr u32 word_bitlength_mask = word_bitlength - 1;

  static constexpr u32 block_bitlength = sizeof(word_t) * 8 * word_cnt;
  static constexpr u32 block_bitlength_log2 = dtl::ct::log_2_u32<block_bitlength>::value;
  static constexpr u32 block_bitlength_mask = block_bitlength - 1;


  // Inspect the given hash function
  using hash_value_t = decltype(Hasher<key_t, 0>::hash(42)); // TODO find out why NVCC complains
  static_assert(std::is_integral<hash_value_t>::value, "Hash function must return an integral type.");
  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;


  // The number of hash functions to use.
  static constexpr u32 k = K;

  static constexpr u64 max_m = 256ull * 1024 * 1024 * 8; // FIXME

  using block_t = multiword_block<key_t, word_t, word_cnt, s, k,
      Hasher, hash_value_t,
      /* recursive counters */
      0, 1, word_cnt>;
  using addr_t = bloomfilter_addressing_logic<block_addressing, hash_value_t, block_t>;

  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  const addr_t addr;
  const allocator_t allocator;
  std::vector<word_t, allocator_t> word_array;
  //===----------------------------------------------------------------------===//


  /// C'tor
  explicit
  blocked_bloomfilter_logic(const size_t length,
                        const allocator_t allocator = allocator_t())
      : allocator(allocator),
        addr(length),
        word_array(addr.length() / word_bitlength, 0, this->allocator) {
    if (addr.length() > max_m) throw std::invalid_argument("Length must not exceed 'max_m'.");
  }

  /// Copy c'tor
  blocked_bloomfilter_logic(const blocked_bloomfilter_logic&) = default;
  blocked_bloomfilter_logic(const blocked_bloomfilter_logic& other,
                 const allocator_t& allocator)
      : addr(other.addr),
        allocator(allocator),
        word_array(other.word_array.begin(), other.word_array.end(), this->allocator) { }


  ~blocked_bloomfilter_logic() {
    word_array.clear();
    word_array.shrink_to_fit();
  }


  /// Creates a copy of the bloomfilter (allows to specify a different allocator)
  template<typename AllocOfCopy = Alloc>
  blocked_bloomfilter_logic<Tk, Hasher, Tw, Wc, s, K, block_addressing, AllocOfCopy>
  make_copy(AllocOfCopy alloc = AllocOfCopy()) const {
    using return_t = blocked_bloomfilter_logic<Tk, Hasher, Tw, Wc, s, K, block_addressing, AllocOfCopy>;
    return_t bf_copy(addr.length(), alloc);
    bf_copy.word_array.clear();
    bf_copy.word_array.insert(bf_copy.word_array.begin(), word_array.begin(), word_array.end());
    return bf_copy;
  }


  /// Creates a copy of the bloomfilter (allows to specify a different allocator)
  template<typename AllocOfCopy = Alloc>
  blocked_bloomfilter_logic<Tk, Hasher, Tw, Wc, s, K, block_addressing, AllocOfCopy>*
  make_heap_copy(AllocOfCopy alloc = AllocOfCopy()) const {
    using bf_t = blocked_bloomfilter_logic<Tk, Hasher, Tw, Wc, s, K, block_addressing, AllocOfCopy>;
    bf_t* bf_copy = new bf_t(addr.length(), alloc);
    bf_copy->word_array.clear();
    bf_copy->word_array.insert(bf_copy->word_array.begin(), word_array.begin(), word_array.end());
    return bf_copy;
  }



  //===----------------------------------------------------------------------===//
  // Insert
  //===----------------------------------------------------------------------===//
  __forceinline__ __host__
  void
  insert(const key_t key) noexcept {
    const hash_value_t block_addressing_hash_val = Hasher<const key_t, 0>::hash(key);
    const hash_value_t block_idx = addr.get_block_idx(block_addressing_hash_val);
    const hash_value_t bitvector_word_idx = block_idx << word_cnt_log2;

    auto block_ptr = &word_array[bitvector_word_idx];

    multiword_block<key_t, word_t, word_cnt, s, k,
        Hasher, hash_value_t,
        /* recursive counter */
        1, 0, word_cnt>
        ::insert(key, block_ptr);
  }


  //===----------------------------------------------------------------------===//
  // Contains
  //===----------------------------------------------------------------------===//
  __forceinline__ __host__ __device__
  u1
  contains(const key_t key) const noexcept {
    const hash_value_t block_addressing_hash_val = Hasher<const key_t, 0>::hash(key);
    const hash_value_t block_idx = addr.get_block_idx(block_addressing_hash_val);
    const hash_value_t bitvector_word_idx = block_idx << word_cnt_log2;

    const auto block_ptr = &word_array[bitvector_word_idx];

    u1 found = multiword_block<key_t, word_t, word_cnt, s, k,
        Hasher, hash_value_t,
        /* recursive counter */
        1, 0, word_cnt>
        ::contains(key, block_ptr);
    return found;
  }


  //===----------------------------------------------------------------------===//
  // Contains (SIMD)
  //===----------------------------------------------------------------------===//
  template<u64 n> // the vector length
  __forceinline__ __host__
  typename vec<word_t, n>::mask
  contains_vec(const vec<key_t, n>& keys) const noexcept {
    using key_vt = vec<key_t, n>;
    using hash_value_vt = vec<hash_value_t, n>;
    using word_vt = vec<word_t, n>;

    const hash_value_vt block_addressing_hash_vals = Hasher<key_vt, 0>::hash(keys);
    const hash_value_vt block_idxs = addr.get_block_idxs(block_addressing_hash_vals);
    const hash_value_vt bitvector_word_idx = block_idxs << word_cnt_log2;

    auto found = multiword_block<key_t, word_t, word_cnt, s, k,
        Hasher, hash_value_t,
        /* recursive counter */
        1, 0, word_cnt>
        ::contains(keys, &word_array[0], bitvector_word_idx);
    return found;
  }


  //===----------------------------------------------------------------------===//
  // Batch-wise Contains
  //===----------------------------------------------------------------------===//
  template<u64 vector_len = dtl::simd::lane_count<key_t>>
  __forceinline__
  $u64
  batch_contains(const key_t* keys, u32 key_cnt,
                 $u32* match_positions, u32 match_offset) const {
    return internal::dispatch<blocked_bloomfilter_logic, vector_len>
             ::batch_contains(*this,
                              keys, key_cnt,
                              match_positions, match_offset);
  }


  /// Returns length in bits.
  __forceinline__
  size_t
  length() const noexcept {
    return addr.length();
  }


  u64
  popcnt() const noexcept {
    return std::accumulate(word_array.begin(), word_array.end(), 0ull,
                           [](u64 cntr, word_t word) { return cntr + dtl::bits::pop_count(word); });
  }


  f64
  load_factor() const noexcept {
    f64 m = addr.length();
    return popcnt() / m;
  }


  void
  print_info() const noexcept {
    std::cout << "-- bloomfilter parameters --" << std::endl;
    std::cout << "  k:                    " << k << std::endl;
    std::cout << "  word bitlength:       " << word_bitlength << std::endl;
    std::cout << "  hash value bitlength: " << hash_value_bitlength << std::endl;
    std::cout << "  word count:           " << word_cnt << std::endl;
    std::cout << "  sector count:         " << s << std::endl;
    std::cout << "  max m:                " << max_m << std::endl;
    std::cout << "  max size [MiB]:       " << (max_m / 8.0 / 1024.0 / 1024.0 ) << std::endl;
    std::cout << "dynamic" << std::endl;
    std::cout << "  m:                    " << addr.length() << std::endl;
    f64 size_MiB = addr.length() / 8.0 / 1024.0 / 1024.0;
    if (size_MiB < 1) {
      std::cout << "  size [KiB]:           " << (size_MiB * 1024) << std::endl;
    }
    else {
      std::cout << "  size [MiB]:           " << size_MiB << std::endl;
    }
    std::cout << "  population count:     " << popcnt() << std::endl;
    std::cout << "  load factor:          " << load_factor() << std::endl;
  }


  void
  print() const noexcept {
    std::cout << "-- Bloom filter dump --" << std::endl;
    $u64 i = 0;
    for (const word_t word : word_array) {
      std::cout << std::bitset<word_bitlength>(word);
      i++;
      if (i % (128 / word_bitlength) == 0) {
        std::cout << std::endl;
      }
      else {
        std::cout << " ";
      }
    }
    std::cout << std::endl;
  }


};

} // namespace dtl
