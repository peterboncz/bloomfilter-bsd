#include <new>

#include <dtl/dtl.hpp>
#include <dtl/filter/filter.hpp>
#include <dtl/filter/model/optimizer.hpp>

#include "skyline_matrix_builder.hpp"
#include "util.hpp"


namespace dtl {
namespace filter {
namespace model {


//===----------------------------------------------------------------------===//

void
build_skyline_matrix(dtl::filter::model::calibration_data& data) {


  u64 n_lo_log2 = 10;
  u64 n_hi_log2 = 24;
//  u64 n_hi_log2 = 14; // TODO remove

  u64 tw_lo_log2 = 4;
  u64 tw_hi_log2 = 24;
//  u64 tw_hi_log2 = 10; // TODO remove



  // all valid values for n
  const std::vector<$u64> n_values = [&]() {
    std::set<$u64> n_vals;

    for ($u64 n_log2 = n_lo_log2; n_log2 <= n_hi_log2; n_log2++) {
//      const std::vector<$f64> exp {
//          n_log2 +  0 * 0.0625,
//          n_log2 +  1 * 0.0625,
//          n_log2 +  2 * 0.0625,
//          n_log2 +  3 * 0.0625,
//          n_log2 +  4 * 0.0625,
//          n_log2 +  5 * 0.0625,
//          n_log2 +  6 * 0.0625,
//          n_log2 +  7 * 0.0625,
//          n_log2 +  8 * 0.0625,
//          n_log2 +  9 * 0.0625,
//          n_log2 + 10 * 0.0625,
//          n_log2 + 11 * 0.0625,
//          n_log2 + 12 * 0.0625,
//          n_log2 + 13 * 0.0625,
//          n_log2 + 14 * 0.0625,
//          n_log2 + 15 * 0.0625,
//          n_log2 + 16 * 0.0625,
//      };
//      const std::vector<$f64> exp {
//          n_log2 + 0 * 0.125,
//          n_log2 + 1 * 0.125,
//          n_log2 + 2 * 0.125,
//          n_log2 + 3 * 0.125,
//          n_log2 + 4 * 0.125,
//          n_log2 + 5 * 0.125,
//          n_log2 + 6 * 0.125,
//          n_log2 + 7 * 0.125,
//          n_log2 + 8 * 0.125,
//      };
//      const std::vector<$f64> exp {
//          n_log2 + 0 * 0.25,
//          n_log2 + 1 * 0.25,
//          n_log2 + 2 * 0.25,
//          n_log2 + 3 * 0.25,
//          n_log2 + 4 * 0.25,
//      };
//      const std::vector<$f64> exp {
//          n_log2 + 0 * 0.5,
//          n_log2 + 1 * 0.5,
//      };
      const std::vector<$f64> exp {
          n_log2 + 0 * 0.5,
      };

      for (auto e : exp) {
        u64 n = std::pow(2, e);
        n_vals.insert(n);
      }
    }
    std::vector<$u64> ret_val(n_vals.begin(), n_vals.end());
    return ret_val;
  }();


  // all valid values for n
  const std::vector<$u64> tw_values = [&]() {
    std::set<$u64> tw_vals;

    for ($u64 tw_log2 = tw_lo_log2; tw_log2 <= tw_hi_log2; tw_log2++) {
//      const std::vector<$f64> exp {
//          tw_log2 +  0 * 0.0625,
//          tw_log2 +  1 * 0.0625,
//          tw_log2 +  2 * 0.0625,
//          tw_log2 +  3 * 0.0625,
//          tw_log2 +  4 * 0.0625,
//          tw_log2 +  5 * 0.0625,
//          tw_log2 +  6 * 0.0625,
//          tw_log2 +  7 * 0.0625,
//          tw_log2 +  8 * 0.0625,
//          tw_log2 +  9 * 0.0625,
//          tw_log2 + 10 * 0.0625,
//          tw_log2 + 11 * 0.0625,
//          tw_log2 + 12 * 0.0625,
//          tw_log2 + 13 * 0.0625,
//          tw_log2 + 14 * 0.0625,
//          tw_log2 + 15 * 0.0625,
//          tw_log2 + 16 * 0.0625,
//      };
//      const std::vector<$f64> exp {
//          tw_log2 + 0 * 0.125,
//          tw_log2 + 1 * 0.125,
//          tw_log2 + 2 * 0.125,
//          tw_log2 + 3 * 0.125,
//          tw_log2 + 4 * 0.125,
//          tw_log2 + 5 * 0.125,
//          tw_log2 + 6 * 0.125,
//          tw_log2 + 7 * 0.125,
//          tw_log2 + 8 * 0.125,
//      };
//      const std::vector<$f64> exp {
//          tw_log2 + 0 * 0.25,
//          tw_log2 + 1 * 0.25,
//          tw_log2 + 2 * 0.25,
//          tw_log2 + 3 * 0.25,
//          tw_log2 + 4 * 0.25,
//      };
//      const std::vector<$f64> exp {
//          tw_log2 + 0 * 0.5,
//          tw_log2 + 1 * 0.5,
//      };
      const std::vector<$f64> exp {
          tw_log2 + 0 * 0.5,
      };
      for (auto e : exp) {
        u64 tw = std::pow(2, e);
        tw_vals.insert(tw);
      }
    }
    std::vector<$u64> ret_val(tw_vals.begin(), tw_vals.end());
    return ret_val;
  }();


  const auto config_set = [&]() {
    std::set<dtl::blocked_bloomfilter_config> supported_configs;
    for (auto c : data.get_bbf_configs()) {
      // test if the current build support this configuration
      try {
        dtl::filter::filter f(c, 4096 + (c.addr_mode == dtl::block_addressing::MAGIC) + 128);
        supported_configs.insert(c);
      }
      catch (...) {
        std::cerr << "Filter not supported by this build: " << c << std::endl;
      }
    }
    return supported_configs;
  }();

  const std::vector<dtl::blocked_bloomfilter_config> configs(config_set.begin(), config_set.end());

  std::cout << "Number of valid Bloom filter configurations: " << configs.size() << std::endl;

  auto find_opt = [&](u64 n, u64 tw) { // TODO also support unit nanoseconds
    // limit bits per element [4, 20]
    u64 m_lo = n * 4;
    u64 m_hi = n * 20;

    skyline_matrix::entry_t ret_val;
    $u1 first = true;

    for (auto& c : configs) {
      timing t = {tw, tw}; // TODO also support unit nanoseconds
      optimizer opt(n, t, m_lo, m_hi, c, data);
      while (opt.step()) { }

      if (first) {
        ret_val.config_ = c;
        ret_val.m_ = opt.get_result_m();
        ret_val.overhead_ = opt.get_result_overhead();
        first = false;
      }
      else {
        if (ret_val.overhead_ > opt.get_result_overhead()) {
          ret_val.config_ = c;
          ret_val.m_ = opt.get_result_m();
          ret_val.overhead_ = opt.get_result_overhead();
        }
      }
    }
    return ret_val;
  };

  $u64 cntr = 0;
  u64 n_cnt = n_values.size();
  u64 tw_cnt = tw_values.size();
  u64 matrix_entry_cnt = n_values.size() * tw_values.size();

  u64 skyline_size = skyline_matrix::size_in_bytes(n_cnt, tw_cnt);
  std::vector<$i8> skyline_matrix_mem(skyline_size);
  skyline_matrix* skyline_matrix_ptr = new(&skyline_matrix_mem[0]) skyline_matrix;
  skyline_matrix_ptr->meta_data_.n_values_count_ = n_cnt;
  skyline_matrix_ptr->meta_data_.tw_values_count_ = tw_cnt;

  for ($u64 i = 0; i < n_cnt; i++) {
    skyline_matrix_ptr->n_values_begin()[i] = n_values[i];
  }

  for ($u64 i = 0; i < tw_cnt; i++) {
    skyline_matrix_ptr->tw_values_begin()[i] = tw_values[i];
  }



  using n_tw_idx_pair = std::pair<$u64, $u64>;
  std::vector<n_tw_idx_pair> n_tw_idx_pairs(matrix_entry_cnt);


  std::cout << "n values: ";
  for (auto n : n_values) {
    std::cout << n << " ";
  }
  std::cout << std::endl;

  std::cout << "tw values: ";
  for (auto tw : tw_values) {
    std::cout << tw << " ";
  }
  std::cout << std::endl;

  std::cout << "Size of skyline matrix: " << (n_cnt * tw_cnt) << " entries." << std::endl;


  for ($u64 n_idx = 0; n_idx < n_cnt; n_idx++) {
    for ($u64 tw_idx = 0; tw_idx < tw_cnt; tw_idx++) {
      n_tw_idx_pairs.emplace_back(std::make_pair(n_idx, tw_idx));
    }
  }

  {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(n_tw_idx_pairs.begin(), n_tw_idx_pairs.end(), g);
  }


  std::function<void(const n_tw_idx_pair, std::ostream&)> fn = [&](const n_tw_idx_pair n_tw_idx, std::ostream& os) -> void {
    const auto n_idx = n_tw_idx.first;
    const auto tw_idx = n_tw_idx.second;
    skyline_matrix_ptr->entries_begin()[(tw_idx * n_cnt) + n_idx] = find_opt(n_values[n_idx], tw_values[tw_idx]);
  };
  dispatch<n_tw_idx_pair>(n_tw_idx_pairs, fn);


  {
    for ($u64 n_idx = 0; n_idx < n_cnt; n_idx++) {
      for ($u64 tw_idx = 0; tw_idx < tw_cnt; tw_idx++) {
        const auto& entry = skyline_matrix_ptr->entries_begin()[(tw_idx * n_cnt) + n_idx];
        std::cout << "n=" << n_values[n_idx] << ", tw=" << tw_values[tw_idx]
                  << ", " << entry.config_
                  << ", m=" << (entry.m_ / 1024 / 8) << " [KiB]"
                  << ", overhead=" << entry.overhead_.cycles_per_lookup
                  << std::endl;
      }
    }
  }

  data.put_skyline_matrix(*skyline_matrix_ptr);

}

//===----------------------------------------------------------------------===//


} // namespace model
} // namespace filter
} // namespace dtl
