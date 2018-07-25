#pragma once

#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_config.hpp>
#include <dtl/filter/cuckoofilter/cuckoofilter_config.hpp>
#include <dtl/filter/blocked_bloomfilter/fpr.hpp>


#include "calibration_data.hpp"
#include "cost_fn.hpp"
#include "timing.hpp"

namespace dtl {
namespace filter {
namespace model {


//===----------------------------------------------------------------------===//
/// Finds the value for m with the least overhead.
class optimizer {

  // search terminates if the improvement is less than 'epsilon' cycles.
  static constexpr auto epsilon = 0.001; // [cycles]

  u64 n_;
  const timing tw_;

  // the search range
  u64 m_from_;
  u64 m_to_;

  blocked_bloomfilter_config config_;
  const calibration_data& calibration_data_;

  // the current m
  $u64 m_current_;
  // the number of iterations
  $u64 it_;
  // minimum found?
  $u1 done_;
  $u64 step_size_;

  // the function to minimize
  timing
  f(u64 m) const {
    return cost_fn::overhead(n_, tw_, m, config_, calibration_data_);
  }

  // derivative of f()
  $f64
  d(u64 m) const {
    // numerical differentiation (symmetric difference quotient)
    u64 h = 1ull * 1024 * 8; // TODO remove magic number
    auto a = f(m + h).cycles_per_lookup;
    auto b = f(m - h).cycles_per_lookup;
    f64 df = (a - b) / (2 * h);
    return df;
  }


public:

  /// C'tor
  optimizer(u64 n, const timing& tw,
            u64 m_from, u64 m_to,
            const blocked_bloomfilter_config& config,
            const calibration_data& calibration_data)
      : n_(n), tw_(tw), m_from_(m_from), m_to_(m_to), config_(config),
        calibration_data_(calibration_data),
        m_current_(((m_to - m_from) / 2) + m_from), it_(0), done_(false), step_size_((m_to - m_from) / 4) {
    step();
  }


  /// Execute one iteration of the optimization process.
  /// Returns true if the global minimum has been found, false otherwise.
  u1
  step() {
    if (done_) return false;
    it_++;

    const auto m_previous_ = m_current_;
    const auto df = d(m_current_);
    m_current_ = (df < 0) ? m_current_ + step_size_
                          : m_current_ - step_size_;

    if (m_current_ < m_from_) {
      m_current_ = m_from_;
      done_ = true;
      return false;
    }

    if (m_current_ > m_to_) {
      m_current_ = m_to_;
      done_ = true;
      return false;
    }

    step_size_ /= 2;

    const auto o_previous = f(m_previous_);
    const auto o_current = f(m_current_);

    if (std::abs(o_current.cycles_per_lookup - o_previous.cycles_per_lookup) < epsilon
        || step_size_ == 0) {
      done_ = true;
      return false;
    }
    return true;
  }

  /// Returns the (optimized) value for m.
  u64
  get_result_m() const {
    return m_current_;
  }

  /// Returns the overhead value.
  timing
  get_result_overhead() const {
    return f(m_current_);
  }

  u64
  get_iteration() {
    return it_;
  }

  u1
  is_done() {
    return done_;
  }

};
//===----------------------------------------------------------------------===//


} // namespace model
} // namespace filter
} // namespace dtl
