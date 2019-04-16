#include <algorithm>
#include <set>
#include <thread>
#include <iomanip>

#include <dtl/dtl.hpp>
#include <amsfilter_model/internal/util.hpp>
#include <amsfilter_model/fpr.hpp>
//===----------------------------------------------------------------------===//
// Tool to generate a LuT for the false-positive rates (see file fpr_lut.hpp).
//===----------------------------------------------------------------------===//
static constexpr u64 bpe_min = 1;
static constexpr u64 bpe_max = 64;

static constexpr u64 bpe_x100_min = bpe_min * 100;
static constexpr u64 bpe_x100_step_size = 10;
static_assert(100 % bpe_x100_step_size == 0, "Invalid step size.");

static constexpr u64 bpe_intermediates = 100 / bpe_x100_step_size;
static constexpr u64 lut_size = (bpe_max - bpe_min) * bpe_intermediates + 1;
//===----------------------------------------------------------------------===//
$i32
main() {

  auto valid_configs = amsfilter::model::get_valid_configs();
  std::set<$u32> valid_configs_encoded;
  for (auto& c : valid_configs) {
    // Ignore the addressing mode.
    auto config = c;
    config.addr_mode = dtl::block_addressing::POWER_OF_TWO;
    auto encoded_config = amsfilter::model::config_to_int(config);
    valid_configs_encoded.insert(encoded_config);
  }

  std::cout <<
      "#pragma once\n"
      "\n"
      "//===----------------------------------------------------------------------===//\n"
      "//===-------------------- GENERATED FILE. DO NOT EDIT. --------------------===//\n"
      "//===----------------------------------------------------------------------===//\n"
      "\n"
      "#include <algorithm>\n"
      "#include <sstream>\n"
      "#include <stdexcept>\n"
      "#include <cmath>\n"
      "#include <dtl/dtl.hpp>\n"
      "#include <dtl/filter/blocked_bloomfilter/block_addressing_logic.hpp>\n"
      "#include <amsfilter/amsfilter.hpp>\n"
      "#include <amsfilter_model/internal/util.hpp>\n"
      "\n"
      "namespace amsfilter {\n"
      "//===----------------------------------------------------------------------===//\n";
  std::cout << "// All valid filter configurations (encoded as integers)."
      << std::endl;
  std::cout << "static constexpr std::size_t config_cnt = "
      << valid_configs_encoded.size() << ";" << std::endl;
  std::cout << "static const $u32 configs[config_cnt] {\n  ";
  for (auto& enc : valid_configs_encoded) {
    std::cout << enc << ",";
  }
  std::cout << "};" << std::endl;

  std::cout
      << "//===----------------------------------------------------------------------===//"
      << std::endl;

  std::cout
      << "// FPR lookup table for varying bits-per-element (bpe) rates.\n"
      << "// The bpe varies in [" << bpe_min << ".0, " << bpe_max << ".0] with a step"
      << " size of " << (bpe_x100_step_size / 100.0) << "."
      << std::endl;
  std::cout << "static const $f64 fpr_lut[" << valid_configs_encoded.size()
      << "][" << lut_size << "] {" << std::endl;

  for (auto& enc : valid_configs_encoded) {
    auto c = amsfilter::model::config_from_int(enc);
    std::cout << "  // w=" << c.word_cnt_per_block
        << ", s=" << c.sector_cnt
        << ", z=" << c.zone_cnt
        << ", k=" << c.k
        << std::endl;
    std::cout << "  {";

    $u1 is_first = true;
    for (std::size_t i = 0; i < lut_size; ++i) {
      if (is_first) {
        is_first = false;
      }
      else {
        std::cout << ",";
      }
      auto bpe_x100 = bpe_x100_min + (i * bpe_x100_step_size);
      auto bpe = (bpe_x100 / 100.0);
      static constexpr u64 m = 8ull * 1024 * 1024 * 8;
      u64 n = static_cast<u64>(m / bpe);
      std::cout << amsfilter::fpr_fast(c, m, n);
    }

    std::cout << "}," << std::endl;
  }
  std::cout << "};" << std::endl;

  // Generate the FPR lookup function.
  std::cout <<
      "//===----------------------------------------------------------------------===//\n"
      "$f64\n"
      "lookup_fpr(const Config& c, u64 m, u64 n) {\n"
      "  static constexpr u64 bpe_min = " << bpe_min << ";\n"
      "  static constexpr u64 bpe_max = " << bpe_max << ";\n"
      "  static constexpr u64 bpe_x100_step_size = " << bpe_x100_step_size << ";\n"
      "\n"
      "  // The addressing mode is ignored.\n"
      "  Config config = c;\n"
      "  config.addr_mode = dtl::block_addressing::POWER_OF_TWO;\n"
      "\n"
      "  f64 b = (m * 1.0) / n;\n"
      "  f64 bpe = std::min(\n"
      "      std::max(b, bpe_min * 1.0),\n"
      "      bpe_max * 1.0\n"
      "      );\n"
      "  u64 fpr_idx = static_cast<u64>(\n"
      "      std::round((bpe - bpe_min) * (100/bpe_x100_step_size)));\n"
      "  auto enc = model::config_to_int(config);\n"
      "  auto search = std::lower_bound(configs, configs + config_cnt, enc);\n"
      "  auto lut_idx = static_cast<u64>(std::distance(configs, search));\n"
      "  if (lut_idx >= config_cnt) lut_idx = config_cnt - 1;\n"
      "  if (configs[lut_idx] != enc) {\n"
      "    std::stringstream err;\n"
      "    err << \"Unknown filter configuration: \" << c;\n"
      "    throw std::invalid_argument(err.str());\n"
      "  }\n"
      "  return fpr_lut[lut_idx][fpr_idx];\n"
      "}\n";


  std::cout
      << "//===----------------------------------------------------------------------===//"
      << std::endl;


  u32 k_ignored = 1;
  std::set<$u32> valid_configs_encoded_ignoring_k;
  for (auto& c : valid_configs) {
    auto config = c;
    // Ignore the addressing mode.
    config.addr_mode = dtl::block_addressing::POWER_OF_TWO;
    // Ignore the k.
    config.k = k_ignored;
    auto encoded_config = amsfilter::model::config_to_int(config);
    valid_configs_encoded_ignoring_k.insert(encoded_config);
  }

  std::cout << "// All valid filter configurations IGNORING k (encoded as integers)."
      << std::endl;
  std::cout << "static constexpr std::size_t optimal_k_config_cnt = "
      << valid_configs_encoded_ignoring_k.size() << ";" << std::endl;
  std::cout << "static const $u32 optimal_k_configs[optimal_k_config_cnt] {\n  ";
  for (auto& enc : valid_configs_encoded_ignoring_k) {
    std::cout << enc << ",";
  }
  std::cout << "};" << std::endl;

  std::cout
      << "//===----------------------------------------------------------------------===//"
      << std::endl;

  std::cout
      << "// Optimal k lookup table for varying bits-per-element (bpe) rates.\n"
      << "// The bpe varies in [" << bpe_min << ".0, " << bpe_max << ".0] with a step"
      << " size of " << (bpe_x100_step_size / 100.0) << "."
      << std::endl;
  std::cout << "static const $u32 optimal_k_lut[" << valid_configs_encoded.size()
      << "][" << lut_size << "] {" << std::endl;


  for (auto& enc : valid_configs_encoded_ignoring_k) {
    auto c = amsfilter::model::config_from_int(enc);
    std::cout << "  // w=" << c.word_cnt_per_block
        << ", s=" << c.sector_cnt
        << ", z=" << c.zone_cnt
        << std::endl;
    std::cout << "  {";

    $u1 is_first = true;
    for (std::size_t i = 0; i < lut_size; ++i) {
      if (is_first) {
        is_first = false;
      }
      else {
        std::cout << ",";
      }
      auto bpe_x100 = bpe_x100_min + (i * bpe_x100_step_size);
      auto bpe = (bpe_x100 / 100.0);
      std::cout << amsfilter::optimal_k(c, bpe) << "u";
    }

    std::cout << "}," << std::endl;
  }
  std::cout << "};" << std::endl;

  // Generate the optimal-k lookup function.
  std::cout <<
      "//===----------------------------------------------------------------------===//\n"
      "$u32\n"
      "lookup_optimal_k(const Config& c, f64 bits_per_element) {\n"
      "  static constexpr u64 bpe_min = " << bpe_min << ";\n"
      "  static constexpr u64 bpe_max = " << bpe_max << ";\n"
      "  static constexpr u64 bpe_x100_step_size = " << bpe_x100_step_size << ";\n"
      "\n"
      "  Config config = c;\n"
      "  // The addressing mode is ignored.\n"
      "  config.addr_mode = dtl::block_addressing::POWER_OF_TWO;\n"
      "  // The given k is ignored.\n"
      "  config.k = " << k_ignored << ";\n"
      "\n"
      "  f64 bpe = std::min(\n"
      "      std::max(bits_per_element, bpe_min * 1.0),\n"
      "      bpe_max * 1.0\n"
      "      );\n"
      "  u64 opt_k_idx = static_cast<u64>(\n"
      "      std::round((bpe - bpe_min) * (100/bpe_x100_step_size)));\n"
      "  auto enc = model::config_to_int(config);\n"
      "  auto search = std::lower_bound(\n"
      "    optimal_k_configs, optimal_k_configs + optimal_k_config_cnt, enc);\n"
      "  auto lut_idx = static_cast<u64>(std::distance(optimal_k_configs, search));\n"
      "  if (lut_idx >= optimal_k_config_cnt) lut_idx = optimal_k_config_cnt - 1;\n"
      "  if (optimal_k_configs[lut_idx] != enc) {\n"
      "    std::stringstream err;\n"
      "    err << \"Unknown filter configuration: \" << c;\n"
      "    throw std::invalid_argument(err.str());\n"
      "  }\n"
      "  return optimal_k_lut[lut_idx][opt_k_idx];\n"
      "}\n";

  std::cout <<
      "//===----------------------------------------------------------------------===//\n"
          "} // namespace amsfilter" << std::endl;

  return 0;
}
//===----------------------------------------------------------------------===//
