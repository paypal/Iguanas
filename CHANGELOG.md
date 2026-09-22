# Changelog

All notable changes to this project are documented in this file. Format
loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- `RandomForestClassifier` as a third rule-generation backend alongside
  XGBoost and LightGBM, including monotone-constraint support via
  `monotonic_cst` and `scale_pos_weight` steering via `class_weight`.
- `leaf_selection` option (`"max_gain"` / `"all_positive"`) controlling how
  many rules are extracted per tree.
- Coverage-aware candidate metrics: `lift`, `wracc`, `laplace`, `m_estimate`,
  plus complexity metrics `count_conditions` / `count_features`.
- `combine_rules_budgeted`: budgeted maximum-coverage composition for
  matching a fixed alert-rate budget.
- Benchmarking harness (`benchmarks/`) implementing a nested
  generate/select/test protocol with a runtime leak guard.

### Fixed
- `combine_rules_a_star` admissibility bugs: the search is now an exact
  best-first branch-and-bound with a provably admissible bound, replacing the
  previous non-admissible heuristic and default hill-climbing behaviour.
- Operator-precedence bug in `add_missing_value_conditions` and an invalid
  polars `concat` mode.
- ONNX opset/IR version pinning, decoupling generated models from whatever
  `onnx`/`onnxruntime` version happens to be installed.
- A potential `UnboundLocalError` in `compute_counterfactual` when a
  condition's operator isn't one of the six recognised comparison operators.
- mypy `--strict` errors and ruff formatting across the package.

### Changed
- Claim-hygiene pass: corrected docstrings and docs to match actual code
  behaviour (parallelism model, security notes on `eval()`-based evaluation,
  fairness/monitoring/cross-validation scope).
- Test suite now has 100% line *and* branch coverage (`branch = true` in
  `[tool.coverage.run]`).

### Docs
- Added JOSS software paper and budgeted-composition methods paper drafts
  under `paper/`.

## [1.3.1] and earlier
See git history for changes prior to this file's introduction.
