# Changelog

All notable changes to this project are documented in this file. Format
loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [1.4.0] - 2026-09-24

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
- `extract_rules` now raises a clear `ValueError` (instead of an internal
  `AttributeError`) when called with `all_features_constrained=True` on an
  estimator that wasn't fitted with a non-zero monotone constraint for every
  feature.
- mypy `--strict` errors and ruff formatting across the package.

### Changed
- Claim-hygiene pass: corrected docstrings and docs to match actual code
  behaviour (parallelism model, security notes on `eval()`-based evaluation,
  fairness/monitoring/cross-validation scope).
- Test suite now has 100% line *and* branch coverage (`branch = true` in
  `[tool.coverage.run]`).
- Trimmed the PyPI sdist from ~11MB/1500+ files down to <1MB by excluding the
  `benchmarks/` and `paper/` research/paper directories, which aren't part of
  the installable package.

### Docs
- Added JOSS software paper and budgeted-composition methods paper drafts
  under `paper/`.

## [1.3.1] and earlier
See git history for changes prior to this file's introduction.
