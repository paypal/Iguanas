<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="https://paypal.github.io/Iguanas/_static/IGUANAS_LOGO.png">
  <img alt="Iguanas Logo" src="https://paypal.github.io/Iguanas/_static/IGUANAS_LOGO.png">
</picture>

# Iguanas: A Lightning-Fast Rule Generation Python Library


| | |
|:--|:-:|
| Package | [![PyPI version](https://img.shields.io/pypi/v/iguanas)](https://pypi.org/project/iguanas/) [![Python versions](https://img.shields.io/pypi/pyversions/iguanas)](https://pypi.org/project/iguanas/) |
| Quality | [![License](https://img.shields.io/github/license/paypal/iguanas)](https://github.com/paypal/iguanas/blob/main/LICENSE) [![Coverage](https://img.shields.io/codecov/c/github/paypal/iguanas)](https://codecov.io/gh/paypal/iguanas) |
| Documentation | [![Documentation](https://img.shields.io/badge/docs-online-blue)](https://paypal.github.io/Iguanas/) |
| Code style | [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |
| Downloads | [![Downloads](https://static.pepy.tech/badge/iguanas)](https://pepy.tech/project/iguanas) [![Downloads/Month](https://static.pepy.tech/badge/iguanas/month)](https://pepy.tech/project/iguanas) |
| Community | [![GitHub Stars](https://img.shields.io/github/stars/paypal/Iguanas?style=social)](https://github.com/paypal/Iguanas) [![Contributors](https://img.shields.io/github/contributors/paypal/Iguanas)](https://github.com/paypal/Iguanas/graphs/contributors) [![Last Commit](https://img.shields.io/github/last-commit/paypal/Iguanas)](https://github.com/paypal/Iguanas/commits/main) |


📚 **[Full Documentation](https://paypal.github.io/Iguanas/)**


## What is Iguanas?

Iguanas is a library built on top of Polars, designed to streamline the entire rule-based system development workflow — from raw data to production-ready rules — leveraging **Polars' blazing-fast multi-core processing**.

Built by the PSP Data Team at PayPal, Iguanas makes rule generation, evaluation, and selection both **faster and simpler**.

## ⚡ Key Features

- **🚀 Lightning Fast**: Built on Polars for multi-core parallel processing
- **🎯 End-to-End**: Generate, evaluate, combine, and select rules in one library
- **📦 Production Ready**: Lightweight rule strings that deploy anywhere
- **🔧 Flexible**: Sequential and parallel grid search strategies
- **🔗 Composable**: Chain generation → evaluation → selection with a few function calls
- **🎓 Easy to Learn**: Simple functional API with clear, consistent signatures

## 🛠️ What Can Iguanas Do?

### ⚙️ Rule Generation
Generate interpretable rules from labelled datasets using XGBoost tree extraction:
- `rule_grid_search_sequential` - Single-process grid search over weight transformations and scale_pos_weight values
- `rule_grid_search_parallel_weights` - Parallel grid search parallelised over weight transformations
- `rule_grid_search_parallel_scales` - Parallel grid search parallelised over scale_pos_weight values
- `extract_rules` - Extract rules from a fitted XGBoost model (with optional monotone constraints)
- `extract_rule_by_max_gain` - Extract the highest-gain rule path from a single tree
- `extract_rule_with_monotone_constraints` - Extract a rule path respecting monotone constraints

### 📊 Metrics
Compute classification performance metrics for rule predictions:
- `compute_metrics` - Compute a full metrics table (accuracy, precision, recall, F-beta, TP/FP/TN/FN, flagged %) for a set of rules
- `compute_single_metric` - Compute a single scalar metric (accuracy, precision, recall or F-beta) — optimised for hot-path evaluation

### 🔍 Rule Evaluation
Evaluate rules on data and filter by performance:
- `apply_rules` - Evaluate rule expressions on a DataFrame and return a boolean prediction matrix
- `apply_and_filter_by_performance` - Evaluate rules and filter by user-defined metric thresholds
- `select_diverse_top_rules` - Select top-performing rules while removing highly correlated duplicates
- `apply_filter_and_deduplicate_rules` - Complete end-to-end pipeline: evaluate → filter → deduplicate

### 🔀 Rule Combination
Combine individual rules into compound rules to improve performance:
- `combine_rules_full_search` - Exhaustive search over all rule pairs
- `combine_rules_cumulative` - Incrementally combine rules with a running candidate
- `combine_rules_greedy` - Greedy combination selecting the best pair at each step
- `combine_rules_beam_search` - Beam search combination balancing quality and efficiency
- `combine_rules_a_star` - A* search combination using a heuristic cost function

### ✂️ Rule Selection
Deduplicate and prune rule sets:
- `filter_rules_by_feature_overlap` - Remove rules that share too many features with higher-importance rules
- `filter_correlated_rules` - Remove rules whose predictions are highly correlated
- `select_best_rule_per_column_combination` - Keep only the best-performing rule for each unique column combination
- `extract_feature_names_from_rule` - Parse a rule string and return the feature names it references

### 🔬 Rule Analysis
Inspect and report on rule sets:
- `generate_rule_performance_report` - Generate a combined performance and structure report for a rule set
- `parse_conditions` - Parse a rule expression into its constituent conditions
- `parse_levels` - Parse a rule expression into a structured level-by-level representation
- `rebuild_from_levels` - Reconstruct a rule string from a level representation

### 🖊️ Rule Formatting
Clean up rule expressions for display or logging:
- `simplify_rule` - Simplify a rule expression by removing redundant conditions
- `rule_to_sql` - Convert a rule expression to a SQL `WHERE` clause with an optional table alias

### 📐 Monotone Constraints
Infer feature directionality to guide rule generation:
- `infer_monotone_constraints_from_correlations` - Infer monotone constraints (±1) from feature–target correlations
- `infer_monotone_constraints_from_stumps` - 
Infer monotone constraints (±1) from decision stumps

### ⚖️ Sample Weight Transformations
Generate sample weight schedules to steer rule learning:
- `generate_increasing_weights` - Weights that increase with feature value (power, log families)
- `generate_decreasing_weights` - Weights that decrease with feature value (reciprocal families)
- `generate_weights` - Generate both increasing and decreasing weight schedules in one call
- `select_uncorrelated_weights` - Select a diverse subset of weight columns by searching for a correlation threshold that yields approximately `num_weights` uncorrelated columns

### 🔁 Rule Cross-Validation
Validate rule stability across held-out folds without re-generating rules:
- `validate_rules_cv` - Evaluate rules across K folds and return per-metric mean, std, and min — flags overfitted rules by their high variance
- `identify_unstable_rules` - Return the names of rules whose cross-validated metric variance exceeds a threshold

### 💬 Rule Explanation
Inspect and explain individual rule predictions:
- `verbalize_rule` - Convert a rule expression to a plain-English sentence
- `compute_coverage_overlap` - Compute pairwise Jaccard overlap between rule predictions
- `compute_counterfactual` - Find the minimal feature changes needed to un-flag a sample

### 🗂️ Rule Registry
Store and compare named rule snapshots across experiments:
- `RuleRegistry` - Save, load, delete, and list named rule snapshots (with optional JSON persistence)
- `filter_rule_pairs_by_overlap` - Return rule pairs whose Jaccard overlap falls within a `[min_overlap, max_overlap]` range (e.g. disjoint pairs, near-redundant pairs, or everything in between)

### 🚀 Deployment
Export rules and score data efficiently at scale:
- `apply_rules_lazy` - Evaluate rule expressions on a Polars `LazyFrame` for out-of-core scoring
- `rules_to_onnx` - Convert rule strings to a portable ONNX binary classifier (servable by any ONNX-compatible runtime)

### 📤 ONNX Export
Convert any fitted rule or ruleset into a self-contained ONNX model:

```python
import numpy as np
from iguanas.onnx_converter import rules_to_onnx
import onnxruntime as ort

# From a fitted RuleClassifier / RulesetClassifier
export = clf.export()  # {"rule": "...", "feature_cols": [...]}
model = rules_to_onnx(export["rule"])  # single rule string

# Or pass a list of rules (OR'd together)
model = rules_to_onnx(
    ['(X["age"] >= 30.0) & (X["income"] > 50000.0)',
     '(X["credit_score"] >= 720.0)'],
    dtype="f32",  # or "f64" for double precision
)

# Score with onnxruntime
sess = ort.InferenceSession(model.SerializeToString())
X = np.array([[35.0, 60000.0, 700.0],
              [25.0, 30000.0, 640.0]], dtype=np.float32)
predictions = sess.run(None, {"X": X})[0]  # int64 array: [1, 0]

# Feature ordering is stored in model metadata
feature_map = {p.key: p.value for p in model.metadata_props}
# {"feature_0": "age", "feature_1": "income", "feature_2": "credit_score"}
```

The exported model has:
- **Input** `X`: `[N, num_features]` tensor (float32 or float64)
- **Output** `prediction`: `[N]` int64 tensor (0 or 1)
- **Metadata**: feature-name-to-column-index mapping in `metadata_props`

### ⚖️ Fairness
Audit rule performance across demographic subgroups:
- `compute_subgroup_metrics` - Compute precision, recall, and all other metrics broken down by a protected attribute column
- `compute_disparate_impact_ratio` - Compute the ratio of positive prediction rates between subgroups to surface disparate impact

### 📈 Rule Monitoring
Track rule performance drift between a reference period and a current period:
- `compare_rule_metrics` - Compare per-rule metrics between two `compute_metrics` outputs and flag rules that have degraded beyond optional thresholds

## 🚀 Quick Start

```python
import polars as pl
import numpy as np
from xgboost import XGBClassifier

from iguanas.weight_transformations import generate_weights
from iguanas.rule_generation import rule_grid_search_parallel_weights
from iguanas.rule_evaluation import apply_filter_and_deduplicate_rules

# 1. Load your data
X_train = pl.DataFrame({
    "age":    [25, 45, 35, 50, 30, 55, 40, 28],
    "income": [30000, 80000, 50000, 90000, 40000, 95000, 70000, 35000],
})
y_train = pl.Series([0, 1, 0, 1, 0, 1, 1, 0])

# 2. Generate sample weight transformations
weights = generate_weights(X_train["income"])

# 3. Run a parallel grid search to extract rules
estimator = XGBClassifier(max_depth=2, n_estimators=5, random_state=42)
scale_pos_weights = np.logspace(0, 1, 5)

rules_df = rule_grid_search_parallel_weights(
    estimator, X_train, y_train,
    scale_pos_weights=scale_pos_weights,
    sample_weights_df=weights,
    n_jobs=-1,
)

# 4. Evaluate, filter, and deduplicate rules
R, metrics, selected_rules = apply_filter_and_deduplicate_rules(
    X_train, y_train, rules_df,
    metric_thresholds=[
        {"name": "precision", "operator": ">=", "value": 0.6},
        {"name": "recall",    "operator": ">=", "value": 0.5},
    ],
    max_corr=0.8,
)

print(selected_rules)
```

## 📦 Installation

Requires Python 3.10 or higher.

```bash
pip install iguanas
```

Or install from source:

```bash
git clone https://github.com/paypal/iguanas.git
cd iguanas
pip install -e .    # Install in editable/development mode
```

## 📚 Documentation

For detailed documentation, tutorials, and API reference, visit:

**[https://paypal.github.io/iguanas/](https://paypal.github.io/iguanas/)**

## 🎯 Use Cases

Iguanas is perfect for:

- **Fraud Detection** - Generate high-precision rules to flag suspicious transactions
- **Risk Scoring** - Build interpretable rule sets for credit or operational risk
- **Compliance & Policy** - Encode business policies as auditable rule expressions
- **Anomaly Detection** - Surface rare but meaningful patterns in labelled data
- **Model Explainability** - Extract human-readable rules from gradient boosted models

## 🏢 Used By

Iguanas powers rule-based systems at:
- PayPal (internal use)

## 🤝 Contributing

We welcome contributions! Please check out our [contributing guidelines](https://github.com/paypal/iguanas/blob/master/CONTRIBUTING.md).

## 📄 License

Iguanas is licensed under the Apache License 2.0. See [LICENSE](https://github.com/paypal/iguanas/blob/master/LICENSE) file for details.

## 🙏 Credits

Developed by the PSP Data Team at PayPal.

---

**Built by data scientists, for data scientists**