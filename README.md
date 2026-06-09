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
| Community | [![GitHub Stars](https://img.shields.io/github/stars/paypal/Iguanas?style=social)](https://github.com/paypal/Iguanas) [![GitHub Forks](https://img.shields.io/github/forks/paypal/Iguanas?style=social)](https://github.com/paypal/Iguanas) [![Contributors](https://img.shields.io/github/contributors/paypal/Iguanas)](https://github.com/paypal/Iguanas/graphs/contributors) [![Last Commit](https://img.shields.io/github/last-commit/paypal/Iguanas)](https://github.com/paypal/Iguanas/commits/main) |


📚 **[Full Documentation](https://paypal.github.io/iguanas/)**


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
- `compute_metrics` - Compute a full metrics table (precision, recall, F-beta, TP/FP/TN/FN, flagged %) for a set of rules
- `compute_single_metric` - Compute a single scalar metric (precision, recall, accuracy, or F-beta) — optimised for hot-path evaluation

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

### 📐 Monotone Constraints
Infer feature directionality to guide rule generation:
- `infer_monotone_constraints_from_correlations` - Infer monotone constraints (±1) from feature–target correlations
- `infer_monotone_constraints_from_stumps` - 
Infer monotone constraints (±1) from decision stumps

### ⚖️ Sample Weight Transformations
Generate sample weight schedules to steer rule learning:
- `generate_increasing_weight` - Weights that increase with feature value (power, log families)
- `generate_decreasing_weight` - Weights that decrease with feature value (reciprocal families)
- `generate_all_weight` - Generate both increasing and decreasing weight schedules in one call

## 🚀 Quick Start

```python
import polars as pl
import numpy as np
from xgboost import XGBClassifier

from iguanas.weight_transformations import generate_all_weight
from iguanas.rule_generation import rule_grid_search_parallel_weights
from iguanas.rule_evaluation import apply_filter_and_deduplicate_rules

# 1. Load your data
X_train = pl.DataFrame({
    "age":    [25, 45, 35, 50, 30, 55, 40, 28],
    "income": [30000, 80000, 50000, 90000, 40000, 95000, 70000, 35000],
})
y_train = pl.Series([0, 1, 0, 1, 0, 1, 1, 0])

# 2. Generate sample weight transformations
weights = generate_all_weight(X_train["income"])

# 3. Run a parallel grid search to extract rules
estimator = XGBClassifier(max_depth=2, n_estimators=5, random_state=42)
scale_pos_weight_vec = np.logspace(0, 1, 5)

rules_df = rule_grid_search_parallel_weights(
    estimator, X_train, y_train,
    scale_pos_weight_vec=scale_pos_weight_vec,
    weights_train_vec=weights,
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
