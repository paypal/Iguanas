Quick Start
===========

This guide will get you started with Iguanas in minutes.

Basic Example
-------------

Here's a simple example showing the core Iguanas workflow:

.. code-block:: python

   import polars as pl
   import numpy as np
   from xgboost import XGBClassifier

   # Import Iguanas modules
   from iguanas.rule_generation import rule_grid_search_parallel_scales
   from iguanas.rule_evaluation import apply_rules
   from iguanas.rule_selection import filter_correlated_rules
   from iguanas.metrics import compute_metrics

   # 1. Load your data (example with synthetic data)
   X_train = pl.DataFrame({
      'age': [25, 45, 35, 50, 30, 55, 40, 28],
      'income': [30000, 80000, 50000, 90000, 40000, 95000, 70000, 35000],
      'credit_score': [650, 720, 680, 750, 660, 780, 710, 640]
   })
   y_train = pl.Series([0, 1, 0, 1, 0, 1, 1, 0])

   # 3. Configure XGBoost estimator for rule extraction
   estimator = XGBClassifier(
      max_depth=1,  # Decision stumps for simple rules
      n_estimators=10,
      random_state=42
   )

   # 4. Generate rules using grid search
   scale_pos_weights = np.logspace(0, 1, 5)  # Try different class balance weights
   rules_df = rule_grid_search_parallel_scales(
      estimator=estimator,
      X_train=X_train,
      y_train=y_train,
      scale_pos_weights=scale_pos_weights,
      n_jobs=-1,
      verbose=1
   )

   # 5. Apply rules to your data
   rules = rules_df['rule'].unique().to_list()
   R_train = apply_rules(X_train, rules)

   # 6. Compute performance metrics
   metrics = compute_metrics(R_train, y_train)
   print(metrics.select(['rule', 'precision', 'recall', 'f1']).head(10))

   # 7. Filter correlated rules to keep only diverse, high-performing rules
   importance = dict(metrics[['rule', 'f1']].rows())
   uncorrelated_rules = filter_correlated_rules(R_train, importance, max_corr=0.8)

   print(f"Original rules: {len(rules)}")
   print(f"Filtered rules: {len(uncorrelated_rules)}")

Understanding the API
---------------------

Iguanas is organized into modular components that work together in a typical workflow:

**1. Rule Generation** (:doc:`api/rule_generation`)
   Generate rules from your data using XGBoost or LightGBM decision trees:
   
   - ``rule_grid_search()``: Parallelized grid search over weight transformations and scale_pos_weight values
   - ``extract_rules()``: Extract rules from a fitted XGBoost/LightGBM model
   - ``extract_rule_by_max_gain()``: Extract the highest-gain rule path from a single tree

**2. Rule Evaluation** (:doc:`api/rule_evaluation`)
   Apply rules to data and evaluate their performance:
   
   - ``apply_rules()``: Evaluate rule expressions on DataFrames
   - ``apply_and_filter_by_performance()``: Filter rules by precision/recall thresholds
   - ``select_diverse_top_rules()``: Select top performing non-correlated rules

**3. Metrics** (:doc:`api/metrics`)
   Compute comprehensive performance metrics:
   
   - ``compute_metrics()``: Calculate precision, recall, F-scores, TPVE metrics
   - Supports both count-based and weighted metrics

**4. Rule Selection** (:doc:`api/rule_selection`)
   Filter and select rules based on similarity and correlation:
   
   - ``filter_correlated_rules()``: Remove highly correlated rules
   - ``filter_rules_by_feature_overlap()``: Filter rules with similar feature usage
   - ``extract_feature_names_from_rule()``: Extract features used in rules

**5. Rule Combination** (:doc:`api/rule_combination`)
   Combine rules to create more powerful composite rules:
   
   - ``combine_rules_full_search()``: Generate all combinations
   - ``combine_rules_greedy()``: Greedy search for best combinations
   - ``combine_rules_beam_search()``: Beam search algorithm
   - ``combine_rules_a_star()``: A* search algorithm

**6. Rule Analysis** (:doc:`api/rule_analysis`)
   Analyze rules at hierarchical levels:
   
   - ``generate_rule_performance_report()``: Metrics at rule, component, and condition levels

**7. Rule Formatting** (:doc:`api/rule_formatting`)
   Transform and simplify rules:
   
   - ``simplify_rule()``: Remove redundant conditions (e.g. duplicate bounds on the same feature)
   - ``rule_to_sql()``: Convert a rule expression to a SQL ``WHERE`` clause with an optional table alias

**8. Utilities**
   Supporting utilities for rule generation:
   
   - :doc:`api/weight_transformations`: Generate and select sample weight schedules (``generate_weights``, ``select_uncorrelated_weights``)
   - :doc:`api/monotone_constraints`: Infer monotone constraints for XGBoost/LightGBM

**9. Rule Cross-Validation** (:doc:`api/rule_cv`)
   Validate rule stability across held-out folds:
   
   - ``validate_rules_cv()``: Evaluate rules on K folds and return per-metric mean, std, and min
   - ``identify_unstable_rules()``: Flag rules with high cross-fold variance

**10. Rule Explanation** (:doc:`api/rule_explanation`)
   Interpret and explain individual rule predictions:
   
   - ``verbalize_rule()``: Convert a rule expression to plain English
   - ``compute_coverage_overlap()``: Compute pairwise Jaccard overlap between rule predictions
   - ``compute_counterfactual()``: Find the minimal feature changes to un-flag a sample

**11. Rule Fairness** (:doc:`api/rule_fairness`)
   Audit rule performance across demographic subgroups:
   
   - ``compute_subgroup_metrics()``: Per-subgroup precision, recall, and all other metrics
   - ``compute_disparate_impact_ratio()``: Surface disparate impact relative to a reference group

**12. Rule Monitoring** (:doc:`api/rule_monitoring`)
   Track rule performance drift over time:
   
   - ``compare_rule_metrics()``: Compare per-rule metrics between a reference and current period; flag degraded rules

**13. Rule Registry** (:doc:`api/rule_registry`)
   Version-control rulesets across experiments:
   
   - ``RuleRegistry``: Save, load, delete, and compare named rule snapshots (JSON persistence)
   - ``filter_rule_pairs_by_overlap()``: Filter rule pairs by Jaccard overlap range

**14. Classifiers** (:doc:`api/rule_classifier`)
   scikit-learn compatible end-to-end pipelines:
   
   - ``RuleClassifier``: Fit → generate → filter → select single best rule
   - ``RulesetClassifier``: Fit → generate → filter → correlate → greedily combine rules

**15. ONNX Export** (:doc:`api/onnx_converter`)
   Deploy rules to any ONNX-compatible runtime:
   
   - ``rules_to_onnx()``: Convert rule strings to a portable ONNX binary classifier


Next Steps
----------

* Explore :doc:`api/rule_generation` for generating rules from your data
* Check out :doc:`api/rule_evaluation` for applying and filtering rules
* Check out :doc:`api/metrics` for evaluating rule performance
* Learn about :doc:`api/rule_combination` for combining rules
* Dive into :doc:`api/rule_selection` for deduplicating rule sets
* See :doc:`api/rule_classifier` for the scikit-learn compatible pipeline
* See :doc:`api/onnx_converter` for exporting rules to ONNX
* Browse the :doc:`api_reference` for complete API documentation
