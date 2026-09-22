
.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api_reference

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples

.. toctree::
   :maxdepth: 2
   :caption: Evaluation
   :hidden:

   benchmark

Iguanas is a **rule generation and evaluation** library built on top of `Polars <https://pola.rs/>`_,
designed to streamline the entire rule-based system development workflow — from raw data to
production-ready rules.

.. note::

   For data preprocessing and feature engineering prior to rule generation, we recommend using
   `Gators <https://paypal.github.io/gators/index.html>`_ — a complementary library built on top of
   Polars by the same team at PayPal, providing 70+ transformers for cleaning, encoding,
   imputation, scaling, and more.

Built by the PSP Data Team at PayPal, Iguanas makes rule generation, evaluation, and selection
**simpler**, with a single-node execution model that uses multi-threading where it helps.

Key Features
============

* ⚙️ **Vectorised evaluation**: Rules are compiled once to Polars expressions (cached) and applied as columnar, multi-threaded operations
* 🧵 **Thread-parallel grid search**: Grid search is parallelised on a single machine via joblib's threading backend
* 🎯 **End-to-End**: Generate, evaluate, combine, and select rules in one library
* 📦 **Production Ready**: Lightweight rule strings that deploy anywhere, plus ONNX export for runtimes that must not execute Python
* 🔧 **Flexible**: Sequential and thread-parallel grid search strategies
* 🔗 **Composable**: Chain generation → evaluation → selection with a few function calls
* 🎓 **Easy to Learn**: Simple functional API with clear, consistent signatures

.. note::

   **Scope of parallelism.** Iguanas runs on a single node. Grid search uses
   ``joblib.Parallel`` with the ``"threading"`` backend, and rule evaluation relies on
   Polars' internal multi-threading. There is no multiprocessing, no distributed or
   cluster execution (no Dask, Ray or Spark), and no GPU code path. No published
   benchmark accompanies this release, so no throughput or speed-up figure is claimed.

Quick Start
===========

.. code-block:: python

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

What Can Iguanas Do?
====================

* ⚙️ :doc:`Rule Generation <api/rule_generation>` - Extract rules from XGBoost/LightGBM models with grid search
* 📊 :doc:`Metrics <api/metrics>` - Precision, recall, F-beta, MCC, and weighted variants
* 🔍 :doc:`Rule Evaluation <api/rule_evaluation>` - Evaluate, filter, and deduplicate rule sets; lazy scoring via ``apply_rules_lazy`` (compiles rules with ``eval()`` — trusted rule sources only)
* 🔁 :doc:`Rule Cross-Validation <api/rule_cv>` - Check rule stability across K folds (cv mean, std, min per metric; optimistically biased — see the module notes)
* 🔀 :doc:`Rule Combination <api/rule_combination>` - Combine rules with greedy, beam, and A* search
* ✂️ :doc:`Rule Selection <api/rule_selection>` - Prune by feature overlap and correlation
* 🔬 :doc:`Rule Analysis <api/rule_analysis>` - Inspect and report on rule structure
* 💬 :doc:`Rule Explanation <api/rule_explanation>` - Verbalize rules, compute coverage overlap, and generate counterfactual explanations
* 🖊️ :doc:`Rule Formatting <api/rule_formatting>` - Simplify rules, export to SQL, and reverse encoded-feature rules (e.g. from gators) back to the original columns
* ⚖️ :doc:`Rule Fairness <api/rule_fairness>` - Post-hoc bias measurement across demographic subgroups (measured, not optimised)
* 🗂️ :doc:`Rule Registry <api/rule_registry>` - Store named ruleset snapshots and filter pairs by Jaccard overlap
* 📈 :doc:`Rule Monitoring <api/rule_monitoring>` - Flag per-rule performance degradation between reference and current periods (threshold on metric deltas, not a statistical drift test)
* 🤖 :doc:`Classifiers <api/rule_classifier>` - scikit-learn compatible ``RuleClassifier`` and ``RulesetClassifier``
* 📦 :doc:`ONNX Export <api/onnx_converter>` - Convert rules to portable ONNX models for any runtime (parses with ``ast``; executes no Python at scoring time)
* 📐 :doc:`Monotone Constraints <api/monotone_constraints>` - Infer feature directionality
* 🔧 :doc:`Weight Transformations <api/weight_transformations>` - Generate and select sample weight schedules

Use Cases
=========

Iguanas is intended for:

* **Fraud Detection** — Generate high-precision rules to flag suspicious transactions
* **Risk Scoring** — Build interpretable rule sets for credit or operational risk
* **Compliance & Policy** — Encode business policies as auditable rule expressions
* **Anomaly Detection** — Surface rare but meaningful patterns in labelled data
* **Model Explainability** — Extract human-readable rules from gradient boosted models

Credits
-------

Developed by the PSP Data Team at PayPal.

**⚡ Built by data scientists, for data scientists**
