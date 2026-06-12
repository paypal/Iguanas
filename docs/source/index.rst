
.. image:: _static/IGUANAS_LOGO.png
   :align: center
   :alt: Iguanas Logo

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

Iguanas is a **lightning-fast rule generation** library built on top of `Polars <https://pola.rs/>`_,
designed to streamline the entire rule-based system development workflow — from raw data to
production-ready rules — leveraging Polars' blazing-fast multi-core processing.

.. note::

   For data preprocessing and feature engineering prior to rule generation, we recommend using
   `Gators <https://paypal.github.io/gators/index.html>`_ — a complementary library built on top of
   Polars by the same team at PayPal, providing 70+ transformers for cleaning, encoding,
   imputation, scaling, and more.

Built by the PSP Data Team at PayPal, Iguanas makes rule generation, evaluation, and selection
both **faster** and **simpler**.

Key Features
============

* 🚀 **Lightning Fast**: Built on Polars for multi-core parallel processing
* 🎯 **End-to-End**: Generate, evaluate, combine, and select rules in one library
* 📦 **Production Ready**: Lightweight rule strings that deploy anywhere
* 🔧 **Flexible**: Sequential and parallel grid search strategies
* 🔗 **Composable**: Chain generation → evaluation → selection with a few function calls
* 🎓 **Easy to Learn**: Simple functional API with clear, consistent signatures

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
        weights_train_vec=weights,
        n_jobs=-1,
    )

    # 4. Evaluate, filter, and deduplicate rules
    R, metrics, selected_rules = apply_filter_and_deduplicate_rules(
        X_train, y_train, rules_df,
        metrics_threshold=[
            {"name": "precision", "operator": ">=", "value": 0.6},
            {"name": "recall",    "operator": ">=", "value": 0.5},
        ],
        max_corr=0.8,
    )

    print(selected_rules)

What Can Iguanas Do?
====================

* ⚙️ :doc:`Rule Generation <api/rule_generation>` - Extract rules from XGBoost models with grid search
* 📊 :doc:`Metrics <api/metrics>` - Precision, recall, F-beta, and weighted variants
* 🔍 :doc:`Rule Evaluation <api/rule_evaluation>` - Evaluate, filter, and deduplicate rule sets
* 🔀 :doc:`Rule Combination <api/rule_combination>` - Combine rules with greedy, beam, and A* search
* ✂️ :doc:`Rule Selection <api/rule_selection>` - Prune by feature overlap and correlation
* 🔬 :doc:`Rule Analysis <api/rule_analysis>` - Inspect and report on rule structure
* 🖊️ :doc:`Rule Formatting <api/rule_formatting>` - Simplify and clean rule expressions
* 📐 :doc:`Monotone Constraints <api/monotone_constraints>` - Infer feature directionality
* ⚖️ :doc:`Weight Transformations <api/weight_transformations>` - Generate sample weight schedules

Use Cases
=========

Iguanas is perfect for:

* **Fraud Detection** — Generate high-precision rules to flag suspicious transactions
* **Risk Scoring** — Build interpretable rule sets for credit or operational risk
* **Compliance & Policy** — Encode business policies as auditable rule expressions
* **Anomaly Detection** — Surface rare but meaningful patterns in labelled data
* **Model Explainability** — Extract human-readable rules from gradient boosted models

Credits
-------

Developed by the PSP Data Team at PayPal.

**⚡ Built by data scientists, for data scientists**
