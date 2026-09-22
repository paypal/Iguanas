=====================
Rule Generation
=====================

.. currentmodule:: iguanas.rule_generation

Supported backends: ``XGBClassifier``, ``LGBMClassifier`` and scikit-learn's
``RandomForestClassifier``. All three are converted to the same canonical
per-node tree table before rule extraction, so every function below works
unchanged regardless of which fitted estimator is passed in.

Functions
=========

extract_max_gain_rule
-------------------------

.. autofunction:: extract_max_gain_rule

extract_rule_with_monotone_constraints
---------------------------------------

.. autofunction:: extract_rule_with_monotone_constraints

extract_rules
-------------

.. autofunction:: extract_rules

rule_grid_search
---------------------------------

.. autofunction:: rule_grid_search

rule_grid_search_sequential
----------------------------

.. autofunction:: rule_grid_search_sequential

rule_grid_search_parallel_weights
----------------------------------

.. autofunction:: rule_grid_search_parallel_weights

rule_grid_search_parallel_scales
---------------------------------

.. autofunction:: rule_grid_search_parallel_scales
