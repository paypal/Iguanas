Examples
========

This section contains end-to-end examples demonstrating how to generate optimized rulesets using **Iguanas**.

.. toctree::
   :maxdepth: 2
   :hidden:

   examples/titanic_survival_example
   examples/titanic_survival_example_with_preprocessing
   examples/titanic_survival_scikit-learn_api_example

Overview
--------

Each example notebook demonstrates a complete ML workflow using Iguanas rule generation, alone or with and  feature engineering done with the `gators <https://paypal.github.io/gators/index.html>`_ package

Titanic Survival Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Binary classification using advanced feature engineering including string parsing, mathematical features,
and rare category encoding.

* Without feature engineering, the best F1 score from rules alone is 0.640:

:doc:`View Notebook <examples/titanic_survival_example>`

* Using a sklearn API wrapper for Iguanas, the best F1 score from rules alone is 0.642:

:doc:`View Notebook <examples/titanic_survival_scikit-learn_api_example>`

* With feature engineering, the best F1 score from rules done on engineered features is 0.78:

:doc:`View Notebook <examples/titanic_survival_example_with_preprocessing>`
