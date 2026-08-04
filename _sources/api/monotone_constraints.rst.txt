==============================
Monotone Constraints
==============================

The monotone_constraints module provides functions for inferring monotone constraints for XGBoost models.

.. currentmodule:: iguanas.monotone_constraints

Functions
=========

infer_monotone_constraints_from_correlations
---------------------------------------------

.. autofunction:: infer_monotone_constraints_from_correlations

   Compute monotone constraint signs for XGBoost based on feature-target correlations.

   Calculates Pearson correlations between each feature and the target, then
   assigns constraint values:
   
   - **1** for positive correlation
   - **-1** for negative correlation
   - **0** for no correlation

   **Parameters:**
   
   - **X** (*pl.DataFrame*): DataFrame containing features
   - **y** (*pl.Series*): Target series for computing correlations
   
   **Returns:**
   
   - **pl.DataFrame**: DataFrame with columns: feature, pearson_corr, constraint

infer_monotone_constraints_from_stumps
---------------------------------------

.. autofunction:: infer_monotone_constraints_from_stumps

   Determine monotone constraints by training decision stumps for each feature.

   Trains a single-split tree (max_depth=1) for each feature and examines how
   predictions change from min to max value to determine monotonic relationship.

   **Parameters:**
   
   - **stump** (*XGBClassifier*): XGBoost classifier configured as a stump (max_depth=1)
   - **X** (*pl.DataFrame*): Features DataFrame
   - **y** (*pl.Series*): Target series for training
   
   **Returns:**
   
   - **pl.DataFrame**: DataFrame with columns: feature, constraint, pred_at_min, pred_at_max, delta
