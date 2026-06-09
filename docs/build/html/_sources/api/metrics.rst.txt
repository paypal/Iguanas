=====================
Metrics
=====================

.. currentmodule:: iguanas.metrics

Functions
=========

compute_single_metric
----------------------

.. autofunction:: compute_single_metric

compute_metrics
---------------

.. autofunction:: compute_metrics
   
   **Examples:**
   
   .. code-block:: python
   
      import polars as pl
      from iguanas.metrics import compute_metrics
      
      # Count-based metrics only
      metrics_X = compute_metrics(R, y, weights=None)
      
      # Both count and weighted metrics
      metrics_X = compute_metrics(R, y, weights=transaction_amounts)
      
      # Sort by TPVE3 to find best rules
      top_rules = metrics_X.sort("TPVE3", descending=True).head(10)
