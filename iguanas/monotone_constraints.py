import polars as pl
from xgboost import XGBClassifier


def infer_monotone_constraints_from_correlations(X: pl.DataFrame, y: pl.Series) -> pl.DataFrame:
    """Compute monotone constraint signs for XGBoost based on feature-target correlations.

    Parameters
    ----------
    X : pl.DataFrame
        DataFrame containing features.
    y : pl.Series
        Target series for computing correlations.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - feature: Feature name
        - pearson_corr: Pearson correlation with target
        - constraint: Constraint value (1 for positive, -1 for negative, 0 for no correlation)
    """
    # Add y as temporary column for efficient correlation computation
    X_temp = X.with_columns(y.alias("_target"))
    corr_result = X_temp.select([pl.corr(col, "_target").alias(col) for col in X.columns])

    result = pl.DataFrame(
        {
            "feature": X.columns,
            "pearson_corr": [corr_result[col][0] for col in X.columns],
        }
    ).with_columns(
        # Vectorized constraint computation
        pl.when(pl.col("pearson_corr") > 0)
        .then(pl.lit(1))
        .when(pl.col("pearson_corr") < 0)
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .alias("constraint")
    )

    return result


def infer_monotone_constraints_from_stumps(
    stump: XGBClassifier, X: pl.DataFrame, y: pl.Series
) -> pl.DataFrame:
    """Determine monotone constraints by training decision stumps for each feature.

    Trains a single-split tree (max_depth=1) for each feature and examines how
    predictions change from min to max value to determine monotonic relationship.

    Parameters
    ----------
    stump : XGBClassifier
        XGBoost classifier configured as a stump (max_depth=1).
    X : pl.DataFrame
        Features DataFrame.
    y : pl.Series
        Target series for training.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - feature: Feature name
        - constraint: Constraint value (1 for increasing, -1 for decreasing, 0 for flat)
        - pred_at_min: Predicted probability at minimum feature value
        - pred_at_max: Predicted probability at maximum feature value
        - delta: Change in probability (pred_at_max - pred_at_min)
    """
    # Get feature columns and target

    results = []
    for col in X.columns:
        stump.fit(X.select(col), y)
        # Get predictions at min and max values
        min_val = X.select(pl.col(col).min()).item()
        max_val = X.select(pl.col(col).max()).item()

        pred_at_min = stump.predict_proba([[min_val]])[0][1]
        pred_at_max = stump.predict_proba([[max_val]])[0][1]

        # Determine constraint based on prediction direction
        if pred_at_max > pred_at_min:
            constraint = 1
        elif pred_at_max < pred_at_min:
            constraint = -1
        else:
            constraint = 0

        results.append(
            {
                "feature": col,
                "constraint": constraint,
                "pred_at_min": pred_at_min,
                "pred_at_max": pred_at_max,
                "delta": pred_at_max - pred_at_min,
            }
        )

    return pl.DataFrame(results)
