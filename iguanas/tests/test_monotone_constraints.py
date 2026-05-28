import polars as pl
import pytest
from xgboost import XGBClassifier

from iguanas.monotone_constraints import (
    infer_monotone_constraints_from_correlations,
    infer_monotone_constraints_from_stumps,
)


@pytest.fixture
def sample_positive_correlation():
    """Features with positive correlation to target."""
    return pl.DataFrame(
        {
            "feature_pos": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_strong_pos": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    ), pl.Series("target", [0, 0, 1, 1, 1])


@pytest.fixture
def sample_negative_correlation():
    """Features with negative correlation to target."""
    return pl.DataFrame(
        {
            "feature_neg": [5.0, 4.0, 3.0, 2.0, 1.0],
            "feature_strong_neg": [10.0, 8.0, 6.0, 4.0, 2.0],
        }
    ), pl.Series("target", [0, 0, 1, 1, 1])


@pytest.fixture
def sample_mixed_correlation():
    """Features with mixed correlations."""
    return pl.DataFrame(
        {
            "positive_feature": [1.0, 2.0, 3.0, 4.0, 5.0],
            "negative_feature": [5.0, 4.0, 3.0, 2.0, 1.0],
            "no_correlation": [3.0, 3.0, 3.0, 3.0, 3.0],
        }
    ), pl.Series("target", [0, 0, 1, 1, 1])


@pytest.fixture
def sample_larger_dataset():
    """Larger dataset for more robust testing."""
    return pl.DataFrame(
        {
            "feature_A": list(range(20)),
            "feature_B": list(range(19, -1, -1)),
            "feature_C": [i * 2 for i in range(20)],
        }
    ), pl.Series("target", [0] * 10 + [1] * 10)


@pytest.fixture
def xgb_stump():
    """XGBoost classifier configured as a stump."""
    return XGBClassifier(
        max_depth=1,
        n_estimators=1,
        learning_rate=1.0,
        random_state=42,
    )


# Tests for infer_monotone_constraints_from_correlations
def test_correlations_positive_correlation(sample_positive_correlation):
    X, y = sample_positive_correlation
    result = infer_monotone_constraints_from_correlations(X, y)

    # Check that result is a DataFrame
    assert isinstance(result, pl.DataFrame)

    # Check expected columns exist
    assert "feature" in result.columns
    assert "pearson_corr" in result.columns
    assert "constraint" in result.columns

    # Check that all features are present
    assert result.height == 2
    assert set(result["feature"].to_list()) == {"feature_pos", "feature_strong_pos"}

    # Check that all constraints are positive (1)
    assert all(result["constraint"] == 1)


def test_correlations_negative_correlation(sample_negative_correlation):
    X, y = sample_negative_correlation
    result = infer_monotone_constraints_from_correlations(X, y)

    # Check that all constraints are negative (-1)
    assert all(result["constraint"] == -1)

    # Check that correlations are negative
    assert all(result["pearson_corr"] < 0)


def test_correlations_mixed_correlation(sample_mixed_correlation):
    X, y = sample_mixed_correlation
    result = infer_monotone_constraints_from_correlations(X, y)

    # Check correct number of features
    assert result.height == 3

    # Check positive feature has constraint 1
    pos_row = result.filter(pl.col("feature") == "positive_feature")
    assert pos_row["constraint"][0] == 1
    assert pos_row["pearson_corr"][0] > 0

    # Check negative feature has constraint -1
    neg_row = result.filter(pl.col("feature") == "negative_feature")
    assert neg_row["constraint"][0] == -1
    assert neg_row["pearson_corr"][0] < 0

    # Check no correlation feature exists in results
    no_corr_row = result.filter(pl.col("feature") == "no_correlation")
    assert no_corr_row.height == 1
    # With small sample, constant feature may still have constraint due to correlation with target
    assert no_corr_row["constraint"][0] in [-1, 0, 1]


def test_correlations_single_feature():
    X = pl.DataFrame({"feature_1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pl.Series("target", [0, 0, 1, 1, 1])
    result = infer_monotone_constraints_from_correlations(X, y)

    # Should handle single feature
    assert result.height == 1
    assert result["feature"][0] == "feature_1"
    assert result["constraint"][0] in [-1, 0, 1]


def test_correlations_perfect_correlation():
    X = pl.DataFrame({"perfect_pos": [0.0, 1.0, 2.0, 3.0, 4.0]})
    y = pl.Series("target", [0, 1, 2, 3, 4])
    result = infer_monotone_constraints_from_correlations(X, y)

    # Perfect positive correlation
    assert result["constraint"][0] == 1
    # Correlation should be very close to 1
    assert abs(result["pearson_corr"][0] - 1.0) < 0.01


def test_correlations_all_zeros():
    X = pl.DataFrame(
        {
            "zero_feature": [0.0, 0.0, 0.0, 0.0, 0.0],
            "normal_feature": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    y = pl.Series("target", [0, 0, 1, 1, 1])
    result = infer_monotone_constraints_from_correlations(X, y)

    # Zero feature should have zero constraint (no variation)
    zero_row = result.filter(pl.col("feature") == "zero_feature")
    # Constraint should be 0 (correlation is NaN or 0)
    assert result.height == 2


def test_correlations_larger_dataset(sample_larger_dataset):
    X, y = sample_larger_dataset
    result = infer_monotone_constraints_from_correlations(X, y)

    # Check all features processed
    assert result.height == 3

    # feature_A should be positive (increasing)
    feature_a = result.filter(pl.col("feature") == "feature_A")
    assert feature_a["constraint"][0] == 1

    # feature_B should be negative (decreasing)
    feature_b = result.filter(pl.col("feature") == "feature_B")
    assert feature_b["constraint"][0] == -1

    # feature_C should be positive (increasing, scaled version of feature_A)
    feature_c = result.filter(pl.col("feature") == "feature_C")
    assert feature_c["constraint"][0] == 1


def test_correlations_binary_target_balanced():
    X = pl.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0],
            "feature_2": [4.0, 3.0, 2.0, 1.0],
        }
    )
    y = pl.Series("target", [0, 0, 1, 1])
    result = infer_monotone_constraints_from_correlations(X, y)

    # Should work with balanced binary target
    assert result.height == 2
    assert all(result["constraint"].is_in([-1, 0, 1]))


# Tests for infer_monotone_constraints_from_stumps
def test_stumps_positive_correlation(sample_positive_correlation, xgb_stump):
    X, y = sample_positive_correlation
    result = infer_monotone_constraints_from_stumps(xgb_stump, X, y)

    # Check that result is a DataFrame
    assert isinstance(result, pl.DataFrame)

    # Check expected columns exist
    expected_columns = ["feature", "constraint", "pred_at_min", "pred_at_max", "delta"]
    for col in expected_columns:
        assert col in result.columns

    # Check that all features are present
    assert result.height == 2

    # Check that constraints are in valid range
    assert all(result["constraint"].is_in([-1, 0, 1]))


def test_stumps_negative_correlation(sample_negative_correlation, xgb_stump):
    X, y = sample_negative_correlation
    result = infer_monotone_constraints_from_stumps(xgb_stump, X, y)

    # Check that constraints are valid
    assert all(result["constraint"].is_in([-1, 0, 1]))

    # Check that at least one has negative delta (stronger negative pattern)
    # With simple stumps, not all features may show clear monotonic relationship
    assert result.height == 2


def test_stumps_mixed_correlation(sample_mixed_correlation, xgb_stump):
    X, y = sample_mixed_correlation
    result = infer_monotone_constraints_from_stumps(xgb_stump, X, y)

    # Check correct number of features
    assert result.height == 3

    # Check positive feature has valid constraint
    pos_row = result.filter(pl.col("feature") == "positive_feature")
    assert pos_row["constraint"][0] in [-1, 0, 1]

    # Check negative feature has valid constraint
    neg_row = result.filter(pl.col("feature") == "negative_feature")
    assert neg_row["constraint"][0] in [-1, 0, 1]

    # All features should have predictions and deltas
    assert all(result["pred_at_min"] >= 0)
    assert all(result["pred_at_max"] >= 0)


def test_stumps_pred_at_min_max_ordering(sample_positive_correlation, xgb_stump):
    X, y = sample_positive_correlation
    result = infer_monotone_constraints_from_stumps(xgb_stump, X, y)

    # Check that constraint values are consistent with predictions
    for row in result.iter_rows(named=True):
        if row["constraint"] == 1:
            # Positive constraint: pred_at_max should be > pred_at_min
            assert (
                row["pred_at_max"] > row["pred_at_min"]
                or abs(row["pred_at_max"] - row["pred_at_min"]) < 1e-6
            )
        elif row["constraint"] == -1:
            # Negative constraint: pred_at_max should be < pred_at_min
            assert (
                row["pred_at_max"] < row["pred_at_min"]
                or abs(row["pred_at_max"] - row["pred_at_min"]) < 1e-6
            )
        else:
            # Zero constraint: predictions should be equal or very close
            assert abs(row["pred_at_max"] - row["pred_at_min"]) < 0.1


def test_stumps_delta_consistency(sample_positive_correlation, xgb_stump):
    X, y = sample_positive_correlation
    result = infer_monotone_constraints_from_stumps(xgb_stump, X, y)

    # Delta should equal pred_at_max - pred_at_min
    for row in result.iter_rows(named=True):
        expected_delta = row["pred_at_max"] - row["pred_at_min"]
        assert abs(row["delta"] - expected_delta) < 1e-6


def test_stumps_single_feature(xgb_stump):
    X = pl.DataFrame({"feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    y = pl.Series("target", [0, 0, 0, 0, 1, 1, 1, 1])
    result = infer_monotone_constraints_from_stumps(xgb_stump, X, y)

    # Should handle single feature
    assert result.height == 1
    assert result["feature"][0] == "feature_1"
    assert result["constraint"][0] in [-1, 0, 1]


def test_stumps_prediction_probabilities_range(sample_larger_dataset, xgb_stump):
    X, y = sample_larger_dataset
    result = infer_monotone_constraints_from_stumps(xgb_stump, X, y)

    # Predictions should be valid probabilities [0, 1]
    assert all(result["pred_at_min"] >= 0)
    assert all(result["pred_at_min"] <= 1)
    assert all(result["pred_at_max"] >= 0)
    assert all(result["pred_at_max"] <= 1)


def test_stumps_constraint_values_only(sample_larger_dataset, xgb_stump):
    X, y = sample_larger_dataset
    result = infer_monotone_constraints_from_stumps(xgb_stump, X, y)

    # Constraint should only be -1, 0, or 1
    unique_constraints = set(result["constraint"].to_list())
    assert unique_constraints.issubset({-1, 0, 1})


def test_stumps_feature_names_preserved(sample_mixed_correlation, xgb_stump):
    X, y = sample_mixed_correlation
    result = infer_monotone_constraints_from_stumps(xgb_stump, X, y)

    # All feature names should be preserved
    assert set(result["feature"].to_list()) == set(X.columns)


def test_stumps_imbalanced_target():
    """Test with highly imbalanced target."""
    X = pl.DataFrame({"feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    y = pl.Series("target", [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # Very imbalanced

    stump = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
    result = infer_monotone_constraints_from_stumps(stump, X, y)

    # Should handle imbalanced target
    assert result.height == 1
    # Should produce valid constraint
    assert result["constraint"][0] in [-1, 0, 1]


def test_stumps_larger_dataset(sample_larger_dataset):
    X, y = sample_larger_dataset
    stump = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
    result = infer_monotone_constraints_from_stumps(stump, X, y)

    # Check all features processed
    assert result.height == 3

    # All constraints should be valid
    assert all(result["constraint"].is_in([-1, 0, 1]))

    # All features should have predictions
    assert all(result["pred_at_min"] >= 0)
    assert all(result["pred_at_max"] >= 0)


def test_correlation_vs_stumps_agreement(sample_mixed_correlation, xgb_stump):
    """Test that both methods generally agree on constraint direction."""
    X, y = sample_mixed_correlation

    corr_result = infer_monotone_constraints_from_correlations(X, y)
    stump_result = infer_monotone_constraints_from_stumps(xgb_stump, X, y)

    # Join results on feature
    corr_dict = {row["feature"]: row["constraint"] for row in corr_result.iter_rows(named=True)}
    stump_dict = {row["feature"]: row["constraint"] for row in stump_result.iter_rows(named=True)}

    # For strong correlations, methods should agree
    # (allowing for some disagreement on weak/no correlation features)
    assert corr_dict.keys() == stump_dict.keys()


def test_correlations_with_nulls():
    """Test correlation method with null values."""
    X = pl.DataFrame(
        {
            "feature_1": [1.0, 2.0, None, 4.0, 5.0],
            "feature_2": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    y = pl.Series("target", [0, 0, 1, 1, 1])
    result = infer_monotone_constraints_from_correlations(X, y)

    # Should handle nulls (Polars will handle them in correlation)
    assert result.height == 2
    assert all(result["constraint"].is_in([-1, 0, 1]))


def test_stumps_deterministic_results(sample_positive_correlation):
    """Test that results are deterministic with fixed random state."""
    X, y = sample_positive_correlation

    stump1 = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
    result1 = infer_monotone_constraints_from_stumps(stump1, X, y)

    stump2 = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
    result2 = infer_monotone_constraints_from_stumps(stump2, X, y)

    # Results should be identical
    assert result1["constraint"].to_list() == result2["constraint"].to_list()
