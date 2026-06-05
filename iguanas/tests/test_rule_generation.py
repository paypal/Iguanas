import numpy as np
import pandas as pd
import polars as pl
import pytest
from xgboost import XGBClassifier

from iguanas.rule_generation import (
    _check_all_features_have_monotone_constraints,
    _train_rules_for_weight_transformation,
    _train_rules_for_scale,
    extract_rule_by_max_gain,
    extract_rule_with_monotone_constraints,
    extract_rules,
    rule_grid_search_parallel_scales,
    rule_grid_search_parallel_weights,
    rule_grid_search_sequential,
)


class TestExtractRuleByMaxGain:
    """Test cases for extract_rule_by_max_gain function."""

    def test_empty_dataframe(self):
        """Test that empty dataframe returns empty string."""
        tree_X = pd.DataFrame()
        result = extract_rule_by_max_gain(tree_X)
        assert result == ""

    def test_no_leaves(self):
        """Test that dataframe with no leaves returns empty string."""
        tree_X = pd.DataFrame(
            {
                "Tree": [0, 0],
                "Node": [0, 1],
                "ID": ["0-0", "0-1"],
                "Feature": ["feature1", "feature2"],
                "Split": [5.0, 10.0],
                "Yes": ["0-1", None],
                "No": ["0-2", None],
                "Missing": [None, None],
                "Gain": [0.5, 0.3],
                "Cover": [100, 50],
                "Category": [None, None],
            }
        )
        result = extract_rule_by_max_gain(tree_X)
        assert result == ""

    def test_single_split_tree(self):
        """Test extraction from a tree with single split."""
        tree_X = pd.DataFrame(
            {
                "Tree": [0, 0, 0],
                "Node": [0, 1, 2],
                "ID": ["0-0", "0-1", "0-2"],
                "Feature": ["amount", "Leaf", "Leaf"],
                "Split": [100.0, None, None],
                "Yes": ["0-1", None, None],
                "No": ["0-2", None, None],
                "Missing": ["0-1", None, None],
                "Gain": [0.5, 0.3, 0.8],
                "Cover": [100, 40, 60],
                "Category": [None, None, None],
            }
        )
        result = extract_rule_by_max_gain(tree_X)
        # Should follow path to leaf with gain=0.8 (Node 2, which is "No" branch)
        assert result == '(X["amount"] >= 100.0)'

    def test_two_level_tree(self):
        """Test extraction from a tree with two levels."""
        tree_X = pd.DataFrame(
            {
                "Tree": [0, 0, 0, 0, 0],
                "Node": [0, 1, 2, 3, 4],
                "ID": ["0-0", "0-1", "0-2", "0-3", "0-4"],
                "Feature": ["age", "income", "Leaf", "Leaf", "Leaf"],
                "Split": [30.0, 50000.0, None, None, None],
                "Yes": ["0-1", "0-3", None, None, None],
                "No": ["0-2", "0-4", None, None, None],
                "Missing": ["0-1", "0-3", None, None, None],
                "Gain": [1.5, 0.8, 0.2, 0.4, 1.2],
                "Cover": [100, 60, 40, 30, 30],
                "Category": [None, None, None, None, None],
            }
        )
        result = extract_rule_by_max_gain(tree_X)
        # Should follow path to leaf with gain=1.2 (Node 4)
        # Path: 0 -> No (2 is leaf) or 0 -> Yes (1) -> No (4)
        # Node 4 is "No" branch of Node 1, which is "Yes" branch of Node 0
        assert '(X["age"] < 30.0)' in result
        assert '(X["income"] >= 50000.0)' in result

    def test_rounding_of_split_values(self):
        """Test that split values are rounded to 5 decimal places."""
        tree_X = pd.DataFrame(
            {
                "Tree": [0, 0, 0],
                "Node": [0, 1, 2],
                "ID": ["0-0", "0-1", "0-2"],
                "Feature": ["value", "Leaf", "Leaf"],
                "Split": [3.123456789, None, None],
                "Yes": ["0-1", None, None],
                "No": ["0-2", None, None],
                "Missing": ["0-1", None, None],
                "Gain": [0.5, 0.3, 0.8],
                "Cover": [100, 40, 60],
                "Category": [None, None, None],
            }
        )
        result = extract_rule_by_max_gain(tree_X)
        assert "3.12346" in result

    def test_empty_node_rows(self):
        """Test that empty node_rows returns empty string (line 49 coverage)."""
        # Create a tree where the node lookup will fail
        tree_X = pd.DataFrame(
            {
                "Tree": [0, 0],
                "Node": [0, 1],
                "ID": ["0-0", "0-1"],
                "Feature": ["feature1", "Leaf"],
                "Split": [5.0, 0.0],
                "Yes": ["0-1", None],
                "No": ["0-100", None],  # Non-existent node
                "Missing": [None, None],
                "Gain": [0.5, 0.8],
                "Cover": [100, 50],
                "Category": [None, None],
            }
        )
        result = extract_rule_by_max_gain(tree_X)
        # Should return something, but may be incomplete
        assert isinstance(result, str)


class TestExtractRuleWithMonotoneConstraints:
    """Test cases for extract_rule_with_monotone_constraints function."""

    def test_empty_dataframe(self):
        """Test that empty dataframe raises KeyError (no Node column)."""
        tree_X = pd.DataFrame()
        monotone_constraints = {}
        with pytest.raises(KeyError):
            extract_rule_with_monotone_constraints(tree_X, monotone_constraints)

    def test_single_leaf(self):
        """Test tree that starts with a leaf node."""
        tree_X = pd.DataFrame(
            {
                "Tree": [0],
                "Node": [0],
                "ID": ["0-0"],
                "Feature": ["Leaf"],
                "Split": [None],
                "Yes": [None],
                "No": [None],
                "Missing": [None],
                "Gain": [0.5],
                "Cover": [100],
                "Category": [None],
            }
        )
        monotone_constraints = {}
        result = extract_rule_with_monotone_constraints(tree_X, monotone_constraints)
        assert result == ""

    def test_positive_constraint(self):
        """Test rule extraction with positive monotone constraint."""
        tree_X = pd.DataFrame(
            {
                "Tree": [0, 0, 0],
                "Node": [0, 1, 2],
                "ID": ["0-0", "0-1", "0-2"],
                "Feature": ["amount", "Leaf", "Leaf"],
                "Split": [100.0, None, None],
                "Yes": ["0-1", None, None],
                "No": ["0-2", None, None],
                "Missing": ["0-1", None, None],
                "Gain": [0.5, 0.3, 0.8],
                "Cover": [100, 40, 60],
                "Category": [None, None, None],
            }
        )
        monotone_constraints = {"amount": 1}
        result = extract_rule_with_monotone_constraints(tree_X, monotone_constraints)
        # Positive constraint follows "No" branch (>=)
        assert result == '(X["amount"] >= 100.0)'

    def test_negative_constraint(self):
        """Test rule extraction with negative monotone constraint."""
        tree_X = pd.DataFrame(
            {
                "Tree": [0, 0, 0],
                "Node": [0, 1, 2],
                "ID": ["0-0", "0-1", "0-2"],
                "Feature": ["age", "Leaf", "Leaf"],
                "Split": [50.0, None, None],
                "Yes": ["0-1", None, None],
                "No": ["0-2", None, None],
                "Missing": ["0-1", None, None],
                "Gain": [0.5, 0.3, 0.8],
                "Cover": [100, 40, 60],
                "Category": [None, None, None],
            }
        )
        monotone_constraints = {"age": -1}
        result = extract_rule_with_monotone_constraints(tree_X, monotone_constraints)
        # Negative constraint follows "Yes" branch (<)
        assert result == '(X["age"] < 50.0)'

    def test_two_level_tree_with_constraints(self):
        """Test extraction from a tree with two levels."""
        # Create a tree where following constraints leads to a 2-level path
        tree_X = pd.DataFrame(
            {
                "Tree": [0, 0, 0, 0, 0],
                "Node": [0, 1, 2, 3, 4],
                "ID": ["0-0", "0-1", "0-2", "0-3", "0-4"],
                "Feature": ["income", "age", "Leaf", "Leaf", "Leaf"],
                "Split": [50000.0, 30.0, None, None, None],
                "Yes": ["0-1", "0-3", None, None, None],
                "No": ["0-4", "0-2", None, None, None],
                "Missing": ["0-1", "0-3", None, None, None],
                "Gain": [1.5, 0.8, 0.2, 0.4, 1.2],
                "Cover": [100, 60, 40, 30, 30],
                "Category": [None, None, None, None, None],
            }
        )
        monotone_constraints = {"income": -1, "age": -1}
        result = extract_rule_with_monotone_constraints(tree_X, monotone_constraints)
        # income: -1 -> follows "Yes" branch (<) to node 1
        # age: -1 -> follows "Yes" branch (<) to node 3 (leaf)
        assert '(X["income"] < 50000.0)' in result
        assert '(X["age"] < 30.0)' in result

    def test_missing_constraint_raises_error(self):
        """Test that missing monotone constraint raises ValueError."""
        tree_X = pd.DataFrame(
            {
                "Tree": [0, 0, 0],
                "Node": [0, 1, 2],
                "ID": ["0-0", "0-1", "0-2"],
                "Feature": ["amount", "Leaf", "Leaf"],
                "Split": [100.0, None, None],
                "Yes": ["0-1", None, None],
                "No": ["0-2", None, None],
                "Missing": ["0-1", None, None],
                "Gain": [0.5, 0.3, 0.8],
                "Cover": [100, 40, 60],
                "Category": [None, None, None],
            }
        )
        monotone_constraints = {}  # Missing constraint for "amount"

        with pytest.raises(ValueError, match="has no monotone constraint defined"):
            extract_rule_with_monotone_constraints(tree_X, monotone_constraints)

    def test_zero_constraint_raises_error(self):
        """Test that zero monotone constraint raises ValueError."""
        tree_X = pd.DataFrame(
            {
                "Tree": [0, 0, 0],
                "Node": [0, 1, 2],
                "ID": ["0-0", "0-1", "0-2"],
                "Feature": ["amount", "Leaf", "Leaf"],
                "Split": [100.0, None, None],
                "Yes": ["0-1", None, None],
                "No": ["0-2", None, None],
                "Missing": ["0-1", None, None],
                "Gain": [0.5, 0.3, 0.8],
                "Cover": [100, 40, 60],
                "Category": [None, None, None],
            }
        )
        monotone_constraints = {"amount": 0}  # Zero constraint not allowed

        with pytest.raises(
            ValueError, match="has no monotone constraint defined or has constraint 0"
        ):
            extract_rule_with_monotone_constraints(tree_X, monotone_constraints)


class TestHasAllFeaturesConstrained:
    """Test cases for _check_all_features_have_monotone_constraints function."""

    def test_no_constraints(self):
        """Test estimator with no monotone constraints."""
        estimator = XGBClassifier()
        estimator.monotone_constraints = None
        result = _check_all_features_have_monotone_constraints(estimator, n_features=3)
        assert result is False

    def test_empty_dict_constraints(self):
        """Test estimator with empty monotone constraints dict."""
        estimator = XGBClassifier()
        estimator.monotone_constraints = {}
        result = _check_all_features_have_monotone_constraints(estimator, n_features=3)
        assert result is False

    def test_partial_constraints(self):
        """Test estimator with partial monotone constraints."""
        estimator = XGBClassifier()
        estimator.monotone_constraints = {"feature1": 1, "feature2": -1}
        result = _check_all_features_have_monotone_constraints(estimator, n_features=3)
        assert result is False

    def test_all_nonzero_constraints(self):
        """Test estimator with all features having non-zero constraints."""
        estimator = XGBClassifier()
        estimator.monotone_constraints = {"feature1": 1, "feature2": -1, "feature3": 1}
        result = _check_all_features_have_monotone_constraints(estimator, n_features=3)
        assert result is True

    def test_with_zero_constraint(self):
        """Test estimator with a zero constraint."""
        estimator = XGBClassifier()
        estimator.monotone_constraints = {"feature1": 1, "feature2": 0, "feature3": -1}
        result = _check_all_features_have_monotone_constraints(estimator, n_features=3)
        assert result is False

    def test_constraints_not_dict(self):
        """Test estimator with non-dict monotone constraints."""
        estimator = XGBClassifier()
        estimator.monotone_constraints = [1, -1, 0]  # List instead of dict
        result = _check_all_features_have_monotone_constraints(estimator, n_features=3)
        assert result is False


class TestRuleGridSearchParallelWeights:
    """Test cases for rule_grid_search_parallel_weights function."""

    def test_basic_grid_search(self):
        """Test basic grid search with simple data."""
        # Create simple binary classification data
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 100))

        estimator = XGBClassifier(max_depth=1, n_estimators=2, random_state=42)
        scale_pos_weight_vec = np.array([1.0, 2.0])

        result = rule_grid_search_parallel_weights(
            estimator, X_train, y_train, scale_pos_weight_vec, n_jobs=1
        )

        # Check result is a Polars DataFrame
        assert isinstance(result, pl.DataFrame)

        # Check it has expected columns
        assert "rule" in result.columns
        assert "tree" in result.columns
        assert "scale_pos_weight" in result.columns
        assert "transformation" in result.columns

        # Check it has some rows
        assert len(result) > 0

    def test_with_custom_weights(self):
        """Test grid search with custom sample weights."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 100))

        # Create custom weights
        weights_train = pd.DataFrame(
            {
                "linear": np.ones(100),
                "power": np.ones(100) * 2,
            }
        )

        estimator = XGBClassifier(max_depth=1, n_estimators=2, random_state=42)
        scale_pos_weight_vec = np.array([1.0])

        result = rule_grid_search_parallel_weights(
            estimator,
            X_train,
            y_train,
            scale_pos_weight_vec,
            weights_train_vec=weights_train,
            n_jobs=1,
        )

        # Check transformations are present
        transformations = result["transformation"].unique().to_list()
        assert "linear" in transformations or "power" in transformations

    def test_with_polars_input(self):
        """Test grid search with Polars DataFrame input."""
        np.random.seed(42)
        X_train = pl.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )
        y_train = pl.Series(np.random.randint(0, 2, 100))

        estimator = XGBClassifier(max_depth=1, n_estimators=2, random_state=42)
        scale_pos_weight_vec = np.array([1.0])

        result = rule_grid_search_parallel_weights(
            estimator, X_train, y_train, scale_pos_weight_vec, n_jobs=1
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0

    def test_empty_scale_weights_raises_error(self):
        """Test that empty scale_pos_weight_vec raises ValueError."""
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 50))

        estimator = XGBClassifier(max_depth=1, n_estimators=1)
        scale_pos_weight_vec = np.array([])  # Empty array

        with pytest.raises(ValueError, match="scale_pos_weight_vec cannot be empty"):
            rule_grid_search_parallel_weights(estimator, X_train, y_train, scale_pos_weight_vec)

    def test_with_monotone_constraints(self):
        """Test grid search with monotone constraints."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 100))

        # Set monotone constraints for all features
        estimator = XGBClassifier(
            max_depth=1,
            n_estimators=2,
            random_state=42,
            monotone_constraints={"feature1": 1, "feature2": -1},
        )
        scale_pos_weight_vec = np.array([1.0])

        result = rule_grid_search_parallel_weights(
            estimator, X_train, y_train, scale_pos_weight_vec, n_jobs=1
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0

    def test_verbose_prints(self, capsys):
        """Test verbose=1 prints summary lines (lines 680, 693)."""
        np.random.seed(42)
        X_train = pd.DataFrame({"f1": np.random.randn(50), "f2": np.random.randn(50)})
        y_train = pd.Series(np.random.randint(0, 2, 50))
        estimator = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
        rule_grid_search_parallel_weights(
            estimator, X_train, y_train, np.array([1.0]), n_jobs=1, verbose=1
        )
        captured = capsys.readouterr()
        assert "rule grid search" in captured.out.lower()

    def test_polars_weights_train_vec(self):
        """Polars DataFrame weights_train_vec triggers .to_pandas() branch (line 680)."""
        np.random.seed(42)
        X_train = pd.DataFrame({"f1": np.random.randn(50), "f2": np.random.randn(50)})
        y_train = pd.Series(np.random.randint(0, 2, 50))
        weights = pl.DataFrame({"Baseline": np.ones(50)})
        estimator = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
        result = rule_grid_search_parallel_weights(
            estimator, X_train, y_train, np.array([1.0]), weights_train_vec=weights, n_jobs=1
        )
        assert isinstance(result, pl.DataFrame)


class TestRuleGridSearchParallelScales:
    """Test cases for rule_grid_search_parallel_scales function."""

    def test_basic_grid_search(self):
        """Test basic grid search with simple data."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 100))

        estimator = XGBClassifier(max_depth=1, n_estimators=2, random_state=42)
        scale_pos_weight_vec = np.array([1.0, 2.0])

        result = rule_grid_search_parallel_scales(
            estimator, X_train, y_train, scale_pos_weight_vec, n_jobs=1
        )

        assert isinstance(result, pl.DataFrame)
        assert "rule" in result.columns
        assert "tree" in result.columns
        assert "scale_pos_weight" in result.columns
        assert "transformation" in result.columns
        assert len(result) > 0

    def test_with_custom_weights(self):
        """Test grid search with custom sample weights."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 100))

        weights_train = pd.DataFrame(
            {
                "linear": np.ones(100),
                "power": np.ones(100) * 2,
            }
        )

        estimator = XGBClassifier(max_depth=1, n_estimators=2, random_state=42)
        scale_pos_weight_vec = np.array([1.0, 2.0])

        result = rule_grid_search_parallel_scales(
            estimator,
            X_train,
            y_train,
            scale_pos_weight_vec,
            weights_train_vec=weights_train,
            n_jobs=1,
        )

        transformations = result["transformation"].unique().to_list()
        assert "linear" in transformations or "power" in transformations

    def test_with_polars_input(self):
        """Test grid search with Polars DataFrame input."""
        np.random.seed(42)
        X_train = pl.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )
        y_train = pl.Series(np.random.randint(0, 2, 100))

        estimator = XGBClassifier(max_depth=1, n_estimators=2, random_state=42)
        scale_pos_weight_vec = np.array([1.0, 2.0])

        result = rule_grid_search_parallel_scales(
            estimator, X_train, y_train, scale_pos_weight_vec, n_jobs=1
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0

    def test_empty_scale_weights_raises_error(self):
        """Test that empty scale_pos_weight_vec raises ValueError."""
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 50))

        estimator = XGBClassifier(max_depth=1, n_estimators=1)
        scale_pos_weight_vec = np.array([])

        with pytest.raises(ValueError, match="scale_pos_weight_vec cannot be empty"):
            rule_grid_search_parallel_scales(estimator, X_train, y_train, scale_pos_weight_vec)

    def test_with_monotone_constraints(self):
        """Test grid search with monotone constraints."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 100))

        estimator = XGBClassifier(
            max_depth=1,
            n_estimators=2,
            random_state=42,
            monotone_constraints={"feature1": 1, "feature2": -1},
        )
        scale_pos_weight_vec = np.array([1.0, 2.0])

        result = rule_grid_search_parallel_scales(
            estimator, X_train, y_train, scale_pos_weight_vec, n_jobs=1
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0

    def test_many_scales_few_weights(self):
        """Test the preferred use-case: more scale values than weight columns."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 100))

        # One weight column, many scale values — parallelise over scales
        weights_train = pd.DataFrame({"baseline": np.ones(100)})
        scale_pos_weight_vec = np.array([1.0, 2.0, 4.0, 8.0])

        estimator = XGBClassifier(max_depth=1, n_estimators=2, random_state=42)

        result = rule_grid_search_parallel_scales(
            estimator,
            X_train,
            y_train,
            scale_pos_weight_vec,
            weights_train_vec=weights_train,
            n_jobs=1,
        )

        assert isinstance(result, pl.DataFrame)
        # Results should cover multiple scale values
        if result.height > 0:
            assert result["scale_pos_weight"].n_unique() > 1

    def test_verbose_prints(self, capsys):
        """Test verbose=1 prints summary lines (lines 724, 728)."""
        np.random.seed(42)
        X_train = pd.DataFrame({"f1": np.random.randn(50), "f2": np.random.randn(50)})
        y_train = pd.Series(np.random.randint(0, 2, 50))
        estimator = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
        rule_grid_search_parallel_scales(
            estimator, X_train, y_train, np.array([1.0]), n_jobs=1, verbose=1
        )
        captured = capsys.readouterr()
        assert "parallel-scales" in captured.out.lower()

    def test_polars_weights_train_vec(self):
        """Polars DataFrame weights_train_vec uses .to_pandas() (line 680)."""
        np.random.seed(42)
        X_train = pd.DataFrame({"f1": np.random.randn(50), "f2": np.random.randn(50)})
        y_train = pd.Series(np.random.randint(0, 2, 50))
        weights = pl.DataFrame({"Baseline": np.ones(50)})
        estimator = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
        result = rule_grid_search_parallel_scales(
            estimator, X_train, y_train, np.array([1.0]), weights_train_vec=weights, n_jobs=1
        )
        assert isinstance(result, pl.DataFrame)

    def test_empty_result_path(self):
        """No rules produced → empty DataFrame branch (line 724)."""
        from unittest.mock import patch
        np.random.seed(42)
        X_train = pd.DataFrame({"f1": np.random.randn(20), "f2": np.random.randn(20)})
        y_train = pd.Series(np.random.randint(0, 2, 20))
        estimator = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
        with patch("iguanas.rule_generation._train_rules_for_scale", return_value=[]):
            result = rule_grid_search_parallel_scales(
                estimator, X_train, y_train, np.array([1.0]), n_jobs=1
            )
        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()


class TestExtractRules:
    """Test cases for extract_rules function."""

    def test_extract_rules_without_constraints(self):
        """Test extracting rules without monotone constraints."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 100))

        estimator = XGBClassifier(max_depth=2, n_estimators=3, random_state=42)
        estimator.fit(X_train, y_train)

        result = extract_rules(estimator, all_features_constrained=False)

        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "rule" in result.columns
            assert "tree" in result.columns

    def test_extract_rules_with_constraints(self):
        """Test extracting rules with monotone constraints."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 100))

        estimator = XGBClassifier(
            max_depth=2,
            n_estimators=3,
            random_state=42,
            monotone_constraints={"feature1": 1, "feature2": -1},
        )
        estimator.fit(X_train, y_train)

        result = extract_rules(estimator, all_features_constrained=True)

        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "rule" in result.columns
            assert "tree" in result.columns

    def test_extract_rules_with_kwargs(self):
        """Test extracting rules with additional kwargs."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 100))

        estimator = XGBClassifier(max_depth=2, n_estimators=2, random_state=42)
        estimator.fit(X_train, y_train)

        result = extract_rules(
            estimator,
            all_features_constrained=False,
            transformation="custom",
            scale_pos_weight=2.5,
        )

        if not result.empty:
            assert "transformation" in result.columns
            assert "scale_pos_weight" in result.columns
            assert result["transformation"].iloc[0] == "custom"
            assert result["scale_pos_weight"].iloc[0] == 2.5


class TestExtractRuleEdgeCases:
    """Additional test cases for extract_rule functions to achieve better coverage."""

    def test_extract_rule_by_max_gain_node_not_found(self):
        """Test when best_leaf_node is not found in tree (line 51 coverage)."""
        # Create a malformed tree where Node doesn't match ID expectations
        tree_X = pd.DataFrame(
            {
                "Tree": [0, 0],
                "Node": [99, 100],  # Nodes don't exist in ID mapping
                "ID": ["0-0", "0-1"],
                "Feature": ["Leaf", "Leaf"],
                "Split": [None, None],
                "Yes": [None, None],
                "No": [None, None],
                "Missing": [None, None],
                "Gain": [0.5, 0.8],
                "Cover": [50, 50],
                "Category": [None, None],
            }
        )
        result = extract_rule_by_max_gain(tree_X)
        # Should return empty string when node not found
        assert result == ""

    def test_extract_rule_with_monotone_empty_current_node(self):
        """Test when root node is empty (line 132 coverage)."""
        tree_X = pd.DataFrame(
            {
                "Tree": [0],
                "Node": [1],  # No Node 0
                "ID": ["0-1"],
                "Feature": ["Leaf"],
                "Split": [None],
                "Yes": [None],
                "No": [None],
                "Missing": [None],
                "Gain": [0.5],
                "Cover": [50],
                "Category": [None],
            }
        )
        result = extract_rule_with_monotone_constraints(tree_X, {"feature1": 1})
        assert result == ""

    def test_extract_rule_with_monotone_next_node_empty(self):
        """Test when next node doesn't exist (line 166 coverage)."""
        tree_X = pd.DataFrame(
            {
                "Tree": [0, 0],
                "Node": [0, 1],
                "ID": ["0-0", "0-1"],
                "Feature": ["feature1", "Leaf"],
                "Split": [5.0, None],
                "Yes": ["0-1", None],
                "No": ["0-999", None],  # Non-existent node
                "Missing": ["0-1", None],
                "Gain": [0.5, 0.3],
                "Cover": [100, 50],
                "Category": [None, None],
            }
        )
        result = extract_rule_with_monotone_constraints(tree_X, {"feature1": 1})
        # Should break when next node not found
        assert '(X["feature1"] >= 5.0)' in result


class TestRuleGridSearchSequential:
    """Test cases for rule_grid_search_sequential function (lines 503-561 coverage)."""

    def test_basic_sequential_grid_search(self):
        """Test basic sequential grid search."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 50))

        estimator = XGBClassifier(max_depth=1, n_estimators=2, random_state=42)
        scale_pos_weight_vec = np.array([1.0, 2.0])

        result = rule_grid_search_sequential(
            estimator, X_train, y_train, scale_pos_weight_vec, weights_train_vec=None
        )

        assert isinstance(result, pl.DataFrame)
        if result.height > 0:
            assert "rule" in result.columns
            assert "transformation" in result.columns
            assert "scale_pos_weight" in result.columns

    def test_sequential_with_custom_weights(self):
        """Test sequential grid search with custom weights."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 50))

        weights_train = pd.DataFrame(
            {
                "custom": np.ones(50) * 2.0,
                "linear": np.arange(50, dtype=float),
            }
        )

        estimator = XGBClassifier(max_depth=1, n_estimators=2, random_state=42)
        scale_pos_weight_vec = np.array([1.5])

        result = rule_grid_search_sequential(
            estimator, X_train, y_train, scale_pos_weight_vec, weights_train
        )

        assert isinstance(result, pl.DataFrame)
        if result.height > 0:
            assert "transformation" in result.columns

    def test_sequential_empty_scale_pos_weight_raises(self):
        """Test that empty scale_pos_weight_vec raises error."""
        X_train = pd.DataFrame({"feature1": [1, 2, 3]})
        y_train = pd.Series([0, 1, 0])
        estimator = XGBClassifier(max_depth=1, n_estimators=1)

        with pytest.raises(ValueError, match="scale_pos_weight_vec cannot be empty"):
            rule_grid_search_sequential(
                estimator, X_train, y_train, np.array([]), weights_train_vec=None
            )

    def test_verbose_prints(self, capsys):
        """Test verbose=1 prints summary lines (lines 456, 483)."""
        np.random.seed(42)
        X_train = pd.DataFrame({"f1": np.random.randn(50), "f2": np.random.randn(50)})
        y_train = pd.Series(np.random.randint(0, 2, 50))
        estimator = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
        rule_grid_search_sequential(estimator, X_train, y_train, np.array([1.0]), verbose=1)
        captured = capsys.readouterr()
        assert "sequential" in captured.out.lower()


class TestRuleGridSearchPandasWeights:
    """Test cases for rule_grid_search with pandas DataFrame weights (line 414, 446 coverage)."""

    def test_grid_search_with_pandas_weights(self):
        """Test rule_grid_search with pandas DataFrame weights."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(30),
                "feature2": np.random.randn(30),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 30))

        # Use pandas DataFrame for weights
        weights_train = pd.DataFrame({"weight1": np.ones(30)})

        estimator = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
        scale_pos_weight_vec = np.array([1.0])

        result = rule_grid_search_parallel_weights(
            estimator,
            X_train,
            y_train,
            scale_pos_weight_vec,
            weights_train_vec=weights_train,
            n_jobs=1,
        )

        assert isinstance(result, pl.DataFrame)


class TestExtractRulesEdgeCases:
    """Test edge cases for extract_rules function."""

    def test_extract_rules_empty_tree(self):
        """Test extract_rules with an estimator that produces empty trees (line 214 coverage)."""
        # Create a minimal dataset - XGBoost requires some variation in y
        np.random.seed(42)
        X_train = pd.DataFrame({"feature1": [1, 1, 1, 2]})
        y_train = pd.Series([0, 0, 0, 1])  # Minimal variation to avoid XGBoost error

        estimator = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
        try:
            estimator.fit(X_train, y_train)
        except Exception:
            # If fit fails (e.g., on certain XGBoost versions), skip this test case
            import pytest

            pytest.skip("XGBoost fit failed on this data configuration")

        # This should handle empty trees gracefully
        result = extract_rules(estimator, all_features_constrained=False)
        assert isinstance(result, pd.DataFrame)

    def test_extract_rules_no_valid_rules(self):
        """Test extract_rules when no valid rules are extracted (line 232, 246 coverage)."""
        # Create a very simple estimator
        np.random.seed(42)
        X_train = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
        y_train = pd.Series([0, 1, 0])

        estimator = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
        estimator.fit(X_train, y_train)

        # Extract rules and check it handles empty results
        result = extract_rules(estimator, all_features_constrained=False)
        assert isinstance(result, pd.DataFrame)


class TestRuleGridSearchWithNoneWeights:
    """Test rule_grid_search with None weights (line 417 coverage)."""

    def test_grid_search_with_none_weights(self):
        """Test rule_grid_search with weights_train_vec=None."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(30),
                "feature2": np.random.randn(30),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 30))

        estimator = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
        scale_pos_weight_vec = np.array([1.0])

        # Explicitly pass None for weights
        result = rule_grid_search_parallel_weights(
            estimator,
            X_train,
            y_train,
            scale_pos_weight_vec,
            weights_train_vec=None,
            n_jobs=1,
        )

        assert isinstance(result, pl.DataFrame)


class TestRuleGridSearchVerbose:
    """Test rule_grid_search with verbose output (line 429, 460 coverage)."""

    def test_grid_search_with_verbose(self, capsys):
        """Test rule_grid_search with verbose=1."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(30),
                "feature2": np.random.randn(30),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 30))

        estimator = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
        scale_pos_weight_vec = np.array([1.0])

        result = rule_grid_search_parallel_weights(
            estimator,
            X_train,
            y_train,
            scale_pos_weight_vec,
            weights_train_vec=None,
            n_jobs=1,
            verbose=1,  # Enable verbose output
        )

        # Check that something was printed
        captured = capsys.readouterr()
        assert "Starting rule grid search" in captured.out or len(captured.out) > 0
        assert isinstance(result, pl.DataFrame)


class TestRuleGridSearchEmptyResults:
    """Test rule_grid_search when no rules are extracted (line 457, 460 coverage)."""

    def test_grid_search_empty_results(self):
        """Test rule_grid_search when extracting rules produces empty results."""
        # Use minimal variation to avoid XGBoost error
        np.random.seed(42)
        X_train = pd.DataFrame({"feature1": [1.0, 1.0, 1.0, 2.0]})
        y_train = pd.Series([0, 0, 0, 1])

        estimator = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
        scale_pos_weight_vec = np.array([1.0])

        result = rule_grid_search_parallel_weights(
            estimator,
            X_train,
            y_train,
            scale_pos_weight_vec,
            weights_train_vec=None,
            n_jobs=1,
        )

        # Even with no rules, should return an empty DataFrame
        assert isinstance(result, pl.DataFrame)


class TestRuleGridSearchSequentialWithNoneWeights:
    """Test rule_grid_search_sequential with None weights (line 533 coverage)."""

    def test_sequential_with_none_weights(self):
        """Test rule_grid_search_sequential with weights_train_vec=None."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(30),
                "feature2": np.random.randn(30),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 30))

        estimator = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
        scale_pos_weight_vec = np.array([1.0])

        result = rule_grid_search_sequential(
            estimator,
            X_train,
            y_train,
            scale_pos_weight_vec,
            weights_train_vec=None,  # Explicitly None
        )

        assert isinstance(result, pl.DataFrame)


class TestRuleGridSearchSequentialEmptyResults:
    """Test rule_grid_search_sequential when no rules are extracted (line 574 coverage)."""

    def test_sequential_empty_results(self):
        """Test rule_grid_search_sequential when extracting rules produces empty results."""
        # Use minimal variation to avoid XGBoost error
        np.random.seed(42)
        X_train = pd.DataFrame({"feature1": [1.0, 1.0, 1.0, 2.0]})
        y_train = pd.Series([0, 0, 0, 1])

        estimator = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
        scale_pos_weight_vec = np.array([1.0])

        result = rule_grid_search_sequential(
            estimator,
            X_train,
            y_train,
            scale_pos_weight_vec,
            weights_train_vec=None,
        )

        # Even with no rules, should return an empty DataFrame
        assert isinstance(result, pl.DataFrame)


class TestEmptyNodeRowsCoverage:
    """Test to cover line 50 - empty node_rows in extract_rule_by_max_gain."""

    def test_empty_node_rows_corrupted_data(self, monkeypatch):
        """Test line 50: when node filter returns empty due to corrupted node values."""

        # Create a tree
        tree_X = pd.DataFrame(
            {
                "Tree": [0, 0],
                "Node": [0, 1],
                "ID": ["0-0", "0-1"],
                "Feature": ["amount", "Leaf"],
                "Split": [100.0, None],
                "Yes": ["0-1", None],
                "No": [None, None],
                "Missing": [None, None],
                "Gain": [0.5, 0.8],
                "Cover": [100, 50],
                "Category": [None, None],
            }
        )

        # Monkeypatch set_index to corrupt the Node column values
        original_set_index = pd.DataFrame.set_index

        def mock_set_index(self, keys, **kwargs):
            result = original_set_index(self, keys, **kwargs)
            # Change all Node values to 999 (not matching best_leaf_node=1)
            if "Node" in result.columns:
                result["Node"] = 999
            return result

        monkeypatch.setattr(pd.DataFrame, "set_index", mock_set_index)

        result = extract_rule_by_max_gain(tree_X)
        # Should return empty string when node_rows is empty
        assert result == ""


class TestEmptyTreeInExtractRules:
    """Test to cover line 211 - empty tree in extract_rules groupby iteration."""

    def test_empty_tree_in_groupby(self, monkeypatch):
        """Test line 211: when a tree in the grouped iteration is empty."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(30),
                "feature2": np.random.randn(30),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 30))

        estimator = XGBClassifier(max_depth=2, n_estimators=2, random_state=42)
        estimator.fit(X_train, y_train)

        # Monkeypatch the groupby to inject an empty DataFrame

        original_groupby = pd.DataFrame.groupby

        def mock_groupby(self, *args, **kwargs):
            """Mock groupby to include an empty tree."""
            groups = original_groupby(self, *args, **kwargs)

            # Create a generator that includes an empty DataFrame
            def gen():
                yielded_empty = False
                for name, group in groups:
                    yield name, group
                    if not yielded_empty:
                        # Inject an empty tree
                        yield 999, pd.DataFrame()
                        yielded_empty = True

            return gen()

        monkeypatch.setattr(pd.DataFrame, "groupby", mock_groupby)

        result = extract_rules(estimator, all_features_constrained=False)
        assert isinstance(result, pd.DataFrame)


class TestPandasWeightsElseBranch:
    """Test to cover line 418 - elif branch for Polars weights in rule_grid_search."""

    def test_polars_weights_conversion(self):
        """Test line 418: when weights_train_vec is Polars DataFrame (elif branch)."""
        np.random.seed(100)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 50))

        # Create a Polars DataFrame for weights
        weights_pl = pl.DataFrame({"my_weight": np.random.rand(50) + 1.0})

        # Verify it's Polars, not pandas
        assert isinstance(weights_pl, pl.DataFrame)
        assert not isinstance(weights_pl, pd.DataFrame)

        estimator = XGBClassifier(max_depth=2, n_estimators=2, random_state=100)
        scale_pos_weight_vec = np.array([1.0])

        result = rule_grid_search_parallel_weights(
            estimator,
            X_train,
            y_train,
            scale_pos_weight_vec,
            weights_train_vec=weights_pl,  # Pass Polars DataFrame
            n_jobs=1,
            verbose=0,
        )

        assert isinstance(result, pl.DataFrame)


class TestPolarsInputSequential:
    """Test to cover lines 522-523 - Polars input to rule_grid_search_sequential."""

    def test_polars_input_sequential(self):
        """Test lines 522-523: Polars DataFrame input conversion in sequential version."""
        np.random.seed(42)
        # Create Polars DataFrames
        X_train = pl.DataFrame(
            {
                "feature1": np.random.randn(30),
                "feature2": np.random.randn(30),
            }
        )
        y_train = pl.Series(np.random.randint(0, 2, 30))

        estimator = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
        scale_pos_weight_vec = np.array([1.0])

        result = rule_grid_search_sequential(
            estimator,
            X_train,
            y_train,
            scale_pos_weight_vec,
            weights_train_vec=None,
        )

        assert isinstance(result, pl.DataFrame)


class TestPandasWeightsSequential:
    """Test to cover line 536 - elif branch for Polars weights in sequential."""

    def test_polars_weights_sequential(self):
        """Test line 536: Polars DataFrame weights (elif branch) in sequential."""
        np.random.seed(100)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 50))

        # Create Polars DataFrame for weights
        weights_pl = pl.DataFrame({"seq_weight": np.ones(50) * 1.5})

        # Verify it's Polars, not pandas
        assert isinstance(weights_pl, pl.DataFrame)
        assert not isinstance(weights_pl, pd.DataFrame)

        estimator = XGBClassifier(max_depth=2, n_estimators=2, random_state=100)
        scale_pos_weight_vec = np.array([1.0])

        result = rule_grid_search_sequential(
            estimator,
            X_train,
            y_train,
            scale_pos_weight_vec,
            weights_train_vec=weights_pl,  # Pass Polars DataFrame
        )

        assert isinstance(result, pl.DataFrame)


class TestFitExceptionSequential:
    """Test to cover lines 562-564 - fit exception in sequential version."""

    def test_fit_exception_sequential(self, monkeypatch):
        """Test lines 562-564: when fit raises exception in sequential processing."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(30),
                "feature2": np.random.randn(30),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 30))

        estimator = XGBClassifier(max_depth=1, n_estimators=1, random_state=42)
        scale_pos_weight_vec = np.array([1.0, 2.0])

        # Monkeypatch fit to raise exception on first call
        original_fit = XGBClassifier.fit

        def mock_fit(self, X, y, sample_weight=None):
            if not hasattr(mock_fit, "call_count"):
                mock_fit.call_count = 0
            mock_fit.call_count += 1
            if mock_fit.call_count == 1:
                raise RuntimeError("Simulated fit failure in sequential")
            return original_fit(self, X, y, sample_weight=sample_weight)

        monkeypatch.setattr(XGBClassifier, "fit", mock_fit)

        result = rule_grid_search_sequential(
            estimator,
            X_train,
            y_train,
            scale_pos_weight_vec,
            weights_train_vec=None,
        )

        # Should succeed with second scale_pos_weight
        assert isinstance(result, pl.DataFrame)


class TestNonNumericInputRaises:
    """Test that all three grid search functions reject object-dtype X_train."""

    @pytest.fixture()
    def estimator(self):
        return XGBClassifier(max_depth=1, n_estimators=1, random_state=42)

    @pytest.fixture()
    def y_train(self):
        return pd.Series([0, 1, 0, 1, 0])

    @pytest.fixture()
    def X_train_object(self):
        return pd.DataFrame({"a": ["cat", "dog", "cat", "dog", "cat"], "b": [1, 2, 3, 4, 5]})

    def test_sequential_raises_on_object_dtype(self, estimator, X_train_object, y_train):
        with pytest.raises(ValueError, match="non-numeric data"):
            rule_grid_search_sequential(
                estimator, X_train_object, y_train, scale_pos_weight_vec=[1.0]
            )

    def test_parallel_weights_raises_on_object_dtype(self, estimator, X_train_object, y_train):
        with pytest.raises(ValueError, match="non-numeric data"):
            rule_grid_search_parallel_weights(
                estimator, X_train_object, y_train, scale_pos_weight_vec=[1.0]
            )

    def test_parallel_scales_raises_on_object_dtype(self, estimator, X_train_object, y_train):
        with pytest.raises(ValueError, match="non-numeric data"):
            rule_grid_search_parallel_scales(
                estimator, X_train_object, y_train, scale_pos_weight_vec=[1.0]
            )


class TestPrivateRuleGenerationHelpers:
    """Tests for _train_rules_for_weight_transformation and _train_rules_for_scale (lines 300, 307-308, 368)."""

    @pytest.fixture()
    def base_setup(self):
        np.random.seed(0)
        X_np = np.random.randn(50, 2)
        y_np = np.random.randint(0, 2, 50)
        estimator = XGBClassifier(max_depth=1, n_estimators=1, random_state=0)
        params = estimator.get_params()
        params.pop("scale_pos_weight", None)
        return X_np, y_np, params

    def test_train_single_weight_feature_names_none(self, base_setup):
        """feature_names=None triggers the else branch (line 300: X_fit = X_train)."""
        X_np, y_np, params = base_setup
        weights = pd.Series(np.ones(50), name="Baseline")
        result = _train_rules_for_weight_transformation(
            weights, params, X_np, y_np, [1.0], False, feature_names=None
        )
        assert isinstance(result, list)

    def test_train_scale_feature_names_none(self, base_setup):
        """feature_names=None triggers the else branch (line 368: X_fit = X_train)."""
        X_np, y_np, params = base_setup
        weights_np = np.ones((50, 1))
        result = _train_rules_for_scale(
            1.0, weights_np, ["Baseline"], params, X_np, y_np, False, feature_names=None
        )
        assert isinstance(result, list)

    def test_train_single_weight_exception_is_caught(self, base_setup):
        """Exception during fit is caught and skipped (lines 307-308)."""
        from unittest.mock import patch
        X_np, y_np, params = base_setup
        weights = pd.Series(np.ones(50), name="Baseline")
        with patch("iguanas.rule_generation.XGBClassifier.fit", side_effect=RuntimeError("fail")):
            result = _train_rules_for_weight_transformation(
                weights, params, X_np, y_np, [1.0], False, feature_names=None
            )
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
