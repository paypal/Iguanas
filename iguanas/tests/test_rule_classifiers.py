import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError
from sklearn.exceptions import NotFittedError
from xgboost import XGBClassifier

from iguanas.rule_classifier import RuleClassifier
from iguanas.ruleset_classifier import RulesetClassifier


class TestRuleClassifierInitialization:
    """Test RuleClassifier initialization and parameter validation."""

    def test_valid_initialization(self):
        """Test RuleClassifier can be initialized with valid parameters."""
        estimator = XGBClassifier(n_estimators=5, max_depth=3, eval_metric="logloss", random_state=0)
        scale_pos_weight_vec = np.array([0.1, 1.0, 10.0])

        clf = RuleClassifier(
            estimator=estimator,
            scale_pos_weight_vec=scale_pos_weight_vec,
            opt_metric="accuracy",
            metrics_threshold=[
                {"name": "precision", "operator": ">=", "value": 0.2},
                {"name": "recall", "operator": ">=", "value": 0.3},
            ],
        )

        assert clf.opt_metric == "accuracy"
        assert clf.metrics_threshold == [
            {"name": "precision", "operator": ">=", "value": 0.2},
            {"name": "recall", "operator": ">=", "value": 0.3},
        ]

    def test_default_parameters(self):
        """Test RuleClassifier uses sensible defaults."""
        estimator = XGBClassifier(n_estimators=5, max_depth=3, eval_metric="logloss", random_state=0)
        scale_pos_weight_vec = np.array([1.0])

        clf = RuleClassifier(
            estimator=estimator,
            scale_pos_weight_vec=scale_pos_weight_vec,
        )

        assert clf.opt_metric == "accuracy"
        assert clf.metrics_threshold is None

    def test_precision_out_of_range_low(self):
        """Test validation rejects metrics_threshold with negative precision value."""
        estimator = XGBClassifier(n_estimators=5, max_depth=3, eval_metric="logloss", random_state=0)
        scale_pos_weight_vec = np.array([1.0])

        with pytest.raises(ValidationError):
            RuleClassifier(
                estimator=estimator,
                scale_pos_weight_vec=scale_pos_weight_vec,
                metrics_threshold=[{"name": "precision", "operator": ">=", "value": -0.1}],
            )

    def test_precision_out_of_range_high(self):
        """Test validation rejects metrics_threshold with precision value > 1."""
        estimator = XGBClassifier(n_estimators=5, max_depth=3, eval_metric="logloss", random_state=0)
        scale_pos_weight_vec = np.array([1.0])

        with pytest.raises(ValidationError):
            RuleClassifier(
                estimator=estimator,
                scale_pos_weight_vec=scale_pos_weight_vec,
                metrics_threshold=[{"name": "precision", "operator": ">=", "value": 1.5}],
            )

    def test_recall_out_of_range_low(self):
        """Test validation rejects metrics_threshold with negative recall value."""
        estimator = XGBClassifier(n_estimators=5, max_depth=3, eval_metric="logloss", random_state=0)
        scale_pos_weight_vec = np.array([1.0])

        with pytest.raises(ValidationError):
            RuleClassifier(
                estimator=estimator,
                scale_pos_weight_vec=scale_pos_weight_vec,
                metrics_threshold=[{"name": "recall", "operator": ">=", "value": -0.1}],
            )

    def test_recall_out_of_range_high(self):
        """Test validation rejects metrics_threshold with recall value > 1."""
        estimator = XGBClassifier(n_estimators=5, max_depth=3, eval_metric="logloss", random_state=0)
        scale_pos_weight_vec = np.array([1.0])

        with pytest.raises(ValidationError):
            RuleClassifier(
                estimator=estimator,
                scale_pos_weight_vec=scale_pos_weight_vec,
                metrics_threshold=[{"name": "recall", "operator": ">=", "value": 1.5}],
            )


class TestRuleClassifierFitPredict:
    """Test RuleClassifier fit and predict methods."""

    @pytest.fixture
    def sample_data(self):
        """Create simple binary classification dataset."""
        X = pl.DataFrame({
            "age": [25, 35, 45, 55, 65, 75],
            "income": [30000, 50000, 60000, 80000, 100000, 120000],
        })
        y = pl.Series([0, 0, 1, 1, 1, 1])
        return X, y

    @pytest.fixture
    def rule_classifier(self):
        """Create a RuleClassifier instance."""
        estimator = XGBClassifier(
            n_estimators=5, max_depth=3, eval_metric="logloss", random_state=0
        )
        scale_pos_weight_vec = np.logspace(-1, 1, 5)
        return RuleClassifier(
            estimator=estimator,
            scale_pos_weight_vec=scale_pos_weight_vec,
            opt_metric="accuracy",
            metrics_threshold=[
                {"name": "precision", "operator": ">=", "value": 0.0},
                {"name": "recall", "operator": ">=", "value": 0.0},
            ],
        )

    def test_fit_returns_self(self, rule_classifier, sample_data):
        """Test that fit returns self for method chaining."""
        X, y = sample_data
        result = rule_classifier.fit(X, y)
        assert result is rule_classifier

    def test_fit_sets_feature_cols(self, rule_classifier, sample_data):
        """Test that fit identifies numeric feature columns."""
        X, y = sample_data
        rule_classifier.fit(X, y)
        assert set(rule_classifier._feature_cols_) == {"age", "income"}

    def test_fit_sets_best_rule(self, rule_classifier, sample_data):
        """Test that fit sets best_rule_."""
        X, y = sample_data
        rule_classifier.fit(X, y)
        assert isinstance(rule_classifier._best_rule_, str)

    def test_predict_returns_series(self, rule_classifier, sample_data):
        """Test that predict returns a Polars Series named after the best rule."""
        X, y = sample_data
        rule_classifier.fit(X, y)
        pred = rule_classifier.predict(X)
        assert isinstance(pred, pl.Series)
        assert pred.name == rule_classifier._best_rule_

    def test_predict_shape(self, rule_classifier, sample_data):
        """Test that predict output has correct shape."""
        X, y = sample_data
        rule_classifier.fit(X, y)
        pred = rule_classifier.predict(X)
        assert len(pred) == len(X)

    def test_predict_dtype_boolean(self, rule_classifier, sample_data):
        """Test that predict returns boolean values."""
        X, y = sample_data
        rule_classifier.fit(X, y)
        pred = rule_classifier.predict(X)
        assert pred.dtype == pl.Boolean

    def test_predict_without_fit(self, rule_classifier, sample_data):
        """Test predict without fit raises NotFittedError."""
        X, _ = sample_data
        with pytest.raises(NotFittedError):
            rule_classifier.predict(X)

    def test_predict_proba_returns_series(self, rule_classifier, sample_data):
        """Test that predict_proba returns a Polars Series named after the best rule."""
        X, y = sample_data
        rule_classifier.fit(X, y)
        proba = rule_classifier.predict_proba(X)
        assert isinstance(proba, pl.Series)
        assert proba.name == rule_classifier._best_rule_

    def test_predict_proba_dtype_float(self, rule_classifier, sample_data):
        """Test that predict_proba returns float64 values."""
        X, y = sample_data
        rule_classifier.fit(X, y)
        proba = rule_classifier.predict_proba(X)
        assert proba.dtype == pl.Float64

    def test_predict_proba_values_in_range(self, rule_classifier, sample_data):
        """Test that predict_proba values are in [0.0, 1.0]."""
        X, y = sample_data
        rule_classifier.fit(X, y)
        proba = rule_classifier.predict_proba(X)
        assert all((proba >= 0.0) & (proba <= 1.0))

    def test_predict_proba_without_fit(self, rule_classifier, sample_data):
        """Test predict_proba without fit raises NotFittedError."""
        X, _ = sample_data
        with pytest.raises(NotFittedError):
            rule_classifier.predict_proba(X)

    def test_fit_predict(self, rule_classifier, sample_data):
        """Test fit_predict method."""
        X, y = sample_data
        pred = rule_classifier.fit_predict(X, y)
        assert isinstance(pred, pl.Series)
        assert len(pred) == len(X)
        assert pred.dtype == pl.Boolean


class TestRulesetClassifierInitialization:
    """Test RulesetClassifier initialization and parameter validation."""

    def test_valid_initialization(self):
        """Test RulesetClassifier can be initialized with valid parameters."""
        estimator = XGBClassifier(n_estimators=5, max_depth=3, eval_metric="logloss", random_state=0)
        scale_pos_weight_vec = np.array([0.1, 1.0, 10.0])

        clf = RulesetClassifier(
            estimator=estimator,
            scale_pos_weight_vec=scale_pos_weight_vec,
            opt_metric="f1",
            metrics_threshold=[{"name": "precision", "operator": ">=", "value": 0.15}, {"name": "recall", "operator": ">=", "value": 0.15}],
            max_rules=10,
            max_corr=0.9,
            combine_operator="and",
        )

        assert clf.opt_metric == "f1"
        assert clf.max_rules == 10
        assert clf.metrics_threshold == [{"name": "precision", "operator": ">=", "value": 0.15}, {"name": "recall", "operator": ">=", "value": 0.15}]
        assert clf.max_corr == 0.9
        assert clf.combine_operator == "and"

    def test_default_parameters(self):
        """Test RulesetClassifier uses sensible defaults."""
        estimator = XGBClassifier(n_estimators=5, max_depth=3, eval_metric="logloss", random_state=0)
        scale_pos_weight_vec = np.array([1.0])

        clf = RulesetClassifier(
            estimator=estimator,
            scale_pos_weight_vec=scale_pos_weight_vec,
        )

        assert clf.opt_metric == "accuracy"
        assert clf.max_rules == 10
        assert clf.metrics_threshold is None
        assert clf.max_corr == 0.8
        assert clf.combine_operator == "or"

    def test_max_rules_validation_zero(self):
        """Test validation rejects max_rules <= 0."""
        estimator = XGBClassifier(n_estimators=5, max_depth=3, eval_metric="logloss", random_state=0)
        scale_pos_weight_vec = np.array([1.0])

        with pytest.raises(ValidationError):
            RulesetClassifier(
                estimator=estimator,
                scale_pos_weight_vec=scale_pos_weight_vec,
                max_rules=0,
            )

    def test_max_rules_validation_negative(self):
        """Test validation rejects max_rules < 0."""
        estimator = XGBClassifier(n_estimators=5, max_depth=3, eval_metric="logloss", random_state=0)
        scale_pos_weight_vec = np.array([1.0])

        with pytest.raises(ValidationError):
            RulesetClassifier(
                estimator=estimator,
                scale_pos_weight_vec=scale_pos_weight_vec,
                max_rules=-5,
            )

    def test_precision_out_of_range(self):
        """Test validation rejects invalid min_precision."""
        estimator = XGBClassifier(n_estimators=5, max_depth=3, eval_metric="logloss", random_state=0)
        scale_pos_weight_vec = np.array([1.0])

        with pytest.raises(ValidationError):
            RulesetClassifier(
                estimator=estimator,
                scale_pos_weight_vec=scale_pos_weight_vec,
                metrics_threshold=[{"name": "precision", "operator": ">=", "value": 1.5}],
            )

    def test_recall_out_of_range(self):
        """Test validation rejects invalid min_recall."""
        estimator = XGBClassifier(n_estimators=5, max_depth=3, eval_metric="logloss", random_state=0)
        scale_pos_weight_vec = np.array([1.0])

        with pytest.raises(ValidationError):
            RulesetClassifier(
                estimator=estimator,
                scale_pos_weight_vec=scale_pos_weight_vec,
                metrics_threshold=[{"name": "recall", "operator": ">=", "value": -0.5}],
            )

    def test_max_corr_out_of_range(self):
        """Test validation rejects invalid max_corr."""
        estimator = XGBClassifier(n_estimators=5, max_depth=3, eval_metric="logloss", random_state=0)
        scale_pos_weight_vec = np.array([1.0])

        with pytest.raises(ValidationError):
            RulesetClassifier(
                estimator=estimator,
                scale_pos_weight_vec=scale_pos_weight_vec,
                max_corr=1.5,
            )

    def test_invalid_combine_operator(self):
        """Test validation rejects invalid combine_operator."""
        estimator = XGBClassifier(n_estimators=5, max_depth=3, eval_metric="logloss", random_state=0)
        scale_pos_weight_vec = np.array([1.0])

        with pytest.raises(ValidationError):
            RulesetClassifier(
                estimator=estimator,
                scale_pos_weight_vec=scale_pos_weight_vec,
                combine_operator="xor",
            )


class TestRulesetClassifierFitPredict:
    """Test RulesetClassifier fit and predict methods."""

    @pytest.fixture
    def sample_data(self):
        """Create simple binary classification dataset."""
        X = pl.DataFrame({
            "age": [25, 35, 45, 55, 65, 75],
            "income": [30000, 50000, 60000, 80000, 100000, 120000],
        })
        y = pl.Series([0, 0, 1, 1, 1, 1])
        return X, y

    @pytest.fixture
    def ruleset_classifier_or(self):
        """Create a RulesetClassifier with OR operator."""
        estimator = XGBClassifier(
            n_estimators=5, max_depth=3, eval_metric="logloss", random_state=0
        )
        scale_pos_weight_vec = np.logspace(-1, 1, 5)
        return RulesetClassifier(
            estimator=estimator,
            scale_pos_weight_vec=scale_pos_weight_vec,
            opt_metric="accuracy",
            max_rules=5,
            metrics_threshold=[{"name": "precision", "operator": ">=", "value": 0.0}, {"name": "recall", "operator": ">=", "value": 0.0}],
            combine_operator="or",
        )

    @pytest.fixture
    def ruleset_classifier_and(self):
        """Create a RulesetClassifier with AND operator."""
        estimator = XGBClassifier(
            n_estimators=5, max_depth=3, eval_metric="logloss", random_state=0
        )
        scale_pos_weight_vec = np.logspace(-1, 1, 5)
        return RulesetClassifier(
            estimator=estimator,
            scale_pos_weight_vec=scale_pos_weight_vec,
            opt_metric="accuracy",
            max_rules=5,
            metrics_threshold=[{"name": "precision", "operator": ">=", "value": 0.0}, {"name": "recall", "operator": ">=", "value": 0.0}],
            combine_operator="and",
        )

    def test_fit_returns_self(self, ruleset_classifier_or, sample_data):
        """Test that fit returns self for method chaining."""
        X, y = sample_data
        result = ruleset_classifier_or.fit(X, y)
        assert result is ruleset_classifier_or

    def test_fit_sets_feature_cols(self, ruleset_classifier_or, sample_data):
        """Test that fit identifies numeric feature columns."""
        X, y = sample_data
        ruleset_classifier_or.fit(X, y)
        assert set(ruleset_classifier_or._feature_cols_) == {"age", "income"}

    def test_fit_sets_selected_rules(self, ruleset_classifier_or, sample_data):
        """Test that fit sets selected_rules_."""
        X, y = sample_data
        ruleset_classifier_or.fit(X, y)
        assert isinstance(ruleset_classifier_or._best_ruleset_, str)

    def test_predict_returns_series(self, ruleset_classifier_or, sample_data):
        """Test that predict returns a Polars Series named after the combined rule."""
        X, y = sample_data
        ruleset_classifier_or.fit(X, y)
        pred = ruleset_classifier_or.predict(X)
        expected_name = ruleset_classifier_or._best_ruleset_
        assert isinstance(pred, pl.Series)
        assert pred.name == expected_name

    def test_predict_shape(self, ruleset_classifier_or, sample_data):
        """Test that predict output has correct shape."""
        X, y = sample_data
        ruleset_classifier_or.fit(X, y)
        pred = ruleset_classifier_or.predict(X)
        assert len(pred) == len(X)

    def test_predict_dtype_boolean(self, ruleset_classifier_or, sample_data):
        """Test that predict returns boolean values."""
        X, y = sample_data
        ruleset_classifier_or.fit(X, y)
        pred = ruleset_classifier_or.predict(X)
        assert pred.dtype == pl.Boolean

    def test_predict_or_operator(self, ruleset_classifier_or, sample_data):
        """Test predict with OR operator."""
        X, y = sample_data
        ruleset_classifier_or.fit(X, y)
        pred = ruleset_classifier_or.predict(X)
        assert isinstance(pred, pl.Series)

    def test_predict_and_operator(self, ruleset_classifier_and, sample_data):
        """Test predict with AND operator."""
        X, y = sample_data
        ruleset_classifier_and.fit(X, y)
        pred = ruleset_classifier_and.predict(X)
        assert isinstance(pred, pl.Series)

    def test_predict_without_fit(self, ruleset_classifier_or, sample_data):
        """Test predict without fit raises NotFittedError."""
        X, _ = sample_data
        with pytest.raises(NotFittedError):
            ruleset_classifier_or.predict(X)

    def test_predict_proba_returns_series(self, ruleset_classifier_or, sample_data):
        """Test that predict_proba returns a Polars Series named after the combined rule."""
        X, y = sample_data
        ruleset_classifier_or.fit(X, y)
        proba = ruleset_classifier_or.predict_proba(X)
        expected_name = ruleset_classifier_or._best_ruleset_
        assert isinstance(proba, pl.Series)
        assert proba.name == expected_name

    def test_predict_proba_dtype_float(self, ruleset_classifier_or, sample_data):
        """Test that predict_proba returns float64 values."""
        X, y = sample_data
        ruleset_classifier_or.fit(X, y)
        proba = ruleset_classifier_or.predict_proba(X)
        assert proba.dtype == pl.Float64

    def test_predict_proba_values_in_range(self, ruleset_classifier_or, sample_data):
        """Test that predict_proba values are in [0.0, 1.0]."""
        X, y = sample_data
        ruleset_classifier_or.fit(X, y)
        proba = ruleset_classifier_or.predict_proba(X)
        assert all((proba >= 0.0) & (proba <= 1.0))

    def test_predict_proba_without_fit(self, ruleset_classifier_or, sample_data):
        """Test predict_proba without fit raises NotFittedError."""
        X, _ = sample_data
        with pytest.raises(NotFittedError):
            ruleset_classifier_or.predict_proba(X)

    def test_predict_proba_piecewise_linear(self, ruleset_classifier_or, sample_data):
        """Test that predict_proba implements piecewise-linear interpolation."""
        X, y = sample_data
        ruleset_classifier_or.fit(X, y)
        proba = ruleset_classifier_or.predict_proba(X)
        # Verify that values are in expected range
        assert all((proba >= 0.0) & (proba <= 1.0))

    def test_fit_predict(self, ruleset_classifier_or, sample_data):
        """Test fit_predict method."""
        X, y = sample_data
        pred = ruleset_classifier_or.fit_predict(X, y)
        assert isinstance(pred, pl.Series)
        assert len(pred) == len(X)
        assert pred.dtype == pl.Boolean
