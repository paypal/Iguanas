"""Generates non-scaled scores for each rule in a set."""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import math
from typing import Callable
from iguanas.utils.typing import PandasSeriesType, PandasDataFrameType


class PerformanceScorer:
    """
    Generates rule scores from a performance function.    

    Parameters
    ----------
    metric : Callable
        The method/function to calculate the metric used to score the rules. 
        Should have parameters `y_true`, `y_pred` and `sample_weight`.    

    Examples
    --------
    >>> from iguanas.rule_scoring import PerformanceScorer
    >>> import pandas as pd
    >>> from iguanas.metrics import FScore
    >>> f1 = FScore(beta=1)
    >>> X_rules = pd.DataFrame({
    ...     'A': [1, 0, 1, 0],
    ...     'B': [1, 1, 1, 0]
    ... })
    >>> y = pd.Series([
    ...     1, 0, 1, 0
    ... ])
    >>> ps = PerformanceScorer(metric=f1.fit)
    >>> rule_scores = ps.fit(X_rules=X_rules, y=y)
    >>> print(rule_scores)    
    A    1.0
    B    0.8
    dtype: float64
    """

    def __init__(self, metric: Callable):
        self.metric = metric

    def fit(self, X_rules: PandasDataFrameType, y: PandasSeriesType,
            sample_weight=None) -> PandasDataFrameType:
        """
        Generates rule scores from a weighting function.

        Parameters
        ----------
        X_rules : PandasDataFrameType
            The binary columns associated with the rules.
        y : PandasPandasSeriesType
            The binary target column.
        sample_weight : PandasPandasSeriesType, optional
            Row-wise sample_weights to apply. Defaults to None.

        Returns
        -------
        PandasDataFrameType
            The rule scores applied to the dataset.
        """

        scores = self.metric(
            y_true=y, y_preds=X_rules, sample_weight=sample_weight
        )
        rule_scores = pd.Series(scores, X_rules.columns)

        return rule_scores


class LogRegScorer:
    """
    Generates rule scores from the exponentiated coefficients of a trained 
    Logistic Regression model.

    Parameters
    ----------
    *args : tuple, optional
        Positional arguments associated with Sklearn's `LogisisticRegression()`
        class constructor.            
    **kwargs: dict, optional
        Keyword arguments associated with Sklearn's `LogisisticRegression()` 
        class constructor.

    Examples
    --------
    >>> from iguanas.rule_scoring import LogRegScorer
    >>> import pandas as pd    
    >>> X_rules = pd.DataFrame({
    ...     'A': [1, 0, 1, 0],
    ...     'B': [1, 1, 1, 0]
    ... })
    >>> y = pd.Series([
    ...     1, 0, 1, 0
    ... ])
    >>> lrs = LogRegScorer()
    >>> rule_scores = lrs.fit(X_rules=X_rules, y=y)
    >>> print(rule_scores)
    A    2.158956
    B    1.410809
    dtype: float64
    """

    def __init__(self, *args, **kwargs):

        self.args = args
        self.kwargs = kwargs

    def fit(self, X_rules: PandasDataFrameType, y: PandasSeriesType,
            sample_weight=None) -> PandasDataFrameType:
        """
        Generates rule scores from the coefficients of a trained Logistic 
        Regression model.

        Parameters
        ----------
        X_rules : PandasDataFrameType
            The binary columns associated with the rules.
        y : PandasPandasSeriesType
            The binary target column.
        sample_weight : PandasPandasSeriesType, optional
            Row-wise sample_weights to apply. Defaults to None.

        Returns
        -------
        PandasDataFrameType
            The rule scores applied to the dataset.
        """

        lr = LogisticRegression(*self.args, **self.kwargs, random_state=0)
        lr.fit(X=X_rules, y=y, sample_weight=sample_weight)
        scores = np.array(list(map(math.exp, lr.coef_[0])))
        rule_scores = pd.Series(scores, X_rules.columns)

        return rule_scores


class RandomForestScorer:
    """
    Generates rule scores from the feature importance of a trained Random 
    Forest model.

    Parameters
    ----------
    *args : tuple, optional
        Positional arguments associated with Sklearn's 
        `RandomForestClassifier()` class constructor.            
    **kwargs : tuple, optional
        Keyword arguments associated with Sklearn's 
        `RandomForestClassifier()` class constructor.

    Examples
    --------
    >>> from iguanas.rule_scoring import RandomForestScorer
    >>> import pandas as pd
    >>> X_rules = pd.DataFrame({
    ...     'A': [1, 0, 1, 0],
    ...     'B': [1, 1, 1, 0]
    ... })
    >>> y = pd.Series([
    ...     1, 0, 1, 0
    ... ])
    >>> rfs = RandomForestScorer()
    >>> rule_scores = rfs.fit(X_rules=X_rules, y=y)
    >>> print(rule_scores)
    A    0.773762
    B    0.226238
    dtype: float64
    """

    def __init__(self, *args, **kwargs):

        self.args = args
        self.kwargs = kwargs

    def fit(self, X_rules: PandasDataFrameType, y: PandasSeriesType,
            sample_weight=None) -> PandasDataFrameType:
        """
        Generates rule scores from the feature importance of a trained Random
        Forest model.

        Parameters
        ----------
        X_rules : PandasDataFrameType
            The binary columns associated with the rules.
        y : PandasPandasSeriesType
            The binary target column.
        sample_weight : PandasPandasSeriesType, optional
            Row-wise sample_weights to apply. Defaults to None.

        Returns
        -------
        PandasDataFrameType
            The rule scores applied to the dataset.
        """

        rf = RandomForestClassifier(*self.args, **self.kwargs, random_state=0)
        rf.fit(X=X_rules, y=y, sample_weight=sample_weight)
        scores = rf.feature_importances_
        rule_scores = pd.Series(scores, X_rules.columns)

        return rule_scores
