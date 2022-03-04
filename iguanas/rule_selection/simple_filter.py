"""Filters rules based on performance metrics."""
from iguanas.rule_selection._base_filter import _BaseFilter
from iguanas.utils.typing import PandasDataFrameType
import iguanas.utils.utils as utils
from typing import Callable


FILTERING_FUNCTIONS = {
    '>': lambda x, y: x > y,
    '>=': lambda x, y: x >= y,
    '<': lambda x, y: x <= y,
    '<=': lambda x, y: x <= y
}


class SimpleFilter(_BaseFilter):
    """
    Filter rules based on a metric.

    Parameters
    ----------
    threshold : float
        The threshold at which the rules are filtered.
    operator : str
        The operator used to filter the rules. Can be one of the following: 
        '>', '>=', '<', '<='
    metric : Callable
        The method/function which calculates the metric by which the rules are
        filtered.    
    rules : Rules, optional
        An Iguanas `Rules` object containing the rules that need to be 
        filtered. If provided, the rules within the object will be filtered. 
        Defaults to None.

    Attributes
    ----------
    rules_to_keep : List[str]
        List of rules which remain after the filter has been applied.
    rules : Rules
        The Iguanas `Rules` object containing the rules which remain after the
        filter has been applied.

    Examples
    --------
    >>> from iguanas.metrics import FScore
    >>> from iguanas.rule_selection import SimpleFilter
    >>> import pandas as pd
    >>> X_rules = pd.DataFrame({
    ...     'A': [1, 0, 1, 0],
    ...     'B': [1, 1, 1, 0]
    ... })
    >>> y = pd.Series([
    ...     1, 0, 1, 0
    ... ])
    >>> f1 = FScore(beta=1)
    >>> sf = SimpleFilter(
    ...     threshold=0.8,
    ...     operator='>',
    ...     metric=f1.fit    
    ... )
    >>> sf.fit(X_rules=X_rules, y=y)
    >>> print(sf.rules_to_keep)
    ['A']
    >>> X_rules = sf.transform(X_rules=X_rules)
    >>> print(X_rules)
       A
    0  1
    1  0
    2  1
    3  0
    """

    def __init__(self,
                 threshold: float,
                 operator: str,
                 metric: Callable,
                 rules=None):

        if operator not in ['>', '>=', '<', '<=']:
            raise ValueError("`operator` must be '>', '>=', '<' or '<='")
        self.threshold = threshold
        self.operator = operator
        self.metric = metric
        _BaseFilter.__init__(self, rules_to_keep=[], rules=rules)

    def fit(self,
            X_rules: PandasDataFrameType,
            y=None,
            sample_weight=None) -> None:
        """
        Calculates the rules remaining after the filter has been applied.

        Parameters
        ----------
        X_rules : PandasDataFrameType
            The binary columns of the rules applied to a dataset.
        y : PandasSeriesType, optional
            The binary target column.
        sample_weight : PandasSeriesType, optional
            Row-wise weights to apply. Defaults to None.
        """

        utils.check_duplicate_cols(X_rules, 'X_rules')
        metrics = self.metric(X_rules, y, sample_weight)
        filter_func = FILTERING_FUNCTIONS[self.operator]
        mask = filter_func(metrics, self.threshold)
        self.rules_to_keep = X_rules.columns[mask].tolist()
