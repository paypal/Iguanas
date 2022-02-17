"""Filters rules based on performance metrics."""
from iguanas.rule_selection._base_filter import _BaseFilter
from typing import Callable
from iguanas.utils.typing import PandasDataFrameType

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

        metrics = self.metric(X_rules, y, sample_weight)
        filter_func = FILTERING_FUNCTIONS[self.operator]
        mask = filter_func(metrics, self.threshold)
        self.rules_to_keep = X_rules.columns[mask].tolist()
