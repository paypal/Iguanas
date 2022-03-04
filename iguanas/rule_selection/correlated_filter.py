"""Filters correlated rules."""
from iguanas.rule_selection._base_filter import _BaseFilter
from iguanas.correlation_reduction import AgglomerativeClusteringReducer
from iguanas.utils.typing import PandasDataFrameType
import iguanas.utils.utils as utils


class CorrelatedFilter(_BaseFilter):
    """
    Filters correlated rules based on a correlation reduction class (see the
    `correlation_reduction` sub-package).

    Parameters
    ----------
    correlation_reduction_class : AgglomerativeClusteringReducer
        Instatiated class from the `correlation_reduction` sub-package.    
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
    >>> from iguanas.correlation_reduction import AgglomerativeClusteringReducer
    >>> from iguanas.metrics import JaccardSimilarity, FScore
    >>> from iguanas.rule_selection import CorrelatedFilter
    >>> import pandas as pd
    >>> X_rules = pd.DataFrame({
    ...     'A': [1, 0, 1, 0],
    ...     'B': [1, 1, 1, 0]
    ... })
    >>> y = pd.Series([
    ...     1, 0, 1, 0
    ... ])
    >>> js = JaccardSimilarity()
    >>> f1 = FScore(beta=1)
    >>> cf = CorrelatedFilter(
    ...     correlation_reduction_class=AgglomerativeClusteringReducer(
    ...         threshold=0.6,
    ...         strategy='bottom_up',
    ...         similarity_function=js.fit,
    ...         metric=f1.fit
    ...     )
    ... )
    >>> cf.fit(X_rules=X_rules, y=y)
    >>> print(cf.rules_to_keep)
    ['A']
    >>> X_rules = cf.transform(X_rules=X_rules)
    >>> print(X_rules)
       A
    0  1
    1  0
    2  1
    3  0
    """

    def __init__(self,
                 correlation_reduction_class: AgglomerativeClusteringReducer,
                 rules=None):

        self.correlation_reduction_class = correlation_reduction_class
        _BaseFilter.__init__(self, rules_to_keep=[], rules=rules)

    def fit(self,
            X_rules: PandasDataFrameType,
            y=None,
            sample_weight=None) -> None:
        """
        Calculates the uncorrelated rules(using the correlation reduction
        class).

        Parameters
        ----------
        X_rules : PandasDataFrameType
            The binary columns of the rules applied to a dataset.
        y : PandasSeriesType
            Target (if available). Only used in the method/function passed to 
            the `metric` parameter in the `correlation_reduction_class`.
        sample_weight : None
            Row-wise weights to apply (if available). Only used in the 
            method/function passed to the `metric` parameter in the 
            `correlation_reduction_class`.  
        """

        utils.check_duplicate_cols(X_rules, 'X_rules')
        self.correlation_reduction_class.fit(
            X=X_rules, y=y, sample_weight=sample_weight
        )
        self.rules_to_keep = self.correlation_reduction_class.columns_to_keep
