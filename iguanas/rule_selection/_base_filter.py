"""
Base rule filter class. Main filter classes inherit from this one.
"""
from iguanas.rules.rules import Rules
from iguanas.utils.typing import PandasDataFrameType
from typing import List
from copy import deepcopy


class _BaseFilter:
    """
    Base rule filter class. Main filter classes inherit from this one.

    Parameters
    ----------
    rules_to_keep : List[str]
        List of rules which remain after correlated rules have been removed.
    rules : Rules, optional
        An Iguanas `Rules` object containing the rules that need to be 
        filtered. If provided, the rules within the object will be filtered. 
        Defaults to None.
    """

    def __init__(self,
                 rules_to_keep: List[str],
                 rules: Rules) -> None:

        self.rules_to_keep = rules_to_keep
        self.rules = deepcopy(rules)

    def transform(self, X_rules: PandasDataFrameType) -> PandasDataFrameType:
        """
        Applies the filter to the given dataset.

        Parameters
        ----------
        X_rules : PandasDataFrameType
            The binary columns of the rules applied to a dataset.

        Returns
        -------
        PandasDataFrameType
            The binary columns of the filtered rules.
        """

        X_rules = X_rules[self.rules_to_keep]
        if self.rules is not None:
            self.rules.filter_rules(include=self.rules_to_keep)
            self.rule_strings = self.rules.rule_strings
            self.rule_dicts = self.rules.rule_dicts
            self.rule_lambdas = self.rules.rule_lambdas
            self.lambda_kwargs = self.rules.lambda_kwargs
        return X_rules

    def fit_transform(self,
                      X_rules: PandasDataFrameType,
                      y=None,
                      sample_weight=None) -> PandasDataFrameType:
        """
        Fits then applies the filter to the given dataset.

        Parameters
        ----------
        X_rules : PandasDataFrameType
            The binary columns of the rules applied to a dataset.
        y : PandasSeriesType, optional
            The target (if relevant). Defaults to None.
        sample_weight : PandasSeriesType, optional
            Row-wise weights to apply. Defaults to None.

        Returns
        -------
        PandasDataFrameType
            The binary columns of the filtered rules.
        """

        self.fit(X_rules=X_rules, y=y, sample_weight=sample_weight)
        return self.transform(X_rules=X_rules)
