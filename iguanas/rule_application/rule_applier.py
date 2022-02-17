"""Applies rules in the standard Iguanas string format."""
import pandas as pd
import numpy as np
import iguanas.utils as utils
from typing import Dict, Union
from iguanas.utils.types import KoalasDataFrame, KoalasSeries, PandasDataFrame,\
    PandasSeries
from iguanas.utils.typing import KoalasDataFrameType, PandasDataFrameType


class RuleApplier:
    """
    Applies rules (stored in the standard Iguanas string format) to a dataset.

    Parameters
    ----------
    rule_strings : Dict[str, str]
        Set of rules defined using the standard Iguanas string format 
        (values) and their names (keys).        
    """

    def __init__(self,
                 rule_strings: Dict[str, str]):
        self.rule_strings = rule_strings

    def transform(self,
                  X: Union[PandasDataFrameType, KoalasDataFrameType]) -> Union[PandasDataFrameType, KoalasDataFrameType]:
        """
        Applies the set of rules to a dataset, `X`.

        Parameters
        ----------
        X : Union[PandasDataFrameType, KoalasDataFrameType]
            The feature set on which the rules should be applied.                    

        Returns
        -------
            Union[PandasDataFrameType, KoalasDataFrameType]
                The binary columns of the rules.
        """
        utils.check_allowed_types(X, 'X', [PandasDataFrame, KoalasDataFrame])
        X_rules = self._get_X_rules(X)
        return X_rules

    def _get_X_rules(self, X: Union[PandasDataFrameType, KoalasDataFrameType]) -> Union[
            PandasDataFrameType, KoalasDataFrameType]:
        """
        Returns the binary columns of the list of rules applied to the 
        dataset `X`.
        """

        X_rules_list = []
        for rule_name, rule_string in self.rule_strings.items():
            try:
                X_rule = eval(rule_string)
            except KeyError as e:
                raise KeyError(
                    f'Feature {e} in rule `{rule_name}` not found in `X`')
            if utils.is_type(X_rule, (PandasSeries, KoalasSeries)):
                X_rule = X_rule.fillna(False).astype(int)
                X_rule.name = rule_name
            elif isinstance(X_rule, np.ndarray):
                X_rule = X_rule.astype(int)
            X_rules_list.append(X_rule)
        if isinstance(X_rules_list[0], np.ndarray):
            X_rules = pd.DataFrame(np.asarray(X_rules_list)).T
            X_rules.columns = list(self.rule_strings.keys())
        else:
            X_rules = utils.concat(X_rules_list, axis=1, sort=False)
        X_rules.index = X.index
        return X_rules
