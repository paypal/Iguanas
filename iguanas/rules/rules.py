"""
Defines a rule set using one of the standard Iguanas representations. This rule
set can then be reformatted into the other standard Iguanas representations 
using the class methods.
"""

from iguanas.rules._convert_rule_dicts_to_rule_strings import _ConvertRuleDictsToRuleStrings
from iguanas.rules._convert_rule_strings_to_rule_dicts import _ConvertRuleStringsToRuleDicts
from iguanas.rules._convert_rule_dicts_to_rule_lambdas import _ConvertRuleDictsToRuleLambdas
from iguanas.rules._convert_rule_lambdas_to_rule_strings import _ConvertRuleLambdasToRuleStrings
from iguanas.rules._get_rule_attributes import _GetRuleFeatures
from iguanas.rule_application import RuleApplier
from iguanas.utils.typing import KoalasDataFrameType, PandasDataFrameType
from typing import List, Dict, Callable, Union


class Rules(RuleApplier):
    """
    Defines a set of rules using the following representations: string, 
    dictionary, lambda expression.

    One of the above formats must be provided to define the rule set. The
    rules can then be reformatted into one of the other representations.

    Parameters
    ----------
    rule_dicts : Dict[str, dict]
        Set of rules defined using thestandard Iguanas dictionary format
        (values) and their names (keys). Defaults to `None`.
    rule_strings : Dict[str, str] 
        Set of rules defined using the standard Iguanas string format 
        (values) and their names (keys). Defaults to `None`.
    rule_lambdas : Dict[str, Callable[[dict], str]] 
        Set of rules defined using the standard Iguanas lambda expression 
        format (values) and their names (keys). Must be given in 
        conjunction with either `lambda_kwargs` or `lambda_args`. Defaults 
        to `None`.
    lambda_kwargs : Dict[str, dict] 
        For each rule (keys), a dictionary containing the features used in
        the rule (keys) and the current values (values). Only populates 
        when `.as_lambda()` is used with the keyword argument 
        `with_kwargs=True`. Defaults to `None`.
    lambda_args : Dict[str, list] 
        For each rule (keys), a list containing the current values used in
        the rule. Only populates when `.as_lambda()` is used with the 
        keyword argument `with_kwargs=False`. Defaults to `None`.    

    Attributes
    ----------
    rule_dicts : Dict[str, dict]
        Set of rules defined using the standard Iguanas dictionary format 
        (values) and their names (keys).
    rule_strings : Dict[str, str]
        Set of rules defined using the standard Iguanas string format (values)
        and their names (keys).
    rule_lambdas : Dict[str, Callable[[dict], str]]
        Set of rules defined using the standard Iguanas lambda expression 
        format (values) and their names (keys).
    lambda_kwargs : Dict[str, dict]
        For each rule (keys), a dictionary containing the features used in the
        rule (keys) and the current values (values).
    lambda_args : Dict[str, list]
        For each rule (keys), a list containing the current values used in the
        rule.
    rule_features : Dict[str, set]
        For each rule (keys), a set containing the features used in the rule
        (only populates when the `.get_rules_features()` method is used).    
    """

    def __init__(self, rule_dicts=None, rule_strings=None,
                 rule_lambdas=None, lambda_kwargs=None, lambda_args=None):
        if rule_dicts is None and rule_strings is None and rule_lambdas is None:
            raise ValueError(
                '`rule_dicts`, `rule_strings` or `rule_lambdas` must be given')
        if rule_lambdas is not None and lambda_kwargs is None and \
                lambda_args is None:
            raise ValueError(
                '`lambda_kwargs` or `lambda_args` must be given when `rule_lambdas` is provided')
        self.rule_dicts = {} if rule_dicts is None else rule_dicts
        self.rule_strings = {} if rule_strings is None else rule_strings
        self.rule_lambdas = {} if rule_lambdas is None else rule_lambdas
        self.lambda_kwargs = {} if lambda_kwargs is None else lambda_kwargs
        self.lambda_args = {} if lambda_args is None else lambda_args
        self.rule_features = {}
        RuleApplier.__init__(
            self, rule_strings=self.rule_strings,
        )

    def __add__(self, other):

        def _exception_if_similar_keys(dict1: dict, dict2: dict,
                                       rule_format: str) -> None:
            """
            Raises exception if there are similar keys in `dict1` and `dict2`.
            """
            dict1_keys = set(dict1.keys())
            dict2_keys = set(dict2.keys())
            sim_keys = dict1_keys.intersection(dict2_keys)
            if sim_keys:
                raise ValueError(
                    f"""Attempting to add rule sets with similar keys in `{rule_format}`. Similar keys are '{"', '".join(sim_keys)}'.""")

        # Raise exception if there are similar keys in any of the rule formats
        check_list = [
            (self.rule_dicts, other.rule_dicts, 'rule_dicts'),
            (self.rule_strings, other.rule_strings, 'rule_strings'),
            (self.rule_lambdas, other.rule_lambdas, 'rule_lambdas'),
            (self.lambda_kwargs, other.lambda_kwargs, 'lambda_kwargs'),
            (self.lambda_args, other.lambda_args, 'lambda_args')
        ]
        for dict1, dict2, rule_format in check_list:
            _exception_if_similar_keys(dict1, dict2, rule_format)
        # Combine rule formats
        rule_dicts = {**self.rule_dicts, **other.rule_dicts}
        rule_strings = {**self.rule_strings, **other.rule_strings}
        rule_lambdas = {**self.rule_lambdas, **other.rule_lambdas}
        lambda_kwargs = {**self.lambda_kwargs, **other.lambda_kwargs}
        lambda_args = {**self.lambda_args, **other.lambda_args}
        # Return combined rule set
        return Rules(
            rule_dicts=rule_dicts, rule_strings=rule_strings,
            rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs,
            lambda_args=lambda_args
        )

    def __radd__(self, other):
        return self

    def __repr__(self):
        rules_lens = [
            len(self.rule_dicts),
            len(self.rule_strings),
            len(self.rule_lambdas)
        ]
        return f'Rules object containing {max(rules_lens)} rules'

    def as_rule_dicts(self) -> Dict[str, dict]:
        """
        Converts rules into the standard Iguanas dictionary format.

        Returns
        -------
        Dict[str, dict]
            Rules in the standard Iguanas dictionary format.
        """

        if self.rule_strings != {}:
            self._rule_strings_to_rule_dicts()
        elif self.rule_lambdas != {}:
            self._rule_lambdas_to_rule_strings()
            self._rule_strings_to_rule_dicts()
        return self.rule_dicts

    def as_rule_strings(self, as_numpy: bool) -> Dict[str, str]:
        """
        Converts rules into the standard Iguanas string format.

        Parameters
        ----------
        as_numpy : bool
            If `True`, the conditions in the string format will uses Numpy
            rather than Pandas. These rules are generally evaluated more 
            quickly on larger dataset stored as Pandas DataFrames.

        Returns
        -------
        Dict[str, str]
            Rules in the standard Iguanas string format.
        """

        if self.rule_strings != {}:
            self._rule_strings_to_rule_dicts()
        elif self.rule_lambdas != {}:
            self._rule_lambdas_to_rule_strings()
            self._rule_strings_to_rule_dicts()
        self._rule_dicts_to_rule_strings(as_numpy=as_numpy)
        return self.rule_strings

    def as_rule_lambdas(self, as_numpy: bool,
                        with_kwargs: bool) -> Dict[str, Callable[[dict], str]]:
        """
        Converts rules into the standard Iguanas lambda expression format.

        Parameters
        ----------
        as_numpy : bool
            If `True`, the conditions in the string format will uses Numpy 
            rather than Pandas. These rules are generally evaluated more 
            quickly on larger dataset stored as Pandas DataFrames.
        with_kwargs : bool
            If `True`, the string in the lambda expression is created such that
            the inputs are keyword arguments. If `False`, the inputs are
            positional arguments.

        Returns
        -------
        Dict[str, Callable[[dict], str]]
            Rules in the standard Iguanas lambda expression format.
        """

        if self.rule_lambdas != {}:
            self._rule_lambdas_to_rule_strings()
            self._rule_strings_to_rule_dicts()
        elif self.rule_strings != {}:
            self._rule_strings_to_rule_dicts()
        self._rule_dicts_to_rule_lambdas(
            as_numpy=as_numpy, with_kwargs=with_kwargs)
        return self.rule_lambdas

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

        if self.rule_strings == {}:
            self.rule_strings = self.as_rule_strings(as_numpy=False)
        X_rules = RuleApplier.transform(self, X=X)
        return X_rules

    def filter_rules(self, include=None, exclude=None) -> None:
        """
        Filters the rules by their names.

        Parameters
        ----------
        include : List[str], optional
            The list of rule names to keep. Defaults to `None`.
        exclude : List[str], optional
            The list of rule names to drop. Defaults to `None`.

        Raises
        ------
        Exception
            `include` and `exclude` cannot contain similar values.
        """

        if include is not None and exclude is not None:
            intersected = set.intersection(set(include), set(exclude))
            if len(intersected) > 0:
                raise Exception(
                    '`include` and `exclude` contain similar values')
        for d in [self.rule_strings, self.rule_dicts, self.rule_lambdas]:
            if d != {}:
                rule_names = list(d.keys())
                break
        for rule_name in rule_names:
            if (include is not None and rule_name not in include) or \
                    (exclude is not None and rule_name in exclude):
                self.rule_strings.pop(rule_name, None)
                self.rule_dicts.pop(rule_name, None)
                self.rule_lambdas.pop(rule_name, None)
                self.lambda_kwargs.pop(rule_name, None)
                self.lambda_args.pop(rule_name, None)
                self.rule_features.pop(rule_name, None)

    def get_rule_features(self) -> Dict[str, set]:
        """
        Returns the set of unique features present in each rule.

        Returns
        -------
        Dict[str, set]
            Set of unique features (values) in each rule (keys).
        """

        if self.rule_dicts == {}:
            _ = self.as_rule_dicts()
        grf = _GetRuleFeatures(rule_dicts=self.rule_dicts)
        self.rule_feature_set = grf.get()
        return self.rule_feature_set

    def _rule_dicts_to_rule_strings(self, as_numpy: bool) -> None:
        """
        Converts the rules (each being represented in the standard Iguanas
        dictionary format) into the standard Iguanas string format.
        """

        if self.rule_dicts == {}:
            raise ValueError('`rule_dicts` must be given')
        converter = _ConvertRuleDictsToRuleStrings(
            rule_dicts=self.rule_dicts)
        self.rule_strings = converter.convert(as_numpy=as_numpy)

    def _rule_strings_to_rule_dicts(self) -> None:
        """
        Converts the rules (each being represented in the standard Iguanas string
        format) into the standard Iguanas dictionary format.
        """

        if self.rule_strings == {}:
            raise ValueError('`rule_strings` must be given')
        converter = _ConvertRuleStringsToRuleDicts(
            rule_strings=self.rule_strings)
        self.rule_dicts = converter.convert()

    def _rule_dicts_to_rule_lambdas(self, as_numpy: bool,
                                    with_kwargs: bool) -> None:
        """
        Converts the rules (each being represented in the standard Iguanas
        dictionary format) into the standard Iguanas lambda expression format.
        This is useful for optimising rules.
        """
        if self.rule_dicts == {}:
            raise ValueError('`rule_dicts` must be given')
        converter = _ConvertRuleDictsToRuleLambdas(rule_dicts=self.rule_dicts)
        self.rule_lambdas = converter.convert(
            as_numpy=as_numpy, with_kwargs=with_kwargs)
        self.lambda_kwargs = converter.lambda_kwargs
        self.lambda_args = converter.lambda_args
        self.rule_features = converter.rule_features

    def _rule_lambdas_to_rule_strings(self) -> None:
        """
        Converts the rules (each being represented in the standard Iguanas lambda
        expression format) into the standard Iguanas string format.
        """
        if self.rule_lambdas == {}:
            raise ValueError('`rule_lambdas` must be given')
        if self.lambda_kwargs == {} and self.lambda_args == {}:
            raise ValueError('`lambda_kwargs` or `lambda_args` must be given')
        converter = _ConvertRuleLambdasToRuleStrings(
            rule_lambdas=self.rule_lambdas, lambda_kwargs=self.lambda_kwargs,
            lambda_args=self.lambda_args)
        self.rule_strings = converter.convert()
