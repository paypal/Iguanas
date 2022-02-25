"""Class for converting between dict and string representations of rules."""
from typing import Callable, Dict, Tuple, List, Union


class _ConvertRuleDictsToRuleStrings:
    """
    Converts a set of rules (each being represented in the standard Iguanas 
    dictionary format) into the standard Iguanas string format.        

    Parameters
    ----------
    rule_dicts : Dict[str, dict] 
        Set of rules defined using the standard Iguanas dictionary format 
        (values) and their names (keys).

    Attributes
    ----------
    rule_strings : Dict[str, str] 
        Set of rules defined using the standard Iguanas string format (values)
        and their names (keys).
    """

    def __init__(self, rule_dicts: Dict[str, dict]):
        self.rule_dicts = rule_dicts
        self.rule_strings = {}
        self._lambda_kwargs = {}
        self._lambda_args = []
        self._rule_features = []
        self._lambda_kwarg_suffix = 0
        self._condition_lookup = {
            'AND': '&',
            'OR': '|'
        }
        self._operator_lookup = {
            'begins_with': {
                'operator': 'startswith',
                'is_not': False,
                'conversion_function': self._create_condition_string_for_str_based_operators
            },
            'ends_with': {
                'operator': 'endswith',
                'is_not': False,
                'conversion_function': self._create_condition_string_for_str_based_operators
            },
            'contains': {
                'operator': 'contains',
                'is_not': False,
                'conversion_function': self._create_condition_string_for_contains_operator
            },
            'is_null': {
                'operator': 'isna',
                'is_not': False,
                'conversion_function': self._create_condition_string_for_is_null_operator
            },
            'in': {
                'operator': 'isin',
                'is_not': False,
                'conversion_function': self._create_condition_string_for_is_in_operator
            },
            'not_begins_with': {
                'operator': 'startswith',
                'is_not': True,
                'conversion_function': self._create_condition_string_for_str_based_operators
            },
            'not_ends_with': {
                'operator': 'endswith',
                'is_not': True,
                'conversion_function': self._create_condition_string_for_str_based_operators
            },
            'not_contains': {
                'operator': 'contains',
                'is_not': True,
                'conversion_function': self._create_condition_string_for_contains_operator
            },
            'is_not_null': {
                'operator': 'isna',
                'is_not': True,
                'conversion_function': self._create_condition_string_for_is_null_operator
            },
            'not_in': {
                'operator': 'isin',
                'is_not': True,
                'conversion_function': self._create_condition_string_for_is_in_operator
            },
            'equal': {
                'operator': '==',
                'is_not': None,
                'conversion_function': self._create_condition_string_for_standard_operators
            },
            'not_equal': {
                'operator': '!=',
                'is_not': None,
                'conversion_function': self._create_condition_string_for_standard_operators
            },
            'greater': {
                'operator': '>',
                'is_not': None,
                'conversion_function': self._create_condition_string_for_standard_operators
            },
            'greater_or_equal': {
                'operator': '>=',
                'is_not': None,
                'conversion_function': self._create_condition_string_for_standard_operators
            },
            'less': {
                'operator': '<',
                'is_not': None,
                'conversion_function': self._create_condition_string_for_standard_operators
            },
            'less_or_equal': {
                'operator': '<=',
                'is_not': None,
                'conversion_function': self._create_condition_string_for_standard_operators
            },
            'is_empty': {
                'operator': "fillna('')==",
                'is_not': None,
                'conversion_function': self._create_condition_string_for_is_empty_operator
            },
            'is_not_empty': {
                'operator': "fillna('')!=",
                'is_not': None,
                'conversion_function': self._create_condition_string_for_is_empty_operator
            },
            'equal_field': {
                'operator': '==',
                'is_not': None,
                'conversion_function': self._create_condition_string_for_field_level_comparison
            },
            'not_equal_field': {
                'operator': '!=',
                'is_not': None,
                'conversion_function': self._create_condition_string_for_field_level_comparison
            },
            'greater_or_equal_field': {
                'operator': '>=',
                'is_not': None,
                'conversion_function': self._create_condition_string_for_field_level_comparison
            },
            'greater_field': {
                'operator': '>',
                'is_not': None,
                'conversion_function': self._create_condition_string_for_field_level_comparison
            },
            'less_or_equal_field': {
                'operator': '<=',
                'is_not': None,
                'conversion_function': self._create_condition_string_for_field_level_comparison
            },
            'less_field': {
                'operator': '<',
                'is_not': None,
                'conversion_function': self._create_condition_string_for_field_level_comparison
            }
        }

    def convert(self, as_numpy: bool) -> Dict[str, str]:
        """
        Converts a set of rules (each being represented in the standard Iguanas 
        dictionary format) into the standard Iguanas string format.        

        Parameters
        ----------
        as_numpy : bool
            If True, the conditions in the string format will uses Numpy rather
            than Pandas. These rules are generally evaluated more quickly on 
            larger dataset stored as Pandas DataFrames. 

        Returns
        -------
        Dict[str, str]
            Set of rules defined using the standard Iguanas string format.
        """

        for rule_name, rule_dict in self.rule_dicts.items():
            rule_string = self._convert_rule(
                rule_dict=rule_dict, as_numpy=as_numpy)
            self.rule_strings[rule_name] = rule_string
        return self.rule_strings

    def _convert_to_lambda(self, as_numpy: bool,
                           with_kwargs: bool) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Converts a set of rules (each being represented in the standard Iguanas 
        dictionary format) into the standard Iguanas lambda expression format.

        Returns
        -------
        Tuple[Dict, Dict, Dict, Dict]
            The `rule_lambdas`, `lambda_kwargs`, `lambda_args` and `rule_features`.
        """

        def _make_lambda(rule_lambda_str: str,
                         with_kwargs: bool) -> Callable[[Dict], str]:
            """Generates lambda expression for the given rule string"""

            if with_kwargs:
                rule_lambda = lambda **kwargs: rule_lambda_str.format(**kwargs)
            else:
                rule_lambda = lambda *args: rule_lambda_str.format(*args)
            return rule_lambda

        rule_lambdas = {}
        lambda_kwargs = {}
        lambda_args = {}
        rule_features = {}
        for rule_name, rule_dict in self.rule_dicts.items():
            rule_lambda_str = self._recurse_convert_rule_dict_conditions(
                rule_dict=rule_dict, as_numpy=as_numpy, as_lambda=True, with_kwargs=with_kwargs)
            rule_lambda_str = rule_lambda_str[1:-1]
            rule_lambda = _make_lambda(rule_lambda_str=rule_lambda_str,
                                       with_kwargs=with_kwargs)
            rule_lambdas[rule_name] = rule_lambda
            lambda_kwargs[rule_name] = self._lambda_kwargs.copy()
            lambda_args[rule_name] = self._lambda_args.copy()
            rule_features[rule_name] = self._rule_features.copy()
            self._lambda_kwargs, self._lambda_args, self._rule_features = {}, [], []
        return rule_lambdas, lambda_kwargs, lambda_args, rule_features

    def _convert_rule(self, rule_dict: Dict[str, Dict], as_numpy: bool) -> str:
        """
        Converts a rule stored in the standard Iguanas dictionary format into the
        standard Iguanas string format.
        """

        rule_string = self._recurse_convert_rule_dict_conditions(
            rule_dict=rule_dict, as_numpy=as_numpy, as_lambda=False, with_kwargs=False)
        rule_string = rule_string[1:-1]
        return rule_string

    def _recurse_convert_rule_dict_conditions(self, rule_dict: Dict[str, Dict],
                                              as_numpy: bool, as_lambda: bool,
                                              with_kwargs: bool) -> str:
        """Recursively converts a rule dictionary to rule string"""

        str_condition = self._condition_lookup[rule_dict['condition']]
        rules_list = rule_dict['rules']
        str_rule_list = self._convert_rule_dict_conditions(
            rules_list=rules_list, as_numpy=as_numpy, as_lambda=as_lambda,
            with_kwargs=with_kwargs)
        rule_string = f'({str_condition.join(str_rule_list)})'
        return rule_string

    def _convert_rule_dict_conditions(self, rules_list: List[Dict],
                                      as_numpy: bool, as_lambda: bool,
                                      with_kwargs: bool) -> List[str]:
        """
        Loops through list of dictionary conditions - if it is a single 
        condition, converts that to the string format; if not, then it goes to 
        the next level down in the rule.
        """

        str_rule_list = []
        for rule in rules_list:
            rule_keys = list(rule.keys())
            rule_keys.sort()
            if rule_keys == ['condition', 'rules']:
                rule_string = self._recurse_convert_rule_dict_conditions(
                    rule_dict=rule, as_numpy=as_numpy, as_lambda=as_lambda,
                    with_kwargs=with_kwargs)
                str_rule_list.append(rule_string)
            else:
                rule_string = self._create_condition_string_from_dict(
                    condition_dict=rule, as_numpy=as_numpy, as_lambda=as_lambda,
                    with_kwargs=with_kwargs)
                str_rule_list.append(rule_string)
        return str_rule_list

    def _create_condition_string_from_dict(self, condition_dict: Dict[str, Union[str, float]],
                                           as_numpy: bool, as_lambda: bool,
                                           with_kwargs: bool) -> str:
        """Converts a single condition dictionary to the string format"""

        field = condition_dict['field']
        operator = condition_dict['operator']
        value = condition_dict['value']
        # Only add numeric values as formattable inputs to lambda expressions
        # and isinstance(value, (float, int)):
        if as_lambda and type(value) in (float, int):
            str_value = self._parse_value_lambda(
                condition_dict=condition_dict, field=field, with_kwargs=with_kwargs)
        else:
            str_value = self._parse_value_string(condition_dict=condition_dict)
        if operator in self._operator_lookup.keys():
            str_operator = self._operator_lookup[operator]['operator']
            is_not = self._operator_lookup[operator]['is_not']
            conversion_function = self._operator_lookup[operator]['conversion_function']
            condition_string = conversion_function(
                field, str_value, str_operator, as_numpy, is_not)
        else:
            raise Exception(
                'Operator not currently supported in Iguanas. Rule cannot be parsed.')
        return condition_string

    def _parse_value_lambda(self, condition_dict: Dict[str, Union[str, float]], field: str,
                            with_kwargs: bool) -> str:
        """
        Parses the values from the dictionary (depending on whether the end 
        format is the standard string format or the standard lambda expression 
        format).
        """
        actual_value = condition_dict['value']
        if actual_value is None:
            return None
        if with_kwargs:
            features = list(self._lambda_kwargs.keys())
            if field in features:
                field_tag = f'{field}%{self._lambda_kwarg_suffix}'
                self._lambda_kwarg_suffix += 1
            else:
                field_tag = field
            self._lambda_kwargs[field_tag] = actual_value
            value = '{' + f'{field_tag}' + '}'
        else:
            value = '{}'
            self._rule_features.append(field)
            self._lambda_args.append(actual_value)
        if isinstance(actual_value, str):
            value = f"'{value}'"
        return value

    @staticmethod
    def _parse_value_string(condition_dict: Dict[str, Union[str, float]]) -> Union[str, float]:
        """
        Parses the values from the dictionary (depending on whether the end 
        format is the standard string format or the standard lambda expression 
        format).
        """

        value = condition_dict['value']
        if isinstance(value, str):
            value = f"'{value}'"
        return value

    @staticmethod
    def _create_condition_string_for_str_based_operators(field: str,
                                                         value: str,
                                                         operator: str,
                                                         _unused: None,
                                                         is_not: bool) -> str:
        """
        Returns the string representation of a condition, given the field, 
        operator and value, for Pandas-based string operators.
        """

        prefix = '~' if is_not else ''
        condition_string = f"({prefix}X['{field}'].str.{operator}({value}, na=False))"
        return condition_string

    @staticmethod
    def _create_condition_string_for_contains_operator(field: str,
                                                       value: str,
                                                       operator: str,
                                                       _unused: None,
                                                       is_not: bool) -> str:
        """
        Returns the string representation of a condition, given the field, 
        operator and value, for the Pandas-based contain operator.
        """

        prefix = '~' if is_not else ''
        condition_string = f"({prefix}X['{field}'].str.{operator}({value}, na=False, regex=False))"
        return condition_string

    @staticmethod
    def _create_condition_string_for_is_in_operator(field: str,
                                                    value: str,
                                                    operator: str,
                                                    _unused: None,
                                                    is_not: bool) -> str:
        """
        Returns the string representation of a condition, given the field, 
        operator and value, for the Pandas `.isin()` operator.
        """

        prefix = '~' if is_not else ''
        condition_string = f"({prefix}X['{field}'].{operator}({value}))"
        return condition_string

    @staticmethod
    def _create_condition_string_for_is_null_operator(field: str,
                                                      _unused: str,
                                                      operator: str,
                                                      as_numpy: bool,
                                                      is_not: bool) -> str:
        """
        Returns the string representation of a condition, given the field, 
        operator and value, for the Pandas `.isna()` operator.
        """

        prefix = '~' if is_not else ''
        if as_numpy:
            condition_string = f"({prefix}pd.isna(X['{field}'].to_numpy(na_value=np.nan)))"
        else:
            condition_string = f"({prefix}X['{field}'].isna())"
        return condition_string

    @staticmethod
    def _create_condition_string_for_standard_operators(field: str,
                                                        value: str,
                                                        operator: str,
                                                        as_numpy: bool,
                                                        _unused: None) -> str:
        """
        Returns the string representation of a condition, given the field, 
        operator and value, for standard operators.
        """

        if as_numpy:
            condition_string = f"(X['{field}'].to_numpy(na_value=np.nan){operator}{value})"
        else:
            condition_string = f"(X['{field}']{operator}{value})"
        return condition_string

    @staticmethod
    def _create_condition_string_for_field_level_comparison(field: str,
                                                            value: str,
                                                            operator: str,
                                                            as_numpy: bool,
                                                            _unused: None) -> str:
        """
        Returns the string representation of a condition, given the field,
        operator and value, for field level comparisons.
        """

        if as_numpy:
            condition_string = f"(X['{field}'].to_numpy(na_value=np.nan){operator}X[{value}].to_numpy(na_value=np.nan))"
        else:
            condition_string = f"(X['{field}']{operator}X[{value}])"
        return condition_string

    @staticmethod
    def _create_condition_string_for_is_empty_operator(field: str,
                                                       _unused: None,
                                                       operator: str,
                                                       _unused1: None,
                                                       is_not: bool) -> str:
        """
        Returns the string representation of a condition, given the field, 
        operator and value, for the pseudo-Pandas isEmpty() operator.
        """

        prefix = '~' if is_not else ''
        condition_string = f"({prefix}X['{field}'].{operator}'')"
        return condition_string
