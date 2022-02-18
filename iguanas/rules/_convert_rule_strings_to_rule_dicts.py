"""Class for converting between string and dict representations of rules."""
from typing import Dict, List, Tuple, Union


class _ConvertRuleStringsToRuleDicts:
    """
    Converts a set of rules (each being represented in the standard Iguanas string
    format) into the standard Iguanas dictionary format.   

    Parameters
    ----------
    rule_strings : Dict[str, str] 
        Set of rules defined using the standard Iguanas string format 
        (values) and their names (keys).

    Attributes
    ----------
    rule_dicts : Dict[str, Dict]
        Set of rules defined using the standard Iguanas dictionary format 
        (values) and their names (keys).
    """

    def __init__(self, rule_strings: Dict[str, str]):
        self.rule_strings = rule_strings
        self.rule_dicts = {}
        self.condition_lookup = {
            '|': 'OR',
            '&': 'AND'
        }
        self.operator_lookup = {
            '].str.startswith': {
                'operator': 'begins_with',
                'not_operator': 'not_begins_with',
                'conversion_function': self._return_value_for_pandas_str_based_operators
            },
            '].str.endswith': {
                'operator': 'ends_with',
                'not_operator': 'not_ends_with',
                'conversion_function': self._return_value_for_pandas_str_based_operators
            },
            '].str.contains': {
                'operator': 'contains',
                'not_operator': 'not_contains',
                'conversion_function': self._return_value_for_pandas_str_based_operators
            },
            '].isin': {
                'operator': 'in',
                'not_operator': 'not_in',
                'conversion_function': self._return_value_for_pandas_is_in_operator
            },
            '.isna(': {
                'operator': 'is_null',
                'not_operator': 'is_not_null',
                'conversion_function': None
            },
            "fillna('')==": {
                'operator': 'is_empty',
                'not_operator': None,
                'conversion_function': self._return_value_for_standard_operators
            },
            "fillna('')!=": {
                'operator': 'is_not_empty',
                'not_operator': None,
                'conversion_function': self._return_value_for_standard_operators
            },
            "==X['": {
                'operator': 'equal_field',
                'not_operator': None,
                'conversion_function': self._return_value_for_field_level_comparison
            },
            "!=X['": {
                'operator': 'not_equal_field',
                'not_operator': None,
                'conversion_function': self._return_value_for_field_level_comparison
            },
            ">=X['": {
                'operator': 'greater_or_equal_field',
                'not_operator': None,
                'conversion_function': self._return_value_for_field_level_comparison
            },
            ">X['": {
                'operator': 'greater_field',
                'not_operator': None,
                'conversion_function': self._return_value_for_field_level_comparison
            },
            "<=X['": {
                'operator': 'less_or_equal_field',
                'not_operator': None,
                'conversion_function': self._return_value_for_field_level_comparison
            },
            "<X['": {
                'operator': 'less_field',
                'not_operator': None,
                'conversion_function': self._return_value_for_field_level_comparison
            },
            '==': {
                'operator': 'equal',
                'not_operator': None,
                'conversion_function': self._return_value_for_standard_operators
            },
            '!=': {
                'operator': 'not_equal',
                'not_operator': None,
                'conversion_function': self._return_value_for_standard_operators
            },
            '>=': {
                'operator': 'greater_or_equal',
                'not_operator': None,
                'conversion_function': self._return_value_for_standard_operators
            },
            '>': {
                'operator': 'greater',
                'not_operator': None,
                'conversion_function': self._return_value_for_standard_operators
            },
            '<=': {
                'operator': 'less_or_equal',
                'not_operator': None,
                'conversion_function': self._return_value_for_standard_operators
            },
            '<': {
                'operator': 'less',
                'not_operator': None,
                'conversion_function': self._return_value_for_standard_operators
            },
        }

    def convert(self) -> Dict[str, dict]:
        """
        Converts a set of rules (each being represented in the standard Iguanas 
        string format) into the standard Iguanas dictionary format.        

        Returns
        -------
        Dict[str, dict]
            Set of rules defined using the standard Iguanas dictionary format.
        """

        for rule_name, rule_string in self.rule_strings.items():
            self.rule_dicts[rule_name] = self._convert_rule(
                rule_string=rule_string)
        return self.rule_dicts

    def _convert_rule(self, rule_string: str) -> dict:
        """
        Converts a rule stored in the standard Iguanas string format into the 
        standard Iguanas dictionary format.
        """

        parent_dict = {
            'condition': None,
            'rules': []
        }
        if ')&(' not in rule_string and ')|(' not in rule_string:
            condition_dict = self._create_condition_dict(
                rule_string)
            parent_dict['rules'].append(condition_dict)
            parent_dict['condition'] = 'AND'
            rule_dict = parent_dict
        else:
            rule_dict = self._recurse_convert_rule_string_conditions(
                rule_string=rule_string, parent_dict=parent_dict)
        return rule_dict

    def _recurse_convert_rule_string_conditions(self, rule_string: str,
                                                parent_dict: dict) -> dict:
        """Recursively converts a rule string to rule dictionary"""

        parentheses_pair_idxs = self._find_top_level_parentheses_idx(
            rule_string)
        # If rule enclosed in brackets, remove those then call function again
        if len(parentheses_pair_idxs) == 1:
            rule_string = rule_string[1:-1]
            rule_dict = self._recurse_convert_rule_string_conditions(
                rule_string=rule_string, parent_dict=parent_dict)
        else:
            conditions_string_list = self._return_conditions_string_list(
                parentheses_pair_idxs, rule_string)
            connecting_cond_list = self._find_connecting_conditions(
                parentheses_pair_idxs, rule_string)
            connecting_condition = self._return_connecting_condition(
                connecting_cond_list, self.condition_lookup)
            parent_dict['condition'] = connecting_condition
            rule_dict = self._convert_rule_string_conditions(
                conditions_string_list, parent_dict)
        return rule_dict

    def _convert_rule_string_conditions(self, conditions_string_list: List[str],
                                        parent_dict: dict) -> dict:
        """
        Loops through list of string conditions - if it is a single condition, 
        converts that to the dictionary format; if not, then it goes to the 
        next level down in the rule.
        """

        for condition_string in conditions_string_list:
            if ')&(' not in condition_string and ')|(' not in condition_string:
                condition_dict = self._create_condition_dict(
                    condition_string)
                parent_dict['rules'].append(condition_dict)
            else:
                child_dict = {
                    'condition': None,
                    'rules': []
                }
                condition_dict = self._recurse_convert_rule_string_conditions(
                    condition_string, child_dict)
                parent_dict['rules'].append(condition_dict)
        return parent_dict

    def _create_condition_dict(self, rule_string: str) -> dict:
        """Converts a single condition string to the dictionary format"""

        if rule_string.startswith('(') and rule_string.endswith(')'):
            rule_string = rule_string[1:-1]
        feature, operator, value = self._extract_components_from_condition_string(
            rule_string)
        condition_dict = self._create_rule_condition_dict(
            feature, operator, value)
        return condition_dict

    def _extract_components_from_condition_string(self,
                                                  condition_string: str) -> Tuple[str, str, str]:
        """
        Extracts the feature, operator and value from a single condition rule 
        string, ready for injection into the dictionary.
        """

        feature = condition_string.split('[')[1].split(']')[0][1:-1]
        for str_operator in self.operator_lookup.keys():
            if str_operator in condition_string:
                operator_conversion_info = self.operator_lookup[str_operator]
                dict_operator = operator_conversion_info['operator']
                dict_not_operator = operator_conversion_info['not_operator']
                dict_operator = dict_not_operator if condition_string.startswith(
                    '~') else dict_operator
                conversion_function = operator_conversion_info['conversion_function']
                if conversion_function is None:
                    value = None
                else:
                    value = conversion_function(
                        condition_string=condition_string, operator=str_operator)
                return feature, dict_operator, value
        raise Exception(
            'Operator not currently supported in Iguanas. Rule cannot be parsed.')

    @staticmethod
    def _find_top_level_parentheses_idx(rule_string: str) -> Dict[int, int]:
        """
        Returns the indices of each matching pair of top level parentheses in 
        the rule string        
        """

        parentheses_pair_idxs = {}
        start_count = 0
        inside_quote_flag = 0
        for i, char in enumerate(rule_string):
            if char == "'":
                if inside_quote_flag == 0:
                    inside_quote_flag = 1
                    continue
                elif inside_quote_flag == 1:
                    inside_quote_flag = 0
                    continue
            if inside_quote_flag == 1:
                continue
            if char == '(':
                if start_count == 0:
                    start_idx = i + 1
                start_count += 1
                continue
            elif char == ')':
                if start_count == 1:
                    parentheses_pair_idxs[start_idx] = i
                start_count -= 1
                continue
        return parentheses_pair_idxs

    @staticmethod
    def _return_conditions_string_list(parentheses_pair_idxs: Dict[int, int],
                                       rule_string: str) -> List[str]:
        """
        Splits the original rule string into individual, top level 
        conditions.
        """

        conditions_string_list = []
        for start, finish in parentheses_pair_idxs.items():
            conditions_string_list.append(rule_string[start:finish])
        return conditions_string_list

    @staticmethod
    def _find_connecting_conditions(parentheses_pair_idxs: Dict[int, int],
                                    rule_string: str) -> List[str]:
        """
        Returns the connecting conditions (AND or OR) for the top level 
        conditions.
        """

        connecting_cond_list = []
        for i in range(0, len(parentheses_pair_idxs) - 1):
            cond_start = list(parentheses_pair_idxs.values())[i] + 1
            cond_end = list(parentheses_pair_idxs.keys())[i+1] - 1
            connecting_cond = rule_string[cond_start:cond_end]
            connecting_cond_list.append(connecting_cond)
        return connecting_cond_list

    @staticmethod
    def _return_connecting_condition(connecting_cond_list: List[str],
                                     condition_lookup: Dict[str, str]) -> str:
        """
        Returns the unique connecting condition for the top level conditions 
        (should always be only one unique connecting condition for a given 
        level).
        """

        connecting_cond_unique = list(set(connecting_cond_list))
        if len(connecting_cond_unique) != 1:
            raise Exception(
                'More than one connecting condition for a given level')
        connecting_condition = connecting_cond_unique[0]
        connecting_dict_condition = condition_lookup[connecting_condition]
        return connecting_dict_condition

    @staticmethod
    def _create_rule_condition_dict(field: str, operator: str,
                                    value: str) -> Dict[str, Union[str, float]]:
        """
        Creates the dictionary representation of a condition given the field, 
        operator and value.
        """

        condition_dict = {
            'field': field,
            'operator': operator,
            'value': value
        }
        return condition_dict

    @staticmethod
    def _return_value_for_pandas_str_based_operators(condition_string: str,
                                                     operator: str) -> str:
        """
        Returns the value from a condition (repesented as a string) for 
        Pandas-based string operators.
        """

        value = condition_string.split(f"{operator}('")[1].split("'")[0]
        return value

    @staticmethod
    def _return_value_for_pandas_is_in_operator(condition_string: str,
                                                operator: str) -> str:
        """
        Returns the value from a condition (repesented as a string) for the 
        Pandas `.isin()` operator.
        """

        str_value = condition_string.split(f"{operator}([")[1].split("]")[0]
        value = str_value.replace("'", "").replace(', ', ',').split(',')
        return value

    @staticmethod
    def _return_value_for_standard_operators(condition_string: str,
                                             operator: str) -> str:
        """
        Returns the value from a condition (repesented as a string) for 
        standard operators.
        """

        value = condition_string.split(operator)[1]
        if value == "''":
            value = None
        elif value.startswith("'") and value.endswith("'"):
            value = value.replace("'", '')
        elif value == 'True':
            value = True
        elif value == 'False':
            value = False
        else:
            value = float(value)
            # If value is an integer, convert to int
            if value % 1 == 0:
                value = int(value)
        return value

    @staticmethod
    def _return_value_for_field_level_comparison(condition_string: str,
                                                 operator: str) -> str:
        """
        Returns the value from a condition (repesented as a string) for field
        level comparisons.
        """

        value = condition_string.split(operator)[1].split("'")[0]
        return value
