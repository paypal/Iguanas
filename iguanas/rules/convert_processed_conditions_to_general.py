"""Class for converting generated rule conditions to system-ready conditions."""
from iguanas.rules._convert_rule_strings_to_rule_dicts import _ConvertRuleStringsToRuleDicts
from iguanas.rules import Rules
from iguanas.utils.typing import PandasDataFrameType
import pandas as pd
from typing import Dict, List, Tuple, Union


class ConvertProcessedConditionsToGeneral:
    """
    Converts the conditions of rules that utilise either imputed or OHE 
    features into conditions that account for this for unprocessed data. This 
    allows the rules to be applied to unprocessed data.

    Parameters
    ----------
    imputed_values : Dict[str, Union[str, float]], optional
        The value used to impute nulls (values) for each feature in the 
        original, unprocessed dataset (keys). Providing this information 
        converts conditions that include null values into two separate  
        conditions - one for the numeric condition, one for the null 
        condition. Defaults to None.
    ohe_categories : Dict[str, str], optional
        The category (values) linked to each OHE column (keys). If the OHE
        column represents when null values are present, include this 
        feature in the `imputed_values` parameter. Providing this 
        information converts conditions that include OHE features into the
        general format - e.g. if a rule condition is 
        `X['ip_country_US']==True`, the converted condition will be 
        `X['ip_country']=='US'`. Defaults to None.

    Attributes
    ----------
    rules : Rules
        Class containing the rule stored in the standard Iguanas string format.
        See the `rules` module for more information.
    """

    def __init__(self, imputed_values=None, ohe_categories=None):
        if imputed_values is None and ohe_categories is None:
            raise ValueError(
                'Either `imputed_values` or `ohe_categories` must be given')
        self.imputed_values = imputed_values
        self.ohe_categories = ohe_categories
        if ohe_categories is not None:
            self.ohe_columns = ohe_categories.keys()
        else:
            self.ohe_columns = []

    def convert(self, rule_strings: Dict[str, str], X=None) -> Dict[str, str]:
        """
        Converts the conditions of rules that utilise either imputed or OHE 
        features into conditions that account for this for unprocessed data. This 
        allows the rules to be applied to unprocessed data.

        Parameters
        ----------
        rule_strings : Dict[str, str] 
            Set of rules defined using the standard Iguanas string format 
            (values) and their names (keys).
        X : PandasDataFrameType, optional 
            The dataset containing the imputed variables. Required when imputed
            numeric variables are present in rules. Defaults to None.

        Returns
        -------
        Dict[str, str]
            Set of generalised rules defined using the standard Iguanas string format.
        """
        general_rule_strings = {}
        for rule_name, rule_string in rule_strings.items():
            self.condition_replacement_dict = {}
            general_rule_string = self._convert_rule_string(
                rule_string=rule_string, X=X)
            general_rule_strings[rule_name] = general_rule_string
        self.rules = Rules(rule_strings=general_rule_strings)
        return general_rule_strings

    def _convert_rule_string(self, rule_string: str, X: PandasDataFrameType) -> str:
        """
        Adds the general condition for each relevant processed condition to the
        condition_replacement_dict, then replaces the latter with the former.
        """

        if rule_string.startswith('(') and rule_string.endswith(')') and \
                ')&(' not in rule_string and ')|(' not in rule_string:
            self._add_to_condition_replacement_dict(
                rule_string=rule_string, X=X)
        else:
            self._recurse_create_condition_replacement_dict(
                rule_string=rule_string, X=X)
        general_rule_string = self._replace_processed_condition_with_general(
            rule_string=rule_string)
        return general_rule_string

    def _replace_processed_condition_with_general(self,
                                                  rule_string: str) -> str:
        """
        Replaces the original, processed condition with the new, general 
        condition
        """

        for processed_condition, general_condition in self.condition_replacement_dict.items():
            rule_string = rule_string.replace(
                processed_condition, general_condition)
        return rule_string

    def _recurse_create_condition_replacement_dict(self, rule_string: str,
                                                   X: PandasDataFrameType) -> None:
        """
        Recursively converts a rule string with processed conditions to a 
        general format.
        """

        parentheses_pair_idxs = _ConvertRuleStringsToRuleDicts._find_top_level_parentheses_idx(
            rule_string)
        # If rule enclosed in brackets, remove those then call function again
        if len(parentheses_pair_idxs) == 1:
            rule_string = rule_string[1:-1]
            self._recurse_create_condition_replacement_dict(
                rule_string=rule_string, X=X)
        else:
            conditions_string_list = _ConvertRuleStringsToRuleDicts._return_conditions_string_list(
                parentheses_pair_idxs, rule_string)
            self._create_condition_replacement_dict(
                conditions_string_list, X)

    def _create_condition_replacement_dict(self,
                                           conditions_string_list: List[str],
                                           X: PandasDataFrameType) -> None:
        """
        Loops through list of string conditions - if it is a single condition, 
        converts that to the general format; if not, then it goes to the next 
        level down in the rule.
        """

        for condition_string in conditions_string_list:
            if ')&(' not in condition_string and ')|(' not in condition_string:
                self._add_to_condition_replacement_dict(
                    rule_string=condition_string, X=X)
            else:
                self._recurse_create_condition_replacement_dict(
                    rule_string=condition_string, X=X)

    def _add_to_condition_replacement_dict(self, rule_string: str,
                                           X=PandasDataFrameType) -> None:
        """
        Adds the general condition for each relevant processed condition to the
        condition_replacement_dict.
        """

        if rule_string.startswith('(') and rule_string.endswith(')'):
            rule_string = rule_string[1:-1]
        feature, operator, value = self._extract_components_from_condition_string(
            rule_string)
        original_rule_string = f"(X['{feature}']{operator}{value})"
        if feature in self.ohe_columns:
            converted_rule_string = self._convert_ohe_condition_to_general(
                feature=feature, operator=operator, value=value,
                ohe_categories=self.ohe_categories, imputed_values=self.imputed_values)
        else:
            converted_rule_string = self._add_null_condition_to_imputed_numeric_rule(
                feature=feature, operator=operator, value=value,
                imputed_values=self.imputed_values, X=X
            )
        if original_rule_string != converted_rule_string:
            self.condition_replacement_dict[original_rule_string] = converted_rule_string

    @staticmethod
    def _extract_components_from_condition_string(condition_string: str) -> Tuple[str, str, float]:
        """
        Extracts the feature, operator and value of a single condition rule.
        """

        feature = condition_string.split('[')[1].split(']')[0][1:-1]
        operator_list = ['>=', '>', '<=', '<', '==', '!=']
        for operator in operator_list:
            if operator in condition_string:
                value = condition_string.split(operator)[1]
                return feature, operator, value
        raise Exception(
            'Operator not currently supported in Iguanas. Rule cannot be parsed.')

    @staticmethod
    def _add_null_condition_to_imputed_numeric_rule(feature: str,
                                                    operator: str,
                                                    value: str,
                                                    imputed_values: Dict[str, Union[str, float]],
                                                    X: PandasDataFrameType) -> str:
        """
        Takes a single condition, numeric rule and adds an additional condition 
        to account for imputed null values.
        """

        value = float(value) if '.' in value else int(value)
        null_value = imputed_values[feature]
        X_rule_str = f"X['{feature}']{operator}{value}"
        is_null_in_rule_str = f'{null_value}{operator}{value}'
        X_is_null_str = f"(X['{feature}']=={null_value})"
        if eval(is_null_in_rule_str):
            if all(eval(X_rule_str) == eval(X_is_null_str)):
                clean_condition = f"(X['{feature}'].isna())"
            elif null_value == value and operator == '>=':
                next_lowest_value = eval(
                    f"X[{X_rule_str}]['{feature}'].drop_duplicates().nsmallest(2).iloc[-1]")
                clean_condition = f"((X['{feature}']>={next_lowest_value})|(X['{feature}'].isna()))"
            elif null_value == value and operator == '<=':
                next_highest_value = eval(
                    f"X[{X_rule_str}]['{feature}'].drop_duplicates().nlargest(2).iloc[-1]")
                clean_condition = f"((X['{feature}']<={next_highest_value})|(X['{feature}'].isna()))"
            else:
                clean_condition = f"((X['{feature}']{operator}{value})|(X['{feature}'].isna()))"
        else:
            clean_condition = f"(X['{feature}']{operator}{value})"
        return clean_condition

    @staticmethod
    def _convert_ohe_condition_to_general(feature: str, operator: str,
                                          value: str, ohe_categories: Dict[str, str],
                                          imputed_values: None) -> str:
        """
        Takes a single condition rule that uses a OHE column and converts it to
        the explicit definition. Also accounts for OHE columns that represent 
        when the feature is null.
        """
        str_operator_lookup = {
            ('==', 'True'): '==',
            ('!=', 'False'): '==',
            ('==', 'False'): '!=',
            ('!=', 'True'): '!=',
        }
        bool_value_null_cond_lookup = {
            (True, '==', 'True'): {'Value': True, 'AddNullCond': False},
            (False, '==', 'True'): {'Value': False, 'AddNullCond': False},
            (True, '!=', 'False'): {'Value': True, 'AddNullCond': False},
            (False, '!=', 'False'): {'Value': False, 'AddNullCond': False},
            (True, '==', 'False'): {'Value': False, 'AddNullCond': True},
            (False, '==', 'False'): {'Value': True, 'AddNullCond': True},
            (True, '!=', 'True'): {'Value': False, 'AddNullCond': True},
            (False, '!=', 'True'): {'Value': True, 'AddNullCond': True},
        }
        category = ohe_categories[feature]
        original_feature = feature.split(f"_{category}")[0]
        if imputed_values is not None:
            imputed_value = imputed_values[original_feature]
            is_null_col = True if imputed_value == category else False
        else:
            is_null_col = False
        if is_null_col:
            gen_operator = str_operator_lookup[operator, value]
            if gen_operator == '==':
                cleaned_condition = f"(X['{original_feature}'].isna())"
            elif gen_operator == '!=':
                cleaned_condition = f"(~X['{original_feature}'].isna())"
        else:
            if isinstance(category, str):
                gen_operator = str_operator_lookup[operator, value]
                cleaned_condition = f"(X['{original_feature}']{gen_operator}'{category}')"
            else:
                bool_value_null_cond = bool_value_null_cond_lookup[category,
                                                                   operator, value]
                gen_value = bool_value_null_cond['Value']
                add_null_cond = bool_value_null_cond['AddNullCond']
                if add_null_cond:
                    cleaned_condition = f"((X['{original_feature}']=={gen_value})|(X['{original_feature}'].isna()))"
                else:
                    cleaned_condition = f"(X['{original_feature}']=={gen_value})"
        return cleaned_condition


class ReturnMappings:
    """
    Generates mapping dictionaries (for imputed values and OHE values) which 
    are required in the `ConvertProcessedConditionsToGeneral` class.
    """

    @staticmethod
    def return_imputed_values_mapping(*args) -> Dict[str, Union[str, float]]:
        """
        Returns a dictionary of the value used to impute nulls for each feature
        in the original, unprocessed dataset.

        Parameters
        ----------
        *args: tuple
            Set of lists, each of which has two elements - the first containing
            the list of fields imputed, the second containing the imputed
            value.

        Returns
        -------
        Dict[str, Union[str, float]]
            The value used to impute nulls for each feature.
        """

        imputed_values_dict = {}
        for features_list, imputed_value in args:
            for feature in features_list:
                imputed_values_dict[feature] = imputed_value
        return imputed_values_dict

    @staticmethod
    def return_ohe_categories_mapping(pre_ohe_cols: List[str], post_ohe_cols: List[str],
                                      pre_ohe_dtypes: Dict[str, str]) -> Dict[str, str]:
        """
        Return a dictionary of the category linked to each One Hot Encoded 
        column.

        Parameters
        ----------
        pre_ohe_cols : List[str]
            List of column names from the dataset before one hot encoding was
            applied.
        post_ohe_cols : List[str]
            List of column names from the dataset after one hot encoding was
            applied.
        pre_ohe_dtypes : Dict[str. str]
            The datatype (values) of each column (keys) from the dataset before
            one hot encoding was applied.

        Returns
        -------
        Dict[str, str]
            The category linked to each One Hot Encoded column.
        """
        ohe_categories = {}
        ohe_cols = [col for col in post_ohe_cols if col not in pre_ohe_cols]
        orig_cat_cols = [
            col for col in pre_ohe_cols if col not in post_ohe_cols]
        for orig_cat_col in orig_cat_cols:
            ohe_cols_for_cat_col = list(
                filter(lambda x: x.startswith(orig_cat_col), ohe_cols))
            for ohe_col_for_cat_col in ohe_cols_for_cat_col:
                category = ohe_col_for_cat_col.split(f'{orig_cat_col}_')[1]
                # Check datatype of original column (to catch when booleans are saved as text)
                if isinstance(pre_ohe_dtypes[orig_cat_col], (bool, pd.BooleanDtype)):
                    if category == 'True':
                        category = True
                    elif category == 'False':
                        category = False
                ohe_categories[ohe_col_for_cat_col] = category
        return ohe_categories
