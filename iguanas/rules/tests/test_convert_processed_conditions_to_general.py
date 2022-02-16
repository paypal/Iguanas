import pytest
import pandas as pd
import numpy as np
from iguanas.rules import ConvertProcessedConditionsToGeneral, ReturnMappings


@pytest.fixture
def _data():
    X_orig = pd.DataFrame(
        {
            'A': np.array([-1, 1, np.nan, 2] * 250),
            'country': ['GB', 'US', 'FR', np.nan] * 250,
            'is_existing_user': [True, False, False, np.nan] * 250
        }
    )
    imputed_values = {
        'A': 0,
        'country': 'missing',
        'is_existing_user': 'missing'
    }
    ohe_categories = {
        'country_GB': 'GB',
        'country_US': 'US',
        'country_FR': 'FR',
        'country_missing': 'missing',
        'is_existing_user_True': True,
        'is_existing_user_False': False,
        'is_existing_user_missing': 'missing'
    }
    X_processed = X_orig.fillna(imputed_values)
    X_processed = pd.get_dummies(X_processed)
    rule_strings = {
        'nan_low': "(X['A']>=0)",
        'nan_middle': "(X['A']>=-1)",
        'nan_high': "(X['A']<=0)",
        'no_nan': "(X['A']>=2)",
        'is_GB': "(X['country_GB']==True)",
        'is_not_FR': "(X['country_FR']==False)",
        'is_missing': "(X['country_missing']==True)",
        'is_not_missing': "(X['country_missing']==False)",
        'bool_true': "(X['is_existing_user_True']==True)",
        'bool_false': "(X['is_existing_user_False']==True)",
        'bool_true_or_missing': "(X['is_existing_user_False']!=True)",
        'bool_false_or_missing': "(X['is_existing_user_True']!=True)",
        'bool_missing': "(X['is_existing_user_missing']==True)",
        'bool_not_missing': "(X['is_existing_user_missing']==False)",
        'bool_missing_': "(X['is_existing_user_missing']!=False)",
        'bool_not_missing_': "(X['is_existing_user_missing']!=True)",
        'combined_rule_0': "(X['A']>=0)&(X['A']>=-1)",
        'combined_rule_1': "(X['A']>=0)&(X['A']<=0)",
        'combined_rule_2': "(X['A']>=0)&(X['A']>=2)",
        'combined_rule_3': "(X['A']>=0)&(X['country_GB']==True)",
        'combined_rule_4': "(X['A']>=0)&(X['country_FR']==False)",
        'combined_rule_5': "(X['A']>=0)&(X['country_missing']==True)",
        'combined_rule_6': "(X['A']>=0)&(X['country_missing']==False)",
        'combined_rule_7': "(X['A']>=-1)&(X['A']<=0)",
        'combined_rule_8': "(X['A']>=-1)&(X['A']>=2)",
        'combined_rule_9': "(X['A']>=-1)&(X['country_GB']==True)",
        'combined_rule_10': "(X['A']>=-1)&(X['country_FR']==False)",
        'combined_rule_11': "(X['A']>=-1)&(X['country_missing']==True)",
        'combined_rule_12': "(X['A']>=-1)&(X['country_missing']==False)",
        'combined_rule_13': "(X['A']<=0)&(X['A']>=2)",
        'combined_rule_14': "(X['A']<=0)&(X['country_GB']==True)",
        'combined_rule_15': "(X['A']<=0)&(X['country_FR']==False)",
        'combined_rule_16': "(X['A']<=0)&(X['country_missing']==True)",
        'combined_rule_17': "(X['A']<=0)&(X['country_missing']==False)",
        'combined_rule_18': "(X['A']>=2)&(X['country_GB']==True)",
        'combined_rule_19': "(X['A']>=2)&(X['country_FR']==False)",
        'combined_rule_20': "(X['A']>=2)&(X['country_missing']==True)",
        'combined_rule_21': "(X['A']>=2)&(X['country_missing']==False)",
        'combined_rule_22': "(X['country_GB']==True)&(X['country_FR']==False)",
        'combined_rule_23': "(X['country_GB']==True)&(X['country_missing']==True)",
        'combined_rule_24': "(X['country_GB']==True)&(X['country_missing']==False)",
        'combined_rule_25': "(X['country_FR']==False)&(X['country_missing']==True)",
        'combined_rule_26': "(X['country_FR']==False)&(X['country_missing']==False)",
        'combined_rule_27': "(X['country_missing']==True)&(X['country_missing']==False)",
        'combined_rule_28': "(X['country_missing']==True)|(X['country_missing']==False)"
    }
    return rule_strings, X_processed, imputed_values, ohe_categories


@pytest.fixture
def _instantiate_class(_data):
    _, _, imputed_values, ohe_categories = _data
    c = ConvertProcessedConditionsToGeneral(imputed_values, ohe_categories)
    return c


@pytest.fixture
def _expected_general_rule_strings():
    general_rule_strings = {
        'nan_low': "((X['A']>=1.0)|(X['A'].isna()))",
        'nan_middle': "((X['A']>=-1)|(X['A'].isna()))",
        'nan_high': "((X['A']<=-1.0)|(X['A'].isna()))",
        'no_nan': "(X['A']>=2)",
        'is_GB': "(X['country']=='GB')",
        'is_not_FR': "(X['country']!='FR')",
        'is_missing': "(X['country'].isna())",
        'is_not_missing': "(~X['country'].isna())",
        'bool_true': "(X['is_existing_user']==True)",
        'bool_false': "(X['is_existing_user']==False)",
        'bool_true_or_missing': "((X['is_existing_user']==True)|(X['is_existing_user'].isna()))",
        'bool_false_or_missing': "((X['is_existing_user']==False)|(X['is_existing_user'].isna()))",
        'bool_missing': "(X['is_existing_user'].isna())",
        'bool_not_missing': "(~X['is_existing_user'].isna())",
        'bool_missing_': "(X['is_existing_user'].isna())",
        'bool_not_missing_': "(~X['is_existing_user'].isna())",
        'combined_rule_0': "((X['A']>=1.0)|(X['A'].isna()))&((X['A']>=-1)|(X['A'].isna()))",
        'combined_rule_1': "((X['A']>=1.0)|(X['A'].isna()))&((X['A']<=-1.0)|(X['A'].isna()))",
        'combined_rule_2': "((X['A']>=1.0)|(X['A'].isna()))&(X['A']>=2)",
        'combined_rule_3': "((X['A']>=1.0)|(X['A'].isna()))&(X['country']=='GB')",
        'combined_rule_4': "((X['A']>=1.0)|(X['A'].isna()))&(X['country']!='FR')",
        'combined_rule_5': "((X['A']>=1.0)|(X['A'].isna()))&(X['country'].isna())",
        'combined_rule_6': "((X['A']>=1.0)|(X['A'].isna()))&(~X['country'].isna())",
        'combined_rule_7': "((X['A']>=-1)|(X['A'].isna()))&((X['A']<=-1.0)|(X['A'].isna()))",
        'combined_rule_8': "((X['A']>=-1)|(X['A'].isna()))&(X['A']>=2)",
        'combined_rule_9': "((X['A']>=-1)|(X['A'].isna()))&(X['country']=='GB')",
        'combined_rule_10': "((X['A']>=-1)|(X['A'].isna()))&(X['country']!='FR')",
        'combined_rule_11': "((X['A']>=-1)|(X['A'].isna()))&(X['country'].isna())",
        'combined_rule_12': "((X['A']>=-1)|(X['A'].isna()))&(~X['country'].isna())",
        'combined_rule_13': "((X['A']<=-1.0)|(X['A'].isna()))&(X['A']>=2)",
        'combined_rule_14': "((X['A']<=-1.0)|(X['A'].isna()))&(X['country']=='GB')",
        'combined_rule_15': "((X['A']<=-1.0)|(X['A'].isna()))&(X['country']!='FR')",
        'combined_rule_16': "((X['A']<=-1.0)|(X['A'].isna()))&(X['country'].isna())",
        'combined_rule_17': "((X['A']<=-1.0)|(X['A'].isna()))&(~X['country'].isna())",
        'combined_rule_18': "(X['A']>=2)&(X['country']=='GB')",
        'combined_rule_19': "(X['A']>=2)&(X['country']!='FR')",
        'combined_rule_20': "(X['A']>=2)&(X['country'].isna())",
        'combined_rule_21': "(X['A']>=2)&(~X['country'].isna())",
        'combined_rule_22': "(X['country']=='GB')&(X['country']!='FR')",
        'combined_rule_23': "(X['country']=='GB')&(X['country'].isna())",
        'combined_rule_24': "(X['country']=='GB')&(~X['country'].isna())",
        'combined_rule_25': "(X['country']!='FR')&(X['country'].isna())",
        'combined_rule_26': "(X['country']!='FR')&(~X['country'].isna())",
        'combined_rule_27': "(X['country'].isna())&(~X['country'].isna())",
        'combined_rule_28': "(X['country'].isna())|(~X['country'].isna())"
    }
    return general_rule_strings


class TestConvertProcessedConditionsToGeneral:

    def test_convert(self, _data, _instantiate_class, _expected_general_rule_strings):
        rule_strings, X_processed, _, _ = _data
        expected_general_rule_strings = _expected_general_rule_strings
        c = _instantiate_class
        general_rule_strings = c.convert(rule_strings, X_processed)
        assert general_rule_strings == expected_general_rule_strings

    def test_convert_imputed_only(self, _data):
        _, X_processed, _, _ = _data
        exp_result = {
            'nan_low': "((X['A']>=1.0)|(X['A'].isna()))",
            'nan_middle': "((X['A']>=-1)|(X['A'].isna()))",
            'nan_high': "((X['A']<=-1.0)|(X['A'].isna()))",
            'no_nan': "(X['A']>=1)"
        }
        rule_strings_imp = {
            'nan_low': "(X['A']>=0)",
            'nan_middle': "(X['A']>=-1)",
            'nan_high': "(X['A']<=0)",
            'no_nan': "(X['A']>=1)",
        }
        c = ConvertProcessedConditionsToGeneral(imputed_values={'A': 0})
        result = c.convert(rule_strings_imp, X_processed)
        assert result == exp_result

    def test_convert_ohe_only(self, _data):
        _, X_processed, _, _ = _data
        exp_result = {
            'is_GB': "(X['country']=='GB')",
            'is_not_FR': "(X['country']!='FR')"
        }
        rule_strings_ohe = {
            'is_GB': "(X['country_GB']==True)",
            'is_not_FR': "(X['country_FR']==False)",
        }
        c = ConvertProcessedConditionsToGeneral(
            ohe_categories={'country_GB': 'GB', 'country_FR': 'FR'}
        )
        result = c.convert(rule_strings_ohe, X_processed)
        assert result == exp_result

    def test_convert_rule_string(self, _data, _instantiate_class, _expected_general_rule_strings):
        rule_strings, X_processed, _, _ = _data
        expected_general_rule_strings = _expected_general_rule_strings
        c = _instantiate_class
        general_rule_strings = {}
        for rule_name, rule_string in rule_strings.items():
            c.condition_replacement_dict = {}
            general_rule_string = c._convert_rule_string(
                rule_string=rule_string, X=X_processed)
            general_rule_strings[rule_name] = general_rule_string
        assert general_rule_strings == expected_general_rule_strings

    def test_replace_processed_condition_with_general(self, _instantiate_class):
        c = _instantiate_class
        c.condition_replacement_dict = {'A': 'B', 'C': 'D'}
        rule_string = 'AC'
        general_rule_string = c._replace_processed_condition_with_general(
            rule_string=rule_string)
        assert general_rule_string == 'BD'

    def test_recurse_create_condition_replacement_dict(self, _data, _instantiate_class):
        expected_condition_replacement_dict = {
            "(X['A']>=0)": "((X['A']>=1.0)|(X['A'].isna()))",
            "(X['country_missing']==True)": "(X['country'].isna())"
        }
        rule_strings, X_processed, _, _ = _data
        c = _instantiate_class
        c.condition_replacement_dict = {}
        c._recurse_create_condition_replacement_dict(
            rule_string=f"({rule_strings['combined_rule_5']})", X=X_processed)
        assert c.condition_replacement_dict == expected_condition_replacement_dict

    def test_create_condition_replacement_dict(self, _data, _instantiate_class):
        conditions_string_list = [
            "(X['A']>=0)&(X['A']>=2)",
            "(X['country_missing']==True)"
        ]
        expected_condition_replacement_dict = {
            "(X['A']>=0)": "((X['A']>=1.0)|(X['A'].isna()))",
            "(X['country_missing']==True)": "(X['country'].isna())"
        }
        _, X_processed, _, _ = _data
        c = _instantiate_class
        c.condition_replacement_dict = {}
        c._create_condition_replacement_dict(
            conditions_string_list=conditions_string_list, X=X_processed)
        assert c.condition_replacement_dict == expected_condition_replacement_dict

    def test_add_to_condition_replacement_dict(self, _data, _instantiate_class):
        conditions_string_list = [
            "(X['A']>=0)",
            "(X['A']>=2)",
            "(X['country_missing']==True)"
        ]
        expected_condition_replacement_dict = {
            "(X['A']>=0)": "((X['A']>=1.0)|(X['A'].isna()))",
            "(X['country_missing']==True)": "(X['country'].isna())"
        }
        _, X_processed, _, _ = _data
        c = _instantiate_class
        c.condition_replacement_dict = {}
        for condition_string in conditions_string_list:
            c._add_to_condition_replacement_dict(
                rule_string=condition_string, X=X_processed)
        assert c.condition_replacement_dict == expected_condition_replacement_dict

    def test_extract_components_from_condition_string(self, _instantiate_class):
        rule_strings = {
            'greater_or_equal': "X['A']>=0",
            'greater': "X['A']>0",
            'less_or_equal': "X['A']<=0",
            'less': "X['A']<0",
            'equal': "X['country_GB']==True",
            'not_equal': "X['country_FR']!=True"
        }
        expected_components = {
            'greater_or_equal': ('A', '>=', '0'),
            'greater': ('A', '>', '0'),
            'less_or_equal': ('A', '<=', '0'),
            'less': ('A', '<', '0'),
            'equal': ('country_GB', '==', 'True'),
            'not_equal': ('country_FR', '!=', 'True')
        }
        c = _instantiate_class
        for rule_name, rule_string in rule_strings.items():
            comps = c._extract_components_from_condition_string(
                condition_string=rule_string)
            assert comps == expected_components[rule_name]

    def test_add_null_condition_to_imputed_numeric_rule(self,
                                                        _data,
                                                        _instantiate_class):
        rule_components = {
            'nan_low': ('A', '>=', '-1'),
            'nan_middle': ('A', '>=', '-2'),
            'nan_high': ('A', '<=', '-1'),
            'no_nan': ('A', '>=', '2'),
        }
        _, X_processed, _, _ = _data
        c = _instantiate_class
        imputed_values = {'A': -1}
        expected_general_rule_strings = {
            'nan_low': "((X['A']>=0.0)|(X['A'].isna()))",
            'nan_middle': "((X['A']>=-2)|(X['A'].isna()))",
            'nan_high': "(X['A'].isna())",
            'no_nan': "(X['A']>=2)",
        }
        cleaned_strings = {}
        for rule_name, rule_comp in rule_components.items():
            cleaned_string = c._add_null_condition_to_imputed_numeric_rule(
                *rule_comp, imputed_values=imputed_values, X=X_processed)
            cleaned_strings[rule_name] = cleaned_string
        assert cleaned_strings == expected_general_rule_strings

    def test_convert_ohe_condition_to_general(self, _data, _instantiate_class):
        rule_components = {
            'is_GB': ('country_GB', '==', 'True'),
            'is_not_FR': ('country_FR', '==', 'False'),
            'is_missing': ('country_missing', '==', 'True'),
            'is_not_missing': ('country_missing', '==', 'False'),
        }
        expected_cleaned_strings = {
            'is_GB': "(X['country']=='GB')",
            'is_not_FR': "(X['country']!='FR')",
            'is_missing': "(X['country'].isna())",
            'is_not_missing': "(~X['country'].isna())"
        }
        _, _, imputed_values, ohe_categories = _data
        c = _instantiate_class
        cleaned_strings = {}
        for rule_name, rule_comp in rule_components.items():
            cleaned_string = c._convert_ohe_condition_to_general(
                *rule_comp, ohe_categories=ohe_categories, imputed_values=imputed_values)
            cleaned_strings[rule_name] = cleaned_string
        assert cleaned_strings == expected_cleaned_strings

    def test_error(self, _instantiate_class):
        c = _instantiate_class
        with pytest.raises(Exception):
            c._extract_components_from_condition_string(
                "X['A'].str.contains('abc')")
        with pytest.raises(ValueError, match='Either `imputed_values` or `ohe_categories` must be given'):
            ConvertProcessedConditionsToGeneral(
                imputed_values=None, ohe_categories=None
            )


class TestReturnMappings:
    def test_return_imputed_values_mapping(self):
        exp_result = {
            'A': -1,
            'B': -1,
            'C': 10.5,
            'D': 'missing',
            'E': False
        }
        rm = ReturnMappings()
        imputed_mapping = rm.return_imputed_values_mapping(
            [['A', 'B'], -1], [['C'], 10.5], [['D'], 'missing'], [['E'], False])
        assert imputed_mapping == exp_result

    def test_return_ohe_categories_mapping(self):
        pre_ohe_cols = [
            'num', 'country', 'is_existing_user', 'is_existing_user_text'
        ]
        post_ohe_cols = [
            'num', 'country_GB', 'country_US',
            'is_existing_user_True', 'is_existing_user_False',
            'is_existing_user_text_True', 'is_existing_user_text_False'
        ]
        pre_ohe_dtypes = {
            'num': np.float64,
            'country': object,
            'is_existing_user': pd.BooleanDtype(),
            'is_existing_user_text': object

        }
        exp_result = {
            'country_GB': 'GB',
            'country_US': 'US',
            'is_existing_user_True': True,
            'is_existing_user_False': False,
            'is_existing_user_text_True': 'True',
            'is_existing_user_text_False': 'False'
        }
        rm = ReturnMappings()
        ohe_mapping = rm.return_ohe_categories_mapping(
            pre_ohe_cols=pre_ohe_cols, post_ohe_cols=post_ohe_cols,
            pre_ohe_dtypes=pre_ohe_dtypes
        )
        assert ohe_mapping == exp_result
