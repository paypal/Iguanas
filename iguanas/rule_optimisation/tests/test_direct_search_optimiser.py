from iguanas.rule_optimisation import DirectSearchOptimiser
from iguanas.metrics import FScore, AlertsPerDay
from iguanas.rules import Rules
from iguanas.warnings import RulesNotOptimisedWarning
from iguanas.exceptions import RulesNotOptimisedError
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def _create_data():
    np.random.seed(0)
    X = pd.DataFrame({
        'A': np.random.randint(0, 10, 10000),
        'B': np.random.randint(0, 100, 10000),
        'C': np.random.uniform(0, 1, 10000),
        'D': [True, False] * 5000,
        'E': ['yes', 'no'] * 5000,
        'AllNa': [np.nan] * 10000,
        'ZeroVar': [1] * 10000
    })
    X.loc[10000] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    X['A'] = X['A'].astype('Int64')
    X['B'] = X['B'].astype('Int64')
    X['D'] = X['D'].astype('boolean')
    y = pd.Series(np.random.randint(0, 2, 10001))
    sample_weight = pd.Series(
        np.where((X['A'] > 7).fillna(False) & (y == 0), 100, 1))
    return X, y, sample_weight


@pytest.fixture
def _create_inputs():
    rule_lambdas = {
        'integer': lambda **kwargs: "(X['A']>{A})".format(**kwargs),
        'float': lambda **kwargs: "(X['C']>{C})".format(**kwargs),
        'categoric': lambda **kwargs: "(X['E']=='yes')".format(**kwargs),
        'boolean': lambda **kwargs: "(X['D']==True)".format(**kwargs),
        'is_na': lambda **kwargs: "(X['A']>{A})|(X['A'].isna())".format(**kwargs),
        'mixed': lambda **kwargs: "((X['A']>{A})&(X['C']>{C})&(X['E']=='yes')&(X['D']==True))|(X['C']>{C%0})".format(**kwargs),
        'missing_col': lambda **kwargs: "(X['Z']>{Z})".format(**kwargs),
        'all_na': lambda **kwargs: "(X['AllNa']>{AllNa})".format(**kwargs),
        'zero_var': lambda **kwargs: "(X['ZeroVar']>{ZeroVar})".format(**kwargs),
        'already_optimal': lambda **kwargs: "(X['A']>={A})".format(**kwargs),
        'float_with_zero_var': lambda **kwargs: "(X['C']>{C})&(X['ZeroVar']>={ZeroVar})".format(**kwargs),
        'float_with_all_na_greater': lambda **kwargs: "(X['C']>{C})&(X['AllNa']>{AllNa})".format(**kwargs),
        'float_with_all_na_is_na': lambda **kwargs: "(X['C']>{C})&(X['AllNa'].isna())".format(**kwargs),
        'multi_zero_var': lambda **kwargs: "((X['C']>{C})&(X['ZeroVar']>={ZeroVar}))|((X['A']>{A})&(X['ZeroVar']>={ZeroVar%0}))".format(**kwargs),
    }
    lambda_kwargs = {
        'integer': {'A': 9},
        'float': {'C': 1.5},
        'categoric': {},
        'boolean': {},
        'is_na': {'A': 9},
        'mixed': {'A': 1, 'C': 1.5, 'C%0': 2.5},
        'missing_col': {'Z': 1},
        'all_na': {'AllNa': 5},
        'zero_var': {'ZeroVar': 1},
        'already_optimal': {'A': 0},
        'float_with_zero_var': {'C': 1.5, 'ZeroVar': 1},
        'float_with_all_na_greater': {'C': 1.5, 'AllNa': 1},
        'float_with_all_na_is_na': {'C': 1.5},
        'multi_zero_var': {'C': 1.5, 'ZeroVar': 1, 'A': 9, 'ZeroVar%0': 1}
    }
    return rule_lambdas, lambda_kwargs


@pytest.fixture
def _expected_rule_strings():
    opt_rule_strings = {
        'Nelder-Mead': {
            'integer': "(X['A']>8.549999999999999)",
            'float': "(X['C']>1.5)",
            'is_na': "(X['A']>8.549999999999999)|(X['A'].isna())",
            'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
            'already_optimal': "(X['A']>=0)",
            'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
            'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
            'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
            'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
            'categoric': "(X['E']=='yes')",
            'boolean': "(X['D']==True)"
        },
        'Powell': {
            'integer': "(X['A']>9)",
            'float': "(X['C']>1.5)",
            'is_na': "(X['A']>9)|(X['A'].isna())",
            'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
            'already_optimal': "(X['A']>=0)",
            'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
            'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
            'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
            'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
            'categoric': "(X['E']=='yes')",
            'boolean': "(X['D']==True)"
        },
        'CG': {
            'integer': "(X['A']>9)",
            'float': "(X['C']>1.5)",
            'is_na': "(X['A']>9)|(X['A'].isna())",
            'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
            'already_optimal': "(X['A']>=0)",
            'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
            'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
            'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
            'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
            'categoric': "(X['E']=='yes')",
            'boolean': "(X['D']==True)"
        },
        'BFGS': {
            'integer': "(X['A']>9)",
            'float': "(X['C']>1.5)",
            'is_na': "(X['A']>9)|(X['A'].isna())",
            'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
            'already_optimal': "(X['A']>=0)",
            'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
            'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
            'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
            'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
            'categoric': "(X['E']=='yes')",
            'boolean': "(X['D']==True)"
        },
        'L-BFGS-B': {
            'integer': "(X['A']>9)",
            'float': "(X['C']>1.5)",
            'is_na': "(X['A']>9)|(X['A'].isna())",
            'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
            'already_optimal': "(X['A']>=0)",
            'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
            'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
            'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
            'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
            'categoric': "(X['E']=='yes')",
            'boolean': "(X['D']==True)"
        },
        'TNC': {
            'integer': "(X['A']>9)",
            'float': "(X['C']>1.5)",
            'is_na': "(X['A']>9)|(X['A'].isna())",
            'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
            'already_optimal': "(X['A']>=0)",
            'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
            'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
            'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
            'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
            'categoric': "(X['E']=='yes')",
            'boolean': "(X['D']==True)"
        },
        'COBYLA': {
            'integer': "(X['A']>9)",
            'float': "(X['C']>1.5)",
            'is_na': "(X['A']>9)|(X['A'].isna())",
            'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
            'already_optimal': "(X['A']>=0)",
            'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
            'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
            'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
            'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
            'categoric': "(X['E']=='yes')",
            'boolean': "(X['D']==True)"
        },
        'SLSQP': {
            'integer': "(X['A']>9)",
            'float': "(X['C']>1.5)",
            'is_na': "(X['A']>9)|(X['A'].isna())",
            'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
            'already_optimal': "(X['A']>=0)",
            'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
            'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
            'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
            'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
            'categoric': "(X['E']=='yes')",
            'boolean': "(X['D']==True)"
        },
        'trust-constr': {
            'integer': "(X['A']>9)",
            'float': "(X['C']>1.5)",
            'is_na': "(X['A']>9)|(X['A'].isna())",
            'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
            'already_optimal': "(X['A']>=0)",
            'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
            'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
            'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
            'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
            'categoric': "(X['E']=='yes')",
            'boolean': "(X['D']==True)"
        }
    }
    return opt_rule_strings


@pytest.fixture
def _expected_rule_strings_x0_bounds():
    opt_rule_strings = {'Nelder-Mead': {'integer': "(X['A']>4.5)",
                                        'float': "(X['C']>0.0003641365574362787)",
                                        'is_na': "(X['A']>4.5)|(X['A'].isna())",
                                        'mixed': "((X['A']>5.966666666666709)&(X['C']>0.6372486347111281)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.0003641365574362787)",
                                        'already_optimal': "(X['A']>=0)",
                                        'float_with_zero_var': "(X['C']>0.0003641365574362787)&(X['ZeroVar']>=1.0)",
                                        'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                                        'float_with_all_na_is_na': "(X['C']>0.0003641365574362787)&(X['AllNa'].isna())",
                                        'multi_zero_var': "((X['C']>0.0003641365574362787)&(X['ZeroVar']>=1.0))|((X['A']>6.537012970447561)&(X['ZeroVar']>=1.0))",
                                        'categoric': "(X['E']=='yes')",
                                        'boolean': "(X['D']==True)"},
                        'Powell': {'integer': "(X['A']>0.5015693034320075)",
                                   'float': "(X['C']>0.0010925322093125105)",
                                   'is_na': "(X['A']>0.5015693034320075)|(X['A'].isna())",
                                   'mixed': "((X['A']>8.99973693287774)&(X['C']>0.9999237523200665)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.0014302632042792506)",
                                   'already_optimal': "(X['A']>=0)",
                                   'float_with_zero_var': "(X['C']>0.0010925322093125105)&(X['ZeroVar']>=1.0)",
                                   'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                                   'float_with_all_na_is_na': "(X['C']>0.0010925322093125105)&(X['AllNa'].isna())",
                                   'multi_zero_var': "((X['C']>0.0008760021490056879)&(X['ZeroVar']>=1.0))|((X['A']>2.1232419197928096)&(X['ZeroVar']>=1.0))",
                                   'categoric': "(X['E']=='yes')",
                                   'boolean': "(X['D']==True)"},
                        'CG': {'integer': "(X['A']>4.5)",
                               'float': "(X['C']>0.5001660795697812)",
                               'is_na': "(X['A']>4.5)|(X['A'].isna())",
                               'mixed': "((X['A']>4.5)&(X['C']>0.5001660795697812)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.5001660795697812)",
                               'already_optimal': "(X['A']>=0)",
                               'float_with_zero_var': "(X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0)",
                               'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                               'float_with_all_na_is_na': "(X['C']>0.5001660795697812)&(X['AllNa'].isna())",
                               'multi_zero_var': "((X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0))|((X['A']>4.5)&(X['ZeroVar']>=1.0))",
                               'categoric': "(X['E']=='yes')",
                               'boolean': "(X['D']==True)"},
                        'BFGS': {'integer': "(X['A']>4.5)",
                                 'float': "(X['C']>0.5001660795697812)",
                                 'is_na': "(X['A']>4.5)|(X['A'].isna())",
                                 'mixed': "((X['A']>4.5)&(X['C']>0.5001660795697812)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.5001660795697812)",
                                 'already_optimal': "(X['A']>=0)",
                                 'float_with_zero_var': "(X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0)",
                                 'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                                 'float_with_all_na_is_na': "(X['C']>0.5001660795697812)&(X['AllNa'].isna())",
                                 'multi_zero_var': "((X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0))|((X['A']>4.5)&(X['ZeroVar']>=1.0))",
                                 'categoric': "(X['E']=='yes')",
                                 'boolean': "(X['D']==True)"},
                        'L-BFGS-B': {'integer': "(X['A']>4.5)",
                                     'float': "(X['C']>0.5001660795697812)",
                                     'is_na': "(X['A']>4.5)|(X['A'].isna())",
                                     'mixed': "((X['A']>4.5)&(X['C']>0.5001660795697812)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.5001660795697812)",
                                     'already_optimal': "(X['A']>=0)",
                                     'float_with_zero_var': "(X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0)",
                                     'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                                     'float_with_all_na_is_na': "(X['C']>0.5001660795697812)&(X['AllNa'].isna())",
                                     'multi_zero_var': "((X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0))|((X['A']>4.5)&(X['ZeroVar']>=1.0))",
                                     'categoric': "(X['E']=='yes')",
                                     'boolean': "(X['D']==True)"},
                        'TNC': {'integer': "(X['A']>4.5)",
                                'float': "(X['C']>0.5001660795697812)",
                                'is_na': "(X['A']>4.5)|(X['A'].isna())",
                                'mixed': "((X['A']>4.5)&(X['C']>0.5001660795697812)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.5001660795697812)",
                                'already_optimal': "(X['A']>=0)",
                                'float_with_zero_var': "(X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0)",
                                'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                                'float_with_all_na_is_na': "(X['C']>0.5001660795697812)&(X['AllNa'].isna())",
                                'multi_zero_var': "((X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0))|((X['A']>4.5)&(X['ZeroVar']>=1.0))",
                                'categoric': "(X['E']=='yes')",
                                'boolean': "(X['D']==True)"},
                        'COBYLA': {'integer': "(X['A']>-0.5)",
                                   'float': "(X['C']>-0.4998339204302188)",
                                   'is_na': "(X['A']>-0.5)|(X['A'].isna())",
                                   'mixed': "((X['A']>4.5)&(X['C']>0.5001660795697812)&(X['E']=='yes')&(X['D']==True))|(X['C']>-0.4998339204302188)",
                                   'already_optimal': "(X['A']>=0)",
                                   'float_with_zero_var': "(X['C']>-0.20694070161676625)&(X['ZeroVar']>=0.29289321881345254)",
                                   'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                                   'float_with_all_na_is_na': "(X['C']>-0.4998339204302188)&(X['AllNa'].isna())",
                                   'multi_zero_var': "((X['C']>-0.077968673622132)&(X['ZeroVar']>=0.4218652468080868))|((X['A']>4.407184364166293)&(X['ZeroVar']>=0.4317521033490438))",
                                   'categoric': "(X['E']=='yes')",
                                   'boolean': "(X['D']==True)"},
                        'SLSQP': {'integer': "(X['A']>4.5)",
                                  'float': "(X['C']>0.5001660795697812)",
                                  'is_na': "(X['A']>4.5)|(X['A'].isna())",
                                  'mixed': "((X['A']>4.5)&(X['C']>0.5001660795697812)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.5001660795697812)",
                                  'already_optimal': "(X['A']>=0)",
                                  'float_with_zero_var': "(X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0)",
                                  'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                                  'float_with_all_na_is_na': "(X['C']>0.5001660795697812)&(X['AllNa'].isna())",
                                  'multi_zero_var': "((X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0))|((X['A']>4.5)&(X['ZeroVar']>=1.0))",
                                  'categoric': "(X['E']=='yes')",
                                  'boolean': "(X['D']==True)"},
                        'trust-constr': {'integer': "(X['A']>4.5)",
                                         'float': "(X['C']>0.5001660795697812)",
                                         'is_na': "(X['A']>4.5)|(X['A'].isna())",
                                         'mixed': "((X['A']>4.5)&(X['C']>0.5001660795697812)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.5001660795697812)",
                                         'already_optimal': "(X['A']>=0)",
                                         'float_with_zero_var': "(X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0)",
                                         'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                                         'float_with_all_na_is_na': "(X['C']>0.5001660795697812)&(X['AllNa'].isna())",
                                         'multi_zero_var': "((X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0))|((X['A']>4.5)&(X['ZeroVar']>=1.0))",
                                         'categoric': "(X['E']=='yes')",
                                         'boolean': "(X['D']==True)"}
                        }
    return opt_rule_strings


@ pytest.fixture
def _expected_rule_strings_weighted():
    return {
        'Nelder-Mead': {'integer': "(X['A']>8.549999999999999)",
                        'float': "(X['C']>1.5)",
                        'is_na': "(X['A']>8.549999999999999)|(X['A'].isna())",
                        'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                        'already_optimal': "(X['A']>=0)",
                        'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                        'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                        'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                        'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                        'categoric': "(X['E']=='yes')",
                        'boolean': "(X['D']==True)"},
        'Powell': {'integer': "(X['A']>9)",
                   'float': "(X['C']>1.5)",
                   'is_na': "(X['A']>9)|(X['A'].isna())",
                   'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                   'already_optimal': "(X['A']>=0)",
                   'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                   'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                   'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                   'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                   'categoric': "(X['E']=='yes')",
                   'boolean': "(X['D']==True)"},
        'CG': {'integer': "(X['A']>9)",
               'float': "(X['C']>1.5)",
               'is_na': "(X['A']>9)|(X['A'].isna())",
               'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
               'already_optimal': "(X['A']>=0)",
               'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
               'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
               'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
               'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
               'categoric': "(X['E']=='yes')",
               'boolean': "(X['D']==True)"},
        'BFGS': {'integer': "(X['A']>9)",
                 'float': "(X['C']>1.5)",
                 'is_na': "(X['A']>9)|(X['A'].isna())",
                 'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                 'already_optimal': "(X['A']>=0)",
                 'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                 'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                 'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                 'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                 'categoric': "(X['E']=='yes')",
                 'boolean': "(X['D']==True)"},
        'L-BFGS-B': {'integer': "(X['A']>9)",
                     'float': "(X['C']>1.5)",
                     'is_na': "(X['A']>9)|(X['A'].isna())",
                     'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                     'already_optimal': "(X['A']>=0)",
                     'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                     'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                     'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                     'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                     'categoric': "(X['E']=='yes')",
                     'boolean': "(X['D']==True)"},
        'TNC': {'integer': "(X['A']>9)",
                'float': "(X['C']>1.5)",
                'is_na': "(X['A']>9)|(X['A'].isna())",
                'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                'already_optimal': "(X['A']>=0)",
                'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                'categoric': "(X['E']=='yes')",
                'boolean': "(X['D']==True)"},
        'COBYLA': {'integer': "(X['A']>9)",
                   'float': "(X['C']>1.5)",
                   'is_na': "(X['A']>9)|(X['A'].isna())",
                   'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                   'already_optimal': "(X['A']>=0)",
                   'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                   'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                   'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                   'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                   'categoric': "(X['E']=='yes')",
                   'boolean': "(X['D']==True)"},
        'SLSQP': {'integer': "(X['A']>9)",
                  'float': "(X['C']>1.5)",
                  'is_na': "(X['A']>9)|(X['A'].isna())",
                  'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                  'already_optimal': "(X['A']>=0)",
                  'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                  'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                  'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                  'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                  'categoric': "(X['E']=='yes')",
                  'boolean': "(X['D']==True)"},
        'trust-constr': {'integer': "(X['A']>9)",
                         'float': "(X['C']>1.5)",
                         'is_na': "(X['A']>9)|(X['A'].isna())",
                         'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                         'already_optimal': "(X['A']>=0)",
                         'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                         'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                         'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                         'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                         'categoric': "(X['E']=='yes')",
                         'boolean': "(X['D']==True)"}
    }


@ pytest.fixture
def _expected_rule_strings_weighted_x0_bounds():
    return {
        'Nelder-Mead': {'integer': "(X['A']>4.5)",
                        'float': "(X['C']>0.5056366460650756)",
                        'is_na': "(X['A']>4.5)|(X['A'].isna())",
                        'mixed': "((X['A']>4.544030630550376)&(X['C']>0.5062001821281391)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.5059645205154579)",
                        'already_optimal': "(X['A']>=0)",
                        'float_with_zero_var': "(X['C']>0.5056366460650756)&(X['ZeroVar']>=1.0)",
                        'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                        'float_with_all_na_is_na': "(X['C']>0.5056366460650756)&(X['AllNa'].isna())",
                        'multi_zero_var': "((X['C']>0.0003641365574362787)&(X['ZeroVar']>=1.0))|((X['A']>6.537012970447561)&(X['ZeroVar']>=1.0))",
                        'categoric': "(X['E']=='yes')",
                        'boolean': "(X['D']==True)"},
        'Powell': {'integer': "(X['A']>0.5015693034320075)",
                   'float': "(X['C']>0.14610333258470667)",
                   'is_na': "(X['A']>0.5015693034320075)|(X['A'].isna())",
                   'mixed': "((X['A']>8.677776699611885)&(X['C']>0.9914509400603203)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.21861873480767938)",
                   'already_optimal': "(X['A']>=0)",
                   'float_with_zero_var': "(X['C']>0.14610333258470667)&(X['ZeroVar']>=1.0)",
                   'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                   'float_with_all_na_is_na': "(X['C']>0.14610333258470667)&(X['AllNa'].isna())",
                   'multi_zero_var': "((X['C']>0.000621823691021652)&(X['ZeroVar']>=1.0))|((X['A']>2.1240360255929973)&(X['ZeroVar']>=1.0))",
                   'categoric': "(X['E']=='yes')",
                   'boolean': "(X['D']==True)"},
        'CG': {'integer': "(X['A']>4.5)",
               'float': "(X['C']>0.5001660795697812)",
               'is_na': "(X['A']>4.5)|(X['A'].isna())",
               'mixed': "((X['A']>4.5)&(X['C']>0.5001660795697812)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.5001660795697812)",
               'already_optimal': "(X['A']>=0)",
               'float_with_zero_var': "(X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0)",
               'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
               'float_with_all_na_is_na': "(X['C']>0.5001660795697812)&(X['AllNa'].isna())",
               'multi_zero_var': "((X['C']>0.5001660795697812)&(X['ZeroVar']>=0.9999608585886306))|((X['A']>4.5)&(X['ZeroVar']>=1.0000330891590339))",
               'categoric': "(X['E']=='yes')",
               'boolean': "(X['D']==True)"},
        'BFGS': {'integer': "(X['A']>4.5)",
                 'float': "(X['C']>0.5001660795697812)",
                 'is_na': "(X['A']>4.5)|(X['A'].isna())",
                 'mixed': "((X['A']>4.5)&(X['C']>0.5001660795697812)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.5001660795697812)",
                 'already_optimal': "(X['A']>=0)",
                 'float_with_zero_var': "(X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0)",
                 'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                 'float_with_all_na_is_na': "(X['C']>0.5001660795697812)&(X['AllNa'].isna())",
                 'multi_zero_var': "((X['C']>0.5001660795697812)&(X['ZeroVar']>=0.9999608585886306))|((X['A']>4.5)&(X['ZeroVar']>=1.0000330891590339))",
                 'categoric': "(X['E']=='yes')",
                 'boolean': "(X['D']==True)"},
        'L-BFGS-B': {'integer': "(X['A']>4.5)",
                     'float': "(X['C']>0.5001660795697812)",
                     'is_na': "(X['A']>4.5)|(X['A'].isna())",
                     'mixed': "((X['A']>4.5)&(X['C']>0.5001660795697812)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.5001660795697812)",
                     'already_optimal': "(X['A']>=0)",
                     'float_with_zero_var': "(X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0)",
                     'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                     'float_with_all_na_is_na': "(X['C']>0.5001660795697812)&(X['AllNa'].isna())",
                     'multi_zero_var': "((X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0))|((X['A']>4.5)&(X['ZeroVar']>=1.0))",
                     'categoric': "(X['E']=='yes')",
                     'boolean': "(X['D']==True)"},
        'TNC': {'integer': "(X['A']>4.5)",
                'float': "(X['C']>0.5001660795697812)",
                'is_na': "(X['A']>4.5)|(X['A'].isna())",
                'mixed': "((X['A']>4.5)&(X['C']>0.5001660795697812)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.5001660795697812)",
                'already_optimal': "(X['A']>=0)",
                'float_with_zero_var': "(X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0)",
                'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                'float_with_all_na_is_na': "(X['C']>0.5001660795697812)&(X['AllNa'].isna())",
                'multi_zero_var': "((X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0))|((X['A']>4.5)&(X['ZeroVar']>=1.0))",
                'categoric': "(X['E']=='yes')",
                'boolean': "(X['D']==True)"},
        'COBYLA': {'integer': "(X['A']>-0.5)",
                   'float': "(X['C']>-0.4998339204302188)",
                   'is_na': "(X['A']>-0.5)|(X['A'].isna())",
                   'mixed': "((X['A']>4.5)&(X['C']>0.5001660795697812)&(X['E']=='yes')&(X['D']==True))|(X['C']>-0.4998339204302188)",
                   'already_optimal': "(X['A']>=0)",
                   'float_with_zero_var': "(X['C']>0.15012050213720127)&(X['ZeroVar']>=-0.07234657931438221)",
                   'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                   'float_with_all_na_is_na': "(X['C']>-0.4998339204302188)&(X['AllNa'].isna())",
                   'multi_zero_var': "((X['C']>-0.10251065272063686)&(X['ZeroVar']>=0.3973232677095818))|((X['A']>4.381746741970505)&(X['ZeroVar']>=2.50948766391042))",
                   'categoric': "(X['E']=='yes')",
                   'boolean': "(X['D']==True)"},
        'SLSQP': {'integer': "(X['A']>4.5)",
                  'float': "(X['C']>0.5001660795697812)",
                  'is_na': "(X['A']>4.5)|(X['A'].isna())",
                  'mixed': "((X['A']>4.5)&(X['C']>0.5001660795697812)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.5001660795697812)",
                  'already_optimal': "(X['A']>=0)",
                  'float_with_zero_var': "(X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0)",
                  'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                  'float_with_all_na_is_na': "(X['C']>0.5001660795697812)&(X['AllNa'].isna())",
                  'multi_zero_var': "((X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0))|((X['A']>4.5)&(X['ZeroVar']>=1.0))",
                  'categoric': "(X['E']=='yes')",
                  'boolean': "(X['D']==True)"},
        'trust-constr': {'integer': "(X['A']>4.5)",
                         'float': "(X['C']>0.5001660795697812)",
                         'is_na': "(X['A']>4.5)|(X['A'].isna())",
                         'mixed': "((X['A']>4.5)&(X['C']>0.5001660795697812)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.5001660795697812)",
                         'already_optimal': "(X['A']>=0)",
                         'float_with_zero_var': "(X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0)",
                         'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                         'float_with_all_na_is_na': "(X['C']>0.5001660795697812)&(X['AllNa'].isna())",
                         'multi_zero_var': "((X['C']>0.5001660795697812)&(X['ZeroVar']>=1.0))|((X['A']>4.5)&(X['ZeroVar']>=1.0))",
                         'categoric': "(X['E']=='yes')",
                         'boolean': "(X['D']==True)"}
    }


@ pytest.fixture
def _expected_rule_strings_unlabelled():
    return {
        'Nelder-Mead': {'integer': "(X['A']>9)",
                        'float': "(X['C']>1.5)",
                        'is_na': "(X['A']>9)|(X['A'].isna())",
                        'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                        'already_optimal': "(X['A']>=0.00025)",
                        'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                        'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                        'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                        'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                        'categoric': "(X['E']=='yes')",
                        'boolean': "(X['D']==True)"},
        'Powell': {'integer': "(X['A']>9)",
                   'float': "(X['C']>1.5)",
                   'is_na': "(X['A']>9)|(X['A'].isna())",
                   'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                   'already_optimal': "(X['A']>=177.53363890059836)",
                   'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                   'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                   'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                   'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                   'categoric': "(X['E']=='yes')",
                   'boolean': "(X['D']==True)"},
        'CG': {'integer': "(X['A']>9)",
               'float': "(X['C']>1.5)",
               'is_na': "(X['A']>9)|(X['A'].isna())",
               'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
               'already_optimal': "(X['A']>=5.125898936452442e-05)",
               'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
               'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
               'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
               'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
               'categoric': "(X['E']=='yes')",
               'boolean': "(X['D']==True)"},
        'BFGS': {'integer': "(X['A']>9)",
                 'float': "(X['C']>1.5)",
                 'is_na': "(X['A']>9)|(X['A'].isna())",
                 'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                 'already_optimal': "(X['A']>=5.125898936452442e-05)",
                 'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                 'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                 'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                 'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                 'categoric': "(X['E']=='yes')",
                 'boolean': "(X['D']==True)"},
        'L-BFGS-B': {'integer': "(X['A']>9)",
                     'float': "(X['C']>1.5)",
                     'is_na': "(X['A']>9)|(X['A'].isna())",
                     'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                     'already_optimal': "(X['A']>=3.595679261195249e-06)",
                     'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                     'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                     'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                     'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                     'categoric': "(X['E']=='yes')",
                     'boolean': "(X['D']==True)"},
        'TNC': {'integer': "(X['A']>9)",
                'float': "(X['C']>1.5)",
                'is_na': "(X['A']>9)|(X['A'].isna())",
                'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                'already_optimal': "(X['A']>=1.4901161193847656e-08)",
                'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                'categoric': "(X['E']=='yes')",
                'boolean': "(X['D']==True)"},
        'COBYLA': {'integer': "(X['A']>9)",
                   'float': "(X['C']>1.5)",
                   'is_na': "(X['A']>9)|(X['A'].isna())",
                   'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                   'already_optimal': "(X['A']>=10.0)",
                   'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                   'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                   'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                   'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                   'categoric': "(X['E']=='yes')",
                   'boolean': "(X['D']==True)"},
        'SLSQP': {'integer': "(X['A']>9)",
                  'float': "(X['C']>1.5)",
                  'is_na': "(X['A']>9)|(X['A'].isna())",
                  'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                  'already_optimal': "(X['A']>=11935095848.960007)",
                  'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                  'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                  'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                  'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                  'categoric': "(X['E']=='yes')",
                  'boolean': "(X['D']==True)"},
        'trust-constr': {'integer': "(X['A']>9)",
                         'float': "(X['C']>1.5)",
                         'is_na': "(X['A']>9)|(X['A'].isna())",
                         'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                         'already_optimal': "(X['A']>=2.7020533257768187)",
                         'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                         'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                         'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                         'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                         'categoric': "(X['E']=='yes')",
                         'boolean': "(X['D']==True)"}
    }


@ pytest.fixture
def _expected_rule_strings_unlabelled_x0_bounds():
    return {
        'Nelder-Mead': {'integer': "(X['A']>9)",
                        'float': "(X['C']>1.5)",
                        'is_na': "(X['A']>9)|(X['A'].isna())",
                        'mixed': "((X['A']>7.775353965311879)&(X['C']>0.9884794975944149)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.9884919578072813)",
                        'already_optimal': "(X['A']>=4.5)",
                        'float_with_zero_var': "(X['C']>0.999968022582126)&(X['ZeroVar']>=1.0)",
                        'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                        'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                        'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                        'categoric': "(X['E']=='yes')",
                        'boolean': "(X['D']==True)"},
        'Powell': {'integer': "(X['A']>9)",
                   'float': "(X['C']>0.9882521688325012)",
                   'is_na': "(X['A']>9)|(X['A'].isna())",
                   'mixed': "((X['A']>7.7943442656079)&(X['C']>0.896293283056058)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.9993343463105192)",
                   'already_optimal': "(X['A']>=8.203192518090468)",
                   'float_with_zero_var': "(X['C']>0.9882521688325012)&(X['ZeroVar']>=1.0)",
                   'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                   'float_with_all_na_is_na': "(X['C']>0.9882521688325012)&(X['AllNa'].isna())",
                   'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                   'categoric': "(X['E']=='yes')",
                   'boolean': "(X['D']==True)"},
        'CG': {'integer': "(X['A']>9)",
               'float': "(X['C']>1.5)",
               'is_na': "(X['A']>9)|(X['A'].isna())",
               'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
               'already_optimal': "(X['A']>=4.5)",
               'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
               'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
               'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
               'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
               'categoric': "(X['E']=='yes')",
               'boolean': "(X['D']==True)"},
        'BFGS': {'integer': "(X['A']>9)",
                 'float': "(X['C']>1.5)",
                 'is_na': "(X['A']>9)|(X['A'].isna())",
                 'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                 'already_optimal': "(X['A']>=4.5)",
                 'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                 'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                 'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                 'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                 'categoric': "(X['E']=='yes')",
                 'boolean': "(X['D']==True)"},
        'L-BFGS-B': {'integer': "(X['A']>9)",
                     'float': "(X['C']>1.5)",
                     'is_na': "(X['A']>9)|(X['A'].isna())",
                     'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                     'already_optimal': "(X['A']>=4.5)",
                     'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                     'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                     'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                     'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                     'categoric': "(X['E']=='yes')",
                     'boolean': "(X['D']==True)"},
        'TNC': {'integer': "(X['A']>9)",
                'float': "(X['C']>1.5)",
                'is_na': "(X['A']>9)|(X['A'].isna())",
                'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                'already_optimal': "(X['A']>=4.5)",
                'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                'categoric': "(X['E']=='yes')",
                'boolean': "(X['D']==True)"},
        'COBYLA': {'integer': "(X['A']>9)",
                   'float': "(X['C']>1.5)",
                   'is_na': "(X['A']>9)|(X['A'].isna())",
                   'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                   'already_optimal': "(X['A']>=9.5)",
                   'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                   'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                   'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                   'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                   'categoric': "(X['E']=='yes')",
                   'boolean': "(X['D']==True)"},
        'SLSQP': {'integer': "(X['A']>9)",
                  'float': "(X['C']>1.5)",
                  'is_na': "(X['A']>9)|(X['A'].isna())",
                  'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                  'already_optimal': "(X['A']>=4.5)",
                  'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                  'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                  'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                  'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                  'categoric': "(X['E']=='yes')",
                  'boolean': "(X['D']==True)"},
        'trust-constr': {'integer': "(X['A']>9)",
                         'float': "(X['C']>1.5)",
                         'is_na': "(X['A']>9)|(X['A'].isna())",
                         'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                         'already_optimal': "(X['A']>=4.5)",
                         'float_with_zero_var': "(X['C']>1.5)&(X['ZeroVar']>=1)",
                         'float_with_all_na_greater': "(X['C']>1.5)&(X['AllNa']>1)",
                         'float_with_all_na_is_na': "(X['C']>1.5)&(X['AllNa'].isna())",
                         'multi_zero_var': "((X['C']>1.5)&(X['ZeroVar']>=1))|((X['A']>9)&(X['ZeroVar']>=1))",
                         'categoric': "(X['E']=='yes')",
                         'boolean': "(X['D']==True)"}
    }


@ pytest.fixture
def _expected_X_rules_mean():
    X_rules_means = {
        'Nelder-Mead': {
            'integer': 0.10428957104289571,
            'float': 0.0,
            'is_na': 0.10438956104389562,
            'mixed': 0.0,
            'already_optimal': 0.9999000099990001,
            'float_with_zero_var': 0.0,
            'float_with_all_na_greater': 0.0,
            'float_with_all_na_is_na': 0.0,
            'multi_zero_var': 0.0,
            'categoric': 0.49995000499950004,
            'boolean': 0.49995000499950004
        },
        'Powell': {
            'integer': 0.0,
            'float': 0.0,
            'is_na': 9.999000099990002e-05,
            'mixed': 0.0,
            'already_optimal': 0.9999000099990001,
            'float_with_zero_var': 0.0,
            'float_with_all_na_greater': 0.0,
            'float_with_all_na_is_na': 0.0,
            'multi_zero_var': 0.0,
            'categoric': 0.49995000499950004,
            'boolean': 0.49995000499950004
        },
        'CG': {
            'integer': 0.0,
            'float': 0.0,
            'is_na': 9.999000099990002e-05,
            'mixed': 0.0,
            'already_optimal': 0.9999000099990001,
            'float_with_zero_var': 0.0,
            'float_with_all_na_greater': 0.0,
            'float_with_all_na_is_na': 0.0,
            'multi_zero_var': 0.0,
            'categoric': 0.49995000499950004,
            'boolean': 0.49995000499950004
        },
        'BFGS': {
            'integer': 0.0,
            'float': 0.0,
            'is_na': 9.999000099990002e-05,
            'mixed': 0.0,
            'already_optimal': 0.9999000099990001,
            'float_with_zero_var': 0.0,
            'float_with_all_na_greater': 0.0,
            'float_with_all_na_is_na': 0.0,
            'multi_zero_var': 0.0,
            'categoric': 0.49995000499950004,
            'boolean': 0.49995000499950004
        },
        'L-BFGS-B': {
            'integer': 0.0,
            'float': 0.0,
            'is_na': 9.999000099990002e-05,
            'mixed': 0.0,
            'already_optimal': 0.9999000099990001,
            'float_with_zero_var': 0.0,
            'float_with_all_na_greater': 0.0,
            'float_with_all_na_is_na': 0.0,
            'multi_zero_var': 0.0,
            'categoric': 0.49995000499950004,
            'boolean': 0.49995000499950004
        },
        'TNC': {
            'integer': 0.0,
            'float': 0.0,
            'is_na': 9.999000099990002e-05,
            'mixed': 0.0,
            'already_optimal': 0.9999000099990001,
            'float_with_zero_var': 0.0,
            'float_with_all_na_greater': 0.0,
            'float_with_all_na_is_na': 0.0,
            'multi_zero_var': 0.0,
            'categoric': 0.49995000499950004,
            'boolean': 0.49995000499950004
        },
        'COBYLA': {
            'integer': 0.0,
            'float': 0.0,
            'is_na': 9.999000099990002e-05,
            'mixed': 0.0,
            'already_optimal': 0.9999000099990001,
            'float_with_zero_var': 0.0,
            'float_with_all_na_greater': 0.0,
            'float_with_all_na_is_na': 0.0,
            'multi_zero_var': 0.0,
            'categoric': 0.49995000499950004,
            'boolean': 0.49995000499950004
        },
        'SLSQP': {
            'integer': 0.0,
            'float': 0.0,
            'is_na': 9.999000099990002e-05,
            'mixed': 0.0,
            'already_optimal': 0.9999000099990001,
            'float_with_zero_var': 0.0,
            'float_with_all_na_greater': 0.0,
            'float_with_all_na_is_na': 0.0,
            'multi_zero_var': 0.0,
            'categoric': 0.49995000499950004,
            'boolean': 0.49995000499950004
        },
        'trust-constr': {
            'integer': 0.0,
            'float': 0.0,
            'is_na': 9.999000099990002e-05,
            'mixed': 0.0,
            'already_optimal': 0.9999000099990001,
            'float_with_zero_var': 0.0,
            'float_with_all_na_greater': 0.0,
            'float_with_all_na_is_na': 0.0,
            'multi_zero_var': 0.0,
            'categoric': 0.49995000499950004,
            'boolean': 0.49995000499950004
        }
    }
    return X_rules_means


@ pytest.fixture
def _expected_X_rules_mean_x0_bounds():
    X_rules_means = {
        'Nelder-Mead': {'integer': 0.5061493850614939,
                        'float': 0.9998000199980002,
                        'is_na': 0.5062493750624938,
                        'mixed': 0.9998000199980002,
                        'already_optimal': 0.9999000099990001,
                        'float_with_zero_var': 0.9998000199980002,
                        'float_with_all_na_greater': 0.0,
                        'float_with_all_na_is_na': 0.9998000199980002,
                        'multi_zero_var': 0.9998000199980002,
                        'categoric': 0.49995000499950004,
                        'boolean': 0.49995000499950004},
        'Powell': {'integer': 0.9032096790320968,
                   'float': 0.9988001199880012,
                   'is_na': 0.9033096690330967,
                   'mixed': 0.9986001399860014,
                   'already_optimal': 0.9999000099990001,
                   'float_with_zero_var': 0.9988001199880012,
                   'float_with_all_na_greater': 0.0,
                   'float_with_all_na_is_na': 0.9988001199880012,
                   'multi_zero_var': 0.9996000399960004,
                   'categoric': 0.49995000499950004,
                   'boolean': 0.49995000499950004},
        'CG': {'integer': 0.5061493850614939,
               'float': 0.49865013498650135,
               'is_na': 0.5062493750624938,
               'mixed': 0.49865013498650135,
               'already_optimal': 0.9999000099990001,
               'float_with_zero_var': 0.49865013498650135,
               'float_with_all_na_greater': 0.0,
               'float_with_all_na_is_na': 0.49865013498650135,
               'multi_zero_var': 0.7498250174982501,
               'categoric': 0.49995000499950004,
               'boolean': 0.49995000499950004},
        'BFGS': {'integer': 0.5061493850614939,
                 'float': 0.49865013498650135,
                 'is_na': 0.5062493750624938,
                 'mixed': 0.49865013498650135,
                 'already_optimal': 0.9999000099990001,
                 'float_with_zero_var': 0.49865013498650135,
                 'float_with_all_na_greater': 0.0,
                 'float_with_all_na_is_na': 0.49865013498650135,
                 'multi_zero_var': 0.7498250174982501,
                 'categoric': 0.49995000499950004,
                 'boolean': 0.49995000499950004},
        'L-BFGS-B': {'integer': 0.5061493850614939,
                     'float': 0.49865013498650135,
                     'is_na': 0.5062493750624938,
                     'mixed': 0.49865013498650135,
                     'already_optimal': 0.9999000099990001,
                     'float_with_zero_var': 0.49865013498650135,
                     'float_with_all_na_greater': 0.0,
                     'float_with_all_na_is_na': 0.49865013498650135,
                     'multi_zero_var': 0.7498250174982501,
                     'categoric': 0.49995000499950004,
                     'boolean': 0.49995000499950004},
        'TNC': {'integer': 0.5061493850614939,
                'float': 0.49865013498650135,
                'is_na': 0.5062493750624938,
                'mixed': 0.49865013498650135,
                'already_optimal': 0.9999000099990001,
                'float_with_zero_var': 0.49865013498650135,
                'float_with_all_na_greater': 0.0,
                'float_with_all_na_is_na': 0.49865013498650135,
                'multi_zero_var': 0.7498250174982501,
                'categoric': 0.49995000499950004,
                'boolean': 0.49995000499950004},
        'COBYLA': {'integer': 0.9999000099990001,
                   'float': 0.9999000099990001,
                   'is_na': 1.0,
                   'mixed': 0.9999000099990001,
                   'already_optimal': 0.9999000099990001,
                   'float_with_zero_var': 0.9999000099990001,
                   'float_with_all_na_greater': 0.0,
                   'float_with_all_na_is_na': 0.9999000099990001,
                   'multi_zero_var': 0.9999000099990001,
                   'categoric': 0.49995000499950004,
                   'boolean': 0.49995000499950004},
        'SLSQP': {'integer': 0.5061493850614939,
                  'float': 0.49865013498650135,
                  'is_na': 0.5062493750624938,
                  'mixed': 0.49865013498650135,
                  'already_optimal': 0.9999000099990001,
                  'float_with_zero_var': 0.49865013498650135,
                  'float_with_all_na_greater': 0.0,
                  'float_with_all_na_is_na': 0.49865013498650135,
                  'multi_zero_var': 0.7498250174982501,
                  'categoric': 0.49995000499950004,
                  'boolean': 0.49995000499950004},
        'trust-constr': {'integer': 0.5061493850614939,
                         'float': 0.49865013498650135,
                         'is_na': 0.5062493750624938,
                         'mixed': 0.49865013498650135,
                         'already_optimal': 0.9999000099990001,
                         'float_with_zero_var': 0.49865013498650135,
                         'float_with_all_na_greater': 0.0,
                         'float_with_all_na_is_na': 0.49865013498650135,
                         'multi_zero_var': 0.7498250174982501,
                         'categoric': 0.49995000499950004,
                         'boolean': 0.49995000499950004}
    }
    return X_rules_means


@ pytest.fixture
def _expected_X_rules_mean_weighted():
    return {
        'Nelder-Mead': {'integer': 0.10428957104289571,
                        'float': 0.0,
                        'is_na': 0.10438956104389562,
                        'mixed': 0.0,
                        'already_optimal': 0.9999000099990001,
                        'float_with_zero_var': 0.0,
                        'float_with_all_na_greater': 0.0,
                        'float_with_all_na_is_na': 0.0,
                        'multi_zero_var': 0.0,
                        'categoric': 0.49995000499950004,
                        'boolean': 0.49995000499950004},
        'Powell': {'integer': 0.0,
                   'float': 0.0,
                   'is_na': 9.999000099990002e-05,
                   'mixed': 0.0,
                   'already_optimal': 0.9999000099990001,
                   'float_with_zero_var': 0.0,
                   'float_with_all_na_greater': 0.0,
                   'float_with_all_na_is_na': 0.0,
                   'multi_zero_var': 0.0,
                   'categoric': 0.49995000499950004,
                   'boolean': 0.49995000499950004},
        'CG': {'integer': 0.0,
               'float': 0.0,
               'is_na': 9.999000099990002e-05,
               'mixed': 0.0,
               'already_optimal': 0.9999000099990001,
               'float_with_zero_var': 0.0,
               'float_with_all_na_greater': 0.0,
               'float_with_all_na_is_na': 0.0,
               'multi_zero_var': 0.0,
               'categoric': 0.49995000499950004,
               'boolean': 0.49995000499950004},
        'BFGS': {'integer': 0.0,
                 'float': 0.0,
                 'is_na': 9.999000099990002e-05,
                 'mixed': 0.0,
                 'already_optimal': 0.9999000099990001,
                 'float_with_zero_var': 0.0,
                 'float_with_all_na_greater': 0.0,
                 'float_with_all_na_is_na': 0.0,
                 'multi_zero_var': 0.0,
                 'categoric': 0.49995000499950004,
                 'boolean': 0.49995000499950004},
        'L-BFGS-B': {'integer': 0.0,
                     'float': 0.0,
                     'is_na': 9.999000099990002e-05,
                     'mixed': 0.0,
                     'already_optimal': 0.9999000099990001,
                     'float_with_zero_var': 0.0,
                     'float_with_all_na_greater': 0.0,
                     'float_with_all_na_is_na': 0.0,
                     'multi_zero_var': 0.0,
                     'categoric': 0.49995000499950004,
                     'boolean': 0.49995000499950004},
        'TNC': {'integer': 0.0,
                'float': 0.0,
                'is_na': 9.999000099990002e-05,
                'mixed': 0.0,
                'already_optimal': 0.9999000099990001,
                'float_with_zero_var': 0.0,
                'float_with_all_na_greater': 0.0,
                'float_with_all_na_is_na': 0.0,
                'multi_zero_var': 0.0,
                'categoric': 0.49995000499950004,
                'boolean': 0.49995000499950004},
        'COBYLA': {'integer': 0.0,
                   'float': 0.0,
                   'is_na': 9.999000099990002e-05,
                   'mixed': 0.0,
                   'already_optimal': 0.9999000099990001,
                   'float_with_zero_var': 0.0,
                   'float_with_all_na_greater': 0.0,
                   'float_with_all_na_is_na': 0.0,
                   'multi_zero_var': 0.0,
                   'categoric': 0.49995000499950004,
                   'boolean': 0.49995000499950004},
        'SLSQP': {'integer': 0.0,
                  'float': 0.0,
                  'is_na': 9.999000099990002e-05,
                  'mixed': 0.0,
                  'already_optimal': 0.9999000099990001,
                  'float_with_zero_var': 0.0,
                  'float_with_all_na_greater': 0.0,
                  'float_with_all_na_is_na': 0.0,
                  'multi_zero_var': 0.0,
                  'categoric': 0.49995000499950004,
                  'boolean': 0.49995000499950004},
        'trust-constr': {'integer': 0.0,
                         'float': 0.0,
                         'is_na': 9.999000099990002e-05,
                         'mixed': 0.0,
                         'already_optimal': 0.9999000099990001,
                         'float_with_zero_var': 0.0,
                         'float_with_all_na_greater': 0.0,
                         'float_with_all_na_is_na': 0.0,
                         'multi_zero_var': 0.0,
                         'categoric': 0.49995000499950004,
                         'boolean': 0.49995000499950004}
    }


@ pytest.fixture
def _expected_X_rules_mean_weighted_x0_bounds():
    return {
        'Nelder-Mead': {'integer': 0.5061493850614939,
                        'float': 0.49375062493750627,
                        'is_na': 0.5062493750624938,
                        'mixed': 0.49365063493650635,
                        'already_optimal': 0.9999000099990001,
                        'float_with_zero_var': 0.49375062493750627,
                        'float_with_all_na_greater': 0.0,
                        'float_with_all_na_is_na': 0.49375062493750627,
                        'multi_zero_var': 0.9998000199980002,
                        'categoric': 0.49995000499950004,
                        'boolean': 0.49995000499950004},
        'Powell': {'integer': 0.9032096790320968,
                   'float': 0.8528147185281472,
                   'is_na': 0.9033096690330967,
                   'mixed': 0.7774222577742226,
                   'already_optimal': 0.9999000099990001,
                   'float_with_zero_var': 0.8528147185281472,
                   'float_with_all_na_greater': 0.0,
                   'float_with_all_na_is_na': 0.8528147185281472,
                   'multi_zero_var': 0.9997000299970003,
                   'categoric': 0.49995000499950004,
                   'boolean': 0.49995000499950004},
        'CG': {'integer': 0.5061493850614939,
               'float': 0.49865013498650135,
               'is_na': 0.5062493750624938,
               'mixed': 0.49865013498650135,
               'already_optimal': 0.9999000099990001,
               'float_with_zero_var': 0.49865013498650135,
               'float_with_all_na_greater': 0.0,
               'float_with_all_na_is_na': 0.49865013498650135,
               'multi_zero_var': 0.49865013498650135,
               'categoric': 0.49995000499950004,
               'boolean': 0.49995000499950004},
        'BFGS': {'integer': 0.5061493850614939,
                 'float': 0.49865013498650135,
                 'is_na': 0.5062493750624938,
                 'mixed': 0.49865013498650135,
                 'already_optimal': 0.9999000099990001,
                 'float_with_zero_var': 0.49865013498650135,
                 'float_with_all_na_greater': 0.0,
                 'float_with_all_na_is_na': 0.49865013498650135,
                 'multi_zero_var': 0.49865013498650135,
                 'categoric': 0.49995000499950004,
                 'boolean': 0.49995000499950004},
        'L-BFGS-B': {'integer': 0.5061493850614939,
                     'float': 0.49865013498650135,
                     'is_na': 0.5062493750624938,
                     'mixed': 0.49865013498650135,
                     'already_optimal': 0.9999000099990001,
                     'float_with_zero_var': 0.49865013498650135,
                     'float_with_all_na_greater': 0.0,
                     'float_with_all_na_is_na': 0.49865013498650135,
                     'multi_zero_var': 0.7498250174982501,
                     'categoric': 0.49995000499950004,
                     'boolean': 0.49995000499950004},
        'TNC': {'integer': 0.5061493850614939,
                'float': 0.49865013498650135,
                'is_na': 0.5062493750624938,
                'mixed': 0.49865013498650135,
                'already_optimal': 0.9999000099990001,
                'float_with_zero_var': 0.49865013498650135,
                'float_with_all_na_greater': 0.0,
                'float_with_all_na_is_na': 0.49865013498650135,
                'multi_zero_var': 0.7498250174982501,
                'categoric': 0.49995000499950004,
                'boolean': 0.49995000499950004},
        'COBYLA': {'integer': 0.9999000099990001,
                   'float': 0.9999000099990001,
                   'is_na': 1.0,
                   'mixed': 0.9999000099990001,
                   'already_optimal': 0.9999000099990001,
                   'float_with_zero_var': 0.8487151284871512,
                   'float_with_all_na_greater': 0.0,
                   'float_with_all_na_is_na': 0.9999000099990001,
                   'multi_zero_var': 0.9999000099990001,
                   'categoric': 0.49995000499950004,
                   'boolean': 0.49995000499950004},
        'SLSQP': {'integer': 0.5061493850614939,
                  'float': 0.49865013498650135,
                  'is_na': 0.5062493750624938,
                  'mixed': 0.49865013498650135,
                  'already_optimal': 0.9999000099990001,
                  'float_with_zero_var': 0.49865013498650135,
                  'float_with_all_na_greater': 0.0,
                  'float_with_all_na_is_na': 0.49865013498650135,
                  'multi_zero_var': 0.7498250174982501,
                  'categoric': 0.49995000499950004,
                  'boolean': 0.49995000499950004},
        'trust-constr': {'integer': 0.5061493850614939,
                         'float': 0.49865013498650135,
                         'is_na': 0.5062493750624938,
                         'mixed': 0.49865013498650135,
                         'already_optimal': 0.9999000099990001,
                         'float_with_zero_var': 0.49865013498650135,
                         'float_with_all_na_greater': 0.0,
                         'float_with_all_na_is_na': 0.49865013498650135,
                         'multi_zero_var': 0.7498250174982501,
                         'categoric': 0.49995000499950004,
                         'boolean': 0.49995000499950004}
    }


@ pytest.fixture
def _expected_X_rules_mean_unlabelled():
    return {
        'Nelder-Mead': {'integer': 0.0,
                        'float': 0.0,
                        'is_na': 9.999000099990002e-05,
                        'mixed': 0.0,
                        'already_optimal': 0.9032096790320968,
                        'float_with_zero_var': 0.0,
                        'float_with_all_na_greater': 0.0,
                        'float_with_all_na_is_na': 0.0,
                        'multi_zero_var': 0.0,
                        'categoric': 0.49995000499950004,
                        'boolean': 0.49995000499950004},
        'Powell': {'integer': 0.0,
                   'float': 0.0,
                   'is_na': 9.999000099990002e-05,
                   'mixed': 0.0,
                   'already_optimal': 0.0,
                   'float_with_zero_var': 0.0,
                   'float_with_all_na_greater': 0.0,
                   'float_with_all_na_is_na': 0.0,
                   'multi_zero_var': 0.0,
                   'categoric': 0.49995000499950004,
                   'boolean': 0.49995000499950004},
        'CG': {'integer': 0.0,
               'float': 0.0,
               'is_na': 9.999000099990002e-05,
               'mixed': 0.0,
               'already_optimal': 0.9032096790320968,
               'float_with_zero_var': 0.0,
               'float_with_all_na_greater': 0.0,
               'float_with_all_na_is_na': 0.0,
               'multi_zero_var': 0.0,
               'categoric': 0.49995000499950004,
               'boolean': 0.49995000499950004},
        'BFGS': {'integer': 0.0,
                 'float': 0.0,
                 'is_na': 9.999000099990002e-05,
                 'mixed': 0.0,
                 'already_optimal': 0.9032096790320968,
                 'float_with_zero_var': 0.0,
                 'float_with_all_na_greater': 0.0,
                 'float_with_all_na_is_na': 0.0,
                 'multi_zero_var': 0.0,
                 'categoric': 0.49995000499950004,
                 'boolean': 0.49995000499950004},
        'L-BFGS-B': {'integer': 0.0,
                     'float': 0.0,
                     'is_na': 9.999000099990002e-05,
                     'mixed': 0.0,
                     'already_optimal': 0.9032096790320968,
                     'float_with_zero_var': 0.0,
                     'float_with_all_na_greater': 0.0,
                     'float_with_all_na_is_na': 0.0,
                     'multi_zero_var': 0.0,
                     'categoric': 0.49995000499950004,
                     'boolean': 0.49995000499950004},
        'TNC': {'integer': 0.0,
                'float': 0.0,
                'is_na': 9.999000099990002e-05,
                'mixed': 0.0,
                'already_optimal': 0.9032096790320968,
                'float_with_zero_var': 0.0,
                'float_with_all_na_greater': 0.0,
                'float_with_all_na_is_na': 0.0,
                'multi_zero_var': 0.0,
                'categoric': 0.49995000499950004,
                'boolean': 0.49995000499950004},
        'COBYLA': {'integer': 0.0,
                   'float': 0.0,
                   'is_na': 9.999000099990002e-05,
                   'mixed': 0.0,
                   'already_optimal': 0.0,
                   'float_with_zero_var': 0.0,
                   'float_with_all_na_greater': 0.0,
                   'float_with_all_na_is_na': 0.0,
                   'multi_zero_var': 0.0,
                   'categoric': 0.49995000499950004,
                   'boolean': 0.49995000499950004},
        'SLSQP': {'integer': 0.0,
                  'float': 0.0,
                  'is_na': 9.999000099990002e-05,
                  'mixed': 0.0,
                  'already_optimal': 0.0,
                  'float_with_zero_var': 0.0,
                  'float_with_all_na_greater': 0.0,
                  'float_with_all_na_is_na': 0.0,
                  'multi_zero_var': 0.0,
                  'categoric': 0.49995000499950004,
                  'boolean': 0.49995000499950004},
        'trust-constr': {'integer': 0.0,
                         'float': 0.0,
                         'is_na': 9.999000099990002e-05,
                         'mixed': 0.0,
                         'already_optimal': 0.7034296570342966,
                         'float_with_zero_var': 0.0,
                         'float_with_all_na_greater': 0.0,
                         'float_with_all_na_is_na': 0.0,
                         'multi_zero_var': 0.0,
                         'categoric': 0.49995000499950004,
                         'boolean': 0.49995000499950004}
    }


@ pytest.fixture
def _expected_X_rules_mean_unlabelled_x0_bounds():
    return {
        'Nelder-Mead': {'integer': 0.0,
                        'float': 0.0,
                        'is_na': 9.999000099990002e-05,
                        'mixed': 0.0098990100989901,
                        'already_optimal': 0.5061493850614939,
                        'float_with_zero_var': 9.999000099990002e-05,
                        'float_with_all_na_greater': 0.0,
                        'float_with_all_na_is_na': 0.0,
                        'multi_zero_var': 0.0,
                        'categoric': 0.49995000499950004,
                        'boolean': 0.49995000499950004},
        'Powell': {'integer': 0.0,
                   'float': 0.0098990100989901,
                   'is_na': 9.999000099990002e-05,
                   'mixed': 0.009799020097990201,
                   'already_optimal': 0.10428957104289571,
                   'float_with_zero_var': 0.0098990100989901,
                   'float_with_all_na_greater': 0.0,
                   'float_with_all_na_is_na': 0.0098990100989901,
                   'multi_zero_var': 0.0,
                   'categoric': 0.49995000499950004,
                   'boolean': 0.49995000499950004},
        'CG': {'integer': 0.0,
               'float': 0.0,
               'is_na': 9.999000099990002e-05,
               'mixed': 0.0,
               'already_optimal': 0.5061493850614939,
               'float_with_zero_var': 0.0,
               'float_with_all_na_greater': 0.0,
               'float_with_all_na_is_na': 0.0,
               'multi_zero_var': 0.0,
               'categoric': 0.49995000499950004,
               'boolean': 0.49995000499950004},
        'BFGS': {'integer': 0.0,
                 'float': 0.0,
                 'is_na': 9.999000099990002e-05,
                 'mixed': 0.0,
                 'already_optimal': 0.5061493850614939,
                 'float_with_zero_var': 0.0,
                 'float_with_all_na_greater': 0.0,
                 'float_with_all_na_is_na': 0.0,
                 'multi_zero_var': 0.0,
                 'categoric': 0.49995000499950004,
                 'boolean': 0.49995000499950004},
        'L-BFGS-B': {'integer': 0.0,
                     'float': 0.0,
                     'is_na': 9.999000099990002e-05,
                     'mixed': 0.0,
                     'already_optimal': 0.5061493850614939,
                     'float_with_zero_var': 0.0,
                     'float_with_all_na_greater': 0.0,
                     'float_with_all_na_is_na': 0.0,
                     'multi_zero_var': 0.0,
                     'categoric': 0.49995000499950004,
                     'boolean': 0.49995000499950004},
        'TNC': {'integer': 0.0,
                'float': 0.0,
                'is_na': 9.999000099990002e-05,
                'mixed': 0.0,
                'already_optimal': 0.5061493850614939,
                'float_with_zero_var': 0.0,
                'float_with_all_na_greater': 0.0,
                'float_with_all_na_is_na': 0.0,
                'multi_zero_var': 0.0,
                'categoric': 0.49995000499950004,
                'boolean': 0.49995000499950004},
        'COBYLA': {'integer': 0.0,
                   'float': 0.0,
                   'is_na': 9.999000099990002e-05,
                   'mixed': 0.0,
                   'already_optimal': 0.0,
                   'float_with_zero_var': 0.0,
                   'float_with_all_na_greater': 0.0,
                   'float_with_all_na_is_na': 0.0,
                   'multi_zero_var': 0.0,
                   'categoric': 0.49995000499950004,
                   'boolean': 0.49995000499950004},
        'SLSQP': {'integer': 0.0,
                  'float': 0.0,
                  'is_na': 9.999000099990002e-05,
                  'mixed': 0.0,
                  'already_optimal': 0.5061493850614939,
                  'float_with_zero_var': 0.0,
                  'float_with_all_na_greater': 0.0,
                  'float_with_all_na_is_na': 0.0,
                  'multi_zero_var': 0.0,
                  'categoric': 0.49995000499950004,
                  'boolean': 0.49995000499950004},
        'trust-constr': {'integer': 0.0,
                         'float': 0.0,
                         'is_na': 9.999000099990002e-05,
                         'mixed': 0.0,
                         'already_optimal': 0.5061493850614939,
                         'float_with_zero_var': 0.0,
                         'float_with_all_na_greater': 0.0,
                         'float_with_all_na_is_na': 0.0,
                         'multi_zero_var': 0.0,
                         'categoric': 0.49995000499950004,
                         'boolean': 0.49995000499950004}
    }


@ pytest.fixture
def _exp_orig_rule_performances():
    return {
        'already_optimal': 0.6657771847898598,
        'integer': 0.0,
        'float': 0.0,
        'is_na': 0.0,
        'mixed': 0.0,
        'float_with_zero_var': 0.0,
        'float_with_all_na_greater': 0.0,
        'float_with_all_na_is_na': 0.0,
        'multi_zero_var': 0.0
    }


@ pytest.fixture
def _exp_orig_rule_performances_weighted():
    return {
        'integer': 0.0,
        'float': 0.0,
        'is_na': 0.0,
        'mixed': 0.0,
        'already_optimal': 0.08504038992467365,
        'float_with_zero_var': 0.0,
        'float_with_all_na_greater': 0.0,
        'float_with_all_na_is_na': 0.0,
        'multi_zero_var': 0.0
    }


@ pytest.fixture
def _exp_orig_rule_performances_unlabelled():
    return {
        'integer': -100.0,
        'float': -100.0,
        'is_na': -98.01,
        'mixed': -100.0,
        'already_optimal': -980100.0,
        'float_with_zero_var': -100.0,
        'float_with_all_na_greater': -100.0,
        'float_with_all_na_is_na': -100.0,
        'multi_zero_var': -100.0
    }


@ pytest.fixture
def _exp_opt_rule_performances():
    return {
        'Nelder-Mead': {
            'already_optimal': 0.6657771847898598,
            'integer': 0.18332504558262888,
            'is_na': 0.18329466357308585,
            'float': 0.0,
            'mixed': 0.0,
            'float_with_zero_var': 0.0,
            'float_with_all_na_greater': 0.0,
            'float_with_all_na_is_na': 0.0,
            'multi_zero_var': 0.0
        },
        'Powell': {
            'already_optimal': 0.6657771847898598,
            'integer': 0.0,
            'float': 0.0,
            'is_na': 0.0,
            'mixed': 0.0,
            'float_with_zero_var': 0.0,
            'float_with_all_na_greater': 0.0,
            'float_with_all_na_is_na': 0.0,
            'multi_zero_var': 0.0
        },
        'CG': {
            'already_optimal': 0.6657771847898598,
            'integer': 0.0,
            'float': 0.0,
            'is_na': 0.0,
            'mixed': 0.0,
            'float_with_zero_var': 0.0,
            'float_with_all_na_greater': 0.0,
            'float_with_all_na_is_na': 0.0,
            'multi_zero_var': 0.0
        },
        'BFGS': {
            'already_optimal': 0.6657771847898598,
            'integer': 0.0,
            'float': 0.0,
            'is_na': 0.0,
            'mixed': 0.0,
            'float_with_zero_var': 0.0,
            'float_with_all_na_greater': 0.0,
            'float_with_all_na_is_na': 0.0,
            'multi_zero_var': 0.0
        },
        'L-BFGS-B': {
            'already_optimal': 0.6657771847898598,
            'integer': 0.0,
            'float': 0.0,
            'is_na': 0.0,
            'mixed': 0.0,
            'float_with_zero_var': 0.0,
            'float_with_all_na_greater': 0.0,
            'float_with_all_na_is_na': 0.0,
            'multi_zero_var': 0.0
        },
        'TNC': {
            'already_optimal': 0.6657771847898598,
            'integer': 0.0,
            'float': 0.0,
            'is_na': 0.0,
            'mixed': 0.0,
            'float_with_zero_var': 0.0,
            'float_with_all_na_greater': 0.0,
            'float_with_all_na_is_na': 0.0,
            'multi_zero_var': 0.0
        },
        'COBYLA': {
            'already_optimal': 0.6657771847898598,
            'integer': 0.0,
            'float': 0.0,
            'is_na': 0.0,
            'mixed': 0.0,
            'float_with_zero_var': 0.0,
            'float_with_all_na_greater': 0.0,
            'float_with_all_na_is_na': 0.0,
            'multi_zero_var': 0.0
        },
        'SLSQP': {
            'already_optimal': 0.6657771847898598,
            'integer': 0.0,
            'float': 0.0,
            'is_na': 0.0,
            'mixed': 0.0,
            'float_with_zero_var': 0.0,
            'float_with_all_na_greater': 0.0,
            'float_with_all_na_is_na': 0.0,
            'multi_zero_var': 0.0
        },
        'trust-constr': {
            'already_optimal': 0.6657771847898598,
            'integer': 0.0,
            'float': 0.0,
            'is_na': 0.0,
            'mixed': 0.0,
            'float_with_zero_var': 0.0,
            'float_with_all_na_greater': 0.0,
            'float_with_all_na_is_na': 0.0,
            'multi_zero_var': 0.0
        }
    }


@ pytest.fixture
def _exp_opt_rule_performances_x0_bounds():
    return {
        'Nelder-Mead': {'integer': 0.4988062077198568,
                        'float': 0.6658216025085062,
                        'is_na': 0.4987565900726152,
                        'mixed': 0.6658216025085062,
                        'already_optimal': 0.6657771847898598,
                        'float_with_zero_var': 0.6658216025085062,
                        'float_with_all_na_greater': 0.0,
                        'float_with_all_na_is_na': 0.6658216025085062,
                        'multi_zero_var': 0.6658216025085062},
        'Powell': {'integer': 0.6422306211224418,
                   'float': 0.66506442352627,
                   'is_na': 0.6421848260125499,
                   'mixed': 0.6650196968685318,
                   'already_optimal': 0.6657771847898598,
                   'float_with_zero_var': 0.66506442352627,
                   'float_with_all_na_greater': 0.0,
                   'float_with_all_na_is_na': 0.66506442352627,
                   'multi_zero_var': 0.6655101087609262},
        'CG': {'integer': 0.4988062077198568,
               'float': 0.5005512679162072,
               'is_na': 0.4987565900726152,
               'mixed': 0.5005512679162072,
               'already_optimal': 0.6657771847898598,
               'float_with_zero_var': 0.5005512679162072,
               'float_with_all_na_greater': 0.0,
               'float_with_all_na_is_na': 0.5005512679162072,
               'multi_zero_var': 0.6008487468972696},
        'BFGS': {'integer': 0.4988062077198568,
                 'float': 0.5005512679162072,
                 'is_na': 0.4987565900726152,
                 'mixed': 0.5005512679162072,
                 'already_optimal': 0.6657771847898598,
                 'float_with_zero_var': 0.5005512679162072,
                 'float_with_all_na_greater': 0.0,
                 'float_with_all_na_is_na': 0.5005512679162072,
                 'multi_zero_var': 0.6008487468972696},
        'L-BFGS-B': {'integer': 0.4988062077198568,
                     'float': 0.5005512679162072,
                     'is_na': 0.4987565900726152,
                     'mixed': 0.5005512679162072,
                     'already_optimal': 0.6657771847898598,
                     'float_with_zero_var': 0.5005512679162072,
                     'float_with_all_na_greater': 0.0,
                     'float_with_all_na_is_na': 0.5005512679162072,
                     'multi_zero_var': 0.6008487468972696},
        'TNC': {'integer': 0.4988062077198568,
                'float': 0.5005512679162072,
                'is_na': 0.4987565900726152,
                'mixed': 0.5005512679162072,
                'already_optimal': 0.6657771847898598,
                'float_with_zero_var': 0.5005512679162072,
                'float_with_all_na_greater': 0.0,
                'float_with_all_na_is_na': 0.5005512679162072,
                'multi_zero_var': 0.6008487468972696},
        'COBYLA': {'integer': 0.6657771847898598,
                   'float': 0.6657771847898598,
                   'is_na': 0.6657327729971316,
                   'mixed': 0.6657771847898598,
                   'already_optimal': 0.6657771847898598,
                   'float_with_zero_var': 0.6657771847898598,
                   'float_with_all_na_greater': 0.0,
                   'float_with_all_na_is_na': 0.6657771847898598,
                   'multi_zero_var': 0.6657771847898598},
        'SLSQP': {'integer': 0.4988062077198568,
                  'float': 0.5005512679162072,
                  'is_na': 0.4987565900726152,
                  'mixed': 0.5005512679162072,
                  'already_optimal': 0.6657771847898598,
                  'float_with_zero_var': 0.5005512679162072,
                  'float_with_all_na_greater': 0.0,
                  'float_with_all_na_is_na': 0.5005512679162072,
                  'multi_zero_var': 0.6008487468972696},
        'trust-constr': {'integer': 0.4988062077198568,
                         'float': 0.5005512679162072,
                         'is_na': 0.4987565900726152,
                         'mixed': 0.5005512679162072,
                         'already_optimal': 0.6657771847898598,
                         'float_with_zero_var': 0.5005512679162072,
                         'float_with_all_na_greater': 0.0,
                         'float_with_all_na_is_na': 0.5005512679162072,
                         'multi_zero_var': 0.6008487468972696}
    }


@ pytest.fixture
def _exp_opt_rule_performances_weighted():
    return {
        'Nelder-Mead': {'integer': 0.020277579157728768,
                        'float': 0.0,
                        'is_na': 0.020277207392197127,
                        'mixed': 0.0,
                        'already_optimal': 0.08504038992467365,
                        'float_with_zero_var': 0.0,
                        'float_with_all_na_greater': 0.0,
                        'float_with_all_na_is_na': 0.0,
                        'multi_zero_var': 0.0},
        'Powell': {'integer': 0.0,
                   'float': 0.0,
                   'is_na': 0.0,
                   'mixed': 0.0,
                   'already_optimal': 0.08504038992467365,
                   'float_with_zero_var': 0.0,
                   'float_with_all_na_greater': 0.0,
                   'float_with_all_na_is_na': 0.0,
                   'multi_zero_var': 0.0},
        'CG': {'integer': 0.0,
               'float': 0.0,
               'is_na': 0.0,
               'mixed': 0.0,
               'already_optimal': 0.08504038992467365,
               'float_with_zero_var': 0.0,
               'float_with_all_na_greater': 0.0,
               'float_with_all_na_is_na': 0.0,
               'multi_zero_var': 0.0},
        'BFGS': {'integer': 0.0,
                 'float': 0.0,
                 'is_na': 0.0,
                 'mixed': 0.0,
                 'already_optimal': 0.08504038992467365,
                 'float_with_zero_var': 0.0,
                 'float_with_all_na_greater': 0.0,
                 'float_with_all_na_is_na': 0.0,
                 'multi_zero_var': 0.0},
        'L-BFGS-B': {'integer': 0.0,
                     'float': 0.0,
                     'is_na': 0.0,
                     'mixed': 0.0,
                     'already_optimal': 0.08504038992467365,
                     'float_with_zero_var': 0.0,
                     'float_with_all_na_greater': 0.0,
                     'float_with_all_na_is_na': 0.0,
                     'multi_zero_var': 0.0},
        'TNC': {'integer': 0.0,
                'float': 0.0,
                'is_na': 0.0,
                'mixed': 0.0,
                'already_optimal': 0.08504038992467365,
                'float_with_zero_var': 0.0,
                'float_with_all_na_greater': 0.0,
                'float_with_all_na_is_na': 0.0,
                'multi_zero_var': 0.0},
        'COBYLA': {'integer': 0.0,
                   'float': 0.0,
                   'is_na': 0.0,
                   'mixed': 0.0,
                   'already_optimal': 0.08504038992467365,
                   'float_with_zero_var': 0.0,
                   'float_with_all_na_greater': 0.0,
                   'float_with_all_na_is_na': 0.0,
                   'multi_zero_var': 0.0},
        'SLSQP': {'integer': 0.0,
                  'float': 0.0,
                  'is_na': 0.0,
                  'mixed': 0.0,
                  'already_optimal': 0.08504038992467365,
                  'float_with_zero_var': 0.0,
                  'float_with_all_na_greater': 0.0,
                  'float_with_all_na_is_na': 0.0,
                  'multi_zero_var': 0.0},
        'trust-constr': {'integer': 0.0,
                         'float': 0.0,
                         'is_na': 0.0,
                         'mixed': 0.0,
                         'already_optimal': 0.08504038992467365,
                         'float_with_zero_var': 0.0,
                         'float_with_all_na_greater': 0.0,
                         'float_with_all_na_is_na': 0.0,
                         'multi_zero_var': 0.0}
    }


@ pytest.fixture
def _exp_opt_rule_performances_weighted_x0_bounds():
    return {
        'Nelder-Mead': {'integer': 0.044601398352576996,
                        'float': 0.08315554286290255,
                        'is_na': 0.044601001610048124,
                        'mixed': 0.08315693982461446,
                        'already_optimal': 0.08504038992467365,
                        'float_with_zero_var': 0.08315554286290255,
                        'float_with_all_na_greater': 0.0,
                        'float_with_all_na_is_na': 0.08315554286290255,
                        'multi_zero_var': 0.08504111456691237},
        'Powell': {'integer': 0.07737844641675759,
                   'float': 0.08649740043410227,
                   'is_na': 0.07737778159635708,
                   'mixed': 0.08528603220316795,
                   'already_optimal': 0.08504038992467365,
                   'float_with_zero_var': 0.08649740043410227,
                   'float_with_all_na_greater': 0.0,
                   'float_with_all_na_is_na': 0.08649740043410227,
                   'multi_zero_var': 0.08500775431600115},
        'CG': {'integer': 0.044601398352576996,
               'float': 0.08286183610147839,
               'is_na': 0.044601001610048124,
               'mixed': 0.08286183610147839,
               'already_optimal': 0.08504038992467365,
               'float_with_zero_var': 0.08286183610147839,
               'float_with_all_na_greater': 0.0,
               'float_with_all_na_is_na': 0.08286183610147839,
               'multi_zero_var': 0.08286183610147839},
        'BFGS': {'integer': 0.044601398352576996,
                 'float': 0.08286183610147839,
                 'is_na': 0.044601001610048124,
                 'mixed': 0.08286183610147839,
                 'already_optimal': 0.08504038992467365,
                 'float_with_zero_var': 0.08286183610147839,
                 'float_with_all_na_greater': 0.0,
                 'float_with_all_na_is_na': 0.08286183610147839,
                 'multi_zero_var': 0.08286183610147839},
        'L-BFGS-B': {'integer': 0.044601398352576996,
                     'float': 0.08286183610147839,
                     'is_na': 0.044601001610048124,
                     'mixed': 0.08286183610147839,
                     'already_optimal': 0.08504038992467365,
                     'float_with_zero_var': 0.08286183610147839,
                     'float_with_all_na_greater': 0.0,
                     'float_with_all_na_is_na': 0.08286183610147839,
                     'multi_zero_var': 0.06533455226154716},
        'TNC': {'integer': 0.044601398352576996,
                'float': 0.08286183610147839,
                'is_na': 0.044601001610048124,
                'mixed': 0.08286183610147839,
                'already_optimal': 0.08504038992467365,
                'float_with_zero_var': 0.08286183610147839,
                'float_with_all_na_greater': 0.0,
                'float_with_all_na_is_na': 0.08286183610147839,
                'multi_zero_var': 0.06533455226154716},
        'COBYLA': {'integer': 0.08504038992467365,
                   'float': 0.08504038992467365,
                   'is_na': 0.0850396652947843,
                   'mixed': 0.08504038992467365,
                   'already_optimal': 0.08504038992467365,
                   'float_with_zero_var': 0.0865677707782971,
                   'float_with_all_na_greater': 0.0,
                   'float_with_all_na_is_na': 0.08504038992467365,
                   'multi_zero_var': 0.08504038992467365},
        'SLSQP': {'integer': 0.044601398352576996,
                  'float': 0.08286183610147839,
                  'is_na': 0.044601001610048124,
                  'mixed': 0.08286183610147839,
                  'already_optimal': 0.08504038992467365,
                  'float_with_zero_var': 0.08286183610147839,
                  'float_with_all_na_greater': 0.0,
                  'float_with_all_na_is_na': 0.08286183610147839,
                  'multi_zero_var': 0.06533455226154716},
        'trust-constr': {'integer': 0.044601398352576996,
                         'float': 0.08286183610147839,
                         'is_na': 0.044601001610048124,
                         'mixed': 0.08286183610147839,
                         'already_optimal': 0.08504038992467365,
                         'float_with_zero_var': 0.08286183610147839,
                         'float_with_all_na_greater': 0.0,
                         'float_with_all_na_is_na': 0.08286183610147839,
                         'multi_zero_var': 0.06533455226154716}
    }


@ pytest.fixture
def _exp_opt_rule_performances_unlabelled():
    return {
        'Nelder-Mead': {'integer': -100.0,
                        'float': -100.0,
                        'is_na': -98.01,
                        'mixed': -100.0,
                        'already_optimal': -797984.8899999999,
                        'float_with_zero_var': -100.0,
                        'float_with_all_na_greater': -100.0,
                        'float_with_all_na_is_na': -100.0,
                        'multi_zero_var': -100.0},
        'Powell': {'integer': -100.0,
                   'float': -100.0,
                   'is_na': -98.01,
                   'mixed': -100.0,
                   'already_optimal': -100.0,
                   'float_with_zero_var': -100.0,
                   'float_with_all_na_greater': -100.0,
                   'float_with_all_na_is_na': -100.0,
                   'multi_zero_var': -100.0},
        'CG': {'integer': -100.0,
               'float': -100.0,
               'is_na': -98.01,
               'mixed': -100.0,
               'already_optimal': -797984.8899999999,
               'float_with_zero_var': -100.0,
               'float_with_all_na_greater': -100.0,
               'float_with_all_na_is_na': -100.0,
               'multi_zero_var': -100.0},
        'BFGS': {'integer': -100.0,
                 'float': -100.0,
                 'is_na': -98.01,
                 'mixed': -100.0,
                 'already_optimal': -797984.8899999999,
                 'float_with_zero_var': -100.0,
                 'float_with_all_na_greater': -100.0,
                 'float_with_all_na_is_na': -100.0,
                 'multi_zero_var': -100.0},
        'L-BFGS-B': {'integer': -100.0,
                     'float': -100.0,
                     'is_na': -98.01,
                     'mixed': -100.0,
                     'already_optimal': -797984.8899999999,
                     'float_with_zero_var': -100.0,
                     'float_with_all_na_greater': -100.0,
                     'float_with_all_na_is_na': -100.0,
                     'multi_zero_var': -100.0},
        'TNC': {'integer': -100.0,
                'float': -100.0,
                'is_na': -98.01,
                'mixed': -100.0,
                'already_optimal': -797984.8899999999,
                'float_with_zero_var': -100.0,
                'float_with_all_na_greater': -100.0,
                'float_with_all_na_is_na': -100.0,
                'multi_zero_var': -100.0},
        'COBYLA': {'integer': -100.0,
                   'float': -100.0,
                   'is_na': -98.01,
                   'mixed': -100.0,
                   'already_optimal': -100.0,
                   'float_with_zero_var': -100.0,
                   'float_with_all_na_greater': -100.0,
                   'float_with_all_na_is_na': -100.0,
                   'multi_zero_var': -100.0},
        'SLSQP': {'integer': -100.0,
                  'float': -100.0,
                  'is_na': -98.01,
                  'mixed': -100.0,
                  'already_optimal': -100.0,
                  'float_with_zero_var': -100.0,
                  'float_with_all_na_greater': -100.0,
                  'float_with_all_na_is_na': -100.0,
                  'multi_zero_var': -100.0},
        'trust-constr': {'integer': -100.0,
                         'float': -100.0,
                         'is_na': -98.01,
                         'mixed': -100.0,
                         'already_optimal': -480942.25,
                         'float_with_zero_var': -100.0,
                         'float_with_all_na_greater': -100.0,
                         'float_with_all_na_is_na': -100.0,
                         'multi_zero_var': -100.0}
    }


@ pytest.fixture
def _exp_opt_rule_performances_unlabelled_x0_bounds():
    return {
        'Nelder-Mead': {'integer': -100.0,
                        'float': -100.0,
                        'is_na': -98.01,
                        'mixed': -0.009999999999999929,
                        'already_optimal': -246214.44,
                        'float_with_zero_var': -98.01,
                        'float_with_all_na_greater': -100.0,
                        'float_with_all_na_is_na': -100.0,
                        'multi_zero_var': -100.0},
        'Powell': {'integer': -100.0,
                   'float': -0.009999999999999929,
                   'is_na': -98.01,
                   'mixed': -0.039999999999999716,
                   'already_optimal': -8892.49,
                   'float_with_zero_var': -0.009999999999999929,
                   'float_with_all_na_greater': -100.0,
                   'float_with_all_na_is_na': -0.009999999999999929,
                   'multi_zero_var': -100.0},
        'CG': {'integer': -100.0,
               'float': -100.0,
               'is_na': -98.01,
               'mixed': -100.0,
               'already_optimal': -246214.44,
               'float_with_zero_var': -100.0,
               'float_with_all_na_greater': -100.0,
               'float_with_all_na_is_na': -100.0,
               'multi_zero_var': -100.0},
        'BFGS': {'integer': -100.0,
                 'float': -100.0,
                 'is_na': -98.01,
                 'mixed': -100.0,
                 'already_optimal': -246214.44,
                 'float_with_zero_var': -100.0,
                 'float_with_all_na_greater': -100.0,
                 'float_with_all_na_is_na': -100.0,
                 'multi_zero_var': -100.0},
        'L-BFGS-B': {'integer': -100.0,
                     'float': -100.0,
                     'is_na': -98.01,
                     'mixed': -100.0,
                     'already_optimal': -246214.44,
                     'float_with_zero_var': -100.0,
                     'float_with_all_na_greater': -100.0,
                     'float_with_all_na_is_na': -100.0,
                     'multi_zero_var': -100.0},
        'TNC': {'integer': -100.0,
                'float': -100.0,
                'is_na': -98.01,
                'mixed': -100.0,
                'already_optimal': -246214.44,
                'float_with_zero_var': -100.0,
                'float_with_all_na_greater': -100.0,
                'float_with_all_na_is_na': -100.0,
                'multi_zero_var': -100.0},
        'COBYLA': {'integer': -100.0,
                   'float': -100.0,
                   'is_na': -98.01,
                   'mixed': -100.0,
                   'already_optimal': -100.0,
                   'float_with_zero_var': -100.0,
                   'float_with_all_na_greater': -100.0,
                   'float_with_all_na_is_na': -100.0,
                   'multi_zero_var': -100.0},
        'SLSQP': {'integer': -100.0,
                  'float': -100.0,
                  'is_na': -98.01,
                  'mixed': -100.0,
                  'already_optimal': -246214.44,
                  'float_with_zero_var': -100.0,
                  'float_with_all_na_greater': -100.0,
                  'float_with_all_na_is_na': -100.0,
                  'multi_zero_var': -100.0},
        'trust-constr': {'integer': -100.0,
                         'float': -100.0,
                         'is_na': -98.01,
                         'mixed': -100.0,
                         'already_optimal': -246214.44,
                         'float_with_zero_var': -100.0,
                         'float_with_all_na_greater': -100.0,
                         'float_with_all_na_is_na': -100.0,
                         'multi_zero_var': -100.0}
    }


@ pytest.fixture
def _instantiate(_create_data, _create_inputs):
    X, _, _ = _create_data
    rule_lambdas, lambda_kwargs = _create_inputs
    f1 = FScore(beta=1)
    with pytest.warns(
            RulesNotOptimisedWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `x0` for these rules") and \
        pytest.warns(
            RulesNotOptimisedWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `x0` for these rules") and\
        pytest.warns(
            RulesNotOptimisedWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `bounds` for these rules") and \
        pytest.warns(
            RulesNotOptimisedWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `bounds` for these rules"):
        ro = DirectSearchOptimiser(
            rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs,
            method='Nelder-mead',
            x0=DirectSearchOptimiser.create_x0(X, lambda_kwargs),
            bounds=DirectSearchOptimiser.create_bounds(X, lambda_kwargs),
            metric=f1.fit
        )
    return ro


@ pytest.fixture
def _instantiate_unlabelled(_create_data, _create_inputs):
    X, _, _ = _create_data
    rule_lambdas, lambda_kwargs = _create_inputs
    apd = AlertsPerDay(10, 10)
    with pytest.warns(
            RulesNotOptimisedWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `x0` for these rules") and \
        pytest.warns(
            RulesNotOptimisedWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `x0` for these rules") and\
        pytest.warns(
            RulesNotOptimisedWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `bounds` for these rules") and \
        pytest.warns(
            RulesNotOptimisedWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `bounds` for these rules"):
        ro = DirectSearchOptimiser(
            rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs,
            method='Nelder-mead',
            x0=DirectSearchOptimiser.create_x0(X, lambda_kwargs),
            bounds=DirectSearchOptimiser.create_bounds(X, lambda_kwargs),
            metric=apd.fit
        )
    return ro


def test_fit(_create_data, _create_inputs, _expected_rule_strings,
             _expected_X_rules_mean, _exp_orig_rule_performances,
             _exp_opt_rule_performances):
    f1 = FScore(1)
    X, y, _ = _create_data
    exp_rule_strings = _expected_rule_strings
    exp_X_rules_mean = _expected_X_rules_mean
    exp_orig_rule_performances = _exp_orig_rule_performances
    exp_opt_rule_performances = _exp_opt_rule_performances
    rule_lambdas, lambda_kwargs = _create_inputs
    _fit(
        rule_lambdas, lambda_kwargs, X, y, None, f1.fit, exp_rule_strings,
        exp_X_rules_mean, exp_orig_rule_performances,
        exp_opt_rule_performances, None, None
    )


def test_fit_transform(_create_data, _create_inputs, _expected_rule_strings,
                       _expected_X_rules_mean, _exp_orig_rule_performances,
                       _exp_opt_rule_performances):
    f1 = FScore(1)
    X, y, _ = _create_data
    exp_rule_strings = _expected_rule_strings
    exp_X_rules_mean = _expected_X_rules_mean
    exp_orig_rule_performances = _exp_orig_rule_performances
    exp_opt_rule_performances = _exp_opt_rule_performances
    rule_lambdas, lambda_kwargs = _create_inputs
    _fit(
        rule_lambdas, lambda_kwargs, X, y, None, f1.fit, exp_rule_strings,
        exp_X_rules_mean, exp_orig_rule_performances,
        exp_opt_rule_performances, None, None, fit_transform=True
    )


def test_fit_with_x0_and_bounds(_create_data, _create_inputs,
                                _expected_rule_strings_x0_bounds,
                                _expected_X_rules_mean_x0_bounds,
                                _exp_orig_rule_performances,
                                _exp_opt_rule_performances_x0_bounds):
    f1 = FScore(1)
    X, y, _ = _create_data
    exp_rule_strings = _expected_rule_strings_x0_bounds
    exp_X_rules_mean = _expected_X_rules_mean_x0_bounds
    exp_orig_rule_performances = _exp_orig_rule_performances
    exp_opt_rule_performances = _exp_opt_rule_performances_x0_bounds
    rule_lambdas, lambda_kwargs = _create_inputs
    _fit(
        rule_lambdas, lambda_kwargs, X, y, None, f1.fit, exp_rule_strings,
        exp_X_rules_mean, exp_orig_rule_performances,
        exp_opt_rule_performances, True, True
    )


def test_fit_weighted(_create_data, _create_inputs,
                      _expected_rule_strings_weighted,
                      _expected_X_rules_mean_weighted,
                      _exp_orig_rule_performances_weighted,
                      _exp_opt_rule_performances_weighted):
    f1 = FScore(1)
    X, y, sample_weight = _create_data
    exp_rule_strings = _expected_rule_strings_weighted
    exp_X_rules_mean = _expected_X_rules_mean_weighted
    exp_orig_rule_performances = _exp_orig_rule_performances_weighted
    exp_opt_rule_performances = _exp_opt_rule_performances_weighted
    rule_lambdas, lambda_kwargs = _create_inputs
    _fit(
        rule_lambdas, lambda_kwargs, X, y, sample_weight, f1.fit, exp_rule_strings,
        exp_X_rules_mean, exp_orig_rule_performances,
        exp_opt_rule_performances, None, None
    )


def test_fit_weighted_with_x0_and_bounds(_create_data, _create_inputs,
                                         _expected_rule_strings_weighted_x0_bounds,
                                         _expected_X_rules_mean_weighted_x0_bounds,
                                         _exp_orig_rule_performances_weighted,
                                         _exp_opt_rule_performances_weighted_x0_bounds):
    f1 = FScore(1)
    X, y, sample_weight = _create_data
    exp_rule_strings = _expected_rule_strings_weighted_x0_bounds
    exp_X_rules_mean = _expected_X_rules_mean_weighted_x0_bounds
    exp_orig_rule_performances = _exp_orig_rule_performances_weighted
    exp_opt_rule_performances = _exp_opt_rule_performances_weighted_x0_bounds
    rule_lambdas, lambda_kwargs = _create_inputs
    _fit(
        rule_lambdas, lambda_kwargs, X, y, sample_weight, f1.fit, exp_rule_strings,
        exp_X_rules_mean, exp_orig_rule_performances,
        exp_opt_rule_performances, True, True
    )


def test_fit_unlabelled(_create_data, _create_inputs,
                        _expected_rule_strings_unlabelled,
                        _expected_X_rules_mean_unlabelled,
                        _exp_orig_rule_performances_unlabelled,
                        _exp_opt_rule_performances_unlabelled):
    apd = AlertsPerDay(10, 10)
    X, y, _ = _create_data
    exp_rule_strings = _expected_rule_strings_unlabelled
    exp_X_rules_mean = _expected_X_rules_mean_unlabelled
    exp_orig_rule_performances = _exp_orig_rule_performances_unlabelled
    exp_opt_rule_performances = _exp_opt_rule_performances_unlabelled
    rule_lambdas, lambda_kwargs = _create_inputs
    _fit(
        rule_lambdas, lambda_kwargs, X, None, None, apd.fit, exp_rule_strings,
        exp_X_rules_mean, exp_orig_rule_performances,
        exp_opt_rule_performances, None, None
    )


def test_fit_unlabelled_with_x0_and_bounds(_create_data, _create_inputs,
                                           _expected_rule_strings_unlabelled_x0_bounds,
                                           _expected_X_rules_mean_unlabelled_x0_bounds,
                                           _exp_orig_rule_performances_unlabelled,
                                           _exp_opt_rule_performances_unlabelled_x0_bounds):
    apd = AlertsPerDay(10, 10)
    X, y, _ = _create_data
    exp_rule_strings = _expected_rule_strings_unlabelled_x0_bounds
    exp_X_rules_mean = _expected_X_rules_mean_unlabelled_x0_bounds
    exp_orig_rule_performances = _exp_orig_rule_performances_unlabelled
    exp_opt_rule_performances = _exp_opt_rule_performances_unlabelled_x0_bounds
    rule_lambdas, lambda_kwargs = _create_inputs
    _fit(
        rule_lambdas, lambda_kwargs, X, None, None, apd.fit, exp_rule_strings,
        exp_X_rules_mean, exp_orig_rule_performances,
        exp_opt_rule_performances, True, True
    )


def test_transform(_instantiate):
    ro = _instantiate
    X = pd.DataFrame({
        'A': [1, 2, 0, 1, 0, 2]
    })
    exp_X_rules = pd.DataFrame({
        'Rule': [0, 1, 0, 0, 0, 1]
    })
    ro.rule_strings = {'Rule': "(X['A']>1)"}
    X_rules = ro.transform(X)
    assert all(X_rules == exp_X_rules)


def test_create_bounds(_create_data, _create_inputs):
    exp_bounds = {
        'integer': [(0.0, 9.0)],
        'float': [(0.0003641365574362787, 0.9999680225821261)],
        'is_na': [(0.0, 9.0)],
        'mixed': [(0.0, 9.0),
                  (0.0003641365574362787, 0.9999680225821261),
                  (0.0003641365574362787, 0.9999680225821261)],
        'all_na': [(0.0, 0.0)],
        'zero_var': [(1.0, 1.0)],
        'already_optimal': [(0.0, 9.0)],
        'float_with_zero_var': [(0.0003641365574362787, 0.9999680225821261),
                                (1.0, 1.0)],
        'float_with_all_na_greater': [(0.0003641365574362787, 0.9999680225821261),
                                      (0.0, 0.0)],
        'float_with_all_na_is_na': [(0.0003641365574362787, 0.9999680225821261)],
        'multi_zero_var': [(0.0003641365574362787, 0.9999680225821261),
                           (1.0, 1.0),
                           (0.0, 9.0),
                           (1.0, 1.0)]
    }
    X, _, _ = _create_data
    _, lambda_kwargs = _create_inputs
    with pytest.warns(
            RulesNotOptimisedWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `bounds` for these rules") and \
        pytest.warns(
            RulesNotOptimisedWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `bounds` for these rules"):
        bounds = DirectSearchOptimiser.create_bounds(X, lambda_kwargs)
        assert bounds == exp_bounds


def test_create_x0(_create_data, _create_inputs):
    exp_x0 = {
        'integer': np.array([4.5]),
        'float': np.array([0.50016608]),
        'is_na': np.array([4.5]),
        'mixed': np.array([4.5, 0.50016608, 0.50016608]),
        'all_na': np.array([0.]),
        'zero_var': np.array([1.]),
        'already_optimal': np.array([4.5]),
        'float_with_zero_var': np.array([0.50016608, 1.]),
        'float_with_all_na_greater': np.array([0.50016608, 0.]),
        'float_with_all_na_is_na': np.array([0.50016608]),
        'multi_zero_var': np.array([0.50016608, 1., 4.5, 1.])
    }
    X, _, _ = _create_data
    _, lambda_kwargs = _create_inputs
    with pytest.warns(
            RulesNotOptimisedWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `x0` for these rules") and \
        pytest.warns(
            RulesNotOptimisedWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `x0` for these rules"):
        x0 = DirectSearchOptimiser.create_x0(X, lambda_kwargs)
        for rule_name in x0.keys():
            np.testing.assert_array_almost_equal(
                x0[rule_name], exp_x0[rule_name])


def test_create_initial_simplexes(_create_data, _create_inputs):
    exp_simplexes = {
        'Origin-based': {
            'integer': {'initial_simplex': np.array([[0.],
                                                     [9.]])},
            'float': {'initial_simplex': np.array([[3.64136557e-04],
                                                   [9.99968023e-01]])},
            'is_na': {'initial_simplex': np.array([[0.],
                                                   [9.]])},
            'mixed': {'initial_simplex': np.array([[0.00000000e+00, 3.64136557e-04, 3.64136557e-04],
                                                   [9.00000000e+00, 0.00000000e+00,
                                                    0.00000000e+00],
                                                   [0.00000000e+00, 9.99968023e-01,
                                                    0.00000000e+00],
                                                   [0.00000000e+00, 0.00000000e+00, 9.99968023e-01]])},
            'all_na': {'initial_simplex': np.array([[0.],
                                                    [0.]])},
            'zero_var': {'initial_simplex': np.array([[1.],
                                                      [1.]])},
            'already_optimal': {'initial_simplex': np.array([[0.],
                                                             [9.]])},
            'float_with_zero_var': {'initial_simplex': np.array([[3.64136557e-04, 1.00000000e+00],
                                                                 [9.99968023e-01,
                                                                  0.00000000e+00],
                                                                 [0.00000000e+00, 1.00000000e+00]])},
            'float_with_all_na_greater': {'initial_simplex': np.array([[3.64136557e-04, 0.00000000e+00],
                                                                       [9.99968023e-01,
                                                                        0.00000000e+00],
                                                                       [0.00000000e+00, 0.00000000e+00]])},
            'float_with_all_na_is_na': {'initial_simplex': np.array([[3.64136557e-04],
                                                                     [9.99968023e-01]])},
            'multi_zero_var': {'initial_simplex': np.array([[3.64136557e-04, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                                            [9.99968023e-01, 0.00000000e+00,
                                                             0.00000000e+00, 0.00000000e+00],
                                                            [0.00000000e+00, 1.00000000e+00,
                                                             0.00000000e+00, 0.00000000e+00],
                                                            [0.00000000e+00, 0.00000000e+00,
                                                             9.00000000e+00, 0.00000000e+00],
                                                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])}
        },
        'Minimum-based': {
            'integer': {'initial_simplex': np.array([[0.],
                                                     [9.]])},
            'float': {'initial_simplex': np.array([[3.64136557e-04],
                                                   [9.99968023e-01]])},
            'is_na': {'initial_simplex': np.array([[0.],
                                                   [9.]])},
            'mixed': {'initial_simplex': np.array([[0.00000000e+00, 3.64136557e-04, 3.64136557e-04],
                                                   [9.00000000e+00, 3.64136557e-04,
                                                    3.64136557e-04],
                                                   [0.00000000e+00, 9.99968023e-01,
                                                    3.64136557e-04],
                                                   [0.00000000e+00, 3.64136557e-04, 9.99968023e-01]])},
            'all_na': {'initial_simplex': np.array([[0.],
                                                    [0.]])},
            'zero_var': {'initial_simplex': np.array([[1.],
                                                      [1.]])},
            'already_optimal': {'initial_simplex': np.array([[0.],
                                                             [9.]])},
            'float_with_zero_var': {'initial_simplex': np.array([[3.64136557e-04, 1.00000000e+00],
                                                                 [9.99968023e-01,
                                                                  1.00000000e+00],
                                                                 [3.64136557e-04, 1.00000000e+00]])},
            'float_with_all_na_greater': {'initial_simplex': np.array([[3.64136557e-04, 0.00000000e+00],
                                                                       [9.99968023e-01,
                                                                        0.00000000e+00],
                                                                       [3.64136557e-04, 0.00000000e+00]])},
            'float_with_all_na_is_na': {'initial_simplex': np.array([[3.64136557e-04],
                                                                     [9.99968023e-01]])},
            'multi_zero_var': {'initial_simplex': np.array([[3.64136557e-04, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                                            [9.99968023e-01, 1.00000000e+00,
                                                             0.00000000e+00, 1.00000000e+00],
                                                            [3.64136557e-04, 1.00000000e+00,
                                                             0.00000000e+00, 1.00000000e+00],
                                                            [3.64136557e-04, 1.00000000e+00,
                                                             9.00000000e+00, 1.00000000e+00],
                                                            [3.64136557e-04, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00]])}
        },
        'Random-based': {
            'integer': {'initial_simplex': np.array([[4.94426086],
                                                     [6.443141]])},
            'float': {'initial_simplex': np.array([[0.54950884],
                                                   [0.71598511]])},
            'is_na': {'initial_simplex': np.array([[4.94426086],
                                                   [6.443141]])},
            'mixed': {'initial_simplex': np.array([[4.94426086, 0.42427461, 0.96460846],
                                                   [6.443141, 0.64664804,
                                                       0.38403706],
                                                   [5.43029526, 0.43821543,
                                                       0.79256697],
                                                   [4.9088526, 0.89267531, 0.52957824]])},
            'all_na': {'initial_simplex': np.array([[0.],
                                                    [0.]])},
            'zero_var': {'initial_simplex': np.array([[1.],
                                                      [1.]])},
            'already_optimal': {'initial_simplex': np.array([[4.94426086],
                                                             [6.443141]])},
            'float_with_zero_var': {'initial_simplex': np.array([[0.54950884, 1.],
                                                                 [0.71598511, 1.],
                                                                 [0.60349127, 1.]])},
            'float_with_all_na_greater': {'initial_simplex': np.array([[0.54950884, 0.],
                                                                       [0.71598511, 0.],
                                                                       [0.60349127, 0.]])},
            'float_with_all_na_is_na': {'initial_simplex': np.array([[0.54950884],
                                                                     [0.71598511]])},
            'multi_zero_var': {'initial_simplex': np.array([[0.54950884, 1., 7.13265087, 1.],
                                                            [0.71598511, 1.,
                                                             4.76481433, 1.],
                                                            [0.60349127, 1.,
                                                             5.11751345, 1.],
                                                            [0.54557615, 1.,
                                                             8.33870011, 1.],
                                                            [0.42427461, 1., 0.63996385, 1.]])}
        }
    }
    X, _, _ = _create_data
    _, lambda_kwargs = _create_inputs
    for shape in ['Origin-based', 'Minimum-based', 'Random-based']:
        with pytest.warns(
                RulesNotOptimisedWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `initial_simplex` for these rules") and \
            pytest.warns(
                RulesNotOptimisedWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `initial_simplex` for these rules"):
            initial_simplex = DirectSearchOptimiser.create_initial_simplexes(
                X, lambda_kwargs, shape)
            for rule_name in initial_simplex.keys():
                np.testing.assert_array_almost_equal(
                    initial_simplex[rule_name]['initial_simplex'],
                    exp_simplexes[shape][rule_name]['initial_simplex']
                )
    with pytest.raises(
            ValueError,
            match='`shape` must be either "Origin-based", "Minimum-based" or "Random-based"'):
        DirectSearchOptimiser.create_initial_simplexes(
            X, lambda_kwargs, 'ERROR')


def test_optimise_rules(_instantiate, _create_inputs, _create_data):
    X, y, _ = _create_data
    # exp_rule_strings are different to ones seen in test_fit* since there's
    # no comparison to the original rules in this test
    exp_rule_strings = {
        'integer': "(X['A']>4.5)",
        'float': "(X['C']>0.0003641365574362787)",
        'is_na': "(X['A']>4.5)|(X['A'].isna())",
        'mixed': "((X['A']>5.966666666666709)&(X['C']>0.6372486347111281)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.0003641365574362787)",
        'already_optimal': "(X['A']>=4.5)",
        'float_with_zero_var': "(X['C']>0.0003641365574362787)&(X['ZeroVar']>=1.0)",
        'float_with_all_na_greater': "(X['C']>0.5001660795697812)&(X['AllNa']>0.0)",
        'float_with_all_na_is_na': "(X['C']>0.0003641365574362787)&(X['AllNa'].isna())",
        'multi_zero_var': "((X['C']>0.0003641365574362787)&(X['ZeroVar']>=1.0))|((X['A']>6.537012970447561)&(X['ZeroVar']>=1.0))"
    }
    ro = _instantiate
    rule_lambdas, lambda_kwargs = _create_inputs
    rules_to_drop = [
        'missing_col', 'categoric', 'boolean', 'all_na', 'zero_var'
    ]
    rule_lambdas_ = {
        rule_name: rule_lambda for rule_name, rule_lambda in rule_lambdas.items() if rule_name not in rules_to_drop}
    lambda_kwargs_ = {
        rule_name: lambda_kwarg for rule_name, lambda_kwarg in lambda_kwargs.items() if rule_name not in rules_to_drop}
    opt_rule_strings = ro._optimise_rules(
        rule_lambdas_, lambda_kwargs_, X, y, None
    )
    assert opt_rule_strings == exp_rule_strings


def test_optimise_rules_weighted(_instantiate, _create_inputs, _create_data):
    X, y, sample_weight = _create_data
    exp_rule_strings = {
        'integer': "(X['A']>4.5)",
        'float': "(X['C']>0.5056366460650756)",
        'is_na': "(X['A']>4.5)|(X['A'].isna())",
        'mixed': "((X['A']>4.544030630550376)&(X['C']>0.5062001821281391)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.5059645205154579)",
        'already_optimal': "(X['A']>=4.5)",
        'float_with_zero_var': "(X['C']>0.5056366460650756)&(X['ZeroVar']>=1.0)",
        'float_with_all_na_greater': "(X['C']>0.5001660795697812)&(X['AllNa']>0.0)",
        'float_with_all_na_is_na': "(X['C']>0.5056366460650756)&(X['AllNa'].isna())",
        'multi_zero_var': "((X['C']>0.0003641365574362787)&(X['ZeroVar']>=1.0))|((X['A']>6.537012970447561)&(X['ZeroVar']>=1.0))"
    }
    ro = _instantiate
    rule_lambdas, lambda_kwargs = _create_inputs
    rules_to_drop = [
        'missing_col', 'categoric', 'boolean', 'all_na', 'zero_var'
    ]
    rule_lambdas_ = {
        rule_name: rule_lambda for rule_name, rule_lambda in rule_lambdas.items() if rule_name not in rules_to_drop}
    lambda_kwargs_ = {
        rule_name: lambda_kwarg for rule_name, lambda_kwarg in lambda_kwargs.items() if rule_name not in rules_to_drop}
    opt_rule_strings = ro._optimise_rules(
        rule_lambdas_, lambda_kwargs_, X, y, sample_weight)
    assert opt_rule_strings == exp_rule_strings


def test_optimise_rules_unlabelled(_instantiate_unlabelled, _create_inputs, _create_data):
    X, _, _ = _create_data
    exp_rule_strings = {
        'integer': "(X['A']>4.5)",
        'float': "(X['C']>0.9999680225821261)",
        'is_na': "(X['A']>4.5)|(X['A'].isna())",
        'mixed': "((X['A']>7.775353965311879)&(X['C']>0.9884794975944149)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.9884919578072813)",
        'already_optimal': "(X['A']>=4.5)",
        'float_with_zero_var': "(X['C']>0.999968022582126)&(X['ZeroVar']>=1.0)",
        'float_with_all_na_greater': "(X['C']>0.5001660795697812)&(X['AllNa']>0.0)",
        'float_with_all_na_is_na': "(X['C']>0.9999680225821261)&(X['AllNa'].isna())",
        'multi_zero_var': "((X['C']>0.9999680225821261)&(X['ZeroVar']>=1.0))|((X['A']>4.70320544242861)&(X['ZeroVar']>=1.0))"
    }
    ro = _instantiate_unlabelled
    rule_lambdas, lambda_kwargs = _create_inputs
    rules_to_drop = [
        'missing_col', 'categoric', 'boolean', 'all_na', 'zero_var'
    ]
    rule_lambdas_ = {
        rule_name: rule_lambda for rule_name, rule_lambda in rule_lambdas.items() if rule_name not in rules_to_drop}
    lambda_kwargs_ = {
        rule_name: lambda_kwarg for rule_name, lambda_kwarg in lambda_kwargs.items() if rule_name not in rules_to_drop}
    opt_rule_strings = ro._optimise_rules(
        rule_lambdas_, lambda_kwargs_, X, None, None
    )
    assert opt_rule_strings == exp_rule_strings


def test_optimise_rules_numpy(_instantiate, _create_data):
    X, y, _ = _create_data
    exp_rule_strings = {
        'already_optimal': "(X['A'].to_numpy(na_value=np.nan)>=4.5)",
        'float': "(X['C'].to_numpy(na_value=np.nan)>0.0003641365574362787)",
        'integer': "(X['A'].to_numpy(na_value=np.nan)>4.5)",
        'is_na': "(X['A'].to_numpy(na_value=np.nan)>4.5)|(pd.isna(X['A'].to_numpy(na_value=np.nan)))",
        'mixed': "((X['A'].to_numpy(na_value=np.nan)>5.966666666666709)&(X['C'].to_numpy(na_value=np.nan)>0.6372486347111281)&(X['E'].to_numpy(na_value=np.nan)=='yes')&(X['D'].to_numpy(na_value=np.nan)==True))|(X['C'].to_numpy(na_value=np.nan)>0.0003641365574362787)"
    }
    ro = _instantiate
    rules = Rules(rule_strings=exp_rule_strings)
    rule_lambdas = rules.as_rule_lambdas(as_numpy=True, with_kwargs=True)
    lambda_kwargs = rules.lambda_kwargs
    opt_rule_strings = ro._optimise_rules(
        rule_lambdas, lambda_kwargs, X, y, None
    )
    assert opt_rule_strings == exp_rule_strings


def test_optimise_single_rule(_create_inputs, _instantiate, _create_data):
    rule_lambdas, lambda_kwargs = _create_inputs
    X, y, _ = _create_data
    exp_result = 'integer', "(X['A']>4.5)"
    ro = _instantiate
    rule_name = 'integer'
    result = ro._optimise_single_rule(
        rule_name=rule_name, rule_lambda=rule_lambdas[rule_name],
        lambda_kwargs=lambda_kwargs, X=X, y=y, sample_weight=None
    )
    assert result == exp_result


def test_return_kwargs_for_minimize(_instantiate):
    exp_kwargs = {
        'x0': np.array([4.5]),
        'jac': None,
        'hess': None,
        'hessp': None,
        'bounds': [(0, 9)],
        'constraints': (),
        'tol': None,
        'callback': None,
        'options': None
    }
    ro = _instantiate
    minimize_kwargs = ro._return_kwargs_for_minimize('integer')
    assert minimize_kwargs == exp_kwargs
    ro.x0 = None
    ro.constraints = None
    exp_kwargs['x0'] = np.array(
        list(ro.orig_lambda_kwargs['integer'].values()))
    exp_kwargs['constraints'] = ()
    minimize_kwargs = ro._return_kwargs_for_minimize('integer')
    assert minimize_kwargs == exp_kwargs
    ro.options = []
    with pytest.raises(TypeError, match='`options` must be a dictionary with each element aligning with a rule.'):
        ro._return_kwargs_for_minimize('integer')


def test_return_opt_param_for_rule(_instantiate):
    ro = _instantiate
    param = ro._return_opt_param_for_rule('constraints', None, 'integer')
    assert param == ()
    param = ro._return_opt_param_for_rule('x0', None, 'integer')
    assert param == np.array(list(ro.orig_lambda_kwargs['integer'].values()))
    param = ro._return_opt_param_for_rule('tol', None, 'integer')
    assert param is None
    param = ro._return_opt_param_for_rule('tol', {'integer': 0.1}, 'integer')
    assert param == 0.1
    with pytest.raises(TypeError, match='`options` must be a dictionary with each element aligning with a rule.'):
        ro._return_opt_param_for_rule('options', [], 'integer')


def test_param_base_calc(_instantiate, _create_data, _create_inputs):

    def _bounds_func(X_min, X_max):
        X_min = np.where(np.isnan(X_min), 0, X_min)
        X_max = np.where(np.isnan(X_max), 0, X_max)
        return list(zip(X_min, X_max))

    exp_bounds = {
        'integer': [(0.0, 9.0)],
        'float': [(0.0003641365574362787, 0.9999680225821261)],
        'is_na': [(0.0, 9.0)],
        'mixed': [(0.0, 9.0),
                  (0.0003641365574362787, 0.9999680225821261),
                  (0.0003641365574362787, 0.9999680225821261)],
        'all_na': [(0.0, 0.0)],
        'zero_var': [(1.0, 1.0)],
        'already_optimal': [(0.0, 9.0)],
        'float_with_zero_var': [(0.0003641365574362787, 0.9999680225821261),
                                (1.0, 1.0)],
        'float_with_all_na_greater': [(0.0003641365574362787, 0.9999680225821261),
                                      (0.0, 0.0)],
        'float_with_all_na_is_na': [(0.0003641365574362787, 0.9999680225821261)],
        'multi_zero_var': [(0.0003641365574362787, 0.9999680225821261),
                           (1.0, 1.0),
                           (0.0, 9.0),
                           (1.0, 1.0)]
    }
    ro = _instantiate
    X, _, _ = _create_data
    _, lambda_kwargs = _create_inputs
    with pytest.warns(
            RulesNotOptimisedWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `bounds` for these rules") and \
        pytest.warns(
            RulesNotOptimisedWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `bounds` for these rules"):
        bounds = ro._param_base_calc(
            X, lambda_kwargs, 'bounds', _bounds_func
        )
    assert bounds == exp_bounds


def test_errors(_create_data, _instantiate):
    X, y, _ = _create_data
    ro = _instantiate
    with pytest.raises(TypeError, match='`X` must be a pandas.core.frame.DataFrame. Current type is list.'):
        ro.fit(X=[], y=y)
    with pytest.raises(TypeError, match='`y` must be a pandas.core.series.Series. Current type is list.'):
        ro.fit(X=X, y=[])
    with pytest.raises(TypeError, match='`sample_weight` must be a pandas.core.series.Series. Current type is list.'):
        ro.fit(X=X, y=y, sample_weight=[])
    X = pd.DataFrame({'ZeroVar': [0, 0, 0]})
    y = pd.Series([0, 1, 0])
    with pytest.raises(RulesNotOptimisedError, match='There are no optimisable rules in the set'):
        with pytest.warns(RulesNotOptimisedWarning, match='Rules `zero_var` have all zero variance features based on the dataset `X` - unable to optimise these rules'):
            ro.fit(X=X, y=y)


def _fit(rule_lambdas, lambda_kwargs, X, y, sample_weight, metric, exp_rule_strings,
         exp_X_rules_mean, exp_orig_rule_performances,
         exp_opt_rule_performances, x0, bounds, fit_transform=False):
    methods = [
        'Nelder-Mead',
        'Powell',
        'CG',
        'BFGS',
        'L-BFGS-B',
        'TNC',
        'COBYLA',
        'SLSQP',
        'trust-constr'
    ]
    if x0:
        with pytest.warns(
                RulesNotOptimisedWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `x0` for these rules") and \
            pytest.warns(
                RulesNotOptimisedWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `x0` for these rules"):
            x0 = DirectSearchOptimiser.create_x0(X, lambda_kwargs)
    if bounds:
        with pytest.warns(
                RulesNotOptimisedWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `bounds` for these rules") and \
            pytest.warns(
                RulesNotOptimisedWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `bounds` for these rules"):
            bounds = DirectSearchOptimiser.create_bounds(X, lambda_kwargs)
    for method in methods:
        with pytest.warns(RulesNotOptimisedWarning,
                          match="Rules `missing_col` use features that are missing from `X` - unable to optimise or apply these rules") and \
                pytest.warns(RulesNotOptimisedWarning,
                             match="Rules `categoric`, `boolean`, `all_na` have no optimisable conditions - unable to optimise these rules") and \
                pytest.warns(RulesNotOptimisedWarning,
                             match="Rules `zero_var` have all zero variance features based on the dataset `X` - unable to optimise these rules"):
            ro = DirectSearchOptimiser(
                rule_lambdas=rule_lambdas,
                lambda_kwargs=lambda_kwargs,
                metric=metric,
                x0=x0,
                bounds=bounds,
                num_cores=2,
                verbose=1,
                method=method,
            )
            assert ro.__repr__() == 'DirectSearchOptimiser object with 14 rules to optimise'
            if fit_transform:
                X_rules = ro.fit_transform(
                    X=X, y=y, sample_weight=sample_weight)
            else:
                X_rules = ro.fit(X=X, y=y, sample_weight=sample_weight)
            assert ro.__repr__() == 'DirectSearchOptimiser object with 9 optimised rules and 2 unoptimisable rules'
            assert ro.rule_strings == ro.rules.rule_strings == exp_rule_strings[method]
            assert ro.rule_names == list(exp_rule_strings[method].keys())
            assert X_rules.mean().to_dict() == exp_X_rules_mean[method]
            assert ro.orig_rule_performances == exp_orig_rule_performances
            assert ro.opt_rule_performances == exp_opt_rule_performances[method]
            assert ro.rule_names_missing_features == ['missing_col']
            assert ro.rule_names_no_opt_conditions == ['categoric', 'boolean']
            assert ro.rule_names_zero_var_features == ['all_na', 'zero_var']
