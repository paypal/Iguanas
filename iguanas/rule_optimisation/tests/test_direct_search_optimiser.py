import pytest
from iguanas.rule_optimisation import DirectSearchOptimiser
from iguanas.metrics import FScore, AlertsPerDay
from iguanas.rules import Rules
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
        'already_optimal': {'A': 0}
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
        },
        'Powell': {
            'integer': "(X['A']>9)",
            'float': "(X['C']>1.5)",
            'is_na': "(X['A']>9)|(X['A'].isna())",
            'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
            'already_optimal': "(X['A']>=0)",
        },
        'CG': {
            'integer': "(X['A']>9)",
            'float': "(X['C']>1.5)",
            'is_na': "(X['A']>9)|(X['A'].isna())",
            'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
            'already_optimal': "(X['A']>=0)",
        },
        'BFGS': {
            'integer': "(X['A']>9)",
            'float': "(X['C']>1.5)",
            'is_na': "(X['A']>9)|(X['A'].isna())",
            'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
            'already_optimal': "(X['A']>=0)",
        },
        'L-BFGS-B': {
            'integer': "(X['A']>9)",
            'float': "(X['C']>1.5)",
            'is_na': "(X['A']>9)|(X['A'].isna())",
            'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
            'already_optimal': "(X['A']>=0)",
        },
        'TNC': {
            'integer': "(X['A']>9)",
            'float': "(X['C']>1.5)",
            'is_na': "(X['A']>9)|(X['A'].isna())",
            'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
            'already_optimal': "(X['A']>=0)",
        },
        'COBYLA': {
            'integer': "(X['A']>9)",
            'float': "(X['C']>1.5)",
            'is_na': "(X['A']>9)|(X['A'].isna())",
            'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
            'already_optimal': "(X['A']>=0)",
        },
        'SLSQP': {
            'integer': "(X['A']>9)",
            'float': "(X['C']>1.5)",
            'is_na': "(X['A']>9)|(X['A'].isna())",
            'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
            'already_optimal': "(X['A']>=0)",
        },
        'trust-constr': {
            'integer': "(X['A']>9)",
            'float': "(X['C']>1.5)",
            'is_na': "(X['A']>9)|(X['A'].isna())",
            'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
            'already_optimal': "(X['A']>=0)",
        }
    }
    return opt_rule_strings


@pytest.fixture
def _expected_rule_strings_x0_bounds():
    opt_rule_strings = {
        'Nelder-Mead': {'integer': "(X['A']>4.5)",
                        'float': "(X['C']>0.0003641365574362787)",
                        'is_na': "(X['A']>4.5)|(X['A'].isna())",
                        'mixed': "((X['A']>5.966666666666709)&(X['C']>0.6367846977638774)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.0003641365574362787)",
                        'already_optimal': "(X['A']>=0)",
                        },
        'Powell': {'integer': "(X['A']>0.5015693034320075)",
                   'float': "(X['C']>0.0010925322097171326)",
                   'is_na': "(X['A']>0.5015693034320075)|(X['A'].isna())",
                   'mixed': "((X['A']>8.999736742129546)&(X['C']>0.9999237550255271)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.0014300920295867209)",
                   'already_optimal': "(X['A']>=0)",
                   },
        'CG': {'integer': "(X['A']>4.5)",
               'float': "(X['C']>0.49980194301234493)",
               'is_na': "(X['A']>4.5)|(X['A'].isna())",
               'mixed': "((X['A']>4.5)&(X['C']>0.49980194301234493)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.49980194301234493)",
               'already_optimal': "(X['A']>=0)",
               },
        'BFGS': {'integer': "(X['A']>4.5)",
                 'float': "(X['C']>0.49980194301234493)",
                 'is_na': "(X['A']>4.5)|(X['A'].isna())",
                 'mixed': "((X['A']>4.5)&(X['C']>0.49980194301234493)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.49980194301234493)",
                 'already_optimal': "(X['A']>=0)",
                 },
        'L-BFGS-B': {'integer': "(X['A']>4.5)",
                     'float': "(X['C']>0.49980194301234493)",
                     'is_na': "(X['A']>4.5)|(X['A'].isna())",
                     'mixed': "((X['A']>4.5)&(X['C']>0.49980194301234493)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.49980194301234493)",
                     'already_optimal': "(X['A']>=0)",
                     },
        'TNC': {'integer': "(X['A']>4.5)",
                'float': "(X['C']>0.49980194301234493)",
                'is_na': "(X['A']>4.5)|(X['A'].isna())",
                'mixed': "((X['A']>4.5)&(X['C']>0.49980194301234493)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.49980194301234493)",
                'already_optimal': "(X['A']>=0)",
                },
        'COBYLA': {'integer': "(X['A']>-0.5)",
                   'float': "(X['C']>-0.5001980569876551)",
                   'is_na': "(X['A']>-0.5)|(X['A'].isna())",
                   'mixed': "((X['A']>4.5)&(X['C']>0.49980194301234493)&(X['E']=='yes')&(X['D']==True))|(X['C']>-0.5001980569876551)",
                   'already_optimal': "(X['A']>=0)",
                   },
        'SLSQP': {'integer': "(X['A']>4.5)",
                  'float': "(X['C']>0.49980194301234493)",
                  'is_na': "(X['A']>4.5)|(X['A'].isna())",
                  'mixed': "((X['A']>4.5)&(X['C']>0.49980194301234493)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.49980194301234493)",
                  'already_optimal': "(X['A']>=0)",
                  },
        'trust-constr': {'integer': "(X['A']>4.5)",
                         'float': "(X['C']>0.49980194301234493)",
                         'is_na': "(X['A']>4.5)|(X['A'].isna())",
                         'mixed': "((X['A']>4.5)&(X['C']>0.49980194301234493)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.49980194301234493)",
                         'already_optimal': "(X['A']>=0)",
                         }
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
                        },
        'Powell': {'integer': "(X['A']>9)",
                   'float': "(X['C']>1.5)",
                   'is_na': "(X['A']>9)|(X['A'].isna())",
                   'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                   'already_optimal': "(X['A']>=0)",
                   },
        'CG': {'integer': "(X['A']>9)",
               'float': "(X['C']>1.5)",
               'is_na': "(X['A']>9)|(X['A'].isna())",
               'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
               'already_optimal': "(X['A']>=0)",
               },
        'BFGS': {'integer': "(X['A']>9)",
                 'float': "(X['C']>1.5)",
                 'is_na': "(X['A']>9)|(X['A'].isna())",
                 'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                 'already_optimal': "(X['A']>=0)",
                 },
        'L-BFGS-B': {'integer': "(X['A']>9)",
                     'float': "(X['C']>1.5)",
                     'is_na': "(X['A']>9)|(X['A'].isna())",
                     'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                     'already_optimal': "(X['A']>=0)",
                     },
        'TNC': {'integer': "(X['A']>9)",
                'float': "(X['C']>1.5)",
                'is_na': "(X['A']>9)|(X['A'].isna())",
                'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                'already_optimal': "(X['A']>=0)",
                },
        'COBYLA': {'integer': "(X['A']>9)",
                   'float': "(X['C']>1.5)",
                   'is_na': "(X['A']>9)|(X['A'].isna())",
                   'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                   'already_optimal': "(X['A']>=0)",
                   },
        'SLSQP': {'integer': "(X['A']>9)",
                  'float': "(X['C']>1.5)",
                  'is_na': "(X['A']>9)|(X['A'].isna())",
                  'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                  'already_optimal': "(X['A']>=0)",
                  },
        'trust-constr': {'integer': "(X['A']>9)",
                         'float': "(X['C']>1.5)",
                         'is_na': "(X['A']>9)|(X['A'].isna())",
                         'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                         'already_optimal': "(X['A']>=0)",
                         }
    }


@ pytest.fixture
def _expected_rule_strings_weighted_x0_bounds():
    return {
        'Nelder-Mead': {'integer': "(X['A']>4.5)",
                        'float': "(X['C']>0.5056589970320209)",
                        'is_na': "(X['A']>4.5)|(X['A'].isna())",
                        'mixed': "((X['A']>4.4047598379629616)&(X['C']>0.5146510613102393)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.5054918821848204)",
                        'already_optimal': "(X['A']>=0)",
                        },
        'Powell': {'integer': "(X['A']>0.5015693034320075)",
                   'float': "(X['C']>0.14610333258470695)",
                   'is_na': "(X['A']>0.5015693034320075)|(X['A'].isna())",
                   'mixed': "((X['A']>8.691672481219253)&(X['C']>0.9917905018089224)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.21864412707936817)",
                   'already_optimal': "(X['A']>=0)",
                   },
        'CG': {'integer': "(X['A']>4.5)",
               'float': "(X['C']>0.49980194301234493)",
               'is_na': "(X['A']>4.5)|(X['A'].isna())",
               'mixed': "((X['A']>4.5)&(X['C']>0.49980194301234493)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.49980194301234493)",
               'already_optimal': "(X['A']>=0)",
               },
        'BFGS': {'integer': "(X['A']>4.5)",
                 'float': "(X['C']>0.49980194301234493)",
                 'is_na': "(X['A']>4.5)|(X['A'].isna())",
                 'mixed': "((X['A']>4.5)&(X['C']>0.49980194301234493)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.49980194301234493)",
                 'already_optimal': "(X['A']>=0)",
                 },
        'L-BFGS-B': {'integer': "(X['A']>4.5)",
                     'float': "(X['C']>0.49980194301234493)",
                     'is_na': "(X['A']>4.5)|(X['A'].isna())",
                     'mixed': "((X['A']>4.5)&(X['C']>0.49980194301234493)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.49980194301234493)",
                     'already_optimal': "(X['A']>=0)",
                     },
        'TNC': {'integer': "(X['A']>4.5)",
                'float': "(X['C']>0.49980194301234493)",
                'is_na': "(X['A']>4.5)|(X['A'].isna())",
                'mixed': "((X['A']>4.5)&(X['C']>0.49980194301234493)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.49980194301234493)",
                'already_optimal': "(X['A']>=0)",
                },
        'COBYLA': {'integer': "(X['A']>-0.5)",
                   'float': "(X['C']>-0.5001980569876551)",
                   'is_na': "(X['A']>-0.5)|(X['A'].isna())",
                   'mixed': "((X['A']>4.5)&(X['C']>0.49980194301234493)&(X['E']=='yes')&(X['D']==True))|(X['C']>-0.5001980569876551)",
                   'already_optimal': "(X['A']>=0)",
                   },
        'SLSQP': {'integer': "(X['A']>4.5)",
                  'float': "(X['C']>0.49980194301234493)",
                  'is_na': "(X['A']>4.5)|(X['A'].isna())",
                  'mixed': "((X['A']>4.5)&(X['C']>0.49980194301234493)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.49980194301234493)",
                  'already_optimal': "(X['A']>=0)",
                  },
        'trust-constr': {'integer': "(X['A']>4.5)",
                         'float': "(X['C']>0.49980194301234493)",
                         'is_na': "(X['A']>4.5)|(X['A'].isna())",
                         'mixed': "((X['A']>4.5)&(X['C']>0.49980194301234493)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.49980194301234493)",
                         'already_optimal': "(X['A']>=0)",
                         }
    }


@ pytest.fixture
def _expected_rule_strings_unlabelled():
    return {
        'Nelder-Mead': {'integer': "(X['A']>9)",
                        'float': "(X['C']>1.5)",
                        'is_na': "(X['A']>9)|(X['A'].isna())",
                        'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                        'already_optimal': "(X['A']>=0.00025)",
                        },
        'Powell': {'integer': "(X['A']>9)",
                   'float': "(X['C']>1.5)",
                   'is_na': "(X['A']>9)|(X['A'].isna())",
                   'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                   'already_optimal': "(X['A']>=177.53363890059836)",
                   },
        'CG': {'integer': "(X['A']>9)",
               'float': "(X['C']>1.5)",
               'is_na': "(X['A']>9)|(X['A'].isna())",
               'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
               'already_optimal': "(X['A']>=5.125898936452442e-05)",
               },
        'BFGS': {'integer': "(X['A']>9)",
                 'float': "(X['C']>1.5)",
                 'is_na': "(X['A']>9)|(X['A'].isna())",
                 'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                 'already_optimal': "(X['A']>=5.125898936452442e-05)",
                 },
        'L-BFGS-B': {'integer': "(X['A']>9)",
                     'float': "(X['C']>1.5)",
                     'is_na': "(X['A']>9)|(X['A'].isna())",
                     'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                     'already_optimal': "(X['A']>=3.595679261195249e-06)",
                     },
        'TNC': {'integer': "(X['A']>9)",
                'float': "(X['C']>1.5)",
                'is_na': "(X['A']>9)|(X['A'].isna())",
                'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                'already_optimal': "(X['A']>=1.4901161193847656e-08)",
                },
        'COBYLA': {'integer': "(X['A']>9)",
                   'float': "(X['C']>1.5)",
                   'is_na': "(X['A']>9)|(X['A'].isna())",
                   'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                   'already_optimal': "(X['A']>=10.0)",
                   },
        'SLSQP': {'integer': "(X['A']>9)",
                  'float': "(X['C']>1.5)",
                  'is_na': "(X['A']>9)|(X['A'].isna())",
                  'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                  'already_optimal': "(X['A']>=11935095848.960007)",
                  },
        'trust-constr': {'integer': "(X['A']>9)",
                         'float': "(X['C']>1.5)",
                         'is_na': "(X['A']>9)|(X['A'].isna())",
                         'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                         'already_optimal': "(X['A']>=2.7020533257768187)",
                         }
    }


@ pytest.fixture
def _expected_rule_strings_unlabelled_x0_bounds():
    return {
        'Nelder-Mead': {'integer': "(X['A']>9)",
                        'float': "(X['C']>1.5)",
                        'is_na': "(X['A']>9)|(X['A'].isna())",
                        'mixed': "((X['A']>8.617018900812145)&(X['C']>0.9999680225821261)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.9887349505409877)",
                        'already_optimal': "(X['A']>=4.5)",
                        },
        'Powell': {'integer': "(X['A']>9)",
                   'float': "(X['C']>0.9882521688325011)",
                   'is_na': "(X['A']>9)|(X['A'].isna())",
                   'mixed': "((X['A']>8.04798865353395)&(X['C']>0.8669412930016224)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.9955975102523807)",
                   'already_optimal': "(X['A']>=8.203192518090468)",
                   },
        'CG': {'integer': "(X['A']>9)",
               'float': "(X['C']>1.5)",
               'is_na': "(X['A']>9)|(X['A'].isna())",
               'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
               'already_optimal': "(X['A']>=4.5)",
               },
        'BFGS': {'integer': "(X['A']>9)",
                 'float': "(X['C']>1.5)",
                 'is_na': "(X['A']>9)|(X['A'].isna())",
                 'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                 'already_optimal': "(X['A']>=4.5)",
                 },
        'L-BFGS-B': {'integer': "(X['A']>9)",
                     'float': "(X['C']>1.5)",
                     'is_na': "(X['A']>9)|(X['A'].isna())",
                     'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                     'already_optimal': "(X['A']>=4.5)",
                     },
        'TNC': {'integer': "(X['A']>9)",
                'float': "(X['C']>1.5)",
                'is_na': "(X['A']>9)|(X['A'].isna())",
                'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                'already_optimal': "(X['A']>=4.5)",
                },
        'COBYLA': {'integer': "(X['A']>9)",
                   'float': "(X['C']>1.5)",
                   'is_na': "(X['A']>9)|(X['A'].isna())",
                   'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                   'already_optimal': "(X['A']>=9.5)",
                   },
        'SLSQP': {'integer': "(X['A']>9)",
                  'float': "(X['C']>1.5)",
                  'is_na': "(X['A']>9)|(X['A'].isna())",
                  'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                  'already_optimal': "(X['A']>=4.5)",
                  },
        'trust-constr': {'integer': "(X['A']>9)",
                         'float': "(X['C']>1.5)",
                         'is_na': "(X['A']>9)|(X['A'].isna())",
                         'mixed': "((X['A']>1)&(X['C']>1.5)&(X['E']=='yes')&(X['D']==True))|(X['C']>2.5)",
                         'already_optimal': "(X['A']>=4.5)",
                         }}


@ pytest.fixture
def _expected_X_rules_mean():
    X_rules_means = {
        'Nelder-Mead': {'integer': 0.10428957104289571,
                        'float': 0.0,
                        'is_na': 0.10438956104389562,
                        'mixed': 0.0,
                        'already_optimal': 0.9999000099990001},
        'Powell': {'integer': 0.0,
                   'float': 0.0,
                   'is_na': 9.999000099990002e-05,
                   'mixed': 0.0,
                   'already_optimal': 0.9999000099990001},
        'CG': {'integer': 0.0,
               'float': 0.0,
               'is_na': 9.999000099990002e-05,
               'mixed': 0.0,
               'already_optimal': 0.9999000099990001},
        'BFGS': {'integer': 0.0,
                 'float': 0.0,
                 'is_na': 9.999000099990002e-05,
                 'mixed': 0.0,
                 'already_optimal': 0.9999000099990001},
        'L-BFGS-B': {'integer': 0.0,
                     'float': 0.0,
                     'is_na': 9.999000099990002e-05,
                     'mixed': 0.0,
                     'already_optimal': 0.9999000099990001},
        'TNC': {'integer': 0.0,
                'float': 0.0,
                'is_na': 9.999000099990002e-05,
                'mixed': 0.0,
                'already_optimal': 0.9999000099990001},
        'COBYLA': {'integer': 0.0,
                   'float': 0.0,
                   'is_na': 9.999000099990002e-05,
                   'mixed': 0.0,
                   'already_optimal': 0.9999000099990001},
        'SLSQP': {'integer': 0.0,
                  'float': 0.0,
                  'is_na': 9.999000099990002e-05,
                  'mixed': 0.0,
                  'already_optimal': 0.9999000099990001},
        'trust-constr': {'integer': 0.0,
                         'float': 0.0,
                         'is_na': 9.999000099990002e-05,
                         'mixed': 0.0,
                         'already_optimal': 0.9999000099990001}
    }
    return X_rules_means


@ pytest.fixture
def _expected_X_rules_mean_x0_bounds():
    X_rules_means = {
        'Nelder-Mead': {'integer': 0.5061493850614939,
                        'float': 0.9998000199980002,
                        'is_na': 0.5062493750624938,
                        'mixed': 0.9998000199980002,
                        'already_optimal': 0.9999000099990001},
        'Powell': {'integer': 0.9032096790320968,
                   'float': 0.9988001199880012,
                   'is_na': 0.9033096690330967,
                   'mixed': 0.9986001399860014,
                   'already_optimal': 0.9999000099990001},
        'CG': {'integer': 0.5061493850614939,
               'float': 0.4991500849915009,
               'is_na': 0.5062493750624938,
               'mixed': 0.4991500849915009,
               'already_optimal': 0.9999000099990001},
        'BFGS': {'integer': 0.5061493850614939,
                 'float': 0.4991500849915009,
                 'is_na': 0.5062493750624938,
                 'mixed': 0.4991500849915009,
                 'already_optimal': 0.9999000099990001},
        'L-BFGS-B': {'integer': 0.5061493850614939,
                     'float': 0.4991500849915009,
                     'is_na': 0.5062493750624938,
                     'mixed': 0.4991500849915009,
                     'already_optimal': 0.9999000099990001},
        'TNC': {'integer': 0.5061493850614939,
                'float': 0.4991500849915009,
                'is_na': 0.5062493750624938,
                'mixed': 0.4991500849915009,
                'already_optimal': 0.9999000099990001},
        'COBYLA': {'integer': 0.9999000099990001,
                   'float': 0.9999000099990001,
                   'is_na': 1.0,
                   'mixed': 0.9999000099990001,
                   'already_optimal': 0.9999000099990001},
        'SLSQP': {'integer': 0.5061493850614939,
                  'float': 0.4991500849915009,
                  'is_na': 0.5062493750624938,
                  'mixed': 0.4991500849915009,
                  'already_optimal': 0.9999000099990001},
        'trust-constr': {'integer': 0.5061493850614939,
                         'float': 0.4991500849915009,
                         'is_na': 0.5062493750624938,
                         'mixed': 0.4991500849915009,
                         'already_optimal': 0.9999000099990001}
    }
    return X_rules_means


@ pytest.fixture
def _expected_X_rules_mean_weighted():
    return {
        'Nelder-Mead': {'integer': 0.10428957104289571,
                        'float': 0.0,
                        'is_na': 0.10438956104389562,
                        'mixed': 0.0,
                        'already_optimal': 0.9999000099990001},
        'Powell': {'integer': 0.0,
                   'float': 0.0,
                   'is_na': 9.999000099990002e-05,
                   'mixed': 0.0,
                   'already_optimal': 0.9999000099990001},
        'CG': {'integer': 0.0,
               'float': 0.0,
               'is_na': 9.999000099990002e-05,
               'mixed': 0.0,
               'already_optimal': 0.9999000099990001},
        'BFGS': {'integer': 0.0,
                 'float': 0.0,
                 'is_na': 9.999000099990002e-05,
                 'mixed': 0.0,
                 'already_optimal': 0.9999000099990001},
        'L-BFGS-B': {'integer': 0.0,
                     'float': 0.0,
                     'is_na': 9.999000099990002e-05,
                     'mixed': 0.0,
                     'already_optimal': 0.9999000099990001},
        'TNC': {'integer': 0.0,
                'float': 0.0,
                'is_na': 9.999000099990002e-05,
                'mixed': 0.0,
                'already_optimal': 0.9999000099990001},
        'COBYLA': {'integer': 0.0,
                   'float': 0.0,
                   'is_na': 9.999000099990002e-05,
                   'mixed': 0.0,
                   'already_optimal': 0.9999000099990001},
        'SLSQP': {'integer': 0.0,
                  'float': 0.0,
                  'is_na': 9.999000099990002e-05,
                  'mixed': 0.0,
                  'already_optimal': 0.9999000099990001},
        'trust-constr': {'integer': 0.0,
                         'float': 0.0,
                         'is_na': 9.999000099990002e-05,
                         'mixed': 0.0,
                         'already_optimal': 0.9999000099990001}
    }


@ pytest.fixture
def _expected_X_rules_mean_weighted_x0_bounds():
    return {
        'Nelder-Mead': {'integer': 0.5061493850614939,
                        'float': 0.49375062493750627,
                        'is_na': 0.5062493750624938,
                        'mixed': 0.4938506149385061,
                        'already_optimal': 0.9999000099990001},
        'Powell': {'integer': 0.9032096790320968,
                   'float': 0.8528147185281472,
                   'is_na': 0.9033096690330967,
                   'mixed': 0.7774222577742226,
                   'already_optimal': 0.9999000099990001},
        'CG': {'integer': 0.5061493850614939,
               'float': 0.4991500849915009,
               'is_na': 0.5062493750624938,
               'mixed': 0.4991500849915009,
               'already_optimal': 0.9999000099990001},
        'BFGS': {'integer': 0.5061493850614939,
                 'float': 0.4991500849915009,
                 'is_na': 0.5062493750624938,
                 'mixed': 0.4991500849915009,
                 'already_optimal': 0.9999000099990001},
        'L-BFGS-B': {'integer': 0.5061493850614939,
                     'float': 0.4991500849915009,
                     'is_na': 0.5062493750624938,
                     'mixed': 0.4991500849915009,
                     'already_optimal': 0.9999000099990001},
        'TNC': {'integer': 0.5061493850614939,
                'float': 0.4991500849915009,
                'is_na': 0.5062493750624938,
                'mixed': 0.4991500849915009,
                'already_optimal': 0.9999000099990001},
        'COBYLA': {'integer': 0.9999000099990001,
                   'float': 0.9999000099990001,
                   'is_na': 1.0,
                   'mixed': 0.9999000099990001,
                   'already_optimal': 0.9999000099990001},
        'SLSQP': {'integer': 0.5061493850614939,
                  'float': 0.4991500849915009,
                  'is_na': 0.5062493750624938,
                  'mixed': 0.4991500849915009,
                  'already_optimal': 0.9999000099990001},
        'trust-constr': {'integer': 0.5061493850614939,
                         'float': 0.4991500849915009,
                         'is_na': 0.5062493750624938,
                         'mixed': 0.4991500849915009,
                         'already_optimal': 0.9999000099990001}
    }


@ pytest.fixture
def _expected_X_rules_mean_unlabelled():
    return {
        'Nelder-Mead': {'integer': 0.0,
                        'float': 0.0,
                        'is_na': 9.999000099990002e-05,
                        'mixed': 0.0,
                        'already_optimal': 0.9032096790320968},
        'Powell': {'integer': 0.0,
                   'float': 0.0,
                   'is_na': 9.999000099990002e-05,
                   'mixed': 0.0,
                   'already_optimal': 0.0},
        'CG': {'integer': 0.0,
               'float': 0.0,
               'is_na': 9.999000099990002e-05,
               'mixed': 0.0,
               'already_optimal': 0.9032096790320968},
        'BFGS': {'integer': 0.0,
                 'float': 0.0,
                 'is_na': 9.999000099990002e-05,
                 'mixed': 0.0,
                 'already_optimal': 0.9032096790320968},
        'L-BFGS-B': {'integer': 0.0,
                     'float': 0.0,
                     'is_na': 9.999000099990002e-05,
                     'mixed': 0.0,
                     'already_optimal': 0.9032096790320968},
        'TNC': {'integer': 0.0,
                'float': 0.0,
                'is_na': 9.999000099990002e-05,
                'mixed': 0.0,
                'already_optimal': 0.9032096790320968},
        'COBYLA': {'integer': 0.0,
                   'float': 0.0,
                   'is_na': 9.999000099990002e-05,
                   'mixed': 0.0,
                   'already_optimal': 0.0},
        'SLSQP': {'integer': 0.0,
                  'float': 0.0,
                  'is_na': 9.999000099990002e-05,
                  'mixed': 0.0,
                  'already_optimal': 0.0},
        'trust-constr': {'integer': 0.0,
                         'float': 0.0,
                         'is_na': 9.999000099990002e-05,
                         'mixed': 0.0,
                         'already_optimal': 0.7034296570342966}
    }


@ pytest.fixture
def _expected_X_rules_mean_unlabelled_x0_bounds():
    return {
        'Nelder-Mead': {'integer': 0.0,
                        'float': 0.0,
                        'is_na': 9.999000099990002e-05,
                        'mixed': 0.0098990100989901,
                        'already_optimal': 0.5061493850614939},
        'Powell': {'integer': 0.0,
                   'float': 0.0098990100989901,
                   'is_na': 9.999000099990002e-05,
                   'mixed': 0.009999000099990002,
                   'already_optimal': 0.10428957104289571},
        'CG': {'integer': 0.0,
               'float': 0.0,
               'is_na': 9.999000099990002e-05,
               'mixed': 0.0,
               'already_optimal': 0.5061493850614939},
        'BFGS': {'integer': 0.0,
                 'float': 0.0,
                 'is_na': 9.999000099990002e-05,
                 'mixed': 0.0,
                 'already_optimal': 0.5061493850614939},
        'L-BFGS-B': {'integer': 0.0,
                     'float': 0.0,
                     'is_na': 9.999000099990002e-05,
                     'mixed': 0.0,
                     'already_optimal': 0.5061493850614939},
        'TNC': {'integer': 0.0,
                'float': 0.0,
                'is_na': 9.999000099990002e-05,
                'mixed': 0.0,
                'already_optimal': 0.5061493850614939},
        'COBYLA': {'integer': 0.0,
                   'float': 0.0,
                   'is_na': 9.999000099990002e-05,
                   'mixed': 0.0,
                   'already_optimal': 0.0},
        'SLSQP': {'integer': 0.0,
                  'float': 0.0,
                  'is_na': 9.999000099990002e-05,
                  'mixed': 0.0,
                  'already_optimal': 0.5061493850614939},
        'trust-constr': {'integer': 0.0,
                         'float': 0.0,
                         'is_na': 9.999000099990002e-05,
                         'mixed': 0.0,
                         'already_optimal': 0.5061493850614939}
    }


@ pytest.fixture
def _exp_orig_rule_performances():
    return {
        'already_optimal': 0.6657771847898598,
        'integer': 0.0,
        'float': 0.0,
        'is_na': 0.0,
        'mixed': 0.0
    }


@ pytest.fixture
def _exp_orig_rule_performances_weighted():
    return {
        'already_optimal': 0.08504038992467365,
        'integer': 0.0,
        'float': 0.0,
        'is_na': 0.0,
        'mixed': 0.0
    }


@ pytest.fixture
def _exp_orig_rule_performances_unlabelled():
    return {
        'is_na': -98.01,
        'integer': -100.0,
        'float': -100.0,
        'mixed': -100.0,
        'already_optimal': -980100.0
    }


@ pytest.fixture
def _exp_opt_rule_performances():
    return {
        'Nelder-Mead': {'already_optimal': 0.6657771847898598,
                        'integer': 0.18332504558262888,
                        'is_na': 0.18329466357308585,
                        'float': 0.0,
                        'mixed': 0.0},
        'Powell': {'already_optimal': 0.6657771847898598,
                   'integer': 0.0,
                   'float': 0.0,
                   'is_na': 0.0,
                   'mixed': 0.0},
        'CG': {'already_optimal': 0.6657771847898598,
               'integer': 0.0,
               'float': 0.0,
               'is_na': 0.0,
               'mixed': 0.0},
        'BFGS': {'already_optimal': 0.6657771847898598,
                 'integer': 0.0,
                 'float': 0.0,
                 'is_na': 0.0,
                 'mixed': 0.0},
        'L-BFGS-B': {'already_optimal': 0.6657771847898598,
                     'integer': 0.0,
                     'float': 0.0,
                     'is_na': 0.0,
                     'mixed': 0.0},
        'TNC': {'already_optimal': 0.6657771847898598,
                'integer': 0.0,
                'float': 0.0,
                'is_na': 0.0,
                'mixed': 0.0},
        'COBYLA': {'already_optimal': 0.6657771847898598,
                   'integer': 0.0,
                   'float': 0.0,
                   'is_na': 0.0,
                   'mixed': 0.0},
        'SLSQP': {'already_optimal': 0.6657771847898598,
                  'integer': 0.0,
                  'float': 0.0,
                  'is_na': 0.0,
                  'mixed': 0.0},
        'trust-constr': {'already_optimal': 0.6657771847898598,
                         'integer': 0.0,
                         'float': 0.0,
                         'is_na': 0.0,
                         'mixed': 0.0}
    }


@ pytest.fixture
def _exp_opt_rule_performances_x0_bounds():
    return {
        'Nelder-Mead': {'float': 0.6658216025085062,
                        'mixed': 0.6658216025085062,
                        'integer': 0.4988062077198568,
                        'already_optimal': 0.6657771847898598,
                        'is_na': 0.4987565900726152},
        'Powell': {'float': 0.66506442352627,
                   'mixed': 0.6650196968685318,
                   'integer': 0.6422306211224418,
                   'already_optimal': 0.6657771847898598,
                   'is_na': 0.6421848260125499},
        'CG': {'float': 0.5009016229212583,
               'mixed': 0.5009016229212583,
               'integer': 0.4988062077198568,
               'already_optimal': 0.6657771847898598,
               'is_na': 0.4987565900726152},
        'BFGS': {'float': 0.5009016229212583,
                 'mixed': 0.5009016229212583,
                 'integer': 0.4988062077198568,
                 'already_optimal': 0.6657771847898598,
                 'is_na': 0.4987565900726152},
        'L-BFGS-B': {'float': 0.5009016229212583,
                     'mixed': 0.5009016229212583,
                     'integer': 0.4988062077198568,
                     'already_optimal': 0.6657771847898598,
                     'is_na': 0.4987565900726152},
        'TNC': {'float': 0.5009016229212583,
                'mixed': 0.5009016229212583,
                'integer': 0.4988062077198568,
                'already_optimal': 0.6657771847898598,
                'is_na': 0.4987565900726152},
        'COBYLA': {'integer': 0.6657771847898598,
                   'float': 0.6657771847898598,
                   'mixed': 0.6657771847898598,
                   'already_optimal': 0.6657771847898598,
                   'is_na': 0.6657327729971316},
        'SLSQP': {'float': 0.5009016229212583,
                  'mixed': 0.5009016229212583,
                  'integer': 0.4988062077198568,
                  'already_optimal': 0.6657771847898598,
                  'is_na': 0.4987565900726152},
        'trust-constr': {'float': 0.5009016229212583,
                         'mixed': 0.5009016229212583,
                         'integer': 0.4988062077198568,
                         'already_optimal': 0.6657771847898598,
                         'is_na': 0.4987565900726152}
    }


@ pytest.fixture
def _exp_opt_rule_performances_weighted():
    return {
        'Nelder-Mead': {'already_optimal': 0.08504038992467365,
                        'integer': 0.020277579157728768,
                        'is_na': 0.020277207392197127,
                        'float': 0.0,
                        'mixed': 0.0},
        'Powell': {'already_optimal': 0.08504038992467365,
                   'integer': 0.0,
                   'float': 0.0,
                   'is_na': 0.0,
                   'mixed': 0.0},
        'CG': {'already_optimal': 0.08504038992467365,
               'integer': 0.0,
               'float': 0.0,
               'is_na': 0.0,
               'mixed': 0.0},
        'BFGS': {'already_optimal': 0.08504038992467365,
                 'integer': 0.0,
                 'float': 0.0,
                 'is_na': 0.0,
                 'mixed': 0.0},
        'L-BFGS-B': {'already_optimal': 0.08504038992467365,
                     'integer': 0.0,
                     'float': 0.0,
                     'is_na': 0.0,
                     'mixed': 0.0},
        'TNC': {'already_optimal': 0.08504038992467365,
                'integer': 0.0,
                'float': 0.0,
                'is_na': 0.0,
                'mixed': 0.0},
        'COBYLA': {'already_optimal': 0.08504038992467365,
                   'integer': 0.0,
                   'float': 0.0,
                   'is_na': 0.0,
                   'mixed': 0.0},
        'SLSQP': {'already_optimal': 0.08504038992467365,
                  'integer': 0.0,
                  'float': 0.0,
                  'is_na': 0.0,
                  'mixed': 0.0},
        'trust-constr': {'already_optimal': 0.08504038992467365,
                         'integer': 0.0,
                         'float': 0.0,
                         'is_na': 0.0,
                         'mixed': 0.0}
    }


@ pytest.fixture
def _exp_opt_rule_performances_weighted_x0_bounds():
    return {
        'Nelder-Mead': {'mixed': 0.08318774358285178,
                        'float': 0.08315554286290255,
                        'integer': 0.044601398352576996,
                        'already_optimal': 0.08504038992467365,
                        'is_na': 0.044601001610048124},
        'Powell': {'float': 0.08649740043410227,
                   'mixed': 0.08528603220316795,
                   'integer': 0.07737844641675759,
                   'already_optimal': 0.08504038992467365,
                   'is_na': 0.07737778159635708},
        'CG': {'float': 0.08281847845891376,
               'mixed': 0.08281847845891376,
               'integer': 0.044601398352576996,
               'already_optimal': 0.08504038992467365,
               'is_na': 0.044601001610048124},
        'BFGS': {'float': 0.08281847845891376,
                 'mixed': 0.08281847845891376,
                 'integer': 0.044601398352576996,
                 'already_optimal': 0.08504038992467365,
                 'is_na': 0.044601001610048124},
        'L-BFGS-B': {'float': 0.08281847845891376,
                     'mixed': 0.08281847845891376,
                     'integer': 0.044601398352576996,
                     'already_optimal': 0.08504038992467365,
                     'is_na': 0.044601001610048124},
        'TNC': {'float': 0.08281847845891376,
                'mixed': 0.08281847845891376,
                'integer': 0.044601398352576996,
                'already_optimal': 0.08504038992467365,
                'is_na': 0.044601001610048124},
        'COBYLA': {'integer': 0.08504038992467365,
                   'float': 0.08504038992467365,
                   'mixed': 0.08504038992467365,
                   'already_optimal': 0.08504038992467365,
                   'is_na': 0.0850396652947843},
        'SLSQP': {'float': 0.08281847845891376,
                  'mixed': 0.08281847845891376,
                  'integer': 0.044601398352576996,
                  'already_optimal': 0.08504038992467365,
                  'is_na': 0.044601001610048124},
        'trust-constr': {'float': 0.08281847845891376,
                         'mixed': 0.08281847845891376,
                         'integer': 0.044601398352576996,
                         'already_optimal': 0.08504038992467365,
                         'is_na': 0.044601001610048124}
    }


@ pytest.fixture
def _exp_opt_rule_performances_unlabelled():
    return {
        'Nelder-Mead': {'is_na': -98.01,
                        'integer': -100.0,
                        'float': -100.0,
                        'mixed': -100.0,
                        'already_optimal': -797984.8899999999},
        'Powell': {'is_na': -98.01,
                   'integer': -100.0,
                   'float': -100.0,
                   'mixed': -100.0,
                   'already_optimal': -100.0},
        'CG': {'is_na': -98.01,
               'integer': -100.0,
               'float': -100.0,
               'mixed': -100.0,
               'already_optimal': -797984.8899999999},
        'BFGS': {'is_na': -98.01,
                 'integer': -100.0,
                 'float': -100.0,
                 'mixed': -100.0,
                 'already_optimal': -797984.8899999999},
        'L-BFGS-B': {'is_na': -98.01,
                     'integer': -100.0,
                     'float': -100.0,
                     'mixed': -100.0,
                     'already_optimal': -797984.8899999999},
        'TNC': {'is_na': -98.01,
                'integer': -100.0,
                'float': -100.0,
                'mixed': -100.0,
                'already_optimal': -797984.8899999999},
        'COBYLA': {'is_na': -98.01,
                   'integer': -100.0,
                   'float': -100.0,
                   'mixed': -100.0,
                   'already_optimal': -100.0},
        'SLSQP': {'is_na': -98.01,
                  'integer': -100.0,
                  'float': -100.0,
                  'mixed': -100.0,
                  'already_optimal': -100.0},
        'trust-constr': {'is_na': -98.01,
                         'integer': -100.0,
                         'float': -100.0,
                         'mixed': -100.0,
                         'already_optimal': -480942.25}}


@ pytest.fixture
def _exp_opt_rule_performances_unlabelled_x0_bounds():
    return {
        'Nelder-Mead': {'mixed': -0.009999999999999929,
                        'float': -100.0,
                        'integer': -100.0,
                        'already_optimal': -246214.44,
                        'is_na': -98.01},
        'Powell': {'mixed': -0.0,
                   'float': -0.009999999999999929,
                   'integer': -100.0,
                   'already_optimal': -8892.49,
                   'is_na': -98.01},
        'CG': {'float': -100.0,
               'mixed': -100.0,
               'integer': -100.0,
               'already_optimal': -246214.44,
               'is_na': -98.01},
        'BFGS': {'float': -100.0,
                 'mixed': -100.0,
                 'integer': -100.0,
                 'already_optimal': -246214.44,
                 'is_na': -98.01},
        'L-BFGS-B': {'float': -100.0,
                     'mixed': -100.0,
                     'integer': -100.0,
                     'already_optimal': -246214.44,
                     'is_na': -98.01},
        'TNC': {'float': -100.0,
                'mixed': -100.0,
                'integer': -100.0,
                'already_optimal': -246214.44,
                'is_na': -98.01},
        'COBYLA': {'is_na': -98.01,
                   'integer': -100.0,
                   'float': -100.0,
                   'mixed': -100.0,
                   'already_optimal': -100.0},
        'SLSQP': {'float': -100.0,
                  'mixed': -100.0,
                  'integer': -100.0,
                  'already_optimal': -246214.44,
                  'is_na': -98.01},
        'trust-constr': {'float': -100.0,
                         'mixed': -100.0,
                         'integer': -100.0,
                         'already_optimal': -246214.44,
                         'is_na': -98.01}}


@ pytest.fixture
def _instantiate(_create_data, _create_inputs):
    X, _, _ = _create_data
    rule_lambdas, lambda_kwargs = _create_inputs
    f1 = FScore(beta=1)
    with pytest.warns(
            UserWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `x0` for these rules") and \
        pytest.warns(
            UserWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `x0` for these rules") and\
        pytest.warns(
            UserWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `bounds` for these rules") and \
        pytest.warns(
            UserWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `bounds` for these rules"):
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
            UserWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `x0` for these rules") and \
        pytest.warns(
            UserWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `x0` for these rules") and\
        pytest.warns(
            UserWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `bounds` for these rules") and \
        pytest.warns(
            UserWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `bounds` for these rules"):
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
        'integer': [(0, 9)],
        'float': [(0.0003641365574362787, 0.9999680225821261)],
        'is_na': [(0, 9)],
        'mixed': [(0, 9),
                  (0.0003641365574362787, 0.9999680225821261),
                  (0.0003641365574362787, 0.9999680225821261)],
        'all_na': [(np.nan, np.nan)],
        'zero_var': [(1.0, 1.0)],
        'already_optimal': [(0, 9)]}
    X, _, _ = _create_data
    _, lambda_kwargs = _create_inputs
    with pytest.warns(
            UserWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `bounds` for these rules") and \
        pytest.warns(
            UserWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `bounds` for these rules"):
        bounds = DirectSearchOptimiser.create_bounds(X, lambda_kwargs)
        # Check `all_na` separately as np.nan != np.nan
        np.testing.assert_array_equal(bounds['all_na'], exp_bounds['all_na'])
        bounds.pop('all_na')
        exp_bounds.pop('all_na')
        assert bounds == exp_bounds


def test_create_x0(_create_data, _create_inputs):
    exp_x0 = {
        'integer': np.array([4.5]),
        'float': np.array([0.49980194]),
        'is_na': np.array([4.5]),
        'mixed': np.array([4.5, 0.49980194, 0.49980194]),
        'all_na': np.array([np.nan]),
        'zero_var': np.array([0.]),
        'already_optimal': np.array([4.5])
    }
    X, _, _ = _create_data
    _, lambda_kwargs = _create_inputs
    with pytest.warns(
            UserWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `x0` for these rules") and \
        pytest.warns(
            UserWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `x0` for these rules"):
        x0 = DirectSearchOptimiser.create_x0(X, lambda_kwargs)
        for rule_name in x0.keys():
            np.testing.assert_array_almost_equal(
                x0[rule_name], exp_x0[rule_name])


def test_create_initial_simplexes(_create_data, _create_inputs):
    exp_simplexes = {
        'Origin-based': {
            'integer': {'initial_simplex': np.array([[0.], [9.]])},
            'float': {'initial_simplex': np.array([[3.64136557e-04], [9.99968023e-01]])},
            'is_na': {'initial_simplex': np.array([[0.], [9.]])},
            'mixed': {'initial_simplex': np.array([
                [0.00000000e+00, 3.64136557e-04, 3.64136557e-04],
                [9.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                [0.00000000e+00, 9.99968023e-01, 0.00000000e+00],
                [0.00000000e+00, 0.00000000e+00, 9.99968023e-01]])
            },
            'all_na': {'initial_simplex': np.array([[np.nan], [np.nan]])},
            'zero_var': {'initial_simplex': np.array([[1.], [1.]])},
            'already_optimal': {'initial_simplex': np.array([[0.], [9.]])}
        },
        'Minimum-based': {
            'integer': {'initial_simplex': np.array([[0], [9.0]])},
            'float': {'initial_simplex': np.array([[0.0003641365574362787], [0.9999680225821261]])},
            'is_na': {'initial_simplex': np.array([[0], [9.0]])},
            'mixed': {'initial_simplex': np.array([
                [0, 0.0003641365574362787, 0.0003641365574362787],
                [9.0, 0.0003641365574362787, 0.0003641365574362787],
                [0.0, 0.9999680225821261, 0.0003641365574362787],
                [0.0, 0.0003641365574362787, 0.9999680225821261]])},
            'all_na': {'initial_simplex': np.array([[np.nan], [np.nan]])},
            'zero_var': {'initial_simplex': np.array([[1.0], [1.0]])},
            'already_optimal': {'initial_simplex': np.array([[0], [9.0]])}
        },
        'Random-based': {
            'integer': {'initial_simplex': np.array([[4.94426086], [6.443141]])},
            'float': {'initial_simplex': np.array([[0.54950884], [0.71598511]])},
            'is_na': {'initial_simplex': np.array([[4.94426086], [6.443141]])},
            'mixed': {'initial_simplex': np.array([
                [4.94426086, 0.42427461, 0.96460846],
                [6.443141, 0.64664804, 0.38403706],
                [5.43029526, 0.43821543, 0.79256697],
                [4.9088526, 0.89267531, 0.52957824]])},
            'all_na': {'initial_simplex': np.array([[np.nan], [np.nan]])},
            'zero_var': {'initial_simplex': np.array([[1.], [1.]])},
            'already_optimal': {'initial_simplex': np.array([[4.94426086], [6.443141]])}
        }
    }
    X, _, _ = _create_data
    _, lambda_kwargs = _create_inputs
    for shape in ['Origin-based', 'Minimum-based', 'Random-based']:
        with pytest.warns(
                UserWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `initial_simplex` for these rules") and \
            pytest.warns(
                UserWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `initial_simplex` for these rules"):
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
    exp_rule_strings = {
        'integer': "(X['A']>4.5)",
        'float': "(X['C']>0.0003641365574362787)",
        'is_na': "(X['A']>4.5)|(X['A'].isna())",
        'mixed': "((X['A']>5.966666666666709)&(X['C']>0.6367846977638774)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.0003641365574362787)",
        'already_optimal': "(X['A']>=4.5)"
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
        rule_lambdas_, lambda_kwargs_, X, y, None)
    assert opt_rule_strings == exp_rule_strings


def test_optimise_rules_weighted(_instantiate, _create_inputs, _create_data):
    X, y, sample_weight = _create_data
    exp_rule_strings = {
        'integer': "(X['A']>4.5)",
        'float': "(X['C']>0.5056589970320209)",
        'is_na': "(X['A']>4.5)|(X['A'].isna())",
        'mixed': "((X['A']>4.4047598379629616)&(X['C']>0.5146510613102393)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.5054918821848204)",
        'already_optimal': "(X['A']>=4.5)"
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
        'mixed': "((X['A']>8.617018900812145)&(X['C']>0.9999680225821261)&(X['E']=='yes')&(X['D']==True))|(X['C']>0.9887349505409877)",
        'already_optimal': "(X['A']>=4.5)"
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
        rule_lambdas_, lambda_kwargs_, X, None, None)
    assert opt_rule_strings == exp_rule_strings


def test_optimise_rules_numpy(_instantiate, _create_data):
    X, y, _ = _create_data
    exp_rule_strings = {
        'already_optimal': "(X['A'].to_numpy(na_value=np.nan)>=4.5)",
        'float': "(X['C'].to_numpy(na_value=np.nan)>0.0003641365574362787)",
        'integer': "(X['A'].to_numpy(na_value=np.nan)>4.5)",
        'is_na': "(X['A'].to_numpy(na_value=np.nan)>4.5)|(pd.isna(X['A'].to_numpy(na_value=np.nan)))",
        'mixed': "((X['A'].to_numpy(na_value=np.nan)>5.966666666666709)&(X['C'].to_numpy(na_value=np.nan)>0.6367846977638774)&(X['E'].to_numpy(na_value=np.nan)=='yes')&(X['D'].to_numpy(na_value=np.nan)==True))|(X['C'].to_numpy(na_value=np.nan)>0.0003641365574362787)"
    }
    ro = _instantiate
    rules = Rules(rule_strings=exp_rule_strings)
    rule_lambdas = rules.as_rule_lambdas(as_numpy=True, with_kwargs=True)
    lambda_kwargs = rules.lambda_kwargs
    opt_rule_strings = ro._optimise_rules(
        rule_lambdas, lambda_kwargs, X, y, None
    )
    assert opt_rule_strings == exp_rule_strings


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
    exp_bounds = {
        'integer': [(0, 9)],
        'float': [(0.0003641365574362787, 0.9999680225821261)],
        'is_na': [(0, 9)],
        'mixed': [(0, 9),
                  (0.0003641365574362787, 0.9999680225821261),
                  (0.0003641365574362787, 0.9999680225821261)],
        'all_na': [(np.nan, np.nan)],
        'zero_var': [(1.0, 1.0)],
        'already_optimal': [(0, 9)]
    }
    ro = _instantiate
    X, _, _ = _create_data
    _, lambda_kwargs = _create_inputs
    with pytest.warns(
            UserWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `bounds` for these rules") and \
        pytest.warns(
            UserWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `bounds` for these rules"):
        bounds = ro._param_base_calc(
            X, lambda_kwargs, 'bounds',
            lambda X_min, X_max: list(zip(X_min, X_max))
        )
    # Check `all_na` separately as np.nan != np.nan
    np.testing.assert_array_equal(bounds['all_na'], exp_bounds['all_na'])
    bounds.pop('all_na')
    exp_bounds.pop('all_na')
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
    with pytest.raises(Exception, match='There are no optimisable rules in the set'):
        with pytest.warns(UserWarning, match='Rules `zero_var` have all zero variance features based on the dataset `X` - unable to optimise these rules'):
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
                UserWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `x0` for these rules") and \
            pytest.warns(
                UserWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `x0` for these rules"):
            x0 = DirectSearchOptimiser.create_x0(X, lambda_kwargs)
    if bounds:
        with pytest.warns(
                UserWarning, match="Rules `categoric`, `boolean` have no optimisable conditions - unable to calculate `bounds` for these rules") and \
            pytest.warns(
                UserWarning, match="Rules `missing_col` use features that are missing from `X` - unable to calculate `bounds` for these rules"):
            bounds = DirectSearchOptimiser.create_bounds(X, lambda_kwargs)
    for method in methods:
        with pytest.warns(UserWarning,
                          match="Rules `missing_col` use features that are missing from `X` - unable to optimise or apply these rules") and \
                pytest.warns(UserWarning,
                             match="Rules `categoric`, `boolean`, `all_na` have no optimisable conditions - unable to optimise these rules") and \
                pytest.warns(UserWarning,
                             match="Rules `zero_var` have all zero variance features based on the dataset `X` - unable to optimise these rules"):
            ro = DirectSearchOptimiser(
                rule_lambdas=rule_lambdas,
                lambda_kwargs=lambda_kwargs,
                metric=metric,
                x0=x0,
                bounds=bounds,
                verbose=1,
                method=method,
            )
            assert ro.__repr__() == 'DirectSearchOptimiser object with 10 rules to optimise'
            if fit_transform:
                X_rules = ro.fit_transform(
                    X=X, y=y, sample_weight=sample_weight)
            else:
                X_rules = ro.fit(X=X, y=y, sample_weight=sample_weight)
            assert ro.__repr__() == 'DirectSearchOptimiser object with 5 rules optimised'
            assert ro.rule_strings == ro.rules.rule_strings == exp_rule_strings[method]
            assert ro.rule_names == list(exp_rule_strings[method].keys())
            assert X_rules.mean().to_dict() == exp_X_rules_mean[method]
            assert ro.orig_rule_performances == exp_orig_rule_performances
            assert ro.opt_rule_performances == exp_opt_rule_performances[method]
            assert ro.rule_names_missing_features == ['missing_col']
            assert ro.rule_names_no_opt_conditions == [
                'categoric', 'boolean', 'all_na']
            assert ro.rule_names_zero_var_features == ['zero_var']
