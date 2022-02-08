import pytest
import numpy as np
import pandas as pd
from iguanas.rule_generation import RuleGeneratorOpt
from sklearn.metrics import precision_score, recall_score
from iguanas.metrics.classification import FScore, Precision
from itertools import product
import random
import math


@pytest.fixture
def create_data():
    def return_random_num(y, fraud_min, fraud_max, nonfraud_min, nonfraud_max, rand_func):
        data = [rand_func(fraud_min, fraud_max) if i == 1 else rand_func(
            nonfraud_min, nonfraud_max) for i in y]
        return data

    random.seed(0)
    np.random.seed(0)
    y = pd.Series(data=[0]*980 + [1]*20, index=list(range(0, 1000)))
    X = pd.DataFrame(data={
        "num_distinct_txn_per_email_1day": [round(max(i, 0)) for i in return_random_num(y, 2, 1, 1, 2, np.random.normal)],
        "num_distinct_txn_per_email_7day": [round(max(i, 0)) for i in return_random_num(y, 4, 2, 2, 3, np.random.normal)],
        "ip_country_us": [round(min(i, 1)) for i in [max(i, 0) for i in return_random_num(y, 0.3, 0.4, 0.5, 0.5, np.random.normal)]],
        "email_kb_distance": [min(i, 1) for i in [max(i, 0) for i in return_random_num(y, 0.2, 0.5, 0.6, 0.4, np.random.normal)]],
        "email_alpharatio":  [min(i, 1) for i in [max(i, 0) for i in return_random_num(y, 0.33, 0.1, 0.5, 0.2, np.random.normal)]],
    },
        index=list(range(0, 1000))
    )
    columns_int = [
        'num_distinct_txn_per_email_1day', 'num_distinct_txn_per_email_7day', 'ip_country_us']
    columns_cat = ['ip_country_us']
    columns_num = ['num_distinct_txn_per_email_1day',
                   'num_distinct_txn_per_email_7day', 'email_kb_distance', 'email_alpharatio']
    weights = y.apply(lambda x: 1000 if x == 1 else 1)
    return [X, y, columns_int, columns_cat, columns_num, weights]


@pytest.fixture
def create_smaller_data():
    random.seed(0)
    np.random.seed(0)
    y = pd.Series(data=[0]*5 + [1]*5, index=list(range(0, 10)))
    X = pd.DataFrame(data={
        'A': [5, 0, 5, 0, 5, 3, 4, 0, 0, 0],
        'B': [0, 1, 0, 1, 0, 1, 0.6, 0.7, 0, 0],
        'C_US': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1]
    },
        index=list(range(0, 10))
    )
    columns_int = ['A']
    columns_cat = ['C']
    columns_num = ['A', 'B']
    weights = y.apply(lambda x: 1000 if x == 1 else 1)
    return [X, y, columns_int, columns_cat, columns_num, weights]


@pytest.fixture
def fs_instantiated():
    f = FScore(0.5)
    return f.fit


@pytest.fixture
def rg_instantiated(fs_instantiated):
    f0dot5 = fs_instantiated
    params = {
        'metric': f0dot5,
        'n_total_conditions': 4,
        'num_rules_keep': 50,
        'n_points': 10,
        'ratio_window': 2,
        'remove_corr_rules': False,
        'verbose': 1
    }
    rg = RuleGeneratorOpt(**params)
    rg._today = '20200204'
    return [rg, params]


@pytest.fixture
def return_dummy_rules():
    def _read(weight_is_none=True):
        if weight_is_none:
            rule_descriptions = pd.DataFrame(
                np.array([["(X['B']>=0.5)", 0.6, 0.6, 1, 0.5, 0.6],
                          ["(X['C_US']==True)", 0.375, 0.6,
                           1, 0.8, 0.4054054054054054],
                          ["(X['A']>=3)", 0.4, 0.4, 1, 0.5, 0.4000000000000001]]),
                columns=['Logic', 'Precision', 'Recall',
                         'nConditions', 'PercDataFlagged', 'Metric'],
                index=['RGO_Rule_20200204_1',
                       'RGO_Rule_20200204_2', 'RGO_Rule_20200204_0'],
            )
            rule_descriptions = rule_descriptions.astype({'Logic': object, 'Precision': float, 'Recall': float,
                                                          'nConditions': int, 'PercDataFlagged': float, 'Metric': float})
            rule_descriptions.index.name = 'Rule'
        else:
            rule_descriptions = pd.DataFrame(
                np.array([["(X['B']>=0.5)", 0.9993337774816788, 0.6, 1, 0.5,
                           0.8819379115710255],
                          ["(X['C_US']==True)", 0.9983361064891847, 0.6, 1, 0.8,
                           0.8813160987074031],
                          ["(X['A']>=3)", 0.9985022466300549, 0.4, 1, 0.5,
                           0.7685213648939442]]),
                columns=['Logic', 'Precision', 'Recall',
                         'nConditions', 'PercDataFlagged', 'Metric'],
                index=['RGO_Rule_20200204_1',
                       'RGO_Rule_20200204_2', 'RGO_Rule_20200204_0'],
            )
            rule_descriptions = rule_descriptions.astype({'Logic': object, 'Precision': float, 'Recall': float,
                                                          'nConditions': int, 'PercDataFlagged': float, 'Metric': float})
            rule_descriptions.index.name = 'Rule'
        X_rules = pd.DataFrame(
            np.array([[0, 1, 1],
                      [1, 1, 0],
                      [0, 1, 1],
                      [1, 1, 0],
                      [0, 1, 1],
                      [1, 1, 1],
                      [1, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 1, 0]], dtype=np.int),
            columns=['RGO_Rule_20200204_1',
                     'RGO_Rule_20200204_2', 'RGO_Rule_20200204_0'],
        )
        rule_combinations = [(('RGO_Rule_20200204_1', 'RGO_Rule_20200204_2'), ("(X['B']>=0.5)", "(X['C_US']==True)")),
                             (('RGO_Rule_20200204_1', 'RGO_Rule_20200204_0'),
                              ("(X['B']>=0.5)", "(X['A']>=3)")),
                             (('RGO_Rule_20200204_2', 'RGO_Rule_20200204_0'), ("(X['C_US']==True)", "(X['A']>=3)"))]
        return rule_descriptions, X_rules, rule_combinations
    return _read


@pytest.fixture
def return_dummy_pairwise_rules():
    rule_descriptions = pd.DataFrame(
        {
            'Rule': ['A', 'B', 'C'],
            'Precision': [1, 0.5, 0]
        }
    )
    rule_descriptions.set_index('Rule', inplace=True)
    pairwise_descriptions = pd.DataFrame(
        {
            'Rule': ['A&B', 'B&C', 'A&C'],
            'Precision': [1, 0.75, 0]
        }
    )
    pairwise_descriptions.set_index('Rule', inplace=True)
    X_rules_pairwise = pd.DataFrame({
        'A&B': range(0, 1000),
        'B&C': range(0, 1000),
        'A&C': range(0, 1000),
    })
    pairwise_to_orig_lookup = {
        'A&B': ['A', 'B'],
        'A&C': ['A', 'C'],
        'B&C': ['B', 'C'],
    }
    return pairwise_descriptions, X_rules_pairwise, pairwise_to_orig_lookup, rule_descriptions


@pytest.fixture
def return_iteration_results():
    iteration_ranges = {
        ('num_distinct_txn_per_email_1day', '>='): (0, 7),
        ('num_distinct_txn_per_email_1day', '<='): (0, 7),
        ('num_distinct_txn_per_email_7day', '>='): (6.0, 12),
        ('num_distinct_txn_per_email_7day', '<='): (0, 6.0),
        ('email_kb_distance', '>='): (0.5, 1.0),
        ('email_kb_distance', '<='): (0.0, 0.5),
        ('email_alpharatio', '>='): (0.5, 1.0),
        ('email_alpharatio', '<='): (0.0, 0.5)
    }
    iteration_arrays = {('num_distinct_txn_per_email_1day', '>='): np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                        ('num_distinct_txn_per_email_1day', '<='): np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                        ('num_distinct_txn_per_email_7day',
                         '>='): np.array([6,  7,  8,  9, 10, 11, 12]),
                        ('num_distinct_txn_per_email_7day', '<='): np.array([0, 1, 2, 3, 4, 5, 6]),
                        ('email_kb_distance',
                         '>='): np.array([0.5, 0.56, 0.61, 0.67, 0.72, 0.78, 0.83, 0.89, 0.94, 1.]),
                        ('email_kb_distance',
                         '<='): np.array([0., 0.056, 0.11, 0.17, 0.22, 0.28, 0.33, 0.39, 0.44,
                                          0.5]),
                        ('email_alpharatio',
                         '>='): np.array([0.5, 0.56, 0.61, 0.67, 0.72, 0.78, 0.83, 0.89, 0.94, 1.]),
                        ('email_alpharatio',
                         '<='): np.array([0., 0.056, 0.11, 0.17, 0.22, 0.28, 0.33, 0.39, 0.44,
                                          0.5])}
    iteration_ranges_3pts = {
        ('num_distinct_txn_per_email_1day', '>='): np.array([0., 4., 7.]),
        ('num_distinct_txn_per_email_1day', '<='):  np.array([0., 4., 7.]),
        ('num_distinct_txn_per_email_7day', '>='):  np.array([6.,  9., 12.]),
        ('num_distinct_txn_per_email_7day', '<='):  np.array([0., 3., 6.]),
        ('email_kb_distance', '>='):  np.array([0.5, 0.75, 1.]),
        ('email_kb_distance', '<='):  np.array([0., 0.25, 0.5]),
        ('email_alpharatio', '>='):  np.array([0.5, 0.75, 1.]),
        ('email_alpharatio', '<='):  np.array([0., 0.25, 0.5])
    }
    fscore_arrays = {('num_distinct_txn_per_email_1day',
                      '>='): np.array([0.02487562, 0.04244482, 0.05393401, 0.01704545, 0.,
                                       0., 0., 0.]),
                     ('num_distinct_txn_per_email_1day',
                      '<='): np.array([0., 0.00608766, 0.02689873, 0.0275634, 0.02590674,
                                       0.02520161, 0.0249004, 0.02487562]),
                     ('num_distinct_txn_per_email_7day',
                      '>='): np.array([0.0304878, 0.04934211, 0., 0., 0.,
                                       0., 0.]),
                     ('num_distinct_txn_per_email_7day',
                      '<='): np.array([0., 0.00903614, 0.01322751, 0.01623377, 0.0248139,
                                       0.02395716, 0.02275161]),
                     ('email_kb_distance',
                      '>='): np.array([0.01290878, 0.01420455, 0.01588983, 0.01509662, 0.0136612,
                                       0.01602564, 0.01798561, 0.0210084, 0.0245098, 0.0154321]),
                     ('email_kb_distance',
                      '<='): np.array([0.10670732, 0.08333333, 0.06835938, 0.06410256, 0.06048387,
                                       0.05901288, 0.05474453, 0.04573171, 0.04298942, 0.04079254]),
                     ('email_alpharatio',
                      '>='): np.array([0.00498008, 0.00327225, 0., 0., 0.,
                                       0., 0., 0., 0., 0.]),
                     ('email_alpharatio',
                      '<='): np.array([0., 0., 0., 0.02232143, 0.04310345,
                                       0.06161972, 0.06157635, 0.05662021, 0.05712366, 0.04429134])}
    return iteration_ranges, iteration_arrays, iteration_ranges_3pts, fscore_arrays


@pytest.fixture
def return_pairwise_info_dict():
    pairwise_info_dict = {"(X['B']>=0.5)&(X['C_US']==True)": {'RuleName1': 'RGO_Rule_20200204_1',
                                                              'RuleName2': 'RGO_Rule_20200204_2',
                                                              'PairwiseRuleName': 'RGO_Rule_20200204_0',
                                                              'PairwiseComponents': ['RGO_Rule_20200204_1', 'RGO_Rule_20200204_2']},
                          "(X['A']>=3)&(X['B']>=0.5)": {'RuleName1': 'RGO_Rule_20200204_1',
                                                        'RuleName2': 'RGO_Rule_20200204_0',
                                                        'PairwiseRuleName': 'RGO_Rule_20200204_1',
                                                        'PairwiseComponents': ['RGO_Rule_20200204_1', 'RGO_Rule_20200204_0']},
                          "(X['A']>=3)&(X['C_US']==True)": {'RuleName1': 'RGO_Rule_20200204_2',
                                                            'RuleName2': 'RGO_Rule_20200204_0',
                                                            'PairwiseRuleName': 'RGO_Rule_20200204_2',
                                                            'PairwiseComponents': ['RGO_Rule_20200204_2', 'RGO_Rule_20200204_0']}}
    return pairwise_info_dict


def test_repr(rg_instantiated):
    rg, _ = rg_instantiated
    exp_repr = "RuleGeneratorOpt(metric=<bound method FScore.fit of FScore with beta=0.5>, n_total_conditions=4, num_rules_keep=50, n_points=10, ratio_window=2, one_cond_rule_opt_metric=<bound method FScore.fit of FScore with beta=1>, remove_corr_rules=False, target_feat_corr_types=None)"
    assert rg.__repr__() == exp_repr
    _ = rg.fit(pd.DataFrame({'A': [1, 0, 0]}), pd.Series([1, 0, 0]))
    assert rg.__repr__() == 'RuleGeneratorOpt object with 1 rules generated'


def test_fit(create_data, rg_instantiated):
    X, y, _, _, _, weights = create_data
    rg, _ = rg_instantiated
    exp_results = [
        ((1000, 86), 8474),
        ((1000, 59), 11281)
    ]
    for i, w in enumerate([None, weights]):
        X_rules = rg.fit(X, y, sample_weight=w)
        assert X_rules.shape == exp_results[i][0]
        assert X_rules.sum().sum() == exp_results[i][1]
        assert rg.rule_names == X_rules.columns.tolist() == list(
            rg.rule_lambdas.keys()) == list(rg.lambda_kwargs.keys()) == list(
                rg.rules.rule_strings.keys())


def test_fit_target_feat_corr_types_infer(create_data, rg_instantiated, fs_instantiated):
    X, y, _, _, _, weights = create_data
    rg, _ = rg_instantiated
    rg.target_feat_corr_types = 'Infer'
    exp_results = [
        ((1000, 30), 1993),
        ((1000, 30), 4602)
    ]
    for i, w in enumerate([None, weights]):
        X_rules = rg.fit(X, y, sample_weight=w)
        assert X_rules.shape == exp_results[i][0]
        assert X_rules.sum().sum() == exp_results[i][1]
        assert rg.rule_names == X_rules.columns.tolist() == list(
            rg.rule_lambdas.keys()) == list(rg.lambda_kwargs.keys()) == list(
                rg.rules.rule_strings.keys())
        assert len(
            [l for l in list(rg.rule_strings.values()) if "X['email_alpharatio']>" in l]) == 0
        assert len(
            [l for l in list(rg.rule_strings.values()) if "X['email_kb_distance']>" in l]) == 0
        assert len(
            [l for l in list(rg.rule_strings.values()) if "X['ip_country_us']==True" in l]) == 0
        assert len([l for l in list(rg.rule_strings.values())
                   if "X['num_distinct_txn_per_email_1day']<" in l]) == 0
        assert len([l for l in list(rg.rule_strings.values())
                   if "X['num_distinct_txn_per_email_7day']<" in l]) == 0


def test_fit_target_feat_corr_types_provided(create_data, rg_instantiated, fs_instantiated):
    X, y, _, _, _, weights = create_data
    rg, _ = rg_instantiated
    rg.target_feat_corr_types = {
        'PositiveCorr': [
            'num_distinct_txn_per_email_1day',
            'num_distinct_txn_per_email_7day'
        ],
        'NegativeCorr': [
            'ip_country_us', 'email_kb_distance', 'email_alpharatio']
    }
    exp_results = [
        ((1000, 30), 1993),
        ((1000, 30), 4602)
    ]
    for i, w in enumerate([None, weights]):
        X_rules = rg.fit(X, y, sample_weight=w)
        assert X_rules.shape == exp_results[i][0]
        assert X_rules.sum().sum() == exp_results[i][1]
        assert rg.rule_names == X_rules.columns.tolist() == list(
            rg.rule_lambdas.keys()) == list(rg.lambda_kwargs.keys()) == list(
                rg.rules.rule_strings.keys())
        assert len(
            [l for l in list(rg.rule_strings.values()) if "X['email_alpharatio']>" in l]) == 0
        assert len(
            [l for l in list(rg.rule_strings.values()) if "X['email_kb_distance']>" in l]) == 0
        assert len(
            [l for l in list(rg.rule_strings.values()) if "X['ip_country_us']==True" in l]) == 0
        assert len([l for l in list(rg.rule_strings.values())
                   if "X['num_distinct_txn_per_email_1day']<" in l]) == 0
        assert len([l for l in list(rg.rule_strings.values())
                   if "X['num_distinct_txn_per_email_7day']<" in l]) == 0


def test_transform(create_data, rg_instantiated):
    exp_result = [
        (1000, 86),
        (1000, 59)
    ]
    X, y, _, _, _, weights = create_data
    rg, _ = rg_instantiated
    for i, w in enumerate([None, weights]):
        _ = rg.fit(X, y, w)
        X_rules = rg.transform(X)
        assert X_rules.shape == exp_result[i]


def test_generate_numeric_one_condition_rules(create_data, rg_instantiated, fs_instantiated):
    exp_rule_strings = [
        {
            'RGO_Rule_20200204_0': "(X['num_distinct_txn_per_email_1day']>=2)", 'RGO_Rule_20200204_1': "(X['num_distinct_txn_per_email_7day']>=7)", 'RGO_Rule_20200204_2': "(X['email_kb_distance']>=0.94)", 'RGO_Rule_20200204_3': "(X['email_alpharatio']>=0.5)",
            'RGO_Rule_20200204_4': "(X['num_distinct_txn_per_email_1day']<=3)", 'RGO_Rule_20200204_5': "(X['num_distinct_txn_per_email_7day']<=4)", 'RGO_Rule_20200204_6': "(X['email_kb_distance']<=0.0)", 'RGO_Rule_20200204_7': "(X['email_alpharatio']<=0.33)"
        },
        {
            'RGO_Rule_20200204_8': "(X['num_distinct_txn_per_email_1day']>=1)", 'RGO_Rule_20200204_9': "(X['num_distinct_txn_per_email_7day']>=7)", 'RGO_Rule_20200204_10': "(X['email_kb_distance']>=0.61)", 'RGO_Rule_20200204_11': "(X['email_alpharatio']>=0.5)",
            'RGO_Rule_20200204_12': "(X['num_distinct_txn_per_email_1day']<=3)", 'RGO_Rule_20200204_13': "(X['num_distinct_txn_per_email_7day']<=5)", 'RGO_Rule_20200204_14': "(X['email_kb_distance']<=0.5)", 'RGO_Rule_20200204_15': "(X['email_alpharatio']<=0.5)"
        }
    ]
    X, y, columns_int, _, columns_num, weights = create_data
    rg, _ = rg_instantiated
    metric = fs_instantiated
    for i, w in enumerate([None, weights]):
        rule_strings, X_rules = rg._generate_numeric_one_condition_rules(
            X, y, columns_num, columns_int, w
        )
        assert X_rules.shape == (1000, 8)
        assert rule_strings == exp_rule_strings[i]


def test_generate_numeric_one_condition_rules_warning(rg_instantiated):
    X = pd.DataFrame({'A': [0, 0, 0]})
    y = pd.Series([0, 1, 0])
    rg, _ = rg_instantiated
    with pytest.warns(UserWarning, match='No numeric one condition rules could be created.'):
        results = rg._generate_numeric_one_condition_rules(
            X, y, ['A'], ['A'], None)
        pd.testing.assert_frame_equal(results[0], pd.DataFrame())
        pd.testing.assert_frame_equal(results[1], pd.DataFrame())


def test_generate_categorical_one_condition_rules(create_data, rg_instantiated):
    exp_rule_strings = [
        {'RGO_Rule_20200204_1': "(X['ip_country_us']==False)"},
        {'RGO_Rule_20200204_3': "(X['ip_country_us']==False)"}
    ]
    X, y, _, columns_cat, _, weights = create_data
    rg, _ = rg_instantiated
    for i, w in enumerate([None, weights]):
        rule_strings, X_rules = rg._generate_categorical_one_condition_rules(
            X, y, columns_cat, w
        )
        assert X_rules.shape == (1000, 1)
        assert rule_strings == exp_rule_strings[i]


def test_generate_categorical_one_condition_rules_warning(rg_instantiated):
    X = pd.DataFrame({'A': [0, 0, 0]})
    y = pd.Series([0, 1, 0])
    rg, _ = rg_instantiated
    with pytest.warns(UserWarning, match='No categorical one condition rules could be created.'):
        results = rg._generate_categorical_one_condition_rules(
            X, y, ['A'], None)
        assert results[0] == {}
        assert results[1].empty


def test_generate_pairwise_rules(return_dummy_rules, create_smaller_data, rg_instantiated,
                                 fs_instantiated):

    def _assert_generate_pairwise_rules(rule_descriptions_shape, X_rules_shape):
        rule_descriptions, X_rules, rule_combinations = return_dummy_rules()
        rule_descriptions, X_rules, _ = rg._generate_pairwise_rules(
            X_rules, y, rule_combinations, w)
        _assert_rule_descriptions_and_X_rules(
            rule_descriptions, X_rules, rule_descriptions_shape, X_rules_shape, X, y, metric, w)

    X, y, _, _, _, weights = create_smaller_data
    rg, _ = rg_instantiated
    metric = fs_instantiated
    for w in [None, weights]:
        _assert_generate_pairwise_rules(
            (3, 6), (10, 3)) if w is None else _assert_generate_pairwise_rules(
                (3, 6), (10, 3))


def test_drop_unnecessary_pairwise_rules(create_data, rg_instantiated, return_dummy_pairwise_rules):
    _, y, _, _, _, _ = create_data
    rg, _ = rg_instantiated
    pairwise_descriptions, X_rules_pairwise, pairwise_to_orig_lookup, rule_descriptions = return_dummy_pairwise_rules
    pairwise_descriptions, X_rules_pairwise = rg._drop_unnecessary_pairwise_rules(
        pairwise_descriptions, X_rules_pairwise, y, pairwise_to_orig_lookup, rule_descriptions
    )
    assert pairwise_descriptions.shape == (1, 1)
    assert X_rules_pairwise.shape == (1000, 1)


def test_generate_n_order_pairwise_rules(return_dummy_rules, create_smaller_data, rg_instantiated,
                                         fs_instantiated):

    exp_rule_strings = [
        {
            'RGO_Rule_20200204_4': "(X['A']>=3)&(X['B']>=0.5)", 'RGO_Rule_20200204_1': "(X['B']>=0.5)", 'RGO_Rule_20200204_2': "(X['C_US']==True)", 'RGO_Rule_20200204_0': "(X['A']>=3)"
        },
        {
            'RGO_Rule_20200204_1': "(X['B']>=0.5)", 'RGO_Rule_20200204_2': "(X['C_US']==True)", 'RGO_Rule_20200204_7': "(X['A']>=3)&(X['B']>=0.5)", 'RGO_Rule_20200204_0': "(X['A']>=3)"
        },
        {
            'RGO_Rule_20200204_10': "(X['A']>=3)&(X['B']>=0.5)", 'RGO_Rule_20200204_1': "(X['B']>=0.5)", 'RGO_Rule_20200204_2': "(X['C_US']==True)", 'RGO_Rule_20200204_0': "(X['A']>=3)"
        },
        {
            'RGO_Rule_20200204_1': "(X['B']>=0.5)", 'RGO_Rule_20200204_2': "(X['C_US']==True)", 'RGO_Rule_20200204_13': "(X['A']>=3)&(X['B']>=0.5)", 'RGO_Rule_20200204_0': "(X['A']>=3)"
        },
    ]
    X, y, _, _, _, weights = create_smaller_data
    rg, _ = rg_instantiated
    rg._rule_name_counter = 3
    for i, (rem_corr_rules, w) in enumerate(list(product([True, False], [None, weights]))):
        rule_descriptions, X_rules, _ = return_dummy_rules(w is None)
        rule_strings = rule_descriptions['Logic'].to_dict()
        rule_strings, X_rules = rg._generate_n_order_pairwise_rules(
            X_rules, y, rule_strings, rem_corr_rules, w
        )
        assert X_rules.shape == (10, 4)
        assert rule_strings == exp_rule_strings[i]


def test_generate_rule_descriptions(rg_instantiated):
    exp_rule_descriptions = pd.DataFrame({
        'Logic': ["(X['A']>0)"],
        'nConditions': [1],
        'Precision': [1.0],
        'Metric': [1.0]
    },
        index=['Rule1']
    )
    p = Precision()
    X_rules = pd.DataFrame({'A': [0, 1, 0, 1]})
    y = pd.Series([0, 1, 0, 1])
    rule_strings = {'Rule1': "(X['A']>0)"}
    rg, _ = rg_instantiated
    rule_descriptions = rg._generate_rule_descriptions(
        X_rules=X_rules, y=y, sample_weight=None, rule_strings=rule_strings,
        metric=p.fit
    )
    pd.testing.assert_frame_equal(rule_descriptions, exp_rule_descriptions)


def test_set_iteration_range(create_data, rg_instantiated, return_iteration_results):
    iteration_ranges, _, _, _ = return_iteration_results
    X, _, columns_int, _, columns_num, _ = create_data
    rg, _ = rg_instantiated
    for col, op in product(columns_num, ['>=', '<=']):
        assert iteration_ranges[(col, op)] == rg._set_iteration_range(
            X[col].values, col, op, 10, 2, columns_int)


def test_set_iteration_array(create_data, rg_instantiated, return_iteration_results):
    iteration_ranges, iteration_arrays, iteration_arrays_3pts, _ = return_iteration_results
    _, _, columns_int, _, columns_num, _ = create_data
    rg, _ = rg_instantiated
    for col, op in product(columns_num, ['>=', '<=']):
        iter_min = iteration_ranges[(col, op)][0]
        iter_max = iteration_ranges[(col, op)][1]
        expected_array = iteration_arrays[(col, op)]
        actual_array = rg._set_iteration_array(
            col, columns_int, iter_min, iter_max, 10)
        np.testing.assert_array_equal(expected_array, actual_array)
    for col, op in product(columns_num, ['>=', '<=']):
        iter_min = iteration_ranges[(col, op)][0]
        iter_max = iteration_ranges[(col, op)][1]
        expected_array = iteration_arrays_3pts[(col, op)]
        actual_array = rg._set_iteration_array(
            col, columns_int, iter_min, iter_max, 3)
        np.testing.assert_array_equal(expected_array, actual_array)


def test_calculate_opt_metric_across_range(create_data, rg_instantiated, return_iteration_results,
                                           fs_instantiated):
    _, iteration_arrays, _, fscore_arrays = return_iteration_results
    X, y, _, _, _, _ = create_data
    rg, _ = rg_instantiated
    metric = fs_instantiated
    for (col, op), x_array in iteration_arrays.items():
        expected_array = fscore_arrays[(col, op)]
        actual_array = rg._calculate_opt_metric_across_range(
            x_array, op, X[col], y, metric, None)
        calculated_array = []
        for x in x_array:
            fscore = metric(y_true=y, y_preds=eval(
                f'X["{col}"]{op}{x}'), sample_weight=None)
            calculated_array.append(fscore)
        calculated_array = np.array(calculated_array)
        np.testing.assert_array_almost_equal(expected_array, actual_array)
        np.testing.assert_array_equal(calculated_array, actual_array)


def test_return_x_of_max_opt_metric(rg_instantiated):
    rg, _ = rg_instantiated
    inputs = [
        (np.array([0, 0, 0, 0]), '>=', np.array([0, 1, 10, 10])),
        (np.array([0, 0.5, 0.5, 0.2]), '>=', np.array([0, 1, 2, 3])),
        (np.array([0, 0.5, 0.5, 0.2]), '<=', np.array([0, 1, 2, 3]))
    ]
    exp_results = [None, 2, 1]
    for i, (opt_metric_iter, operator, x_iter) in enumerate(inputs):
        x = rg._return_x_of_max_opt_metric(opt_metric_iter, operator, x_iter)
        assert x == exp_results[i]


def test_get_rule_combinations_for_loop(return_dummy_rules, rg_instantiated):
    rule_descriptions, _, expected_rule_comb = return_dummy_rules()
    rg, _ = rg_instantiated
    actual_rule_comb = rg._get_rule_combinations_for_loop(
        rule_descriptions, 1, 50)
    np.testing.assert_array_equal(expected_rule_comb, actual_rule_comb)
    num_rules = rule_descriptions.shape[0]
    assert len(actual_rule_comb) == math.factorial(num_rules) / \
        (math.factorial(2) * math.factorial(num_rules - 2))


def test_return_pairwise_information(return_pairwise_info_dict, return_dummy_rules, rg_instantiated):
    expected_pw_info_dict = return_pairwise_info_dict
    _, _, rule_comb = return_dummy_rules()
    rg, _ = rg_instantiated
    actual_pw_info_dict = rg._return_pairwise_information(rule_comb)
    assert actual_pw_info_dict == expected_pw_info_dict


def test_generate_pairwise_df(return_dummy_rules, return_pairwise_info_dict, rg_instantiated):
    _, X_rules, _ = return_dummy_rules()
    expected_pw_info_dict = return_pairwise_info_dict
    rules_names_1, rules_names_2, pairwise_names = [], [], []
    for info_dict in expected_pw_info_dict.values():
        rules_names_1.append(info_dict['RuleName1'])
        rules_names_2.append(info_dict['RuleName2'])
        pairwise_names.append(info_dict['PairwiseRuleName'])
    rg, _ = rg_instantiated
    X_rules_pw = rg._generate_pairwise_df(
        X_rules, rules_names_1, rules_names_2, pairwise_names)
    assert X_rules_pw.shape == (10, 3)
    assert X_rules_pw.columns.tolist() == [
        'RGO_Rule_20200204_0', 'RGO_Rule_20200204_1', 'RGO_Rule_20200204_2'
    ]
    for _, info_dict in expected_pw_info_dict.items():
        X_rule_pw = X_rules_pw[info_dict['PairwiseRuleName']]
        X_rule_pw_calc = X_rules[info_dict['RuleName1']
                                 ] * X_rules[info_dict['RuleName2']]
        np.testing.assert_array_equal(X_rule_pw, X_rule_pw_calc)


def test_return_pairwise_rules_to_drop(rg_instantiated, return_dummy_pairwise_rules):
    rg, _ = rg_instantiated
    pairwise_descriptions, _, pairwise_to_orig_lookup, rule_descriptions = return_dummy_pairwise_rules
    assert rg._return_pairwise_rules_to_drop(
        pairwise_descriptions, pairwise_to_orig_lookup, rule_descriptions) == ['A&B', 'A&C']


def test_sort_rule_dfs_by_opt_metric(rg_instantiated):
    rule_descriptions = pd.DataFrame({
        'Precision': [1.0, 0.5, 0.75, 0.25, 0],
        'Recall': [0.5, 0.7, 1, 0.75, 0],
        'nConditions': [1, 2, 3, 4, 5],
        'PercDataFlagged': [0.2, 0.4, 0.6, 0.45, 0.1],
        'Metric': [0.75, 0.6, 0.85, 0.5, 0]
    },
        index=['A', 'B', 'C', 'D', 'E']
    )
    X_rules = pd.DataFrame({
        'A': [1, 0, 1, 1, 0],
        'B': [1, 0, 0, 1, 1],
        'C': [1, 0, 0, 0, 1],
        'D': [1, 1, 1, 0, 1],
        'E': [1, 0, 0, 0, 1]
    })
    rg, _ = rg_instantiated
    rd, xr = rg._sort_rule_dfs_by_opt_metric(rule_descriptions, X_rules)
    assert rd.index.tolist() == ['C', 'A', 'B', 'D', 'E']
    assert xr.columns.tolist() == ['C', 'A', 'B', 'D', 'E']


def test_errors(create_data, rg_instantiated):
    X, y, _, _, _, _ = create_data
    rg, _ = rg_instantiated
    with pytest.raises(TypeError, match='`X` must be a pandas.core.frame.DataFrame. Current type is list.'):
        rg.fit(X=[], y=y)
    with pytest.raises(TypeError, match='`y` must be a pandas.core.series.Series. Current type is list.'):
        rg.fit(X=X, y=[])
    with pytest.raises(TypeError, match='`sample_weight` must be a pandas.core.series.Series. Current type is list.'):
        rg.fit(X=X, y=y, sample_weight=[])


def _calc_rule_metrics(rule, X, y, metric, sample_weight):
    X_rule = eval(rule).astype(int)
    prec = precision_score(
        y, X_rule, sample_weight=sample_weight, zero_division=0)
    rec = recall_score(y, X_rule, sample_weight=sample_weight,
                       zero_division=0)
    opt_value = metric(X_rule, y, sample_weight=sample_weight)
    perc_data_flagged = X_rule.mean()
    return [prec, rec, perc_data_flagged, opt_value, X_rule]


def _assert_rule_descriptions(rule_descriptions, X, y, metric, sample_weight):
    for _, row in rule_descriptions.iterrows():
        class_results = row.loc[['Precision', 'Recall',
                                 'PercDataFlagged', 'Metric']].values.astype(float)
        rule = row['Logic']
        test_results = _calc_rule_metrics(rule, X, y, metric, sample_weight)
        for i in range(0, len(class_results)):
            assert round(class_results[i], 6) == round(test_results[i], 6)


def _assert_X_rules(X_rules, rule_list, X, y, metric, sample_weight):
    for rule, rule_name in zip(rule_list, X_rules):
        class_result = X_rules[rule_name]
        test_result = _calc_rule_metrics(
            rule, X, y, metric, sample_weight)[-1]
        np.testing.assert_array_equal(class_result, test_result)


def _assert_rule_descriptions_and_X_rules(rule_descriptions, X_rules, rule_descriptions_shape,
                                          X_rules_shape, X, y, metric, sample_weight):
    assert rule_descriptions.shape == rule_descriptions_shape
    assert X_rules.shape == X_rules_shape
    _assert_rule_descriptions(rule_descriptions, X, y, metric, sample_weight)
    _assert_X_rules(
        X_rules, rule_descriptions['Logic'].values, X, y, metric, sample_weight)
