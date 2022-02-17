import pytest
import numpy as np
import pandas as pd
from iguanas.rule_generation import RuleGeneratorDT
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from iguanas.metrics.classification import FScore
import random


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
def fs_instantiated():
    f = FScore(0.5)
    return f.fit


@pytest.fixture
def rg_instantiated(fs_instantiated):
    f0dot5 = fs_instantiated
    params = {
        'metric': f0dot5,
        'n_total_conditions': 4,
        'tree_ensemble': RandomForestClassifier(n_estimators=10, random_state=0),
        'precision_threshold': 0,
        'num_cores': 2,
        'verbose': 1
    }
    rg = RuleGeneratorDT(**params)
    rg._today = '20200204'
    return [rg, params]


@ pytest.fixture
def fit_decision_tree():
    def _fit(X, y, sample_weight=None):
        dt = DecisionTreeClassifier(random_state=0, max_depth=4)
        dt.fit(X, y, sample_weight=sample_weight)
        return dt
    return _fit


def test_fit_dt(create_data, fit_decision_tree):
    X, y, _, _, _, weights = create_data
    for w in [None, weights]:
        dt = fit_decision_tree(X, y, w)
        dt_test = DecisionTreeClassifier(random_state=0, max_depth=4)
        dt_test.fit(X, y, w)
        dt_preds = dt.predict_proba(X)[:, -1]
        dt_test_preds = dt_test.predict_proba(X)[:, -1]
        assert [a == b for a, b in zip(dt_preds, dt_test_preds)]


def test_repr(rg_instantiated):
    rg, _ = rg_instantiated
    exp_repr = "RuleGeneratorDT(metric=<bound method FScore.fit of FScore with beta=0.5>, n_total_conditions=4, tree_ensemble=RandomForestClassifier(n_estimators=10, random_state=0), precision_threshold=0, num_cores=2, target_feat_corr_types=None)"
    assert rg.__repr__() == exp_repr
    _ = rg.fit(pd.DataFrame({'A': [1, 0, 0]}), pd.Series([1, 0, 0]))
    assert rg.__repr__() == 'RuleGeneratorDT object with 1 rules generated'


def test_fit(create_data, rg_instantiated):
    exp_results = [
        pd.Series(
            data=np.array([
                27, 5, 10, 107, 81, 64, 35, 11, 35, 1, 22, 133, 4, 14, 119, 171, 2,
                1, 27, 14, 5, 33, 5, 10, 7, 42, 310, 7, 90, 17, 18, 8, 32, 33, 1,
                2, 77, 172, 33, 58
            ], dtype='int64'),
            index=[
                'RGDT_Rule_20200204_0', 'RGDT_Rule_20200204_1', 'RGDT_Rule_20200204_2', 'RGDT_Rule_20200204_3', 'RGDT_Rule_20200204_4', 'RGDT_Rule_20200204_5', 'RGDT_Rule_20200204_6', 'RGDT_Rule_20200204_7', 'RGDT_Rule_20200204_8',
                'RGDT_Rule_20200204_9', 'RGDT_Rule_20200204_10', 'RGDT_Rule_20200204_11', 'RGDT_Rule_20200204_12', 'RGDT_Rule_20200204_13', 'RGDT_Rule_20200204_14', 'RGDT_Rule_20200204_15', 'RGDT_Rule_20200204_16', 'RGDT_Rule_20200204_17',
                'RGDT_Rule_20200204_18', 'RGDT_Rule_20200204_19', 'RGDT_Rule_20200204_20', 'RGDT_Rule_20200204_21', 'RGDT_Rule_20200204_22', 'RGDT_Rule_20200204_23', 'RGDT_Rule_20200204_24', 'RGDT_Rule_20200204_25', 'RGDT_Rule_20200204_26',
                'RGDT_Rule_20200204_27', 'RGDT_Rule_20200204_28', 'RGDT_Rule_20200204_29', 'RGDT_Rule_20200204_30', 'RGDT_Rule_20200204_31', 'RGDT_Rule_20200204_32', 'RGDT_Rule_20200204_33', 'RGDT_Rule_20200204_34', 'RGDT_Rule_20200204_35',
                'RGDT_Rule_20200204_36', 'RGDT_Rule_20200204_37', 'RGDT_Rule_20200204_38', 'RGDT_Rule_20200204_39'
            ]),
        pd.Series(
            data=np.array([
                161, 219, 166, 125, 107, 205, 133, 225, 228, 110, 163, 43, 21, 170
            ], dtype='int64'),
            index=[
                'RGDT_Rule_20200204_0', 'RGDT_Rule_20200204_1', 'RGDT_Rule_20200204_2',
                'RGDT_Rule_20200204_3', 'RGDT_Rule_20200204_4', 'RGDT_Rule_20200204_5',
                'RGDT_Rule_20200204_6', 'RGDT_Rule_20200204_7', 'RGDT_Rule_20200204_8',
                'RGDT_Rule_20200204_9', 'RGDT_Rule_20200204_10', 'RGDT_Rule_20200204_11',
                'RGDT_Rule_20200204_12', 'RGDT_Rule_20200204_13'
            ])
    ]
    X, y, _, _, _, weights = create_data
    rg, _ = rg_instantiated
    for i, w in enumerate([None, weights]):
        X_rules = rg.fit(X, y, sample_weight=w)
        pd.testing.assert_series_equal(X_rules.sum(), exp_results[i])
        assert rg.rule_names == exp_results[i].index.tolist() == list(
            rg.rule_lambdas.keys()) == list(rg.lambda_kwargs.keys()) == list(
                rg.rules.rule_strings.keys())


def test_fit_target_feat_corr_types_infer(create_data, rg_instantiated):
    exp_results = [
        pd.Series(
            data=np.array([
                32, 46, 11, 107, 81, 332, 69, 35, 71, 36, 70, 203, 182, 85, 119, 217, 27, 121, 218,
                121, 187, 220, 18, 30, 18, 67, 14, 32, 65, 66, 15, 31, 15, 226, 373, 256, 71, 312, 199, 71
            ], dtype='int64'),
            index=[
                'RGDT_Rule_20200204_0', 'RGDT_Rule_20200204_1', 'RGDT_Rule_20200204_2', 'RGDT_Rule_20200204_3', 'RGDT_Rule_20200204_4', 'RGDT_Rule_20200204_5', 'RGDT_Rule_20200204_6', 'RGDT_Rule_20200204_7', 'RGDT_Rule_20200204_8',
                'RGDT_Rule_20200204_9', 'RGDT_Rule_20200204_10', 'RGDT_Rule_20200204_11', 'RGDT_Rule_20200204_12', 'RGDT_Rule_20200204_13', 'RGDT_Rule_20200204_14', 'RGDT_Rule_20200204_15', 'RGDT_Rule_20200204_16', 'RGDT_Rule_20200204_17',
                'RGDT_Rule_20200204_18', 'RGDT_Rule_20200204_19', 'RGDT_Rule_20200204_20', 'RGDT_Rule_20200204_21', 'RGDT_Rule_20200204_22', 'RGDT_Rule_20200204_23', 'RGDT_Rule_20200204_24', 'RGDT_Rule_20200204_25', 'RGDT_Rule_20200204_26',
                'RGDT_Rule_20200204_27', 'RGDT_Rule_20200204_28', 'RGDT_Rule_20200204_29', 'RGDT_Rule_20200204_30', 'RGDT_Rule_20200204_31', 'RGDT_Rule_20200204_32', 'RGDT_Rule_20200204_33', 'RGDT_Rule_20200204_34', 'RGDT_Rule_20200204_35',
                'RGDT_Rule_20200204_36', 'RGDT_Rule_20200204_37', 'RGDT_Rule_20200204_38', 'RGDT_Rule_20200204_39'
            ]),
        pd.Series(
            data=np.array([
                234, 232, 232, 125, 232, 216, 142, 225, 272, 120, 371, 43, 96, 213
            ], dtype='int64'),
            index=[
                'RGDT_Rule_20200204_0', 'RGDT_Rule_20200204_1', 'RGDT_Rule_20200204_2',
                'RGDT_Rule_20200204_3', 'RGDT_Rule_20200204_4', 'RGDT_Rule_20200204_5',
                'RGDT_Rule_20200204_6', 'RGDT_Rule_20200204_7', 'RGDT_Rule_20200204_8',
                'RGDT_Rule_20200204_9', 'RGDT_Rule_20200204_10', 'RGDT_Rule_20200204_11',
                'RGDT_Rule_20200204_12', 'RGDT_Rule_20200204_13'
            ])
    ]
    X, y, _, _, _, weights = create_data
    rg, _ = rg_instantiated
    rg.target_feat_corr_types = 'Infer'
    for i, w in enumerate([None, weights]):
        X_rules = rg.fit(X, y, sample_weight=w)
        pd.testing.assert_series_equal(X_rules.sum(), exp_results[i])
        assert rg.rule_names == exp_results[i].index.tolist() == list(
            rg.rule_lambdas.keys()) == list(rg.lambda_kwargs.keys()) == list(
                rg.rules.rule_strings.keys())
        assert len(
            [l for l in list(rg.rule_strings.keys()) if "X['email_alpharatio']>" in l]) == 0
        assert len(
            [l for l in list(rg.rule_strings.keys()) if "X['email_kb_distance']>" in l]) == 0
        assert len(
            [l for l in list(rg.rule_strings.keys()) if "X['ip_country_us']==True" in l]) == 0
        assert len([l for l in list(rg.rule_strings.keys())
                   if "X['num_distinct_txn_per_email_1day']<" in l]) == 0
        assert len([l for l in list(rg.rule_strings.keys())
                   if "X['num_distinct_txn_per_email_7day']<" in l]) == 0


def test_fit_target_feat_corr_types_provided(create_data, rg_instantiated):
    exp_results = [
        pd.Series(
            data=np.array([
                32, 46, 11, 107, 81, 332, 69, 35, 71, 36, 70, 203, 182, 85, 119, 217, 27, 121, 218,
                121, 187, 220, 18, 30, 18, 67, 14, 32, 65, 66, 15, 31, 15, 226, 373, 256, 71, 312, 199, 71
            ], dtype='int64'),
            index=[
                'RGDT_Rule_20200204_0', 'RGDT_Rule_20200204_1', 'RGDT_Rule_20200204_2', 'RGDT_Rule_20200204_3', 'RGDT_Rule_20200204_4', 'RGDT_Rule_20200204_5', 'RGDT_Rule_20200204_6', 'RGDT_Rule_20200204_7', 'RGDT_Rule_20200204_8',
                'RGDT_Rule_20200204_9', 'RGDT_Rule_20200204_10', 'RGDT_Rule_20200204_11', 'RGDT_Rule_20200204_12', 'RGDT_Rule_20200204_13', 'RGDT_Rule_20200204_14', 'RGDT_Rule_20200204_15', 'RGDT_Rule_20200204_16', 'RGDT_Rule_20200204_17',
                'RGDT_Rule_20200204_18', 'RGDT_Rule_20200204_19', 'RGDT_Rule_20200204_20', 'RGDT_Rule_20200204_21', 'RGDT_Rule_20200204_22', 'RGDT_Rule_20200204_23', 'RGDT_Rule_20200204_24', 'RGDT_Rule_20200204_25', 'RGDT_Rule_20200204_26',
                'RGDT_Rule_20200204_27', 'RGDT_Rule_20200204_28', 'RGDT_Rule_20200204_29', 'RGDT_Rule_20200204_30', 'RGDT_Rule_20200204_31', 'RGDT_Rule_20200204_32', 'RGDT_Rule_20200204_33', 'RGDT_Rule_20200204_34', 'RGDT_Rule_20200204_35',
                'RGDT_Rule_20200204_36', 'RGDT_Rule_20200204_37', 'RGDT_Rule_20200204_38', 'RGDT_Rule_20200204_39'
            ]),
        pd.Series(
            data=np.array([
                234, 232, 232, 125, 232, 216, 142, 225, 272, 120, 371, 43, 96, 213
            ], dtype='int64'),
            index=[
                'RGDT_Rule_20200204_0', 'RGDT_Rule_20200204_1', 'RGDT_Rule_20200204_2',
                'RGDT_Rule_20200204_3', 'RGDT_Rule_20200204_4', 'RGDT_Rule_20200204_5',
                'RGDT_Rule_20200204_6', 'RGDT_Rule_20200204_7', 'RGDT_Rule_20200204_8',
                'RGDT_Rule_20200204_9', 'RGDT_Rule_20200204_10', 'RGDT_Rule_20200204_11',
                'RGDT_Rule_20200204_12', 'RGDT_Rule_20200204_13'
            ])
    ]
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
    for i, w in enumerate([None, weights]):
        X_rules = rg.fit(X, y, sample_weight=w)
        pd.testing.assert_series_equal(X_rules.sum(), exp_results[i])
        assert rg.rule_names == exp_results[i].index.tolist() == list(
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
    exp_results = [
        pd.Series(
            data=np.array([
                27, 5, 10, 107, 81, 64, 35, 11, 35, 1, 22, 133, 4, 14, 119, 171, 2,
                1, 27, 14, 5, 33, 5, 10, 7, 42, 310, 7, 90, 17, 18, 8, 32, 33, 1,
                2, 77, 172, 33, 58
            ], dtype='int64'),
            index=[
                'RGDT_Rule_20200204_0', 'RGDT_Rule_20200204_1', 'RGDT_Rule_20200204_2', 'RGDT_Rule_20200204_3', 'RGDT_Rule_20200204_4', 'RGDT_Rule_20200204_5', 'RGDT_Rule_20200204_6', 'RGDT_Rule_20200204_7', 'RGDT_Rule_20200204_8',
                'RGDT_Rule_20200204_9', 'RGDT_Rule_20200204_10', 'RGDT_Rule_20200204_11', 'RGDT_Rule_20200204_12', 'RGDT_Rule_20200204_13', 'RGDT_Rule_20200204_14', 'RGDT_Rule_20200204_15', 'RGDT_Rule_20200204_16', 'RGDT_Rule_20200204_17',
                'RGDT_Rule_20200204_18', 'RGDT_Rule_20200204_19', 'RGDT_Rule_20200204_20', 'RGDT_Rule_20200204_21', 'RGDT_Rule_20200204_22', 'RGDT_Rule_20200204_23', 'RGDT_Rule_20200204_24', 'RGDT_Rule_20200204_25', 'RGDT_Rule_20200204_26',
                'RGDT_Rule_20200204_27', 'RGDT_Rule_20200204_28', 'RGDT_Rule_20200204_29', 'RGDT_Rule_20200204_30', 'RGDT_Rule_20200204_31', 'RGDT_Rule_20200204_32', 'RGDT_Rule_20200204_33', 'RGDT_Rule_20200204_34', 'RGDT_Rule_20200204_35',
                'RGDT_Rule_20200204_36', 'RGDT_Rule_20200204_37', 'RGDT_Rule_20200204_38', 'RGDT_Rule_20200204_39'
            ]),
        pd.Series(
            data=np.array([
                161, 219, 166, 125, 107, 205, 133, 225, 228, 110, 163, 43, 21, 170
            ], dtype='int64'),
            index=[
                'RGDT_Rule_20200204_0', 'RGDT_Rule_20200204_1', 'RGDT_Rule_20200204_2',
                'RGDT_Rule_20200204_3', 'RGDT_Rule_20200204_4', 'RGDT_Rule_20200204_5',
                'RGDT_Rule_20200204_6', 'RGDT_Rule_20200204_7', 'RGDT_Rule_20200204_8',
                'RGDT_Rule_20200204_9', 'RGDT_Rule_20200204_10', 'RGDT_Rule_20200204_11',
                'RGDT_Rule_20200204_12', 'RGDT_Rule_20200204_13'
            ])
    ]
    X, y, _, _, _, weights = create_data
    rg, _ = rg_instantiated
    for i, w in enumerate([None, weights]):
        _ = rg.fit(X, y, sample_weight=w)
        X_rules = rg.transform(X)
        pd.testing.assert_series_equal(X_rules.sum(), exp_results[i])


def test_extract_rules_from_ensemble(create_data, rg_instantiated):
    exp_results = [
        pd.Series(
            data=np.array([
                27, 5, 10, 107, 81, 64, 35, 11, 35, 1, 22, 133, 4, 14, 119, 171, 2,
                1, 27, 14, 5, 33, 5, 10, 7, 42, 310, 7, 90, 17, 18, 8, 32, 33, 1,
                2, 77, 172, 33, 58
            ], dtype='int64'),
            index=[
                'RGDT_Rule_20200204_0', 'RGDT_Rule_20200204_1', 'RGDT_Rule_20200204_2', 'RGDT_Rule_20200204_3', 'RGDT_Rule_20200204_4', 'RGDT_Rule_20200204_5', 'RGDT_Rule_20200204_6', 'RGDT_Rule_20200204_7', 'RGDT_Rule_20200204_8',
                'RGDT_Rule_20200204_9', 'RGDT_Rule_20200204_10', 'RGDT_Rule_20200204_11', 'RGDT_Rule_20200204_12', 'RGDT_Rule_20200204_13', 'RGDT_Rule_20200204_14', 'RGDT_Rule_20200204_15', 'RGDT_Rule_20200204_16', 'RGDT_Rule_20200204_17',
                'RGDT_Rule_20200204_18', 'RGDT_Rule_20200204_19', 'RGDT_Rule_20200204_20', 'RGDT_Rule_20200204_21', 'RGDT_Rule_20200204_22', 'RGDT_Rule_20200204_23', 'RGDT_Rule_20200204_24', 'RGDT_Rule_20200204_25', 'RGDT_Rule_20200204_26',
                'RGDT_Rule_20200204_27', 'RGDT_Rule_20200204_28', 'RGDT_Rule_20200204_29', 'RGDT_Rule_20200204_30', 'RGDT_Rule_20200204_31', 'RGDT_Rule_20200204_32', 'RGDT_Rule_20200204_33', 'RGDT_Rule_20200204_34', 'RGDT_Rule_20200204_35',
                'RGDT_Rule_20200204_36', 'RGDT_Rule_20200204_37', 'RGDT_Rule_20200204_38', 'RGDT_Rule_20200204_39'
            ]),
        pd.Series(
            data=np.array([
                161, 219, 166, 125, 107, 205, 133, 225, 228, 110, 163, 43, 21, 170
            ], dtype='int64'),
            index=[
                'RGDT_Rule_20200204_40', 'RGDT_Rule_20200204_41', 'RGDT_Rule_20200204_42', 'RGDT_Rule_20200204_43', 'RGDT_Rule_20200204_44', 'RGDT_Rule_20200204_45', 'RGDT_Rule_20200204_46', 'RGDT_Rule_20200204_47',
                'RGDT_Rule_20200204_48', 'RGDT_Rule_20200204_49', 'RGDT_Rule_20200204_50', 'RGDT_Rule_20200204_51', 'RGDT_Rule_20200204_52', 'RGDT_Rule_20200204_53'
            ])
    ]
    X, y, columns_int, columns_cat, _, weights = create_data
    rg, params = rg_instantiated
    for i, w in enumerate([None, weights]):
        rf = params['tree_ensemble']
        rf.max_depth = 4
        rf.fit(X, y, sample_weight=w)
        X_rules = rg._extract_rules_from_ensemble(
            X, y, rf, params['num_cores'], w, columns_int, columns_cat
        )
        pd.testing.assert_series_equal(X_rules.sum(), exp_results[i])


def test_extract_rules_from_ensemble_error(rg_instantiated):
    X = pd.DataFrame({'A': [0, 0, 0]})
    y = pd.Series([0, 1, 0])
    rg, params = rg_instantiated
    rf = params['tree_ensemble']
    rf.fit(X, y, None)
    with pytest.raises(Exception, match='No rules could be generated. Try changing the class parameters.'):
        rg._extract_rules_from_ensemble(
            X, y, rf, params['num_cores'], None, ['A'], ['A'])


def test_extract_rules_from_dt(create_data, rg_instantiated, fit_decision_tree):

    expected_rule_sets_sample_weight_None = set([
        "(X['email_alpharatio']<=0.43817)&(X['email_kb_distance']>0.00092)&(X['num_distinct_txn_per_email_1day']<=1)",
        "(X['email_alpharatio']<=0.43817)&(X['email_kb_distance']>0.00092)&(X['num_distinct_txn_per_email_1day']>=2)",
        "(X['email_alpharatio']<=0.43844)&(X['email_alpharatio']>0.43817)&(X['email_kb_distance']>0.00092)",
        "(X['email_alpharatio']<=0.52347)&(X['email_kb_distance']<=0.00092)&(X['num_distinct_txn_per_email_1day']<=2)&(X['num_distinct_txn_per_email_1day']>=2)",
        "(X['email_alpharatio']<=0.52347)&(X['email_kb_distance']<=0.00092)&(X['num_distinct_txn_per_email_1day']>=3)",
        "(X['email_alpharatio']>0.43844)&(X['email_kb_distance']<=0.29061)&(X['email_kb_distance']>0.00092)&(X['num_distinct_txn_per_email_7day']>=5)"
    ])
    expected_rule_sets_sample_weight_given = set([
        "(X['email_alpharatio']<=0.57456)&(X['num_distinct_txn_per_email_1day']<=3)&(X['num_distinct_txn_per_email_1day']>=1)&(X['num_distinct_txn_per_email_7day']>=1)"
    ])
    X, y, columns_int, columns_cat, _, weights = create_data
    dt = fit_decision_tree(X, y, None)
    rg, _ = rg_instantiated
    rule_sets = rg._extract_rules_from_dt(
        columns=X.columns, decision_tree=dt, columns_int=columns_int, columns_cat=columns_cat)
    assert rule_sets == expected_rule_sets_sample_weight_None
    dt = fit_decision_tree(X, y, weights)
    rg, _ = rg_instantiated
    rule_sets = rg._extract_rules_from_dt(
        columns=X.columns, decision_tree=dt, columns_int=columns_int, columns_cat=columns_cat)
    assert rule_sets == expected_rule_sets_sample_weight_given
    rg, _ = rg_instantiated
    rg.precision_threshold = 1
    rule_sets = rg._extract_rules_from_dt(
        columns=X.columns, decision_tree=dt, columns_int=columns_int, columns_cat=columns_cat)
    assert rule_sets == set()


def test_train_ensemble(rg_instantiated, create_data):
    X, y, _, _, _, weights = create_data
    rg, params = rg_instantiated
    rg.tree_ensemble.max_depth = params['n_total_conditions']
    for w in [None, weights]:
        rg_trained = rg._train_ensemble(
            X, y, tree_ensemble=rg.tree_ensemble, sample_weight=w, verbose=0)
        rf = RandomForestClassifier(
            max_depth=params['n_total_conditions'], random_state=0, n_estimators=100)
        rf.fit(X=X, y=y, sample_weight=w)
        rf_preds = rf.predict_proba(X)[:, -1]
        rg_preds = rg_trained.predict_proba(X)[:, -1]
        assert [a == b for a, b in zip(rf_preds, rg_preds)]


def test_get_dt_attributes(create_data, rg_instantiated, fit_decision_tree):
    expected_results = (
        np.array([1,  2, -1,  4,  5, -1, -1, -1,  9, 10,
                 11, -1, -1, -1, 15, -1, 17, -1, -1]),
        np.array([8,  3, -1,  7,  6, -1, -1, -1, 14, 13,
                 12, -1, -1, -1, 16, -1, 18, -1, -1]),
        np.array([3,  0, -2,  4,  0, -2, -2, -2,  4,  4,
                 0, -2, -2, -2,  1, -2,  3, -2, -2]),
        np.array([
            0.00091782,  1.5, -2.,  0.52346626,  2.5,
            -2., -2., -2.,  0.43843633,  0.43816555,
            1.5, -2., -2., -2.,  4.5,
            -2.,  0.29061292, -2., -2.]),
        np.array([
            0.02, 0.09090909, 0., 0.21875, 0.38888889,
            0.54545455, 0.14285714, 0., 0.01408451, 0.03614458,
            0.03323263, 0.01020408, 0.06666667, 1., 0.00169205,
            0., 0.00917431, 0.06666667, 0.]),
        0.5833333333333334
    )
    X, y, _, _, _, _ = create_data
    rg, _ = rg_instantiated
    dt = fit_decision_tree(X, y, None)
    left, right, features, thresholds, node_precs, tree_prec = rg._get_dt_attributes(
        dt)
    np.testing.assert_array_almost_equal(left, expected_results[0])
    np.testing.assert_array_almost_equal(right, expected_results[1])
    np.testing.assert_array_almost_equal(features, expected_results[2])
    np.testing.assert_array_almost_equal(thresholds, expected_results[3])
    np.testing.assert_array_almost_equal(node_precs, expected_results[4])
    assert tree_prec == expected_results[5]


def test_errors(create_data, rg_instantiated):
    X, y, _, _, _, _ = create_data
    rg, _ = rg_instantiated
    with pytest.raises(TypeError, match='`X` must be a pandas.core.frame.DataFrame. Current type is list.'):
        rg.fit(X=[], y=y)
    with pytest.raises(TypeError, match='`y` must be a pandas.core.series.Series. Current type is list.'):
        rg.fit(X=X, y=[])
    with pytest.raises(TypeError, match='`sample_weight` must be a pandas.core.series.Series. Current type is list.'):
        rg.fit(X=X, y=y, sample_weight=[])

# Methods below are part of _base module ------------------------------------


def test_extract_rules_from_tree(fit_decision_tree, rg_instantiated, create_data):

    expected_rule_sets_sample_weight_None = set([
        "(X['email_alpharatio']<=0.43817)&(X['email_kb_distance']>0.00092)&(X['num_distinct_txn_per_email_1day']<=1)",
        "(X['email_alpharatio']<=0.43817)&(X['email_kb_distance']>0.00092)&(X['num_distinct_txn_per_email_1day']>=2)",
        "(X['email_alpharatio']<=0.43844)&(X['email_alpharatio']>0.43817)&(X['email_kb_distance']>0.00092)",
        "(X['email_alpharatio']<=0.52347)&(X['email_kb_distance']<=0.00092)&(X['num_distinct_txn_per_email_1day']<=2)&(X['num_distinct_txn_per_email_1day']>=2)",
        "(X['email_alpharatio']<=0.52347)&(X['email_kb_distance']<=0.00092)&(X['num_distinct_txn_per_email_1day']>=3)",
        "(X['email_alpharatio']>0.43844)&(X['email_kb_distance']<=0.29061)&(X['email_kb_distance']>0.00092)&(X['num_distinct_txn_per_email_7day']>=5)"
    ])

    expected_rule_sets_sample_weight_given = set([
        "(X['email_alpharatio']<=0.57456)&(X['num_distinct_txn_per_email_1day']<=3)&(X['num_distinct_txn_per_email_1day']>=1)&(X['num_distinct_txn_per_email_7day']>=1)"
    ])
    X, y, columns_int, columns_cat, _, weights = create_data
    dt = fit_decision_tree(X, y, None)
    rg, _ = rg_instantiated
    left, right, features, thresholds, precisions, _ = rg._get_dt_attributes(
        dt)
    rule_sets = rg._extract_rules_from_tree(
        columns=X.columns.tolist(), precision_threshold=rg.precision_threshold,
        columns_int=columns_int, columns_cat=columns_cat, left=left,
        right=right, features=features, thresholds=thresholds,
        precisions=precisions
    )
    assert rule_sets == expected_rule_sets_sample_weight_None
    dt = fit_decision_tree(X, y, weights)
    rg, _ = rg_instantiated
    left, right, features, thresholds, precisions, _ = rg._get_dt_attributes(
        dt)
    rule_sets = rg._extract_rules_from_tree(
        columns=X.columns.tolist(), precision_threshold=rg.precision_threshold,
        columns_int=columns_int, columns_cat=columns_cat, left=left,
        right=right, features=features, thresholds=thresholds,
        precisions=precisions
    )
    assert rule_sets == expected_rule_sets_sample_weight_given
    rg, _ = rg_instantiated
    rg.precision_threshold = 1
    left, right, features, thresholds, precisions, _ = rg._get_dt_attributes(
        dt)
    rule_sets = rg._extract_rules_from_tree(
        columns=X.columns.tolist(), precision_threshold=rg.precision_threshold,
        columns_int=columns_int, columns_cat=columns_cat, left=left,
        right=right, features=features, thresholds=thresholds,
        precisions=precisions
    )
    assert rule_sets == set()
    # Test no leaf nodes
    left = np.array([0, 0, 0])
    rule_sets = rg._extract_rules_from_tree(
        columns=X.columns.tolist(), precision_threshold=rg.precision_threshold,
        columns_int=columns_int, columns_cat=columns_cat, left=left,
        right=right, features=features, thresholds=thresholds,
        precisions=precisions
    )
    assert rule_sets == set()


def test_calc_target_ratio_wrt_features(rg_instantiated):
    X = pd.DataFrame({
        'A': [10, 9, 8, 7, 6, 5],
        'B': [10, 10, 0, 0, 0, 0],
        'C': [0, 1, 2, 3, 4, 5],
        'D': [0, 0, 10, 10, 10, 10]
    })
    y = pd.Series([1, 1, 1, 0, 0, 0])
    expected_result = {
        'PositiveCorr': ['A', 'B'],
        'NegativeCorr': ['C', 'D'],
    }
    rg, _ = rg_instantiated
    target_feat_corr_types = rg._calc_target_ratio_wrt_features(X, y)
    assert target_feat_corr_types == expected_result
