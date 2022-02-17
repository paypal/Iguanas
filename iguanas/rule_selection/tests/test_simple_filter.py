import pytest
from iguanas.rule_selection import SimpleFilter
from iguanas.metrics import FScore, AlertsPerDay
import numpy as np
import pandas as pd
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
    X_rules = pd.DataFrame(data={
        "Rule1": [0]*980 + [1]*6 + [0] * 14,
        "Rule2": [0]*987 + [1]*6 + [0] * 7,
        "Rule3": [0]*993 + [1]*6 + [0] * 1,
        "Rule4": [round(max(i, 0)) for i in return_random_num(y, 0.4, 1, 0.5, 0.6, np.random.uniform)],
        "Rule5": [round(max(i, 0)) for i in return_random_num(y, 0.2, 1, 0, 0.6, np.random.uniform)],
    },
        index=list(range(0, 1000))
    )
    weights = y.apply(lambda x: 10 if x == 1 else 1)
    return X_rules, y, weights


@pytest.fixture
def instantiate_SimpleFilter():
    f4 = FScore(beta=4)
    fr = SimpleFilter(threshold=0.4, operator='>=', metric=f4.fit)
    return fr


@pytest.fixture
def instantiate_SimpleFilter_unlabelled():
    apd = AlertsPerDay(n_alerts_expected_per_day=10,
                       no_of_days_in_file=10)
    fr = SimpleFilter(
        threshold=-100, operator='>=', metric=apd.fit
    )
    return fr


@pytest.fixture
def expected_results_SimpleFilter(create_data):
    X_rules, _, _ = create_data
    expected_results = [
        X_rules[['Rule5']],
        X_rules[['Rule4', 'Rule5']]
    ]
    return expected_results


def test_fit(create_data, instantiate_SimpleFilter, expected_results_SimpleFilter):
    X_rules, y, weights = create_data
    expected_results_SimpleFilter = expected_results_SimpleFilter
    fr = instantiate_SimpleFilter
    # Without weight
    fr.fit(X_rules=X_rules, y=y)
    assert all(fr.rules_to_keep == expected_results_SimpleFilter[0].columns)
    # With weight
    fr.fit(X_rules=X_rules, y=y, sample_weight=weights)
    assert all(fr.rules_to_keep == expected_results_SimpleFilter[1].columns)


def test_fit_unlabelled(create_data, instantiate_SimpleFilter_unlabelled):
    X_rules, _, _ = create_data
    fr = instantiate_SimpleFilter_unlabelled
    fr.fit(X_rules=X_rules)
    assert fr.rules_to_keep == ['Rule1', 'Rule2', 'Rule3', 'Rule5']


def test_transform(create_data, instantiate_SimpleFilter):
    X_rules, _, _ = create_data
    fr = instantiate_SimpleFilter
    fr.rules_to_keep = ['Rule1']
    X_rules_filtered = fr.transform(X_rules)
    pd.testing.assert_frame_equal(X_rules_filtered, X_rules[['Rule1']])


def test_fit_transform(create_data, instantiate_SimpleFilter, expected_results_SimpleFilter):
    X_rules, y, weights = create_data
    expected_results_SimpleFilter = expected_results_SimpleFilter
    fr = instantiate_SimpleFilter
    # Without weight
    X_rules_filtered = fr.fit_transform(
        X_rules=X_rules, y=y)
    assert all(fr.rules_to_keep == expected_results_SimpleFilter[0].columns)
    assert all(X_rules_filtered == expected_results_SimpleFilter[0])
    # With weight
    X_rules_filtered = fr.fit_transform(
        X_rules=X_rules, y=y, sample_weight=weights)
    assert all(fr.rules_to_keep == expected_results_SimpleFilter[1].columns)
    assert all(X_rules_filtered == expected_results_SimpleFilter[1])


def test_fit_transform_unlabelled(create_data,
                                  instantiate_SimpleFilter_unlabelled):
    X_rules, _, _ = create_data
    fr = instantiate_SimpleFilter_unlabelled
    expected_rules_to_keep = ['Rule1', 'Rule2', 'Rule3', 'Rule5']
    X_rules_filtered = fr.fit_transform(X_rules=X_rules)
    assert fr.rules_to_keep == expected_rules_to_keep
    pd.testing.assert_frame_equal(
        X_rules_filtered, X_rules[expected_rules_to_keep])


def test_errors():
    with pytest.raises(ValueError, match="`operator` must be '>', '>=', '<' or '<='"):
        SimpleFilter(threshold=0, operator='==', metric=None)
