import pytest
from iguanas.rule_selection import GreedyFilter
from iguanas.metrics import FScore, Precision
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


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
def instantiate_GreedyFilter():
    f4 = FScore(beta=4)
    p = Precision()
    gf = GreedyFilter(
        metric=f4.fit,
        sorting_metric=p.fit,
        verbose=1
    )
    return gf


@ pytest.fixture
def expected_results_GreedyFilter(create_data):
    X_rules, _, _ = create_data
    expected_results = [
        X_rules[['Rule1', 'Rule2', 'Rule3']],
        X_rules[['Rule1', 'Rule2', 'Rule3', 'Rule5']]
    ]
    return expected_results


@pytest.fixture
def expected_scores():
    expected_scores = [0.9053254437869824, 0.9486607142857143] * 2
    return expected_scores


@ pytest.fixture
def expected_results_return_performance_top_n():
    top_n_no_weight = pd.Series(
        {
            1: 0.31288343558282206, 2: 0.6144578313253012,
            3: 0.9053254437869824, 4: 0.648854961832061, 5: 0.25757575757575757
        },
        name='Metric'
    )
    top_n_weight = pd.Series(
        {
            1: 0.31288343558282206, 2: 0.6144578313253012,
            3: 0.9053254437869824, 4: 0.9486607142857143, 5: 0.776255707762557
        },
        name='Metric'
    )
    return top_n_no_weight, top_n_weight


def test_fit(create_data, instantiate_GreedyFilter, expected_results_GreedyFilter, expected_scores):
    X_rules, y, weights = create_data
    expected_reprs = [
        "GreedyFilter object with 3 rules remaining post-filtering",
        "GreedyFilter object with 4 rules remaining post-filtering"
    ]
    expected_results_GreedyFilter = expected_results_GreedyFilter
    expected_scores = expected_scores
    gf = instantiate_GreedyFilter
    assert gf.__repr__(
    ) == "GreedyFilter(metric=<bound method FScore.fit of FScore with beta=4>, sorting_metric=<bound method Precision.fit of Precision>)"
    # Without weight
    gf.fit(X_rules=X_rules, y=y)
    assert gf.__repr__() == expected_reprs[0]
    assert gf.rules_to_keep == \
        expected_results_GreedyFilter[0].columns.tolist()
    assert gf.score == expected_scores[0]
    # With weight
    gf.fit(X_rules=X_rules, y=y, sample_weight=weights)
    assert gf.__repr__() == expected_reprs[1]
    assert gf.rules_to_keep == \
        expected_results_GreedyFilter[1].columns.tolist()
    assert gf.score == expected_scores[1]


def test_transform(create_data, instantiate_GreedyFilter):
    X_rules, _, _ = create_data
    gf = instantiate_GreedyFilter
    gf.rules_to_keep = ['Rule1']
    X_rules_filtered = gf.transform(X_rules)
    assert (all(X_rules_filtered == X_rules['Rule1'].to_frame()))


def test_fit_transform(create_data, instantiate_GreedyFilter,
                       expected_results_GreedyFilter, expected_scores):
    X_rules, y, weights = create_data
    expected_results_GreedyFilter = expected_results_GreedyFilter
    expected_scores = expected_scores
    gf = instantiate_GreedyFilter
    # Without weight
    X_rules_filtered = gf.fit_transform(
        X_rules=X_rules, y=y)
    assert gf.rules_to_keep == expected_results_GreedyFilter[0].columns.tolist(
    )
    assert all(X_rules_filtered == expected_results_GreedyFilter[0])
    assert gf.score == expected_scores[0]
    # With weight
    X_rules_filtered = gf.fit_transform(
        X_rules=X_rules, y=y, sample_weight=weights)
    assert gf.rules_to_keep == expected_results_GreedyFilter[1].columns.tolist(
    )
    assert all(X_rules_filtered == expected_results_GreedyFilter[1])
    assert gf.score == expected_scores[1]


def test_plot_top_n_performance_on_train(monkeypatch, create_data,
                                         instantiate_GreedyFilter):
    monkeypatch.setattr(plt, 'show', lambda: None)
    X_rules, y, _ = create_data
    gf = instantiate_GreedyFilter
    gf.fit(X_rules, y)
    gf.plot_top_n_performance_on_train()
    assert True


def test_plot_top_n_performance(monkeypatch, create_data,
                                instantiate_GreedyFilter):
    monkeypatch.setattr(plt, 'show', lambda: None)
    X_rules, y, _ = create_data
    gf = instantiate_GreedyFilter
    gf.fit(X_rules, y)
    gf.plot_top_n_performance(
        X_rules=X_rules,
        y=y
    )
    assert True


def test_sort_rules(create_data, instantiate_GreedyFilter):
    p = Precision()
    f4 = FScore(beta=4)
    X_rules, y, _ = create_data
    gf = instantiate_GreedyFilter
    sorted_rules = gf._sort_rules(
        X_rules, y, None, p.fit, f4.fit)
    assert sorted_rules == ['Rule1', 'Rule2', 'Rule3', 'Rule5', 'Rule4']


def test_return_performance_top_n(create_data, instantiate_GreedyFilter, expected_results_return_performance_top_n):
    X_rules, y, weights = create_data
    gf = instantiate_GreedyFilter
    expected_results = expected_results_return_performance_top_n
    expected_top_n_rules = {
        1: ['Rule1'],
        2: ['Rule1', 'Rule2'],
        3: ['Rule1', 'Rule2', 'Rule3'],
        4: ['Rule1', 'Rule2', 'Rule3', 'Rule5'],
        5: ['Rule1', 'Rule2', 'Rule3', 'Rule5', 'Rule4']
    }
    f4 = FScore(beta=4)
    for i, w in enumerate([None, weights]):
        sorted_rules = ['Rule1', 'Rule2', 'Rule3', 'Rule5', 'Rule4']
        top_n_metrics, top_n_rules = gf._return_performance_top_n(
            sorted_rules, X_rules, y, w, f4.fit, 0)
        pd.testing.assert_series_equal(top_n_metrics, expected_results[i])
        assert top_n_rules == expected_top_n_rules


def test_return_top_rules_by_opt_func(instantiate_GreedyFilter,
                                      expected_results_return_performance_top_n,
                                      expected_results_GreedyFilter,
                                      expected_scores):
    gf = instantiate_GreedyFilter
    top_n_list = expected_results_return_performance_top_n
    expected_results = expected_results_GreedyFilter
    expected_scores = expected_scores
    for i, top_n in enumerate(top_n_list):
        rules_to_keep, score = gf._return_top_rules_by_opt_func(
            top_n, ['Rule1', 'Rule2', 'Rule3', 'Rule5', 'Rule4'])
        assert rules_to_keep == expected_results[i].columns.tolist()
        assert score == expected_scores[i]
