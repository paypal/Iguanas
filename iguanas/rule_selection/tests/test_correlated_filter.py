import pytest
from iguanas.rule_selection import CorrelatedFilter
from iguanas.metrics import FScore, CosineSimilarity, JaccardSimilarity
from iguanas.correlation_reduction import AgglomerativeClusteringReducer
import numpy as np
import pandas as pd
from itertools import product


@pytest.fixture
def create_data():
    np.random.seed(0)
    X = pd.DataFrame(
        np.random.randint(0, 10, (1000, 10)),
        columns=[f'col{i}' for i in range(10)]
    )
    y = pd.Series(np.random.randint(0, 2, 1000))
    return X, y


@pytest.fixture
def expected_columns_to_keep():
    return [
        ['col8', 'col9'],
        ['col0', 'col1'],
        ['col8', 'col4'],
        ['col6', 'col0'],
        ['col8'],
        ['col0'],
        ['col8'],
        ['col0'],
        ['col0', 'col1', 'col2', 'col3', 'col4',
            'col5', 'col6', 'col7', 'col8', 'col9'],
        ['col0', 'col1', 'col2', 'col3', 'col4',
            'col5', 'col6', 'col7', 'col8', 'col9'],
        ['col8', 'col4'],
        ['col6', 'col0'],
        ['col0', 'col1', 'col2', 'col3', 'col4',
            'col5', 'col6', 'col7', 'col8', 'col9'],
        ['col0', 'col1', 'col2', 'col3', 'col4',
            'col5', 'col6', 'col7', 'col8', 'col9'],
        ['col8'],
        ['col0'],
    ]


def test_fit(create_data, expected_columns_to_keep):
    X_rules, y = create_data
    expected_results = expected_columns_to_keep
    cs = CosineSimilarity()
    js = JaccardSimilarity()
    fs = FScore(0.5)
    combinations = list(
        product([0.25, 0.75], ['top_down', 'bottom_up'], [cs.fit, js.fit], [fs.fit, None]))
    for i, (threshold, strategy, similarity_function, metric) in enumerate(combinations):
        crc = AgglomerativeClusteringReducer(
            threshold=threshold, strategy=strategy,
            similarity_function=similarity_function,
            metric=metric
        )
        fr = CorrelatedFilter(
            correlation_reduction_class=crc
        )
        if metric is None:
            fr.fit(X_rules=X_rules)
        else:
            fr.fit(X_rules=X_rules, y=y)
        assert sorted(fr.rules_to_keep) == sorted(expected_results[i])


def test_transform(create_data, expected_columns_to_keep):
    X_rules, y = create_data
    expected_results = expected_columns_to_keep
    cs = CosineSimilarity()
    js = JaccardSimilarity()
    fs = FScore(0.5)
    combinations = list(
        product([0.25, 0.75], ['top_down', 'bottom_up'], [cs.fit, js.fit], [fs.fit, None]))
    for i, (threshold, strategy, similarity_function, metric) in enumerate(combinations):
        crc = AgglomerativeClusteringReducer(
            threshold=threshold, strategy=strategy,
            similarity_function=similarity_function,
            metric=metric
        )
        fr = CorrelatedFilter(
            correlation_reduction_class=crc
        )
        if metric is None:
            fr.fit(X_rules=X_rules)
        else:
            fr.fit(X_rules=X_rules, y=y)
        X_rules_reduced = fr.transform(X_rules=X_rules)
        assert sorted(fr.rules_to_keep) == sorted(X_rules_reduced.columns.tolist(
        )) == sorted(expected_results[i])


def test_fit_transform(create_data, expected_columns_to_keep):
    X_rules, y = create_data
    expected_results = expected_columns_to_keep
    cs = CosineSimilarity()
    js = JaccardSimilarity()
    fs = FScore(0.5)
    combinations = list(
        product([0.25, 0.75], ['top_down', 'bottom_up'], [cs.fit, js.fit], [fs.fit, None]))
    for i, (threshold, strategy, similarity_function, metric) in enumerate(combinations):
        crc = AgglomerativeClusteringReducer(
            threshold=threshold, strategy=strategy,
            similarity_function=similarity_function,
            metric=metric
        )
        fr = CorrelatedFilter(
            correlation_reduction_class=crc
        )
        if metric is None:
            X_rules_reduced = fr.fit_transform(X_rules=X_rules)
        else:
            X_rules_reduced = fr.fit_transform(X_rules=X_rules, y=y)
        assert sorted(fr.rules_to_keep) == sorted(X_rules_reduced.columns.tolist(
        )) == sorted(expected_results[i])
