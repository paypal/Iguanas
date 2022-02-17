import pandas as pd
import numpy as np
import pytest
from iguanas.correlation_reduction import AgglomerativeClusteringReducer
from iguanas.metrics.classification import FScore
from iguanas.metrics.pairwise import CosineSimilarity, JaccardSimilarity
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
def similarity_df():
    similarity_df = pd.DataFrame({
        "index": ["A", "B", "C", "D", "E"],
        "A": [1.000000,	0.531925, 0.509824,	0.471540, 0.517857],
        "B": [0.531925, 1.000000, 0.508200,	0.521822, 0.526037],
        "C": [0.509824,	0.508200, 1.000000,	0.493806, 0.502041],
        "D": [0.471540,	0.521822, 0.493806,	1.000000, 0.483528],
        "E": [0.517857,	0.526037, 0.502041,	0.483528, 1.000000]
    })
    similarity_df.set_index('index', inplace=True)
    return similarity_df


@pytest.fixture
def columns_performance():
    columns_performance = pd.Series(
        [0.1, 0.2, 0.5, 0.4, 0.9],
        ["A", "B", "C", "D", "E"]
    )
    return columns_performance


@pytest.fixture
def clusters():
    clusters = pd.Series([0, 0, 0, 1, 0], ["A", "B", "C", "D", "E"])
    return clusters


@pytest.fixture
def cos_sim():
    cs = CosineSimilarity()
    return cs


@pytest.fixture
def jacc_sim():
    js = JaccardSimilarity()
    return js


@pytest.fixture
def dummy_metric(*args, **kwargs):
    return np.array([0.1, 0.2, 0.5, 0.4, 0.9])


@pytest.fixture
def agg_instantiated(cos_sim, dummy_metric):

    cs = cos_sim
    agg = AgglomerativeClusteringReducer(
        threshold=0.5,
        strategy='bottom_up',
        similarity_function=cs.fit,
        metric=dummy_metric,
        print_clustermap=True
    )
    return agg


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


def test_fit(create_data, cos_sim, jacc_sim, expected_columns_to_keep):
    expected_results = expected_columns_to_keep
    X, y = create_data
    cs = cos_sim
    js = jacc_sim
    fs = FScore(0.5)
    combinations = list(
        product([0.25, 0.75], ['top_down', 'bottom_up'], [cs.fit, js.fit], [fs.fit, None]))
    for i, (threshold, strategy, similarity_function, metric) in enumerate(combinations):
        fr = AgglomerativeClusteringReducer(
            threshold, strategy, similarity_function, metric,
            print_clustermap=True
        )
        assert fr.__repr__(
        ) == f'AgglomerativeClusteringReducer(threshold={threshold}, strategy={strategy}, similarity_function={similarity_function}, metric={metric}, print_clustermap=True)'
        if metric is None:
            fr.fit(X)
        else:
            fr.fit(X, y)
        assert sorted(fr.columns_to_keep) == sorted(expected_results[i])
        assert fr.__repr__(
        ) == f'AgglomerativeClusteringReducer object with {len(fr.columns_to_keep)} columns to keep'


def test_fit_2_cols(create_data, jacc_sim):
    X, y = create_data
    js = jacc_sim
    fs = FScore(0.5)
    fr = AgglomerativeClusteringReducer(
        0.5, 'top_down', js.fit, fs.fit,
        print_clustermap=True
    )
    fr.fit(X.iloc[:, :2], y)
    assert sorted(fr.columns_to_keep) == ['col0', 'col1']
    fr = AgglomerativeClusteringReducer(
        0.5, 'bottom_up', js.fit, fs.fit,
        print_clustermap=True
    )
    fr.fit(X.iloc[:, :2], y)
    assert sorted(fr.columns_to_keep) == ['col0']


def test_transform(create_data, cos_sim, jacc_sim, expected_columns_to_keep):
    expected_results = expected_columns_to_keep
    X, y = create_data
    cs = cos_sim
    js = jacc_sim
    fs = FScore(0.5)
    combinations = list(
        product([0.25, 0.75], ['top_down', 'bottom_up'], [cs.fit, js.fit], [fs.fit, None]))
    for i, (threshold, strategy, similarity_function, metric) in enumerate(combinations):
        fr = AgglomerativeClusteringReducer(
            threshold, strategy, similarity_function, metric)
        if metric is None:
            fr.fit(X)
        else:
            fr.fit(X, y)
        X_reduced = fr.transform(X)
        assert sorted(X_reduced.columns.tolist()
                      ) == sorted(expected_results[i])


def test_fit_transform(create_data, cos_sim, jacc_sim, expected_columns_to_keep):
    expected_results = expected_columns_to_keep
    X, y = create_data
    cs = cos_sim
    js = jacc_sim
    fs = FScore(0.5)
    combinations = list(
        product([0.25, 0.75], ['top_down', 'bottom_up'], [cs.fit, js.fit], [fs.fit, None]))
    for i, (threshold, strategy, similarity_function, metric) in enumerate(combinations):
        fr = AgglomerativeClusteringReducer(
            threshold, strategy, similarity_function, metric)
        if metric is None:
            X_reduced = fr.fit_transform(X)
        else:
            X_reduced = fr.fit_transform(X, y)
        assert sorted(fr.columns_to_keep) == sorted(X_reduced.columns.tolist(
        )) == sorted(expected_results[i])


def test_bottom_up(agg_instantiated, clusters, similarity_df):
    agg = agg_instantiated
    clusters = clusters
    similarity_df = similarity_df
    cols_to_drop, columns_to_keep = agg._bottom_up(
        clusters, int(similarity_df.shape[0] / 2),
        similarity_df, []
    )
    assert cols_to_drop == ['B', 'C', 'E']
    assert columns_to_keep == []


def test_top_down(agg_instantiated, clusters, similarity_df):
    agg = agg_instantiated
    clusters = clusters
    similarity_df = similarity_df
    cols_to_drop, columns_to_keep = agg._top_down(
        clusters, 2, similarity_df, []
    )
    assert cols_to_drop == ['A', 'B', 'C', 'E']
    assert columns_to_keep == ['A', 'D']


def test_set_n_clusters(agg_instantiated, similarity_df):
    agg = agg_instantiated
    similarity_df = similarity_df
    for strategy in ['top_down', 'bottom_up']:
        agg.strategy = strategy
        n_clusters = agg._set_n_clusters(similarity_df)
        assert n_clusters == 2


def test_calculate_cluster_median(agg_instantiated, similarity_df):
    agg = agg_instantiated
    similarity_df = similarity_df
    cluster_median = agg._calculate_cluster_median(similarity_df)
    assert round(cluster_median, 6) == 0.509012


def test_get_top_performer(agg_instantiated, columns_performance):
    agg = agg_instantiated
    columns_performance = columns_performance
    top_performer = agg._get_top_performer(['A', 'B'], columns_performance)
    assert top_performer == 'B'


def test_agglomerative_clustering(similarity_df, agg_instantiated):
    similarity_df = similarity_df
    agg = agg_instantiated
    clusters = agg._agglomerative_clustering(similarity_df, 2)
    assert all(clusters.values == np.array([0, 0, 0, 1, 0]))


def test_error(agg_instantiated, create_data, cos_sim):
    agg = agg_instantiated
    cs = cos_sim
    X, _ = create_data
    with pytest.raises(TypeError, match='`X` must be a pandas.core.frame.DataFrame. Current type is str.'):
        agg.fit('X')
    with pytest.raises(ValueError, match="`strategy` must be either 'top_down' or 'bottom_up'"):
        agg = AgglomerativeClusteringReducer(
            threshold=0.5,
            strategy='error',
            similarity_function=cs.fit,
            metric=None,
        )
    agg = agg_instantiated
    X = pd.DataFrame({
        'A': [0, 0, 0],
        'B': [0, 0, 0]
    })
    with pytest.raises(Exception, match="Columns A, B have zero variance, which will result in NaN values for the similarity matrix"):
        agg.fit(X)
