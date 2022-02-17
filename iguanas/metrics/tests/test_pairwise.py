from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import pandas as pd
import pytest
from iguanas.metrics.pairwise import CosineSimilarity, JaccardSimilarity


@pytest.fixture
def create_data():
    arr = np.array([[0, 0, 1, 0, 1],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 1, 1]])
    df = pd.DataFrame(arr.T, columns=['A', 'B', 'C'])
    return arr, df


def test_cosine_similarity(create_data):
    arr, df = create_data
    exp_sim_mat = 1-pairwise_distances(arr, metric='cosine')
    sim = CosineSimilarity()
    sim_df = sim.fit(df)
    assert 5.787693700234703 == sim_df.sum().sum()
    assert all(exp_sim_mat.flatten() == sim_df.values.flatten())


def test_jaccard_similarity(create_data):
    arr, df = create_data
    exp_sim_mat = 1-pairwise_distances(arr.astype(bool), metric='jaccard')
    sim = JaccardSimilarity()
    sim_df = sim.fit(df)
    assert 5 == sim_df.sum().sum()
    assert all(exp_sim_mat.flatten() == sim_df.values.flatten())


def test_repr():
    c = CosineSimilarity()
    j = JaccardSimilarity()
    assert c.__repr__() == 'CosineSimilarity'
    assert j.__repr__() == 'JaccardSimilarity'


def test_errors():
    with pytest.raises(TypeError, match='`X` must be a pandas.core.frame.DataFrame. Current type is str.'):
        sim = JaccardSimilarity()
        sim.fit('X')
    with pytest.raises(TypeError, match='`X` must be a pandas.core.frame.DataFrame. Current type is str.'):
        sim = CosineSimilarity()
        sim.fit('X')
