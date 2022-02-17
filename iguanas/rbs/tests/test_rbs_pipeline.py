import pytest
import pandas as pd
from iguanas.rbs import RBSPipeline
from iguanas.metrics.classification import FScore


@pytest.fixture
def _create_data():
    X = pd.DataFrame({
        'Approve1': [1, 0, 0, 0, 0],
        'Approve2': [0, 1, 0, 0, 0],
        'Decline': [1, 1, 1, 0, 0],
        'Approve3': [1, 1, 1, 1, 0],
    }, index=['a', 'b', 'c', 'd', 'e'])
    return X


@pytest.fixture
def _instantiate():
    f1 = FScore(beta=1)
    config = [
        (0, ['Approve1', 'Approve2']),
        (1, ['Decline']),
        (0, ['Approve3'])
    ]
    rbsp = RBSPipeline(
        config=config,
        final_decision=1,

    )
    return rbsp


def test_predict(_create_data, _instantiate):
    y_pred_exp = pd.Series([0, 0, 1, 0, 1], index=['a', 'b', 'c', 'd', 'e'])
    X = _create_data
    rbsp = _instantiate
    y_pred = rbsp.predict(X)
    assert all(y_pred_exp == y_pred)
    assert y_pred.name is None
    # Test predict after class already instantiated
    y_pred = rbsp.predict(X)
    assert all(y_pred_exp == y_pred)
    assert y_pred.name is None


def test_get_stage_level_preds(_create_data, _instantiate):
    config = [
        (0, ['Approve1', 'Approve2']),
        (1, ['Decline']),
        (0, ['Approve3'])
    ]
    exp_results = pd.DataFrame([
        [-1, 1, -1],
        [-1, 1, -1],
        [0, 1, -1],
        [0, 0, -1],
        [0, 0, 0]
    ],
        columns=[
            'Stage=0, Decision=0', 'Stage=1, Decision=1', 'Stage=2, Decision=0'
    ]
    )
    X = _create_data
    rbsp = _instantiate
    stage_level_preds = rbsp._get_stage_level_preds(X_rules=X, config=config)
    assert all(stage_level_preds == exp_results)
    # Test when config contains no rules
    config = [
        (0, []),
        (1, [])
    ]
    stage_level_preds = rbsp._get_stage_level_preds(X_rules=X, config=config)
    assert stage_level_preds is None


def test_get_pipeline_pred(_instantiate):
    y_pred_exp = pd.Series([0, 0, 1, 0, 1], index=['a', 'b', 'c', 'd', 'e'])
    stage_level_preds = pd.DataFrame([
        [-1, 1, -1],
        [-1, 1, -1],
        [0, 1, -1],
        [0, 0, -1],
        [0, 0, 0]
    ],
        columns=[
            'Stage=0, Decision=0', 'Stage=1, Decision=1', 'Stage=2, Decision=0'
    ]
    )
    rbsp = _instantiate
    y_pred = rbsp._get_pipeline_pred(
        stage_level_preds, ['a', 'b', 'c', 'd', 'e'])
    assert all(y_pred == y_pred_exp)
    assert y_pred.name is None


def test_errors(_create_data):
    X = _create_data
    f1 = FScore(1)
    with pytest.raises(ValueError, match='`config` must be a list'):
        rbsp = RBSPipeline(
            config={},
            final_decision=0,
        )
    with pytest.raises(ValueError, match='`final_decision` must be either 0 or 1'):
        rbsp = RBSPipeline(
            config=[],
            final_decision=2,
        )
    with pytest.raises(TypeError, match='`X_rules` must be a pandas.core.frame.DataFrame. Current type is str.'):
        rbsp = RBSPipeline(
            config=[],
            final_decision=0,
        )
        rbsp.predict('X')
