import pytest
import pandas as pd
from iguanas.rules import Rules
from hyperopt import hp
from hyperopt.pyll.base import Apply
from iguanas.rbs import RBSPipeline
from iguanas.rbs import RBSOptimiser
from iguanas.metrics.classification import FScore
import numpy as np


@pytest.fixture
def _create_data():
    X = pd.DataFrame({
        'Approve1': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
        'Approve2': [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        'Approve3': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        'Decline1': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        'Decline2': [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
        'Decline3': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    })
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
    return X, y


@pytest.fixture
def _create_data_random():
    np.random.seed(0)
    X = pd.DataFrame({
        'Approve1': np.random.randint(0, 2, 100),
        'Approve2': np.random.randint(0, 2, 100),
        'Approve3': np.random.randint(0, 2, 100),
        'Decline1': np.random.randint(0, 2, 100),
        'Decline2': np.random.randint(0, 2, 100),
        'Decline3': np.random.randint(0, 2, 100),
    })
    y = pd.Series(np.random.randint(0, 2, 100))
    sample_weight = (y*10)+1
    return X, y, sample_weight


@pytest.fixture
def _instantiate_config():
    f1 = FScore(beta=1)
    config = [
        (0, ['Approve1', 'Approve2', 'Approve3']),
        (1, ['Decline1', 'Decline2', 'Decline3']),
    ]
    rbsp = RBSPipeline(
        config=config,
        final_decision=0,
    )
    rules = Rules(
        rule_strings={
            'Approve1': "X['1']<0",
            'Approve2': "X['2']<0",
            'Approve3': "X['3']<0",
            'Decline1': "X['1']>0",
            'Decline2': "X['2']>0",
            'Decline3': "X['3']>0"
        }
    )
    rbso = RBSOptimiser(
        pipeline=rbsp,
        metric=f1.fit,
        n_iter=30,
        verbose=0,
        rules=rules
    )
    return rbso


@pytest.fixture
def _instantiate_no_config():
    f1 = FScore(beta=1)
    rbsp = RBSPipeline(
        config=[],
        final_decision=0,
    )
    rules = Rules(
        rule_strings={
            'Approve1': "X['1']<0",
            'Approve2': "X['2']<0",
            'Approve3': "X['3']<0",
            'Decline1': "X['1']>0",
            'Decline2': "X['2']>0",
            'Decline3': "X['3']>0"
        }
    )
    rbso = RBSOptimiser(
        pipeline=rbsp,
        metric=f1.fit,
        n_iter=30,
        neg_pred_rules=['Approve1', 'Approve2', 'Approve3'],
        pos_pred_rules=['Decline1', 'Decline2', 'Decline3'],
        rules=rules,
        verbose=0
    )
    return rbso


@pytest.fixture
def _exp_opt_thresholds():
    opt_thresholds_config_given = {
        'Approve1': 1,
        'Approve2': 0,
        'Approve3': 0,
        'Decline1': 1,
        'Decline2': 0,
        'Decline3': 0
    }
    exp_opt_thresholds_config_omitted = {
        'Approve1%activate': 1,
        'Approve1%stage': 1,
        'Approve2%activate': 1,
        'Approve2%stage': 3,
        'Approve3%activate': 0,
        'Approve3%stage': 0,
        'Decline1%activate': 1,
        'Decline1%stage': 1,
        'Decline2%activate': 0,
        'Decline2%stage': 3,
        'Decline3%activate': 1,
        'Decline3%stage': 2
    }
    return opt_thresholds_config_given, exp_opt_thresholds_config_omitted


def test_fit_config_given(_create_data, _instantiate_config):
    exp_config = [
        (0, ['Approve1']),
        (1, ['Decline1'])
    ]
    X, y = _create_data
    rbso = _instantiate_config
    rbso.fit(X, y)
    assert rbso.config == exp_config
    assert rbso.rules_to_keep == ['Approve1', 'Decline1'] == list(
        rbso.rules.rule_strings.keys())


def test_fit_predict_config_given(_create_data, _instantiate_config):
    exp_config = [
        (0, ['Approve1']),
        (1, ['Decline1'])
    ]
    X, y = _create_data
    rbso = _instantiate_config
    y_pred = rbso.fit_predict(X, y)
    assert rbso.config == exp_config
    assert rbso.rules_to_keep == ['Approve1', 'Decline1'] == list(
        rbso.rules.rule_strings.keys())
    assert all(y_pred == y)


def test_fit_config_omitted(_create_data, _instantiate_no_config):
    exp_config = [
        (0, ['Approve1']),
        (1, ['Decline1', 'Decline3']),
        (0, ['Approve2'])
    ]
    X, y = _create_data
    rbso = _instantiate_no_config
    rbso.fit(X, y)
    assert rbso.config == exp_config
    assert sorted(rbso.rules_to_keep) == [
        'Approve1', 'Approve2', 'Decline1', 'Decline3'
    ] == sorted(list(rbso.rules.rule_strings.keys()))


def test_fit_predict_config_omitted(_create_data, _instantiate_no_config):
    exp_config = [
        (0, ['Approve1']),
        (1, ['Decline1', 'Decline3']),
        (0, ['Approve2'])
    ]
    X, y = _create_data
    rbso = _instantiate_no_config
    y_pred = rbso.fit_predict(X, y)
    assert rbso.config == exp_config
    assert sorted(rbso.rules_to_keep) == [
        'Approve1', 'Approve2', 'Decline1', 'Decline3'
    ] == sorted(list(rbso.rules.rule_strings.keys()))
    assert all(y_pred == y)


def test_fit_weighted_config_given(_create_data_random, _instantiate_config):
    exp_config = [
        (0, []),
        (1, ['Decline1', 'Decline2', 'Decline3'])
    ]
    X, y, sample_weight = _create_data_random
    rbso = _instantiate_config
    rbso.fit(X, y, sample_weight)
    assert rbso.config == exp_config
    assert rbso.rules_to_keep == ['Decline1', 'Decline2', 'Decline3'] == list(
        rbso.rules.rule_strings.keys())


def test_fit_predict_weighted_config_given(_create_data_random, _instantiate_config):
    exp_config = [
        (0, []),
        (1, ['Decline1', 'Decline2', 'Decline3'])
    ]
    X, y, sample_weight = _create_data_random
    rbso = _instantiate_config
    y_pred = rbso.fit_predict(X, y, sample_weight)
    assert rbso.config == exp_config
    assert rbso.rules_to_keep == ['Decline1', 'Decline2', 'Decline3'] == list(
        rbso.rules.rule_strings.keys())
    assert y_pred.sum() == 90


def test_fit_weighted_config_omitted(_create_data_random, _instantiate_no_config):
    exp_config = [
        (1, ['Decline2', 'Decline3', 'Decline1']),
        (0, ['Approve1', 'Approve3'])
    ]
    X, y, sample_weight = _create_data_random
    rbso = _instantiate_no_config
    rbso.fit(X, y, sample_weight)
    assert rbso.config == exp_config
    assert sorted(rbso.rules_to_keep) == [
        'Approve1', 'Approve3', 'Decline1', 'Decline2', 'Decline3'
    ] == sorted(list(rbso.rules.rule_strings.keys()))


def test_fit_predict_weighted_config_omitted(_create_data_random, _instantiate_no_config):
    exp_config = [
        (1, ['Decline2', 'Decline3', 'Decline1']),
        (0, ['Approve1', 'Approve3'])
    ]
    X, y, sample_weight = _create_data_random
    rbso = _instantiate_no_config
    y_pred = rbso.fit_predict(X, y, sample_weight)
    assert rbso.config == exp_config
    assert sorted(rbso.rules_to_keep) == [
        'Approve1', 'Approve3', 'Decline1', 'Decline2', 'Decline3'
    ] == sorted(list(rbso.rules.rule_strings.keys()))
    assert y_pred.sum() == 90


def test_get_space_funcs(_create_data, _instantiate_config):
    X, y = _create_data
    rbso = _instantiate_config
    space_funcs = rbso._get_space_funcs(X)
    for rule, space_func in space_funcs.items():
        assert rule in X.columns
        assert isinstance(space_func, Apply)


def test_optimise_pipeline_config_given(_create_data, _instantiate_config,
                                        _exp_opt_thresholds):

    exp_opt_thresholds, _ = _exp_opt_thresholds
    X, y = _create_data
    space_funcs = {
        rule: hp.choice(rule, [0, 1]) for rule in X.columns
    }
    rbso = _instantiate_config
    opt_thresholds = rbso._optimise_pipeline(X, y, None, space_funcs)
    assert opt_thresholds == exp_opt_thresholds


def test_optimise_pipeline_config_omitted(_create_data, _instantiate_no_config,
                                          _exp_opt_thresholds):

    _, exp_opt_thresholds = _exp_opt_thresholds
    X, y = _create_data
    space_funcs = {
        rule: (hp.choice(f'{rule}%activate', [0, 1]), hp.choice(f'{rule}%stage', list(range(0, X.shape[1])))) for rule in X.columns
    }
    rbso = _instantiate_no_config
    opt_thresholds = rbso._optimise_pipeline(X, y, None, space_funcs)
    assert opt_thresholds == exp_opt_thresholds


def test_generate_config_config_given(_instantiate_config, _exp_opt_thresholds):
    opt_thresholds, _ = _exp_opt_thresholds
    exp_config = [
        (0, ['Approve1']),
        (1, ['Decline1'])
    ]
    rbso = _instantiate_config
    rbso._generate_config(opt_thresholds)
    assert rbso.config == exp_config


def test_generate_config_config_ommited(_instantiate_no_config,
                                        _exp_opt_thresholds):
    _, opt_thresholds = _exp_opt_thresholds
    exp_config = [
        (0, ['Approve1']),
        (1, ['Decline1', 'Decline3']),
        (0, ['Approve2'])
    ]
    rbso = _instantiate_no_config
    rbso._generate_config(opt_thresholds)
    assert rbso.config == exp_config


def test_create_config(_instantiate_no_config):
    space_funcs = {
        'Approve1': [1, 1],
        'Approve2': [1, 3],
        'Approve3': [0, 0],
        'Decline1': [1, 1],
        'Decline2': [0, 3],
        'Decline3': [1, 2]
    }
    exp_config = [
        (0, ['Approve1']),
        (1, ['Decline1', 'Decline3']),
        (0, ['Approve2'])
    ]
    rbso = _instantiate_no_config
    config = rbso._create_config(space_funcs)
    assert config == exp_config


def test_update_config(_instantiate_config, _exp_opt_thresholds):
    opt_thresholds, _ = _exp_opt_thresholds
    exp_config = [
        (0, ['Approve1']),
        (1, ['Decline1'])
    ]
    rbso = _instantiate_config
    config = rbso._update_config(opt_thresholds, rbso.orig_config)
    assert config == exp_config


def test_convert_opt_thr(_instantiate_config, _exp_opt_thresholds):
    _, opt_thresholds = _exp_opt_thresholds
    exp_config = {
        'Approve1': [1, 1],
        'Approve2': [1, 3],
        'Approve3': [0, 0],
        'Decline1': [1, 1],
        'Decline2': [0, 3],
        'Decline3': [1, 2]
    }
    rbso = _instantiate_config
    config = rbso._convert_opt_thr(opt_thresholds)
    assert config == exp_config


def test_errors(_create_data):
    X, y = _create_data
    f1 = FScore(beta=1)
    rbsp = RBSPipeline(
        config=[],
        final_decision=0,
    )
    with pytest.raises(ValueError, match='If `config` not provided in `pipeline`, then one or both of `pos_pred_rules` and `neg_pred_rules` must be given.'):
        rbso = RBSOptimiser(
            pipeline=rbsp,
            metric=f1.fit,
            n_iter=30,
            verbose=0
        )
    with pytest.raises(TypeError, match='`X_rules` must be a pandas.core.frame.DataFrame. Current type is str.'):
        rbso = RBSOptimiser(
            pipeline=rbsp,
            metric=f1.fit,
            pos_pred_rules=['A', 'B'],
            neg_pred_rules=['C', 'D'],
            n_iter=30,
            verbose=0
        )
        rbso.fit('X', y)
    with pytest.raises(TypeError, match='`y` must be a pandas.core.series.Series. Current type is str.'):
        rbso.fit(X, 'y')
