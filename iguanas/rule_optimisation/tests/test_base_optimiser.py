import pytest
from iguanas.rule_optimisation.bayesian_optimiser import BayesianOptimiser
from iguanas.metrics.classification import FScore
from iguanas.rules import Rules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        'already_optimal': lambda **kwargs: "(X['A']>{A})".format(**kwargs),
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
def _instantiate(_create_inputs):
    rule_lambdas, lambda_kwargs = _create_inputs
    f1 = FScore(beta=1)
    ro = BayesianOptimiser(
        rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs,
        metric=f1.fit, n_iter=30, verbose=1
    )
    return ro


def test_plot_performance_uplift(monkeypatch, _instantiate):
    monkeypatch.setattr(plt, 'show', lambda: None)
    ro = _instantiate
    ro.plot_performance_uplift(
        orig_rule_performances={'Rule1': 0.2},
        opt_rule_performances={'Rule1': 0.4}
    )
    assert True


def test_plot_performance_uplift_distribution(monkeypatch, _instantiate):
    monkeypatch.setattr(plt, 'show', lambda: None)
    ro = _instantiate
    ro.plot_performance_uplift_distribution(
        orig_rule_performances={'Rule1': 0.2},
        opt_rule_performances={'Rule1': 0.4}
    )
    assert True


def test_calculate_performance_comparison(_instantiate):
    ro = _instantiate
    orig_rule_performances = {
        'Rule1': 0.1,
        'Rule2': 0.2,
        'Rule3': 0.3
    }
    opt_rule_performances = {
        'Rule1': 0.2,
        'Rule2': 0.4,
        'Rule3': 0.3
    }
    exp_performance_comp = pd.DataFrame(
        np.array([[0.1, 0.2],
                  [0.2, 0.4],
                  [0.3, 0.3]]),
        columns=['OriginalRule', 'OptimisedRule']
    )
    exp_performance_comp.index = ['Rule1', 'Rule2', 'Rule3']
    exp_performance_diff = pd.Series({
        'Rule1': 0.1,
        'Rule2': 0.2,
        'Rule3': 0
    })
    performance_comp, performance_difference = ro._calculate_performance_comparison(orig_rule_performances=orig_rule_performances,
                                                                                    opt_rule_performances=opt_rule_performances)
    pd.testing.assert_frame_equal(performance_comp, exp_performance_comp)
    pd.testing.assert_series_equal(
        performance_difference, exp_performance_diff)


def test_return_X_min_max(_create_data, _instantiate):
    expected_X_min = pd.Series({
        'A': 0.0, 'AllNa': np.nan, 'C': 0.0003641365574362787, 'ZeroVar': 1.0
    })
    expected_X_max = pd.Series({
        'A': 9.0, 'AllNa': np.nan, 'C': 0.9999680225821261, 'ZeroVar': 1.0
    })
    X, _, _ = _create_data
    ro = _instantiate
    X_min, X_max = ro._return_X_min_max(
        X, ['A', 'ZeroVar', 'C', 'AllNa', 'C%0'])
    pd.testing.assert_series_equal(X_min.sort_index(), expected_X_min)
    pd.testing.assert_series_equal(X_max.sort_index(), expected_X_max)


def test_return_rules_missing_features(_create_data, _create_inputs, _instantiate):
    X, _, _ = _create_data
    rule_lambdas, lambda_kwargs = _create_inputs
    r = Rules(rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs)
    ro = _instantiate
    with pytest.warns(UserWarning, match="Rules `missing_col` use features that are missing from `X` - unable to optimise or apply these rules"):
        rule_names_missing_features, rule_features_in_X = ro._return_rules_missing_features(
            rules=r, columns=X.columns, verbose=0)
    assert rule_names_missing_features == ['missing_col']
    assert rule_features_in_X == {'A', 'AllNa', 'C', 'D', 'E', 'ZeroVar'}


def test_return_all_optimisable_rule_features(_instantiate, _create_data, _create_inputs):
    exp_all_features = ['A', 'C', 'C%0', 'Z', 'ZeroVar']
    exp_rule_name_no_opt_conds = ['all_na', 'boolean', 'categoric']
    ro = _instantiate
    X, _,  _ = _create_data
    _, lambda_kwargs = _create_inputs
    with pytest.warns(UserWarning, match="Rules `categoric`, `boolean`, `all_na` have no optimisable conditions - unable to optimise these rules"):
        all_features, rule_names_no_opt_conditions = ro._return_all_optimisable_rule_features(
            lambda_kwargs=lambda_kwargs, X=X, verbose=0)
        all_features.sort()
        rule_names_no_opt_conditions.sort()
        assert all_features == exp_all_features
        assert rule_names_no_opt_conditions == exp_rule_name_no_opt_conds


def test_return_rules_with_zero_var_features(_instantiate, _create_inputs):
    ro = _instantiate
    _, lambda_kwargs = _create_inputs
    X_min = pd.Series({
        'A': 0.0, 'AllNa': np.nan, 'C': 0.0003641365574362787, 'ZeroVar': 1.0
    })
    X_max = pd.Series({
        'A': 9.0, 'AllNa': np.nan, 'C': 0.9999680225821261, 'ZeroVar': 1.0
    })
    with pytest.warns(UserWarning, match="Rules `zero_var` have all zero variance features based on the dataset `X` - unable to optimise these rules"):
        zero_var_rules = ro._return_rules_with_zero_var_features(
            lambda_kwargs=lambda_kwargs, X_min=X_min, X_max=X_max,
            rule_names_no_opt_conditions=['all_na', 'boolean', 'categoric'],
            verbose=0
        )
        assert zero_var_rules == ['zero_var']


def test_return_optimisable_rules(_instantiate, _create_inputs):
    rule_lambdas, lambda_kwargs = _create_inputs
    r = Rules(rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs)
    r.filter_rules(exclude=['missing_col'])
    ro = _instantiate
    rules, non_optimisable_rules = ro._return_optimisable_rules(rules=r, rule_names_no_opt_conditions=[
        'all_na', 'boolean', 'categoric', 'already_optimal'], rule_names_zero_var_features=['zero_var'])
    rule_names_opt = list(rules.rule_lambdas.keys())
    rule_names_non_opt = list(non_optimisable_rules.rule_lambdas.keys())
    rule_names_opt.sort()
    rule_names_non_opt.sort()
    assert rule_names_opt == ['float', 'integer', 'is_na', 'mixed']
    assert rule_names_non_opt == [
        'all_na', 'already_optimal', 'boolean', 'categoric', 'zero_var']


def test_return_orig_rule_if_better_perf(_instantiate):
    orig_rule_performances = {
        'Rule1': 0.5,
        'Rule2': 0.5,
        'Rule3': 0.5,
    }
    opt_rule_performances = {
        'Rule1': 0.3,
        'Rule2': 0.5,
        'Rule3': 0.7,
    }
    orig_rule_strings = {
        'Rule1': "(X['A']>1",
        'Rule2': "(X['A']>2",
        'Rule3': "(X['A']>3"
    }
    opt_rule_strings = {
        'Rule1': "(X['A']>0",
        'Rule2': "(X['A']>4",
        'Rule3': "(X['A']>5"
    }
    orig_X_rules = pd.DataFrame({
        'Rule1': [1, 0, 1],
        'Rule2': [0, 0, 1],
        'Rule3': [1, 0, 0],
    })
    opt_X_rules = pd.DataFrame({
        'Rule1': [1, 1, 1],
        'Rule2': [0, 1, 1],
        'Rule3': [1, 1, 0],
    })
    expected_opt_rule_strings = {
        'Rule1': "(X['A']>1",
        'Rule2': "(X['A']>2",
        'Rule3': "(X['A']>5"
    }
    expected_opt_rule_performances = {
        'Rule1': 0.5,
        'Rule2': 0.5,
        'Rule3': 0.7,
    }
    expected_X_rules = pd.DataFrame({
        'Rule1': [1, 0, 1],
        'Rule2': [0, 1, 1],
        'Rule3': [1, 1, 0],
    })
    ro = _instantiate
    ro.opt_rule_strings = opt_rule_strings
    opt_rule_strings, opt_rule_performances, opt_X_rules = ro._return_orig_rule_if_better_perf(
        orig_rule_performances=orig_rule_performances,
        opt_rule_performances=opt_rule_performances,
        orig_rule_strings=orig_rule_strings,
        opt_rule_strings=opt_rule_strings,
        orig_X_rules=orig_X_rules,
        opt_X_rules=opt_X_rules
    )
    assert opt_rule_strings == expected_opt_rule_strings
    assert opt_rule_performances == expected_opt_rule_performances
    assert all(opt_X_rules == expected_X_rules)


def test_generate_other_rule_formats(_instantiate):
    ro = _instantiate
    ro.rule_strings = {'Rule1': "(X['A']>1)"}
    ro._generate_other_rule_formats()
    assert ro.rule_names == ['Rule1']
    assert ro.rule_lambdas['Rule1'](
        **ro.lambda_kwargs['Rule1']) == "(X['A']>1)"
    assert ro.rules.rule_strings == {'Rule1': "(X['A']>1)"}
    assert ro.rules.rule_lambdas['Rule1'](
        **ro.rules.lambda_kwargs['Rule1']) == "(X['A']>1)"
