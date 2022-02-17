import pytest
import re
import pandas as pd
import numpy as np
from hyperopt import hp
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy
from iguanas.rule_generation import RuleGeneratorDT, RuleGeneratorOpt
from iguanas.rule_optimisation import BayesianOptimiser
from iguanas.rules import Rules
from iguanas.metrics import FScore, JaccardSimilarity, Precision
from iguanas.rule_selection import SimpleFilter, CorrelatedFilter, GreedyFilter, BayesSearchCV
from iguanas.correlation_reduction import AgglomerativeClusteringReducer
from iguanas.rbs import RBSOptimiser, RBSPipeline
from iguanas.pipeline import LinearPipeline, ParallelPipeline, ClassAccessor
from iguanas.rule_generation import RuleGeneratorDT
from iguanas.space import UniformFloat, UniformInteger, Choice

f1 = FScore(1)
js = JaccardSimilarity()
p = Precision()


@pytest.fixture
def _create_data():
    np.random.seed(0)
    X = pd.DataFrame({
        'A': np.random.randint(0, 2, 100),
        'B': np.random.randint(0, 10, 100),
        'C': np.random.normal(0.7, 0.2, 100),
        'D': (np.random.uniform(0, 1, 100) > 0.6).astype(int)
    })
    y = pd.Series((np.random.uniform(0, 1, 100) >
                  0.9).astype(int), name='label')
    sample_weight = (y+1)*10
    return X, y, sample_weight


@pytest.fixture
def _instantiate_classes():
    rf = RandomForestClassifier(n_estimators=10, random_state=0)
    rg_dt = RuleGeneratorDT(
        metric=f1.fit,
        n_total_conditions=4,
        tree_ensemble=rf
    )
    rg_opt = RuleGeneratorOpt(
        metric=f1.fit,
        n_total_conditions=4,
        num_rules_keep=10,
    )
    rule_strings = {
        'Rule1': "(X['A']>0)&(X['C']>0)",
        'Rule2': "(X['B']>0)&(X['D']>0)",
        'Rule3': "(X['D']>0)",
        'Rule4': "(X['C']>0)"
    }
    rules = Rules(rule_strings=rule_strings)
    rule_lambdas = rules.as_rule_lambdas(as_numpy=False, with_kwargs=True)
    ro = BayesianOptimiser(
        rule_lambdas=rule_lambdas,
        lambda_kwargs=rules.lambda_kwargs,
        metric=f1.fit,
        n_iter=5
    )
    sf = SimpleFilter(
        threshold=0.05,
        operator='>=',
        metric=f1.fit
    )
    cf = CorrelatedFilter(
        correlation_reduction_class=AgglomerativeClusteringReducer(
            threshold=0.9,
            strategy='bottom_up',
            similarity_function=js.fit
        )
    )
    gf = GreedyFilter(
        metric=f1.fit,
        sorting_metric=p.fit
    )
    rbs = RBSOptimiser(
        RBSPipeline(
            config=[],
            final_decision=0,
        ),
        metric=f1.fit,
        pos_pred_rules=ClassAccessor('gf', 'rules_to_keep'),
        neg_pred_rules=[],
        n_iter=10
    )
    return rg_dt, rg_opt, ro, sf, cf, gf, rbs


@pytest.fixture
def _cv_datasets(_create_data):
    X, y, sample_weight = _create_data
    cv_datasets = {
        0: (X[:50], X[50:], y[:50], y[50:], sample_weight[:50], sample_weight[50:]),
        1: (X[50:], X[:50], y[50:], y[:50], sample_weight[50:], sample_weight[:50])
    }
    return cv_datasets


@pytest.fixture
def _instantiate_lp_and_bs(_instantiate_classes):
    rg_dt, _, _, _, _, _, rbs = _instantiate_classes
    search_spaces = {
        'rg_dt': {
            'n_total_conditions': UniformInteger(2, 5),
            'target_feat_corr_types': Choice([None])
        },
        'rbs': {
            'n_iter': UniformInteger(10, 15)
        }
    }
    search_spaces_ = {
        'rg_dt': {
            'n_total_conditions': hp.quniform('n_total_conditions', 2, 5, q=1),
            'target_feat_corr_types': hp.choice('target_feat_corr_types', [None])
        },
        'rbs': {
            'n_iter': hp.quniform('n_iter', 10, 15, q=1)
        }
    }
    rbs.pos_pred_rules = ClassAccessor('rg_dt', 'rule_names')
    steps = [
        ('rg_dt', rg_dt),
        ('rbs', rbs)
    ]
    lp = LinearPipeline(steps)
    bs = BayesSearchCV(
        pipeline=lp,
        search_spaces=search_spaces,
        metric=f1.fit,
        cv=3,
        n_iter=1,
        error_score=0,
        verbose=1
    )
    return lp, bs, search_spaces_


@pytest.fixture
def _params_iter():
    params_iter = {
        'rg_dt': {
            'n_total_conditions': 2,
            'target_feat_corr_types': None
        },
        'rbs': {
            'n_iter': 10
        }
    }
    return params_iter


def test_fit_predict_rule_gen_dt(_create_data, _instantiate_classes):
    X, y, sample_weight = _create_data
    rg_dt, _, _, sf, cf, gf, rbs = _instantiate_classes
    search_spaces = {
        'rg_dt': {
            'n_total_conditions': UniformInteger(2, 5),
            'target_feat_corr_types': Choice([None])
        },
        'sf': {
            'threshold': UniformFloat(0, 1),
        },
        'cf': {
            'correlation_reduction_class': Choice(
                [
                    AgglomerativeClusteringReducer(
                        threshold=0.9,
                        strategy='top_down',
                        similarity_function=js.fit,
                        metric=f1.fit
                    ),
                    AgglomerativeClusteringReducer(
                        threshold=0.95,
                        strategy='top_down',
                        similarity_function=js.fit,
                        metric=f1.fit
                    )
                ]
            )
        },
        'gf': {
            'metric': Choice([p.fit, f1.fit])
        },
        'rbs': {
            'n_iter': UniformInteger(10, 15)
        }
    }
    steps = [
        ('rg_dt', rg_dt),
        ('sf', sf),
        ('cf', cf),
        ('gf', gf),
        ('rbs', rbs)
    ]
    lp = LinearPipeline(steps)
    bs = BayesSearchCV(
        pipeline=lp,
        search_spaces=search_spaces,
        metric=f1.fit,
        cv=3,
        n_iter=5,
        error_score=0,
        verbose=1
    )
    with pytest.warns(UserWarning):
        # Test fit/predict/fit_predict, no sample_weight
        bs.fit(X, y)
        assert bs.best_score == 0.22857142857142856
        assert bs.best_index == 2
        assert bs.best_params['cf']['correlation_reduction_class'].threshold == 0.95
        assert str(bs.best_params['gf']['metric']
                   ) == '<bound method Precision.fit of Precision>'
        assert bs.best_params['rbs']['n_iter'] == 15.0
        assert bs.best_params['rg_dt']['n_total_conditions'] == 5.0
        assert bs.best_params['rg_dt']['target_feat_corr_types'] is None
        assert bs.best_params['sf']['threshold'] == 0.2583716619727481
        assert bs.cv_results.shape == (5, 11)
        y_pred = bs.predict(X)
        assert y_pred.mean() == 0.02
        assert f1.fit(y_pred, y) == 0.33333333333333337
        y_pred = bs.fit_predict(X, y)
        assert y_pred.mean() == 0.02
        assert f1.fit(y_pred, y) == 0.33333333333333337
        # Test fit/predict/fit_predict, sample_weight given and
        # sample_weight_in_val = False
        bs.sample_weight_in_val = False
        bs.fit(X, y, sample_weight)
        assert bs.best_score == 0.22857142857142856
        assert bs.best_index == 2
        assert bs.best_params['cf']['correlation_reduction_class'].threshold == 0.95
        assert str(bs.best_params['gf']['metric']
                   ) == '<bound method Precision.fit of Precision>'
        assert bs.best_params['rbs']['n_iter'] == 15.0
        assert bs.best_params['rg_dt']['n_total_conditions'] == 5.0
        assert bs.best_params['rg_dt']['target_feat_corr_types'] is None
        assert bs.best_params['sf']['threshold'] == 0.2583716619727481
        assert bs.cv_results.shape == (5, 11)
        y_pred = bs.predict(X)
        assert y_pred.mean() == 0.02
        assert f1.fit(y_pred, y) == 0.33333333333333337
        y_pred = bs.fit_predict(X, y, sample_weight)
        assert y_pred.mean() == 0.02
        assert f1.fit(y_pred, y) == 0.33333333333333337
        # Test fit/predict/fit_predict, sample_weight given and
        # sample_weight_in_val = True
        bs.sample_weight_in_val = True
        bs.fit(X, y, sample_weight)
        assert bs.best_score == 0.2545454545454546
        assert bs.best_index == 2
        assert bs.best_params['cf']['correlation_reduction_class'].threshold == 0.95
        assert str(bs.best_params['gf']['metric']
                   ) == '<bound method Precision.fit of Precision>'
        assert bs.best_params['rbs']['n_iter'] == 15.0
        assert bs.best_params['rg_dt']['n_total_conditions'] == 5.0
        assert bs.best_params['rg_dt']['target_feat_corr_types'] is None
        assert bs.best_params['sf']['threshold'] == 0.2583716619727481
        assert bs.cv_results.shape == (5, 11)
        y_pred = bs.predict(X)
        assert y_pred.mean() == 0.02
        assert f1.fit(y_pred, y, sample_weight) == 0.33333333333333337
        y_pred = bs.fit_predict(X, y, sample_weight)
        assert y_pred.mean() == 0.02
        assert f1.fit(y_pred, y, sample_weight) == 0.33333333333333337


def test_fit_predict_rule_gen_opt(_create_data, _instantiate_classes):
    X, y, sample_weight = _create_data
    _, rg_opt, _, sf, cf, gf, rbs = _instantiate_classes
    search_spaces = {
        'rg_opt': {
            'n_total_conditions': UniformInteger(2, 5),
        },
        'sf': {
            'threshold': UniformFloat(0, 1),
        },
        'cf': {
            'correlation_reduction_class': Choice(
                [
                    AgglomerativeClusteringReducer(
                        threshold=0.9,
                        strategy='top_down',
                        similarity_function=js.fit,
                        metric=f1.fit
                    ),
                    AgglomerativeClusteringReducer(
                        threshold=0.95,
                        strategy='top_down',
                        similarity_function=js.fit,
                        metric=f1.fit
                    )
                ]
            )
        },
        'gf': {
            'metric': Choice([p.fit, f1.fit])
        },
        'rbs': {
            'n_iter': UniformInteger(10, 15)
        }
    }
    steps = [
        ('rg_opt', rg_opt),
        ('sf', sf),
        ('cf', cf),
        ('gf', gf),
        ('rbs', rbs)
    ]
    lp = LinearPipeline(steps)
    bs = BayesSearchCV(
        pipeline=lp,
        search_spaces=search_spaces,
        metric=f1.fit,
        cv=3,
        n_iter=5,
        error_score=0,
        verbose=1
    )
    with pytest.warns(UserWarning):
        # Test fit/predict/fit_predict, no sample_weight
        bs.fit(X, y)
        assert bs.best_score == 0.08333333333333333
        assert bs.best_index == 4
        assert bs.best_params['rg_opt']['n_total_conditions'] == 4.0
        assert bs.best_params['sf']['threshold'] == 0.1474198119511717
        assert bs.best_params['cf']['correlation_reduction_class'].threshold == 0.95
        assert str(bs.best_params['gf']['metric']
                   ) == '<bound method FScore.fit of FScore with beta=1>'
        assert bs.best_params['rbs']['n_iter'] == 12.0
        bs.pipeline_.get_params()['rbs']['rules_to_keep'] == [
            'RGO_Rule_20220204_25', 'RGO_Rule_20220204_27',
            'RGO_Rule_20220204_41'
        ]
        assert bs.cv_results.shape == (5, 10)
        y_pred = bs.predict(X)
        assert y_pred.mean() == 0.11
        assert f1.fit(y_pred, y) == 0.5714285714285713
        y_pred = bs.fit_predict(X, y)
        assert y_pred.mean() == 0.11
        assert f1.fit(y_pred, y) == 0.5714285714285713
        # Test fit/predict/fit_predict, sample_weight given and
        # sample_weight_in_val = False
        bs.sample_weight_in_val = False
        bs.fit(X, y, sample_weight)
        assert bs.best_score == 0.27777777777777773
        assert bs.best_index == 4
        assert bs.best_params['rg_opt']['n_total_conditions'] == 4.0
        assert bs.best_params['sf']['threshold'] == 0.1474198119511717
        assert bs.best_params['cf']['correlation_reduction_class'].threshold == 0.95
        assert str(bs.best_params['gf']['metric']
                   ) == '<bound method FScore.fit of FScore with beta=1>'
        assert bs.best_params['rbs']['n_iter'] == 12.0
        bs.pipeline_.get_params()['rbs']['rules_to_keep'] == [
            'RGO_Rule_20220204_31', 'RGO_Rule_20220204_24',
            'RGO_Rule_20220204_34'
        ]
        assert bs.cv_results.shape == (5, 10)
        y_pred = bs.predict(X)
        assert y_pred.mean() == 0.13
        assert f1.fit(y_pred, y) == 0.5217391304347826
        y_pred = bs.fit_predict(X, y, sample_weight)
        assert y_pred.mean() == 0.13
        assert f1.fit(y_pred, y) == 0.5217391304347826
        # Test fit/predict/fit_predict, sample_weight given and
        # sample_weight_in_val = True
        bs.sample_weight_in_val = True
        bs.fit(X, y, sample_weight)
        assert bs.best_score == 0.3705379587732529
        assert bs.best_index == 4
        assert bs.best_params['rg_opt']['n_total_conditions'] == 4.0
        assert bs.best_params['sf']['threshold'] == 0.1474198119511717
        assert bs.best_params['cf']['correlation_reduction_class'].threshold == 0.95
        assert str(bs.best_params['gf']['metric']
                   ) == '<bound method FScore.fit of FScore with beta=1>'
        assert bs.best_params['rbs']['n_iter'] == 12.0
        bs.pipeline_.get_params()['rbs']['rules_to_keep'] == [
            'RGO_Rule_20220204_31', 'RGO_Rule_20220204_24',
            'RGO_Rule_20220204_34'
        ]
        assert bs.cv_results.shape == (5, 10)
        y_pred = bs.predict(X)
        assert y_pred.mean() == 0.13
        assert f1.fit(y_pred, y, sample_weight) == 0.6153846153846154
        y_pred = bs.fit_predict(X, y, sample_weight)
        assert y_pred.mean() == 0.13
        assert f1.fit(y_pred, y, sample_weight) == 0.6153846153846154


def test_fit_predict_rule_opt(_create_data, _instantiate_classes):
    X, y, sample_weight = _create_data
    _, _, ro, sf, cf, gf, rbs = _instantiate_classes
    search_spaces = {
        'ro': {
            'metric': Choice([p.fit, f1.fit])
        },
        'sf': {
            'threshold': UniformFloat(0, 1),
        },
        'cf': {
            'correlation_reduction_class': Choice(
                [
                    AgglomerativeClusteringReducer(
                        threshold=0.9,
                        strategy='top_down',
                        similarity_function=js.fit,
                        metric=f1.fit
                    ),
                    AgglomerativeClusteringReducer(
                        threshold=0.95,
                        strategy='top_down',
                        similarity_function=js.fit,
                        metric=f1.fit
                    )
                ]
            )
        },
        'gf': {
            'metric': Choice([p.fit, f1.fit])
        },
        'rbs': {
            'n_iter': UniformInteger(10, 15)
        }
    }
    steps = [
        ('ro', ro),
        ('sf', sf),
        ('cf', cf),
        ('gf', gf),
        ('rbs', rbs)
    ]
    lp = LinearPipeline(steps)
    bs = BayesSearchCV(
        pipeline=lp,
        search_spaces=search_spaces,
        metric=f1.fit,
        cv=3,
        n_iter=10,
        error_score=0,
        verbose=0
    )
    with pytest.warns(UserWarning):
        # Test fit/predict/fit_predict, no sample_weight
        bs.fit(X, y)
        assert bs.best_score == 0.09042145593869733
        assert bs.best_index == 9
        assert bs.best_params['cf']['correlation_reduction_class'].threshold == 0.9
        assert str(bs.best_params['gf']['metric']
                   ) == '<bound method FScore.fit of FScore with beta=1>'
        assert bs.best_params['rbs']['n_iter'] == 13.0
        assert str(bs.best_params['ro']['metric']
                   ) == '<bound method Precision.fit of Precision>'
        assert bs.best_params['sf']['threshold'] == 0.09205449690521583
        assert bs.cv_results.shape == (10, 10)
        y_pred = bs.predict(X)
        assert y_pred.mean() == 0.82
        assert f1.fit(y_pred, y) == 0.1956521739130435
        y_pred = bs.fit_predict(X, y)
        assert y_pred.mean() == 0.82
        assert f1.fit(y_pred, y) == 0.1956521739130435
        # Test fit/predict/fit_predict, sample_weight given and
        # sample_weight_in_val = False
        bs.sample_weight_in_val = False
        bs.fit(X, y, sample_weight)
        assert bs.best_score == 0.1660486960820665
        assert bs.best_index == 9
        assert bs.best_params['cf']['correlation_reduction_class'].threshold == 0.9
        assert str(bs.best_params['gf']['metric']
                   ) == '<bound method FScore.fit of FScore with beta=1>'
        assert bs.best_params['rbs']['n_iter'] == 13.0
        assert str(bs.best_params['ro']['metric']
                   ) == '<bound method Precision.fit of Precision>'
        assert bs.best_params['sf']['threshold'] == 0.09205449690521583
        assert bs.cv_results.shape == (10, 10)
        y_pred = bs.predict(X)
        assert y_pred.mean() == 0.82
        assert f1.fit(y_pred, y) == 0.1956521739130435
        y_pred = bs.fit_predict(X, y, sample_weight)
        assert y_pred.mean() == 0.82
        assert f1.fit(y_pred, y) == 0.1956521739130435
        # Test fit/predict/fit_predict, sample_weight given and
        # sample_weight_in_val = True
        bs.sample_weight_in_val = True
        bs.fit(X, y, sample_weight)
        assert bs.best_score == 0.2756485534322622
        assert bs.best_index == 9
        assert bs.best_params['cf']['correlation_reduction_class'].threshold == 0.9
        assert str(bs.best_params['gf']['metric']
                   ) == '<bound method FScore.fit of FScore with beta=1>'
        assert bs.best_params['rbs']['n_iter'] == 13.0
        assert str(bs.best_params['ro']['metric']
                   ) == '<bound method Precision.fit of Precision>'
        assert bs.best_params['sf']['threshold'] == 0.09205449690521583
        assert bs.cv_results.shape == (10, 10)
        y_pred = bs.predict(X)
        assert y_pred.mean() == 0.82
        assert f1.fit(y_pred, y, sample_weight) == 0.3243243243243243
        y_pred = bs.fit_predict(X, y, sample_weight)
        assert y_pred.mean() == 0.82
        assert f1.fit(y_pred, y, sample_weight) == 0.3243243243243243


def test_fit_predict_parallel_pipeline(_create_data, _instantiate_classes):
    X, y, sample_weight = _create_data
    rg_dt, _, ro, sf, cf, _, rbs = _instantiate_classes
    # Create set of non-fraud rules for optimisation
    rule_strings_nonfraud = {
        'NonFraudRule1': "(X['A']<0)&(X['C']<0)",
        'NonFraudRule2': "(X['B']<0)&(X['D']<0)",
        'NonFraudRule3': "(X['D']<0)",
        'NonFraudRule4': "(X['C']<0)"
    }
    rules_nonfraud = Rules(rule_strings=rule_strings_nonfraud)
    rule_lambdas_nonfraud = rules_nonfraud.as_rule_lambdas(
        as_numpy=False, with_kwargs=True)
    ro_nonfraud = BayesianOptimiser(
        rule_lambdas=rule_lambdas_nonfraud,
        lambda_kwargs=rules_nonfraud.lambda_kwargs,
        metric=f1.fit,
        n_iter=5
    )
    # Define search spaces
    search_spaces = {
        'rg_fraud': {
            'n_total_conditions': UniformInteger(1, 5),
            'target_feat_corr_types': Choice([None, 'Infer'])
        },
        'rg_nonfraud': {
            'n_total_conditions': UniformInteger(1, 5),
            'target_feat_corr_types': Choice([None, 'Infer'])
        },
        'sf_fraud': {
            'threshold': UniformFloat(0, 1)
        },
        'sf_nonfraud': {
            'threshold': UniformFloat(0, 1)
        },
        'cf_fraud': {
            'correlation_reduction_class': Choice(
                [
                    AgglomerativeClusteringReducer(
                        threshold=0.9,
                        strategy='top_down',
                        similarity_function=js.fit,
                        metric=f1.fit
                    ),
                    AgglomerativeClusteringReducer(
                        threshold=0.95,
                        strategy='top_down',
                        similarity_function=js.fit,
                        metric=f1.fit
                    )
                ]
            )
        },
        'cf_nonfraud': {
            'correlation_reduction_class': Choice(
                [
                    AgglomerativeClusteringReducer(
                        threshold=0.9,
                        strategy='top_down',
                        similarity_function=js.fit,
                        metric=f1.fit
                    ),
                    AgglomerativeClusteringReducer(
                        threshold=0.95,
                        strategy='top_down',
                        similarity_function=js.fit,
                        metric=f1.fit
                    )
                ]
            )
        },
    }
    # RBSOptimiser
    rbso = RBSOptimiser(
        pipeline=RBSPipeline(
            config=[],
            final_decision=0,
        ),
        metric=f1.fit,
        n_iter=10,
        pos_pred_rules=ClassAccessor(
            class_tag='cf_fraud',
            class_attribute='rules_to_keep'
        ),
        neg_pred_rules=ClassAccessor(
            class_tag='cf_nonfraud',
            class_attribute='rules_to_keep'
        )
    )
    # Set up pipeline
    # Fraud ----------
    rg_fraud = deepcopy(rg_dt)
    rg_fraud.rule_name_prefix = 'Fraud'
    sf_fraud = deepcopy(sf)
    cf_fraud = deepcopy(cf)
    pp_fraud = ParallelPipeline(
        steps=[
            ('rg_fraud', rg_fraud),
            ('ro_fraud', ro)
        ],
    )
    lp_fraud = LinearPipeline(
        steps=[
            ('pp_fraud', pp_fraud),
            ('sf_fraud', sf_fraud),
            ('cf_fraud', cf_fraud)
        ]
    )
    # Nonfraud ----------
    rg_nonfraud = deepcopy(rg_dt)
    rg_nonfraud.rule_name_prefix = 'NonFraud'
    sf_nonfraud = deepcopy(sf)
    cf_nonfraud = deepcopy(cf)
    pp_nonfraud = ParallelPipeline(
        steps=[
            ('rg_nonfraud', rg_nonfraud),
            ('ro_nonfraud', ro_nonfraud)
        ],
    )
    lp_nonfraud = LinearPipeline(
        steps=[
            ('pp_nonfraud', pp_nonfraud),
            ('sf_nonfraud', sf_nonfraud),
            ('cf_nonfraud', cf_nonfraud)
        ]
    )
    # Overall
    pp_overall = ParallelPipeline(
        steps=[
            ('lp_fraud', lp_fraud),
            ('lp_nonfraud', lp_nonfraud)
        ]
    )
    lp_overall = LinearPipeline(
        steps=[
            ('pp_overall', pp_overall),
            ('rbso', deepcopy(rbso))
        ]
    )
    # Bayes Search CV
    bs = BayesSearchCV(
        pipeline=lp_overall,
        search_spaces=search_spaces,
        metric=f1.fit,
        error_score=0,
        cv=3,
        n_iter=10,
        verbose=1,
        num_cores=3
    )
    # Tests (using num_cores=3 so warnings are supressed)
    # Test fit/predict/fit_predict, no sample_weight
    bs.fit(
        X={
            'lp_fraud': X,
            'lp_nonfraud': X,
        },
        y={
            'lp_fraud': y,
            'lp_nonfraud': 1-y,
            'rbso': y
        }
    )
    assert bs.best_score == 0.21543016370602575
    assert bs.best_index == 3
    assert bs.best_params['cf_fraud']['correlation_reduction_class'].threshold == 0.95
    assert bs.best_params['cf_nonfraud']['correlation_reduction_class'].threshold == 0.9
    assert bs.best_params['rg_fraud'] == {
        'n_total_conditions': 2.0, 'target_feat_corr_types': None
    }
    assert bs.best_params['rg_nonfraud'] == {
        'n_total_conditions': 5.0, 'target_feat_corr_types': None
    }
    assert bs.best_params['sf_fraud'] == {'threshold': 0.05789784369353024}
    assert bs.best_params['sf_nonfraud'] == {'threshold': 0.5286917420754508}
    assert bs.cv_results.shape == (10, 13)
    y_pred = bs.predict(
        X={
            'lp_fraud': X,
            'lp_nonfraud': X,
        }
    )
    assert y_pred.mean() == 0.63
    assert f1.fit(y_pred, y) == 0.24657534246575338
    y_pred = bs.fit_predict(
        X={
            'lp_fraud': X,
            'lp_nonfraud': X,
        },
        y={
            'lp_fraud': y,
            'lp_nonfraud': 1-y,
            'rbso': y
        }
    )
    assert y_pred.mean() == 0.63
    assert f1.fit(y_pred, y) == 0.24657534246575338
    # Test fit/predict/fit_predict, sample_weight given and
    # sample_weight_in_val = False
    bs.sample_weight_in_val = False
    bs.fit(
        X={
            'lp_fraud': X,
            'lp_nonfraud': X,
        },
        y={
            'lp_fraud': y,
            'lp_nonfraud': 1-y,
            'rbso': y
        },
        sample_weight={
            'lp_fraud': sample_weight,
            'lp_nonfraud': None,
            'rbso': sample_weight
        }
    )
    assert bs.best_score == 0.21843434343434343
    assert bs.best_index == 5
    assert bs.best_params['cf_fraud']['correlation_reduction_class'].threshold == 0.9
    assert bs.best_params['cf_nonfraud']['correlation_reduction_class'].threshold == 0.95
    assert bs.best_params['rg_fraud'] == {
        'n_total_conditions': 2.0, 'target_feat_corr_types': 'Infer'}
    assert bs.best_params['rg_nonfraud'] == {
        'n_total_conditions': 4.0, 'target_feat_corr_types': None}
    assert bs.best_params['sf_fraud'] == {'threshold': 0.24333593546839893}
    assert bs.best_params['sf_nonfraud'] == {'threshold': 0.2325140282302649}
    assert bs.cv_results.shape == (10, 13)
    y_pred = bs.predict(
        X={
            'lp_fraud': X,
            'lp_nonfraud': X,
        }
    )
    assert y_pred.mean() == 0.27
    assert f1.fit(y_pred, y) == 0.32432432432432434
    y_pred = bs.fit_predict(
        X={
            'lp_fraud': X,
            'lp_nonfraud': X,
        },
        y={
            'lp_fraud': y,
            'lp_nonfraud': 1-y,
            'rbso': y
        },
        sample_weight={
            'lp_fraud': sample_weight,
            'lp_nonfraud': None,
            'rbso': sample_weight
        }
    )
    assert y_pred.mean() == 0.27
    assert f1.fit(y_pred, y) == 0.32432432432432434
    # Test fit/predict/fit_predict, sample_weight given and
    # sample_weight_in_val = True
    bs.sample_weight_in_val = True
    bs.fit(
        X={
            'lp_fraud': X,
            'lp_nonfraud': X,
        },
        y={
            'lp_fraud': y,
            'lp_nonfraud': 1-y,
            'rbso': y
        },
        sample_weight={
            'lp_fraud': sample_weight,
            'lp_nonfraud': None,
            'rbso': sample_weight
        }
    )
    assert bs.best_score == 0.32174934489925
    assert bs.best_index == 3
    assert bs.best_params['cf_fraud']['correlation_reduction_class'].threshold == 0.95
    assert bs.best_params['cf_nonfraud']['correlation_reduction_class'].threshold == 0.9
    assert bs.best_params['rg_fraud'] == {
        'n_total_conditions': 2.0, 'target_feat_corr_types': None}
    assert bs.best_params['rg_nonfraud'] == {
        'n_total_conditions': 5.0, 'target_feat_corr_types': None}
    assert bs.best_params['sf_fraud'] == {'threshold': 0.05789784369353024}
    assert bs.best_params['sf_nonfraud'] == {'threshold': 0.5286917420754508}
    assert bs.cv_results.shape == (10, 13)
    y_pred = bs.predict(
        X={
            'lp_fraud': X,
            'lp_nonfraud': X,
        }
    )
    assert y_pred.mean() == 0.35
    assert f1.fit(y_pred, y) == 0.3111111111111111
    y_pred = bs.fit_predict(
        X={
            'lp_fraud': X,
            'lp_nonfraud': X,
        },
        y={
            'lp_fraud': y,
            'lp_nonfraud': 1-y,
            'rbso': y
        },
        sample_weight={
            'lp_fraud': sample_weight,
            'lp_nonfraud': None,
            'rbso': sample_weight
        }
    )
    assert y_pred.mean() == 0.35
    assert f1.fit(y_pred, y) == 0.3111111111111111


def test_optimise_params(_cv_datasets, _instantiate_lp_and_bs):
    exp_best_params = {
        'n_iter': 13.0, 'n_total_conditions': 3.0, 'target_feat_corr_types': 0
    }
    exp_cv_results = [
        {
            'Params': {
                'rbs': {'n_iter': 13.0},
                'rg_dt': {'n_total_conditions': 3.0, 'target_feat_corr_types': None}
            },
            'rbs__n_iter': 13.0,
            'rg_dt__n_total_conditions': 3.0,
            'rg_dt__target_feat_corr_types': None,
            'FoldIdx': [0, 1],
            'Scores': np.array([0.21621622, 0.38095238]),
            'MeanScore': 0.2985842985842986,
            'StdDevScore': 0.08236808236808235
        }
    ]
    cv_datasets = _cv_datasets
    lp, bs, search_spaces_ = _instantiate_lp_and_bs
    # With sample_weight_in_val = True
    bs.sample_weight_in_val = True
    best_params, cv_results = bs._optimise_params(
        cv_datasets, lp, search_spaces_)
    assert best_params == exp_best_params
    pd.testing.assert_frame_equal(pd.DataFrame(
        cv_results), pd.DataFrame(exp_cv_results))


def test_objective(_cv_datasets, _instantiate_lp_and_bs):
    cv_datasets = _cv_datasets
    lp, bs, _ = _instantiate_lp_and_bs
    params_iter = {
        'rg_dt': {
            'n_total_conditions': 2,
            'target_feat_corr_types': None
        },
        'rbs': {
            'n_iter': 10
        }
    }
    # With sample_weight_in_val = True
    bs.sample_weight_in_val = True
    bs.cv_results = []
    mean_score = bs._objective((params_iter, lp, cv_datasets))
    assert mean_score == -0.2507836990595611


def test_check_search_spaces_type(_instantiate_lp_and_bs):
    _, bs, _ = _instantiate_lp_and_bs
    search_spaces = {
        'ro': {
            'metric': Choice([p.fit, f1.fit])
        },
        'sf': {
            'threshold': UniformFloat(0, 1),
        },
        'rbs': {
            'n_iter': UniformInteger(10, 15)
        }
    }
    # Check no errors
    bs._check_search_spaces_type(search_spaces)
    # Check errors
    with pytest.raises(TypeError, match="`metric` must be a iguanas.space.spaces.UniformFloat or iguanas.space.spaces.UniformInteger or iguanas.space.spaces.Choice. Current type is int."):
        bs._check_search_spaces_type({'ro': {'metric': 1}})


def test_generate_cv_datasets(_create_data, _instantiate_lp_and_bs):
    _, bs, _ = _instantiate_lp_and_bs
    X, y, sample_weight = _create_data
    exp_results = [
        [
            ((50, 4), 306.3046730719136),
            ((50, 4), 277.2591450699665),
            ((50,), 5),
            ((50,), 5),
            ((50,), 550),
            ((50,), 550)
        ],
        [
            ((50, 4), 277.2591450699665),
            ((50, 4), 306.3046730719136),
            ((50,), 5),
            ((50,), 5),
            ((50,), 550),
            ((50,), 550)
        ]
    ]
    # With weights
    cv_datasets_values = bs._generate_cv_datasets(
        X, y, sample_weight, 2).values()
    results = list(
        map(
            lambda x: [(i.shape, i.sum().sum())
                       for i in x], cv_datasets_values
        )
    )
    assert results == exp_results
    # Without weights
    cv_datasets_values = list(bs._generate_cv_datasets(
        X, y, None, 2).values())
    assert cv_datasets_values[0][-2:] == (None, None)
    assert cv_datasets_values[1][-2:] == (None, None)
    # Remove sample_weight values
    cv_datasets_values[0] = cv_datasets_values[0][:-2]
    cv_datasets_values[1] = cv_datasets_values[1][:-2]
    results = list(
        map(
            lambda x: [(i.shape, i.sum().sum())
                       for i in x], cv_datasets_values
        )
    )
    # Remove sample_weight values
    exp_results[0].pop(-2)
    exp_results[0].pop(-1)
    exp_results[1].pop(-2)
    exp_results[1].pop(-1)
    assert results == exp_results


def test_split_df_into_train_and_val(_instantiate_lp_and_bs, _create_data):
    _, bs, _ = _instantiate_lp_and_bs
    X, y, _ = _create_data
    # Test when dataframe given
    X_train, X_val = bs._split_df_into_train_and_val(
        X, list(range(50)), list(range(50, 100))
    )
    assert X_train.sum().sum() == 269.58142001682694
    assert X_val.sum().sum() == 313.9823981250532
    # Test when series given
    y_train, y_val = bs._split_df_into_train_and_val(
        y, list(range(50)), list(range(50, 100))
    )
    assert y_train.sum() == 5
    assert y_val.sum() == 5
    # Test when dict given
    df_dict = {
        'step_A': X[['A', 'B']],
        'step_B': X[['C', 'D']],
        'step_C': None
    }
    df_train, df_val = bs._split_df_into_train_and_val(
        df_dict, list(range(50)), list(range(50, 100))
    )
    assert df_train['step_A'].sum().sum() == 216
    assert df_train['step_A'].columns.tolist() == ['A', 'B']
    assert df_train['step_B'].sum().sum() == 53.581420016826925
    assert df_train['step_B'].columns.tolist() == ['C', 'D']
    assert df_train['step_C'] is None
    assert df_val['step_A'].sum().sum() == 261
    assert df_val['step_A'].columns.tolist() == ['A', 'B']
    assert df_val['step_B'].sum().sum() == 52.98239812505319
    assert df_val['step_B'].columns.tolist() == ['C', 'D']
    assert df_val['step_C'] is None
    # Test error
    with pytest.raises(TypeError, match='`df` must be a Pandas Series/DataFrame or a dict'):
        bs._split_df_into_train_and_val(
            [], list(range(50)), list(range(50, 100))
        )


def test_convert_search_spaces_to_hyperopt(_instantiate_lp_and_bs):
    _, bs, exp_search_spaces = _instantiate_lp_and_bs
    search_spaces_ = bs._convert_search_spaces_to_hyperopt(bs.search_spaces)
    search_spaces_ == exp_search_spaces
    for step_tag, step_params in search_spaces_.items():
        for param, param_value in step_params.items():
            assert type(param_value) == type(
                exp_search_spaces[step_tag][param])


def test_fit_predict_on_fold(_instantiate_lp_and_bs, _cv_datasets,
                             _instantiate_classes):
    lp, bs, _ = _instantiate_lp_and_bs
    cv_datasets = _cv_datasets
    rg_dt, _, _, sf, _, _, rbs = _instantiate_classes
    steps = [
        ('rg_dt', rg_dt),
        ('sf', sf),
        ('rbs', rbs)
    ]
    lp.steps = steps
    params_iter = {
        'rg_dt': {
            'n_total_conditions': 2,
            'target_feat_corr_types': None
        },
        'rbs': {
            'n_iter': 10
        }
    }
    # With sample_weight_in_val=False
    fold_score = bs._fit_predict_on_fold(
        metric=f1.fit,
        error_score='raise',
        datasets=cv_datasets[0],
        pipeline=lp,
        params_iter=params_iter,
        fold_idx=0,
        sample_weight_in_val=False
    )
    assert fold_score == 0.16666666666666666
    # With sample_weight_in_val=True
    fold_score = bs._fit_predict_on_fold(
        metric=f1.fit,
        error_score='raise',
        datasets=cv_datasets[0],
        pipeline=lp,
        params_iter=params_iter,
        fold_idx=0,
        sample_weight_in_val=True
    )
    assert fold_score == 0.25806451612903225
    # Force errors by setting sf filter threshold to 1
    sf.threshold = 1
    with pytest.raises(Exception, match="No rules remaining for: Pipeline parameter set = {'rg_dt': {'n_total_conditions': 2, 'target_feat_corr_types': None}, 'rbs': {'n_iter': 10}}; Fold index = 0."):
        bs._fit_predict_on_fold(
            metric=f1.fit,
            error_score='raise',
            datasets=cv_datasets[0],
            pipeline=lp,
            params_iter=params_iter,
            fold_idx=0,
            sample_weight_in_val=True
        )
    with pytest.warns(UserWarning, match="No rules remaining for: Pipeline parameter set = {'rg_dt': {'n_total_conditions': 2, 'target_feat_corr_types': None}, 'rbs': {'n_iter': 10}}; Fold index = 0. The metric score for this parameter set & fold will be set to 0"):
        fold_score = bs._fit_predict_on_fold(
            metric=f1.fit,
            error_score=0,
            datasets=cv_datasets[0],
            pipeline=lp,
            params_iter=params_iter,
            fold_idx=0,
            sample_weight_in_val=True
        )
        assert fold_score == 0


def test_update_cv_results(_instantiate_lp_and_bs, _params_iter):
    _, bs, _ = _instantiate_lp_and_bs
    params_iter = _params_iter
    exp_cv_results = [
        {
            'Params': {
                'rg_dt': {'n_total_conditions': 2, 'target_feat_corr_types': None},
                'rbs': {'n_iter': 10}
            },
            'rg_dt__n_total_conditions': 2,
            'rg_dt__target_feat_corr_types': None,
            'rbs__n_iter': 10,
            'FoldIdx': [0, 1, 2],
            'Scores': np.array([0.5, 0.25, 0.75]),
            'MeanScore': 0.5,
            'StdDevScore': 0.1
        }
    ]
    cv_results = bs._update_cv_results(
        [], params_iter, [0, 1, 2], np.array([0.5, 0.25, 0.75]), 0.5, 0.1
    )
    np.testing.assert_array_equal(
        cv_results[0]['Scores'], exp_cv_results[0]['Scores']
    )
    cv_results[0].pop('Scores')
    exp_cv_results[0].pop('Scores')
    assert cv_results == exp_cv_results


def test_reformat_best_params(_instantiate_lp_and_bs):
    _, bs, _ = _instantiate_lp_and_bs
    best_params = {
        'rbs__n_iter': 13.0, 'rg_dt__n_total_conditions': 3.0, 'rg_dt__target_feat_corr_types': 0
    }
    bs._reformat_best_params(best_params, bs.search_spaces)


def test_format_cv_results(_instantiate_lp_and_bs):
    _, bs, _ = _instantiate_lp_and_bs
    cv_results = [
        {
            'Params': {'rbs': {'n_iter': 13.0},
                       'rg_dt': {'n_total_conditions': 4.0, 'target_feat_corr_types': None}},
            'rbs__n_iter': 13.0,
            'rg_dt__n_total_conditions': 4.0,
            'rg_dt__target_feat_corr_types': None,
            'FoldIdx': [0, 1, 2],
            'Scores': np.array([0.3, 0.26086957, 0.25]),
            'MeanScore': 0.2702898550724638,
            'StdDevScore': 0.021471786072505872
        },
        {
            'Params': {'rbs': {'n_iter': 13.0},
                       'rg_dt': {'n_total_conditions': 3.0, 'target_feat_corr_types': None}},
            'rbs__n_iter': 13.0,
            'rg_dt__n_total_conditions': 3.0,
            'rg_dt__target_feat_corr_types': None,
            'FoldIdx': [0, 1, 2],
            'Scores': np.array([0.23076923, 0.14285714, 0.1875]),
            'MeanScore': 0.18704212454212454,
            'StdDevScore': 0.035891419937703416
        },
        {
            'Params': {'rbs': {'n_iter': 15.0},
                       'rg_dt': {'n_total_conditions': 3.0, 'target_feat_corr_types': None}},
            'rbs__n_iter': 15.0,
            'rg_dt__n_total_conditions': 3.0,
            'rg_dt__target_feat_corr_types': None,
            'FoldIdx': [0, 1, 2],
            'Scores': np.array([0.16666667, 0.16, 0.07407407]),
            'MeanScore': 0.13358024691358025,
            'StdDevScore': 0.04216514805393193
        }]
    exp_result = pd.DataFrame([
        {
            'Params': {'rbs': {'n_iter': 13.0},
                       'rg_dt': {'n_total_conditions': 4.0, 'target_feat_corr_types': None}},
            'rbs__n_iter': 13.0,
            'rg_dt__n_total_conditions': 4.0,
            'rg_dt__target_feat_corr_types': None,
            'FoldIdx': [0, 1, 2],
            'Scores': np.array([0.3, 0.26086957, 0.25]),
            'MeanScore': 0.2702898550724638,
            'StdDevScore': 0.021471786072505872
        },
        {
            'Params': {'rbs': {'n_iter': 13.0},
                       'rg_dt': {'n_total_conditions': 3.0, 'target_feat_corr_types': None}},
            'rbs__n_iter': 13.0,
            'rg_dt__n_total_conditions': 3.0,
            'rg_dt__target_feat_corr_types': None,
            'FoldIdx': [0, 1, 2],
            'Scores': np.array([0.23076923, 0.14285714, 0.1875]),
            'MeanScore': 0.18704212454212454,
            'StdDevScore': 0.035891419937703416
        },
        {
            'Params': {'rbs': {'n_iter': 15.0},
                       'rg_dt': {'n_total_conditions': 3.0, 'target_feat_corr_types': None}},
            'rbs__n_iter': 15.0,
            'rg_dt__n_total_conditions': 3.0,
            'rg_dt__target_feat_corr_types': None,
            'FoldIdx': [0, 1, 2],
            'Scores': np.array([0.16666667, 0.16, 0.07407407]),
            'MeanScore': 0.13358024691358025,
            'StdDevScore': 0.04216514805393193
        }
    ])
    result = bs._format_cv_results(cv_results)
    pd.testing.assert_frame_equal(result, exp_result)


def test_error(_create_data, _instantiate_classes):
    X, y, _ = _create_data
    _, _, ro, sf, cf, gf, rbs = _instantiate_classes
    search_spaces = {
        'ro': {
            'metric': Choice([p.fit, f1.fit])
        },
        'sf': {
            'threshold': UniformFloat(0, 1),
        },
        'cf': {
            'correlation_reduction_class': Choice(
                [
                    AgglomerativeClusteringReducer(
                        threshold=0.9,
                        strategy='top_down',
                        similarity_function=js.fit,
                        metric=f1.fit
                    ),
                    AgglomerativeClusteringReducer(
                        threshold=0.95,
                        strategy='top_down',
                        similarity_function=js.fit,
                        metric=f1.fit
                    )
                ]
            )
        },
        'gf': {
            'metric': Choice([p.fit, f1.fit])
        },
        'rbs': {
            'n_iter': UniformInteger(10, 15)
        }
    }
    steps = [
        ('ro', ro),
        ('sf', sf),
        ('cf', cf),
        ('gf', gf),
        ('rbs', rbs)
    ]
    lp = LinearPipeline(steps)
    bs = BayesSearchCV(
        pipeline=lp,
        search_spaces=search_spaces,
        metric=f1.fit,
        cv=3,
        n_iter=10,
        error_score='raise',
        verbose=0
    )
    with pytest.raises(Exception, match=re.escape("No rules remaining for: Pipeline parameter set = {'cf': {'correlation_reduction_class': AgglomerativeClusteringReducer(threshold=0.9, strategy=top_down, similarity_function=<bound method JaccardSimilarity.fit of JaccardSimilarity>, metric=<bound method FScore.fit of FScore with beta=1>, print_clustermap=False)}, 'gf': {'metric': <bound method Precision.fit of Precision>}, 'rbs': {'n_iter': 12.0}, 'ro': {'metric': <bound method Precision.fit of Precision>}, 'sf': {'threshold': 0.4860473230215504}}; Fold index = 0.")):
        bs.fit(X, y)
