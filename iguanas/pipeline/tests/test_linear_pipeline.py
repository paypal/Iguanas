import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from iguanas.pipeline._base_pipeline import DataFrameSizeError

from iguanas.rule_generation import RuleGeneratorDT, RuleGeneratorOpt
from iguanas.rule_optimisation import BayesianOptimiser
from iguanas.rules import Rules
from iguanas.metrics import FScore, JaccardSimilarity, Precision
from iguanas.rule_selection import SimpleFilter, CorrelatedFilter, GreedyFilter
from iguanas.correlation_reduction import AgglomerativeClusteringReducer
from iguanas.rbs import RBSOptimiser, RBSPipeline
from iguanas.pipeline import LinearPipeline, ClassAccessor

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


def test_fit_predict_rule_gen_dt(_create_data, _instantiate_classes):
    X, y, sample_weight = _create_data
    rg_dt, _, _, sf, cf, gf, rbs = _instantiate_classes
    steps = [
        ('rg_dt', rg_dt),
        ('sf', sf),
        ('cf', cf),
        ('gf', gf),
        ('rbs', rbs)
    ]
    rg_dt._today = '20211220'
    lp = LinearPipeline(steps)
    # Test fit/predict/fit_predict, no sample_weight
    lp.fit(X, y)
    assert len(lp.get_params()['sf']['rules_to_keep']) == 43
    assert len(lp.get_params()['cf']['rules_to_keep']) == 41
    assert len(lp.get_params()['gf']['rules_to_keep']) == 10
    assert lp.get_params()['rbs']['rules_to_keep'] == [
        'RGDT_Rule_20211220_26', 'RGDT_Rule_20211220_6', 'RGDT_Rule_20211220_11',
        'RGDT_Rule_20211220_41', 'RGDT_Rule_20211220_36',
        'RGDT_Rule_20211220_40', 'RGDT_Rule_20211220_5'
    ]
    y_pred = lp.predict(X)
    assert y_pred.mean() == 0.13
    assert f1.fit(y_pred, y) == 0.7826086956521738
    y_pred = lp.fit_predict(X, y)
    assert y_pred.mean() == 0.13
    assert f1.fit(y_pred, y) == 0.7826086956521738
    # Test fit/predict/fit_predict, sample_weight given
    lp.fit(X, y, sample_weight)
    assert len(lp.get_params()['sf']['rules_to_keep']) == 40
    assert len(lp.get_params()['cf']['rules_to_keep']) == 38
    assert len(lp.get_params()['gf']['rules_to_keep']) == 10
    assert lp.get_params()['rbs']['rules_to_keep'] == [
        'RGDT_Rule_20211220_25', 'RGDT_Rule_20211220_8',
        'RGDT_Rule_20211220_11', 'RGDT_Rule_20211220_38',
        'RGDT_Rule_20211220_36', 'RGDT_Rule_20211220_37',
        'RGDT_Rule_20211220_7'
    ]
    y_pred = lp.predict(X)
    assert y_pred.mean() == 0.1
    assert f1.fit(y_pred, y, sample_weight) == 0.8421052631578948
    y_pred = lp.fit_predict(X, y, sample_weight)
    assert y_pred.mean() == 0.1
    assert f1.fit(y_pred, y, sample_weight) == 0.8421052631578948


def test_fit_predict_rule_gen_opt(_create_data, _instantiate_classes):
    X, y, sample_weight = _create_data
    _, rg_opt, _, sf, cf, gf, rbs = _instantiate_classes
    steps = [
        ('rg_opt', rg_opt),
        ('sf', sf),
        ('cf', cf),
        ('gf', gf),
        ('rbs', rbs)
    ]
    rg_opt._today = '20211220'
    lp = LinearPipeline(steps)
    # Test fit/predict/fit_predict, no sample_weight
    lp.fit(X, y)
    assert len(lp.get_params()['sf']['rules_to_keep']) == 26
    assert len(lp.get_params()['cf']['rules_to_keep']) == 26
    assert len(lp.get_params()['gf']['rules_to_keep']) == 3
    assert lp.get_params()['rbs']['rules_to_keep'] == [
        'RGO_Rule_20211220_25', 'RGO_Rule_20211220_27', 'RGO_Rule_20211220_41'
    ]
    y_pred = lp.predict(X)
    assert y_pred.mean() == 0.11
    assert f1.fit(y_pred, y) == 0.5714285714285713
    y_pred = lp.fit_predict(X, y)
    assert y_pred.mean() == 0.11
    assert f1.fit(y_pred, y) == 0.5714285714285713
    # Test fit/predict/fit_predict, sample_weight given
    lp.fit(X, y, sample_weight)
    assert len(lp.get_params()['sf']['rules_to_keep']) == 26
    assert len(lp.get_params()['cf']['rules_to_keep']) == 26
    assert len(lp.get_params()['gf']['rules_to_keep']) == 5
    assert lp.get_params()['rbs']['rules_to_keep'] == [
        'RGO_Rule_20211220_31', 'RGO_Rule_20211220_24', 'RGO_Rule_20211220_34'
    ]
    y_pred = lp.predict(X)
    assert y_pred.mean() == 0.13
    assert f1.fit(y_pred, y, sample_weight) == 0.6153846153846154
    y_pred = lp.fit_predict(X, y, sample_weight)
    assert y_pred.mean() == 0.13
    assert f1.fit(y_pred, y, sample_weight) == 0.6153846153846154


def test_fit_predict_rule_opt(_create_data, _instantiate_classes):
    X, y, sample_weight = _create_data
    _, _, ro, sf, cf, gf, rbs = _instantiate_classes
    steps = [
        ('ro', ro),
        ('sf', sf),
        ('cf', cf),
        ('gf', gf),
        ('rbs', rbs)
    ]
    lp = LinearPipeline(steps)
    # Test fit/predict/fit_predict, no sample_weight
    lp.fit(X, y)
    assert len(lp.get_params()['sf']['rules_to_keep']) == 2
    assert len(lp.get_params()['cf']['rules_to_keep']) == 2
    assert len(lp.get_params()['gf']['rules_to_keep']) == 1
    assert lp.get_params()['rbs']['rules_to_keep'] == ['Rule4']
    y_pred = lp.predict(X)
    assert y_pred.mean() == 0.95
    assert f1.fit(y_pred, y) == 0.1904761904761905
    y_pred = lp.fit_predict(X, y)
    assert y_pred.mean() == 0.95
    assert f1.fit(y_pred, y) == 0.1904761904761905
    # Test fit/predict/fit_predict, sample_weight given
    lp.fit(X, y, sample_weight)
    assert len(lp.get_params()['sf']['rules_to_keep']) == 4
    assert len(lp.get_params()['cf']['rules_to_keep']) == 4
    assert len(lp.get_params()['gf']['rules_to_keep']) == 1
    assert lp.get_params()['rbs']['rules_to_keep'] == ['Rule4']
    y_pred = lp.predict(X)
    assert y_pred.mean() == 0.95
    assert f1.fit(y_pred, y, sample_weight) == 0.32
    y_pred = lp.fit_predict(X, y, sample_weight)
    assert y_pred.mean() == 0.95
    assert f1.fit(y_pred, y, sample_weight) == 0.32


def test_fit_transform(_create_data, _instantiate_classes):
    X, y, sample_weight = _create_data
    _, _, ro, sf, cf, gf, _ = _instantiate_classes
    steps = [
        ('ro', ro),
        ('sf', sf),
        ('cf', cf),
        ('gf', gf),
    ]
    lp = LinearPipeline(steps)
    # Test fit_transform, no sample_weight
    X_rules = lp.fit_transform(X, y)
    assert X_rules.columns.tolist() == ['Rule4']
    np.testing.assert_equal(X_rules.mean().values, np.array([0.95]))
    assert len(lp.get_params()['sf']['rules_to_keep']) == 2
    assert len(lp.get_params()['cf']['rules_to_keep']) == 2
    assert len(lp.get_params()['gf']['rules_to_keep']) == 1
    # Test fit_transform, sample_weight given
    X_rules = lp.fit_transform(X, y, sample_weight)
    assert X_rules.columns.tolist() == ['Rule4']
    np.testing.assert_equal(X_rules.mean().values, np.array([0.95]))
    assert len(lp.get_params()['sf']['rules_to_keep']) == 4
    assert len(lp.get_params()['cf']['rules_to_keep']) == 4
    assert len(lp.get_params()['gf']['rules_to_keep']) == 1


def test_get_params(_instantiate_classes):
    _, _, _, sf, _, _, _ = _instantiate_classes
    steps = [
        ('sf', sf),
    ]
    lp = LinearPipeline(steps)
    params = lp.get_params()
    assert params['sf']['threshold'] == 0.05
    assert params['sf']['operator'] == '>='
    assert str(params['sf']['metric']
               ) == '<bound method FScore.fit of FScore with beta=1>'
    assert params['sf']['rules_to_keep'] == []


def test_check_accessor(_instantiate_classes):
    _, _, _, sf, _, _, rbs = _instantiate_classes
    ca = ClassAccessor('sf', 'rules_to_keep')
    sf.rules_to_keep = ['Rule1']
    rbs.pos_pred_rules = ca
    steps = [
        ('sf', sf),
        ('rbs', rbs)
    ]
    lp = LinearPipeline(steps)
    rbs = lp._check_accessor(rbs, steps)
    assert rbs.pos_pred_rules == ['Rule1']


def test_exception_if_no_cols_in_X():
    X = pd.DataFrame([])
    lp = LinearPipeline([])
    with pytest.raises(DataFrameSizeError, match='`X` has been reduced to zero columns after the `rg` step in the pipeline.'):
        lp._exception_if_no_cols_in_X(X, 'rg')
