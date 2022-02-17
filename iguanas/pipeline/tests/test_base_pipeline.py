import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy
from iguanas.pipeline._base_pipeline import DataFrameSizeError
from iguanas.rule_generation import RuleGeneratorDT
from iguanas.rule_optimisation import BayesianOptimiser
from iguanas.rules import Rules
from iguanas.metrics import FScore, JaccardSimilarity, Precision
from iguanas.rule_selection import SimpleFilter, CorrelatedFilter
from iguanas.correlation_reduction import AgglomerativeClusteringReducer
from iguanas.rbs import RBSOptimiser, RBSPipeline
from iguanas.pipeline import LinearPipeline, ParallelPipeline, ClassAccessor

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
    return rg_dt, ro, sf, cf, rbso


def test_fit_predict_linear_and_parallel(_create_data, _instantiate_classes):
    X, y, sample_weight = _create_data
    rg_dt, ro, sf, cf, rbso = _instantiate_classes
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
    # Set up pipeline
    # Fraud ----------
    rg_fraud = deepcopy(rg_dt)
    rg_fraud.rule_name_prefix = 'Fraud'
    sf_fraud = deepcopy(sf)
    cf_fraud = deepcopy(cf)
    pp_fraud = ParallelPipeline(
        steps=[
            ('rg_fraud', rg_fraud),
            ('ro_fraud', deepcopy(ro))
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
            ('ro_nonfraud', deepcopy(ro_nonfraud))
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
    # Tests
    # Test fit/predict/fit_predict, no sample_weight
    lp_overall.fit(
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
    assert len(lp_overall.get_params()['rg_fraud']['rule_names']) == 43
    assert len(lp_overall.get_params()['ro_fraud']['rule_names']) == 4
    assert len(lp_overall.get_params()['sf_fraud']['rules_to_keep']) == 45
    assert len(lp_overall.get_params()['cf_fraud']['rules_to_keep']) == 43
    assert len(lp_overall.get_params()[
        'sf_nonfraud']['rules_to_keep']) == 47
    assert len(lp_overall.get_params()[
        'cf_nonfraud']['rules_to_keep']) == 47
    assert len(lp_overall.get_params()['rg_nonfraud']['rule_names']) == 58
    assert len(lp_overall.get_params()['ro_nonfraud']['rule_names']) == 4
    assert len(lp_overall.get_params()[
        'sf_nonfraud']['rules_to_keep']) == 47
    assert len(lp_overall.get_params()[
        'cf_nonfraud']['rules_to_keep']) == 47
    assert len(lp_overall.get_params()[
        'sf_nonfraud']['rules_to_keep']) == 47
    assert len(lp_overall.get_params()[
        'cf_nonfraud']['rules_to_keep']) == 47
    assert len(lp_overall.get_params()['rbso']['rules_to_keep']) == 47
    assert len(lp_overall.get_params()['rbso']['config']) == 21
    y_pred = lp_overall.predict(
        X={
            'lp_fraud': X,
            'lp_nonfraud': X,
        }
    )
    assert y_pred.mean() == 0.43
    assert f1.fit(y_pred, y) == 0.3018867924528302
    y_pred = lp_overall.fit_predict(
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
    assert y_pred.mean() == 0.43
    assert f1.fit(y_pred, y) == 0.3018867924528302
    # Test fit/predict/fit_predict, sample_weight given
    lp_overall.fit(
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
    assert len(lp_overall.get_params()['rg_fraud']['rule_names']) == 40
    assert len(lp_overall.get_params()['ro_fraud']['rule_names']) == 4
    assert len(lp_overall.get_params()['sf_fraud']['rules_to_keep']) == 44
    assert len(lp_overall.get_params()['cf_fraud']['rules_to_keep']) == 42
    assert len(lp_overall.get_params()[
        'sf_nonfraud']['rules_to_keep']) == 47
    assert len(lp_overall.get_params()[
        'cf_nonfraud']['rules_to_keep']) == 47
    assert len(lp_overall.get_params()['rg_nonfraud']['rule_names']) == 58
    assert len(lp_overall.get_params()['ro_nonfraud']['rule_names']) == 4
    assert len(lp_overall.get_params()[
        'sf_nonfraud']['rules_to_keep']) == 47
    assert len(lp_overall.get_params()[
        'cf_nonfraud']['rules_to_keep']) == 47
    assert len(lp_overall.get_params()[
        'sf_nonfraud']['rules_to_keep']) == 47
    assert len(lp_overall.get_params()[
        'cf_nonfraud']['rules_to_keep']) == 47
    assert len(lp_overall.get_params()['rbso']['rules_to_keep']) == 44
    assert len(lp_overall.get_params()['rbso']['config']) == 22
    y_pred = lp_overall.predict(
        X={
            'lp_fraud': X,
            'lp_nonfraud': X,
        }
    )
    assert y_pred.mean() == 0.29
    assert f1.fit(y_pred, y, sample_weight) == 0.43636363636363634
    y_pred = lp_overall.fit_predict(
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
    assert y_pred.mean() == 0.29
    assert f1.fit(y_pred, y, sample_weight) == 0.43636363636363634


def test_fit_predict_class_accessor_in_list(_create_data, _instantiate_classes):
    X, y, sample_weight = _create_data
    rg_dt, _, _, cf, _ = _instantiate_classes
    rbso = RBSOptimiser(
        pipeline=RBSPipeline(
            config=[
                [
                    0, ClassAccessor(
                        class_tag='cf_nonfraud',
                        class_attribute='rules_to_keep'
                    )
                ],
                [
                    1, ClassAccessor(
                        class_tag='cf_fraud',
                        class_attribute='rules_to_keep'
                    )
                ],
            ],
            final_decision=0,
        ),
        metric=f1.fit,
        n_iter=50
    )
    rg_fraud = deepcopy(rg_dt)
    rg_fraud.rule_name_prefix = 'Fraud'
    rg_nonfraud = deepcopy(rg_dt)
    rg_nonfraud.rule_name_prefix = 'NonFraud'
    lp_fraud = LinearPipeline(
        steps=[
            ('rg_fraud', rg_fraud),
            ('cf_fraud', cf)
        ]
    )
    lp_nonfraud = LinearPipeline(
        steps=[
            ('rg_nonfraud', rg_nonfraud),
            ('cf_nonfraud', cf)
        ]
    )
    pp = ParallelPipeline(
        steps=[
            ('lp_fraud', lp_fraud),
            ('lp_nonfraud', lp_nonfraud)
        ])
    lp = LinearPipeline(
        steps=[
            ('pp', pp),
            ('rbso', rbso)
        ]
    )
    # Without sample_weight
    lp.fit(
        X=X,
        y={
            'lp_fraud': y,
            'lp_nonfraud': 1-y,
            'rbso': y
        }
    )
    y_pred = lp.predict(X)
    assert y_pred.sum() == 33
    y_pred = lp.fit_predict(
        X=X,
        y={
            'lp_fraud': y,
            'lp_nonfraud': 1-y,
            'rbso': y
        }
    )
    assert y_pred.sum() == 33
    # With sample_weight
    lp.fit(
        X=X,
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
    y_pred = lp.predict(X)
    assert y_pred.sum() == 27
    y_pred = lp.fit_predict(
        X=X,
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
    assert y_pred.sum() == 27


def test_fit_predict_class_accessor_in_class(_create_data, _instantiate_classes):
    X, y, sample_weight = _create_data
    # Rule gen
    rg = RuleGeneratorDT(
        metric=f1.fit,
        n_total_conditions=2,
        tree_ensemble=RandomForestClassifier(n_estimators=10, random_state=0),
    )
    # Simple filter
    sf = SimpleFilter(
        threshold=0.1,
        operator='>=',
        metric=f1.fit,
        rules=Rules(
            rule_lambdas=ClassAccessor(
                class_tag='rg',
                class_attribute='rule_lambdas'
            ),
            lambda_kwargs=ClassAccessor(
                class_tag='rg',
                class_attribute='lambda_kwargs'
            )
        )
    )
    # Rule optimiser
    ro = BayesianOptimiser(
        rule_lambdas=ClassAccessor(
            class_tag='sf',
            class_attribute='rule_lambdas'
        ),
        lambda_kwargs=ClassAccessor(
            class_tag='sf',
            class_attribute='lambda_kwargs'
        ),
        metric=f1.fit,
        n_iter=20
    )
    # RBS Optimiser
    rbsp = RBSPipeline(
        config=[],
        final_decision=0,
    )
    rbso = RBSOptimiser(
        pipeline=rbsp,
        pos_pred_rules=ClassAccessor(
            class_tag='ro',
            class_attribute='rule_names'
        ),
        metric=f1.fit,
        n_iter=20
    )
    lp = LinearPipeline(
        steps=[
            ('rg', rg),
            ('sf', sf),
            ('ro', ro),
            ('rbso', rbso)
        ],
        use_init_data=['ro']
    )
    # Without sample_weight
    lp.fit(X=X, y=y)
    y_pred = lp.predict(X=X)
    assert y_pred.sum() == 32
    y_pred = lp.fit_predict(X=X, y=y)
    assert y_pred.sum() == 32
    # With sample_weight
    lp.fit(X=X, y=y, sample_weight=sample_weight)
    y_pred = lp.predict(X=X)
    assert y_pred.sum() == 61
    y_pred = lp.fit_predict(X=X, y=y, sample_weight=sample_weight)
    assert y_pred.sum() == 61


def test_get_params(_instantiate_classes):
    rg_dt, ro, sf, cf, rbso = _instantiate_classes
    # Set up pipeline
    rg_dt.rule_name_prefix = 'Fraud'
    pp = ParallelPipeline(
        steps=[
            ('rg_dt', rg_dt),
            ('ro', ro)
        ],
    )
    lp = LinearPipeline(
        steps=[
            ('pp', pp),
            ('sf', sf),
            ('cf', cf),
            ('rbso', rbso)
        ]
    )
    lp_params = lp.get_params()
    assert len(lp_params) == 6
    assert lp_params['rg_dt'].keys() == rg_dt.__dict__.keys()
    assert lp_params['ro'].keys() == ro.__dict__.keys()
    assert lp_params['pp'].keys() == pp.__dict__.keys()
    assert lp_params['sf'].keys() == sf.__dict__.keys()
    assert lp_params['cf'].keys() == cf.__dict__.keys()
    assert lp_params['rbso'].keys() == rbso.__dict__.keys()


def test_update_kwargs(_instantiate_classes):
    rg_dt, ro, sf, _, _ = _instantiate_classes
    pp = ParallelPipeline(
        steps=[
            ('rg_dt', rg_dt),
            ('ro', ro)
        ]
    )
    lp = LinearPipeline(
        steps=[
            ('pp', pp),
            ('sf', sf)
        ]
    )
    # Test updating step in parent pipeline
    lp._update_kwargs({
        'sf': {
            'threshold': 1
        }
    })
    assert lp.steps[1][1].threshold == 1
    # Test updating step in child pipeline
    lp._update_kwargs({
        'rg_dt': {
            'n_total_conditions': 20
        }
    })
    assert lp.steps[0][1].steps[0][1].n_total_conditions == 20
    # Test error
    with pytest.raises(ValueError, match="Parameter `not_existing` not found in keyword arguments for class in step `rg_dt`"):
        lp._update_kwargs({
            'rg_dt': {
                'not_existing': 20
            }
        })


def test_pipeline_fit(_create_data, _instantiate_classes):
    X, y, sample_weight = _create_data
    rg_dt, _, _, _, _ = _instantiate_classes
    rg_dt._today = '20220127'
    rbso = RBSOptimiser(
        pipeline=RBSPipeline(
            config=[],
            final_decision=0,
        ),
        metric=f1.fit,
        n_iter=10,
        pos_pred_rules=ClassAccessor(
            class_tag='rg_dt',
            class_attribute='rule_names'
        )
    )
    lp = LinearPipeline(steps=[
        ('rg_dt', rg_dt),
        ('rbso', rbso)
    ])
    # No ClassAccessor, datasets as pandas objects
    lp._pipeline_fit(
        step_tag='rg_dt', step=rg_dt, X=X, y=y, sample_weight=sample_weight
    )
    X_rules = rg_dt.transform(X=X)
    assert X_rules.sum().sum() == 326
    # No ClassAccessor, datasets as dicts
    lp._pipeline_fit(
        step_tag='rg_dt', step=rg_dt, X={'rg_dt': X}, y={'rg_dt': y},
        sample_weight={'rg_dt': sample_weight}
    )
    X_rules = rg_dt.transform(X=X)
    assert X_rules.sum().sum() == 326
    # ClassAccessor, datasets as pandas objects
    lp._pipeline_fit(
        step_tag='rbso', step=rbso, X=X_rules, y=y, sample_weight=sample_weight
    )
    assert rbso.rules_to_keep == [
        'RGDT_Rule_20220127_11', 'RGDT_Rule_20220127_37',
        'RGDT_Rule_20220127_33', 'RGDT_Rule_20220127_13',
        'RGDT_Rule_20220127_32', 'RGDT_Rule_20220127_20',
        'RGDT_Rule_20220127_26', 'RGDT_Rule_20220127_27',
        'RGDT_Rule_20220127_4', 'RGDT_Rule_20220127_9',
        'RGDT_Rule_20220127_23', 'RGDT_Rule_20220127_8',
        'RGDT_Rule_20220127_2', 'RGDT_Rule_20220127_17',
        'RGDT_Rule_20220127_39', 'RGDT_Rule_20220127_0',
        'RGDT_Rule_20220127_15', 'RGDT_Rule_20220127_1',
        'RGDT_Rule_20220127_38', 'RGDT_Rule_20220127_6',
        'RGDT_Rule_20220127_12', 'RGDT_Rule_20220127_14',
    ]
    # No need to check ClassAccessor with datasets as dicts as classes that use
    # it will not be first in the pipeline


def test_check_accessor(_instantiate_classes):
    _, _, sf, _, rbso = _instantiate_classes
    sf.rules_to_keep = ['Rule1']
    ca = ClassAccessor('sf', 'rules_to_keep')
    # Check when ClassAccessor is parameter
    rbso = RBSOptimiser(
        pipeline=RBSPipeline(
            config=[],
            final_decision=0,
        ),
        metric=f1.fit,
        n_iter=10,
        pos_pred_rules=ca
    )
    steps = [
        ('sf', sf),
        ('rbs', rbso)
    ]
    lp = LinearPipeline(steps)
    for _, step in lp.steps:
        lp._check_accessor(step)
    assert rbso.pos_pred_rules == ['Rule1']
    # Check when ClassAccessor is within list (that is a parameter)
    ca = ClassAccessor(
        class_tag='sf',
        class_attribute='rules_to_keep'
    )
    rbso = RBSOptimiser(
        pipeline=RBSPipeline(
            config=[
                [
                    0, ca
                ]
            ],
            final_decision=0,
        ),
        metric=f1.fit,
        n_iter=10
    )
    steps = [
        ('sf', sf),
        ('rbs', rbso)
    ]
    lp = LinearPipeline(steps)
    for _, step in lp.steps:
        lp._check_accessor(step)
    assert rbso.config == [[0, ['Rule1']]]


def test_check_accessor_exception(_instantiate_classes):
    _, _, sf, _, _ = _instantiate_classes
    rbso = RBSOptimiser(
        pipeline=RBSPipeline(
            config=[
                (
                    0, ClassAccessor(
                        class_tag='sf',
                        class_attribute='rules_to_keep'
                    )
                )
            ],
            final_decision=0,
        ),
        metric=f1.fit,
        n_iter=10
    )
    steps = [
        ('sf', sf),
        ('rbs', rbso)
    ]
    lp = LinearPipeline(steps)
    with pytest.raises(TypeError, match='`ClassAccessor` object must be within a mutable iterable.'):
        for _, step in lp.steps:
            lp._check_accessor(step)


def test_exception_if_no_cols_in_X():
    X = pd.DataFrame([])
    lp = LinearPipeline([])
    with pytest.raises(DataFrameSizeError, match='`X` has been reduced to zero columns after the `rg` step in the pipeline.'):
        lp._exception_if_no_cols_in_X(X, 'rg')
