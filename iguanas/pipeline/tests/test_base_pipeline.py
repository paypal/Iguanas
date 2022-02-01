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
    param_classes = [
        "<class 'sklearn.ensemble._forest.RandomForestClassifier'>",
        "<class 'method'>",
        "<class 'iguanas.correlation_reduction.agglomerative_clustering_reducer.AgglomerativeClusteringReducer'>",
        "<class 'iguanas.pipeline.class_accessor.ClassAccessor'>"
    ]
    for step_tag, step in (('rg_dt', rg_dt), ('ro', ro), ('sf', sf), ('cf', cf), ('rbso', rbso)):
        for param, param_value in step.__dict__.items():
            if str(type(param_value)) in param_classes:
                assert str(type(lp_params[step_tag][param])) == str(
                    type(step.__dict__[param]))
            else:
                assert lp_params[step_tag][param] == step.__dict__[param]


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
    ca = ClassAccessor('sf', 'rules_to_keep')
    sf.rules_to_keep = ['Rule1']
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


def test_exception_if_no_cols_in_X():
    X = pd.DataFrame([])
    lp = LinearPipeline([])
    with pytest.raises(DataFrameSizeError, match='`X` has been reduced to zero columns after the `rg` step in the pipeline.'):
        lp._exception_if_no_cols_in_X(X, 'rg')
