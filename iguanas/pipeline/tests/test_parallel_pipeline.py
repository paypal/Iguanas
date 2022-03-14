
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy
from iguanas.rule_generation import RuleGeneratorDT
from iguanas.rule_optimisation import BayesianOptimiser
from iguanas.rules import Rules
from iguanas.metrics import FScore, JaccardSimilarity, Precision
from iguanas.rule_selection import SimpleFilter, CorrelatedFilter
from iguanas.correlation_reduction import AgglomerativeClusteringReducer
from iguanas.rbs import RBSOptimiser, RBSPipeline
from iguanas.pipeline import ParallelPipeline, ClassAccessor, LinearPipeline
from iguanas.warnings import NoRulesWarning

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
    return rg_dt, ro, sf, cf, rbso


def test_fit_transform(_create_data, _instantiate_classes):
    X, y, sample_weight = _create_data
    rg_dt, _, _, _, _ = _instantiate_classes
    rg_fraud = deepcopy(rg_dt)
    rg_fraud.rule_name_prefix = 'Fraud'
    rg_nonfraud = deepcopy(rg_dt)
    rg_nonfraud.rule_name_prefix = 'NonFraud'
    pp = ParallelPipeline(
        steps=[
            ('rg_fraud', rg_fraud),
            ('rg_nonfraud', rg_nonfraud)
        ],
        verbose=1  # Set verbose=1
    )
    # Test fit_transform
    X_rules = pp.fit_transform(
        X={
            'rg_fraud': X[['A', 'B']],
            'rg_nonfraud': X[['C', 'D']]
        },
        y={
            'rg_fraud': y,
            'rg_nonfraud': 1-y
        })
    assert X_rules.sum().sum() == 845
    assert X_rules.shape == (100, 56)
    assert len(pp.rule_names) == 56
    assert len(pp.rules.rule_strings) == 56
    # Test transform
    X_rules = pp.transform(
        X={
            'rg_fraud': X[['A', 'B']],
            'rg_nonfraud': X[['C', 'D']]
        }
    )
    assert X_rules.sum().sum() == 845
    assert X_rules.shape == (100, 56)
    assert len(pp.rule_names) == 56
    assert len(pp.rules.rule_strings) == 56
    # Test fit_transform with sample_weight
    pp.verbose = 2  # Set verbose=2
    X_rules = pp.fit_transform(
        X={
            'rg_fraud': X[['A', 'B']],
            'rg_nonfraud': X[['C', 'D']]
        },
        y={
            'rg_fraud': y,
            'rg_nonfraud': 1-y
        },
        sample_weight={
            'rg_fraud': sample_weight,
            'rg_nonfraud': None
        }
    )
    assert X_rules.sum().sum() == 922
    assert X_rules.shape == (100, 59)
    assert len(pp.rule_names) == 59
    assert len(pp.rules.rule_strings) == 59
    # Test transform with sample_weight
    X_rules = pp.transform(
        X={
            'rg_fraud': X[['A', 'B']],
            'rg_nonfraud': X[['C', 'D']]
        }
    )
    assert X_rules.sum().sum() == 922
    assert X_rules.shape == (100, 59)
    assert len(pp.rule_names) == 59
    assert len(pp.rules.rule_strings) == 59


def test_fit_transform_no_rules(_create_data, _instantiate_classes):
    X, y, _ = _create_data
    _, _, sf, _, _ = _instantiate_classes
    X = X[['A']]
    pp = ParallelPipeline(
        steps=[
            ('sf1', sf),
            ('sf2', sf)
        ],
    )
    X_rules = pp.fit_transform(X, y)
    assert X_rules.sum().sum() == 112
    assert X_rules.shape == (100, 2)
    assert len(pp.rule_names) == 2
    assert pp.rules.rule_strings == {}


def test_try_except(_create_data, _instantiate_classes):
    X, y, _ = _create_data
    rg_dt, _, sf, _, _ = _instantiate_classes
    rg_dt1 = deepcopy(rg_dt)
    sf1 = deepcopy(sf)
    X1 = pd.DataFrame({'X': [0, 0, 0]})
    y1 = pd.Series([0, 1, 0])
    pp = ParallelPipeline(
        steps=[
            ('rg_dt', rg_dt),
            ('rg_dt1', rg_dt1)
        ]
    )
    # With generator classes
    with pytest.warns(NoRulesWarning, match='No rules remain in step `rg_dt1` as it raised the following error: "No rules could be generated. Try changing the class parameters."'):
        X_rules = pp.fit_transform(
            X={
                'rg_dt': X,
                'rg_dt1': X1
            },
            y={
                'rg_dt': y,
                'rg_dt1': y1
            },
        )
        assert X_rules.sum().sum() == 352
    with pytest.warns(NoRulesWarning, match='No rules present in step `rg_dt1` - `transform` method cannot be applied for this step.'):
        X_rules = pp.transform(
            X={
                'rg_dt': X,
                'rg_dt1': X1
            }
        )
        assert X_rules.sum().sum() == 352
    # With LinearPipeline classes
    lp = LinearPipeline(
        steps=[
            ('rg_dt', rg_dt),
            ('sf', sf)
        ]
    )
    lp1 = LinearPipeline(
        steps=[
            ('rg_dt1', rg_dt1),
            ('sf1', sf1)
        ]
    )
    pp = ParallelPipeline(
        steps=[
            ('lp', lp),
            ('lp1', lp1)
        ]
    )
    with pytest.warns(NoRulesWarning, match='No rules remain in step `lp1` as it raised the following error: "No rules could be generated. Try changing the class parameters."'):
        X_rules = pp.fit_transform(
            X={
                'lp': X,
                'lp1': X1
            },
            y={
                'lp': y,
                'lp1': y1
            },
        )
        assert X_rules.sum().sum() == 352
    with pytest.warns(NoRulesWarning, match='No rules present in step `lp1` - `transform` method cannot be applied for this step.'):
        X_rules = pp.transform(
            X={
                'lp': X,
                'lp1': X1
            }
        )
        assert X_rules.sum().sum() == 352
    # No rules remaining
    sf.threshold = 1
    lp = LinearPipeline(
        steps=[
            ('rg_dt', rg_dt),
            ('sf', sf)
        ]
    )
    pp = ParallelPipeline(
        steps=[
            ('lp', lp),
            ('lp1', lp1)
        ]
    )
    with pytest.warns(NoRulesWarning) as warnings:
        X_rules = pp.fit_transform(
            X={
                'lp': X,
                'lp1': X1
            },
            y={
                'lp': y,
                'lp1': y1
            },
        )
        pd.testing.assert_frame_equal(X_rules, pd.DataFrame())
        assert pp.rules.rule_strings == {}
    warnings = [w.message.args[0] for w in warnings]
    assert 'No rules remain in step `lp` as it raised the following error: "`X` has been reduced to zero columns after the `sf` step in the pipeline."' in warnings
    assert 'No rules remain in step `lp1` as it raised the following error: "No rules could be generated. Try changing the class parameters."' in warnings
    with pytest.warns(NoRulesWarning) as warnings:
        X_rules = pp.transform(
            X={
                'lp': X,
                'lp1': X1
            }
        )
        pd.testing.assert_frame_equal(X_rules, pd.DataFrame())
        assert pp.rules.rule_strings == {}
    warnings = [w.message.args[0] for w in warnings]
    assert 'No rules present in step `lp` - `transform` method cannot be applied for this step.' in warnings
    assert 'No rules present in step `lp1` - `transform` method cannot be applied for this step.' in warnings
    # Test exception raised
    pp = ParallelPipeline(steps=[('f1', f1)])
    pp.steps_ = pp.steps
    with pytest.raises(AttributeError, match="'FScore' object has no attribute 'transform'"):
        pp.transform(X=X)
