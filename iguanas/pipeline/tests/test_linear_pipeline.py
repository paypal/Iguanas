import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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


def test_fit_predict_use_init_data(_create_data, _instantiate_classes):
    X, y, sample_weight = _create_data
    rg_dt, _, _, _, _, _, _ = _instantiate_classes
    ro = BayesianOptimiser(
        rule_lambdas=ClassAccessor(
            class_tag='rg_dt',
            class_attribute='rule_lambdas'
        ),
        lambda_kwargs=ClassAccessor(
            class_tag='rg_dt',
            class_attribute='lambda_kwargs'
        ),
        metric=f1.fit,
        n_iter=5
    )
    rg_dt._today = '20220201'
    expected_rule_strings = {
        'RGDT_Rule_20220201_0': "(X['A']==False)&(X['B']<=6)&(X['C']>0.6177501837002377)",
        'RGDT_Rule_20220201_1': "(X['A']==False)&(X['B']<=2)&(X['D']==False)",
        'RGDT_Rule_20220201_2': "(X['A']==False)&(X['B']<=2)&(X['B']>=2)&(X['D']==False)",
        'RGDT_Rule_20220201_3': "(X['A']==False)&(X['B']<=4)&(X['C']>0.84748)&(X['D']==True)",
        'RGDT_Rule_20220201_4': "(X['A']==False)&(X['B']<=3)&(X['B']>=0)&(X['C']<=0.6610314524532597)",
        'RGDT_Rule_20220201_5': "(X['A']==False)&(X['B']<=6)&(X['B']>=2)&(X['C']>0.82191)",
        'RGDT_Rule_20220201_6': "(X['A']==False)&(X['B']<=7)&(X['C']>0.85766)&(X['D']==False)",
        'RGDT_Rule_20220201_7': "(X['A']==False)&(X['B']<=7)&(X['B']>=1)&(X['D']==False)",
        'RGDT_Rule_20220201_8': "(X['A']==False)&(X['B']>=1)&(X['C']>0.84748)&(X['D']==True)",
        'RGDT_Rule_20220201_9': "(X['A']==False)&(X['B']>=2)&(X['D']==False)",
        'RGDT_Rule_20220201_10': "(X['A']==False)&(X['C']<=1.1179668118672983)&(X['C']>0.3866744253009945)",
        'RGDT_Rule_20220201_11': "(X['A']==False)&(X['C']<=1.1179668118672983)&(X['C']>0.3866744253009945)",
        'RGDT_Rule_20220201_12': "(X['A']==False)&(X['C']<=0.65965)&(X['D']==False)",
        'RGDT_Rule_20220201_13': "(X['A']==False)&(X['C']<=0.663337811583689)&(X['D']==False)",
        'RGDT_Rule_20220201_14': "(X['A']==False)&(X['C']<=1.1179668118672983)&(X['C']>0.3866744253009945)",
        'RGDT_Rule_20220201_15': "(X['A']==False)&(X['C']<=1.1179668118672983)&(X['C']>0.3866744253009945)&(X['D']==False)",
        'RGDT_Rule_20220201_16': "(X['A']==False)&(X['C']<=1.1179668118672983)&(X['C']>0.3866744253009945)&(X['D']==False)",
        'RGDT_Rule_20220201_17': "(X['A']==False)&(X['C']>0.80832)&(X['D']==True)",
        'RGDT_Rule_20220201_18': "(X['A']==False)&(X['C']>0.6177501837002377)&(X['D']==False)",
        'RGDT_Rule_20220201_19': "(X['A']==False)&(X['C']>0.6177501837002377)",
        'RGDT_Rule_20220201_20': "(X['A']==False)&(X['C']>0.6177501837002377)&(X['D']==False)",
        'RGDT_Rule_20220201_21': "(X['A']==False)&(X['C']>0.6177501837002377)&(X['D']==False)",
        'RGDT_Rule_20220201_22': "(X['A']==True)&(X['B']<=6)&(X['C']<=0.663337811583689)",
        'RGDT_Rule_20220201_23': "(X['A']==True)&(X['B']<=6)&(X['C']<=0.50532)",
        'RGDT_Rule_20220201_24': "(X['A']==True)&(X['B']<=7)&(X['C']>0.2740656256979518)&(X['D']==False)",
        'RGDT_Rule_20220201_25': "(X['A']==True)&(X['B']>=0)&(X['C']<=0.6610314524532597)&(X['D']==False)",
        'RGDT_Rule_20220201_26': "(X['A']==True)&(X['C']<=0.5315)&(X['C']>0.46073)&(X['D']==False)",
        'RGDT_Rule_20220201_27': "(X['A']==True)&(X['C']<=0.663337811583689)&(X['D']==False)",
        'RGDT_Rule_20220201_28': "(X['A']==True)&(X['C']<=0.9194077847929631)&(X['C']>0.2740656256979518)&(X['D']==False)",
        'RGDT_Rule_20220201_29': "(X['A']==True)&(X['C']<=0.65082)&(X['C']>0.63671)&(X['D']==False)",
        'RGDT_Rule_20220201_30': "(X['B']<=7)&(X['C']>0.2740656256979518)",
        'RGDT_Rule_20220201_31': "(X['B']<=5)&(X['C']<=0.51004)&(X['C']>0.45317)&(X['D']==False)",
        'RGDT_Rule_20220201_32': "(X['B']<=7)&(X['B']>=2)&(X['C']<=0.52794)&(X['D']==False)",
        'RGDT_Rule_20220201_33': "(X['B']<=7)&(X['B']>=2)&(X['C']>0.52794)&(X['D']==False)",
        'RGDT_Rule_20220201_34': "(X['B']<=3)&(X['B']>=0)&(X['C']<=0.6610314524532597)&(X['D']==False)",
        'RGDT_Rule_20220201_35': "(X['B']<=6)&(X['C']<=0.663337811583689)&(X['D']==False)",
        'RGDT_Rule_20220201_36': "(X['B']<=8)&(X['C']<=0.65965)&(X['C']>0.62615)&(X['D']==False)",
        'RGDT_Rule_20220201_37': "(X['B']>=0)&(X['C']<=0.6610314524532597)&(X['D']==False)",
        'RGDT_Rule_20220201_38': "(X['B']>=4)&(X['C']<=0.9194077847929631)&(X['C']>0.2740656256979518)&(X['D']==False)",
        'RGDT_Rule_20220201_39': "(X['C']<=0.663337811583689)",
        'RGDT_Rule_20220201_40': "(X['C']<=1.1179668118672983)&(X['C']>0.3866744253009945)",
        'RGDT_Rule_20220201_41': "(X['C']<=0.65965)&(X['C']>0.64871)",
        'RGDT_Rule_20220201_42': "(X['C']<=0.87533)&(X['C']>0.84487)"
    }
    expected_rule_strings_weights = {
        'RGDT_Rule_20220201_0': "(X['A']==False)&(X['B']<=6)&(X['C']>0.6177501837002377)",
        'RGDT_Rule_20220201_1': "(X['A']==False)&(X['B']<=2)&(X['D']==False)",
        'RGDT_Rule_20220201_2': "(X['A']==False)&(X['B']<=7)&(X['B']>=1)&(X['D']==False)",
        'RGDT_Rule_20220201_3': "(X['A']==False)&(X['B']<=4)&(X['C']>0.80832)&(X['D']==True)",
        'RGDT_Rule_20220201_4': "(X['A']==False)&(X['B']<=4)&(X['C']>0.84748)&(X['D']==True)",
        'RGDT_Rule_20220201_5': "(X['A']==False)&(X['B']<=3)&(X['B']>=0)&(X['C']<=0.6610314524532597)",
        'RGDT_Rule_20220201_6': "(X['A']==False)&(X['B']<=6)&(X['B']>=2)&(X['C']>0.82191)",
        'RGDT_Rule_20220201_7': "(X['A']==False)&(X['B']<=7)&(X['C']>0.85766)&(X['D']==False)",
        'RGDT_Rule_20220201_8': "(X['A']==False)&(X['B']<=7)&(X['C']>0.2740656256979518)&(X['D']==False)",
        'RGDT_Rule_20220201_9': "(X['A']==False)&(X['B']>=2)&(X['D']==False)",
        'RGDT_Rule_20220201_10': "(X['A']==False)&(X['C']<=1.1179668118672983)&(X['C']>0.3866744253009945)",
        'RGDT_Rule_20220201_11': "(X['A']==False)&(X['C']<=1.1179668118672983)&(X['C']>0.3866744253009945)",
        'RGDT_Rule_20220201_12': "(X['A']==False)&(X['C']<=0.65965)&(X['D']==False)",
        'RGDT_Rule_20220201_13': "(X['A']==False)&(X['C']<=0.66179)&(X['D']==False)",
        'RGDT_Rule_20220201_14': "(X['A']==False)&(X['C']<=1.1179668118672983)&(X['C']>0.3866744253009945)",
        'RGDT_Rule_20220201_15': "(X['A']==False)&(X['C']<=1.1179668118672983)&(X['C']>0.3866744253009945)&(X['D']==False)",
        'RGDT_Rule_20220201_16': "(X['A']==False)&(X['C']>0.3866744253009945)&(X['D']==False)",
        'RGDT_Rule_20220201_17': "(X['A']==False)&(X['C']>0.6177501837002377)",
        'RGDT_Rule_20220201_18': "(X['A']==False)&(X['C']>0.3866744253009945)&(X['D']==False)",
        'RGDT_Rule_20220201_19': "(X['A']==True)&(X['B']<=6)&(X['C']<=0.663337811583689)",
        'RGDT_Rule_20220201_20': "(X['A']==True)&(X['B']<=7)&(X['B']>=1)&(X['D']==False)",
        'RGDT_Rule_20220201_21': "(X['A']==True)&(X['B']<=6)&(X['C']<=0.50532)",
        'RGDT_Rule_20220201_22': "(X['A']==True)&(X['B']<=7)&(X['C']>0.2740656256979518)&(X['D']==False)",
        'RGDT_Rule_20220201_23': "(X['A']==True)&(X['B']>=0)&(X['C']<=0.6610314524532597)&(X['D']==False)",
        'RGDT_Rule_20220201_24': "(X['A']==True)&(X['C']<=0.663337811583689)&(X['D']==False)",
        'RGDT_Rule_20220201_25': "(X['A']==True)&(X['C']<=0.9194077847929631)&(X['C']>0.2740656256979518)&(X['D']==False)",
        'RGDT_Rule_20220201_26': "(X['B']<=7)&(X['C']>0.2740656256979518)",
        'RGDT_Rule_20220201_27': "(X['B']<=8)&(X['C']<=1.1179668118672983)&(X['C']>0.3866744253009945)&(X['D']==False)",
        'RGDT_Rule_20220201_28': "(X['B']<=3)&(X['B']>=0)&(X['C']<=0.6610314524532597)&(X['D']==False)",
        'RGDT_Rule_20220201_29': "(X['B']<=7)&(X['B']>=2)&(X['C']>0.52794)&(X['D']==False)",
        'RGDT_Rule_20220201_30': "(X['B']<=8)&(X['B']>=2)&(X['C']<=0.84487)&(X['D']==False)",
        'RGDT_Rule_20220201_31': "(X['B']<=6)&(X['C']<=0.663337811583689)&(X['D']==False)",
        'RGDT_Rule_20220201_32': "(X['B']<=8)&(X['C']<=1.1179668118672983)&(X['C']>0.3866744253009945)&(X['D']==False)",
        'RGDT_Rule_20220201_33': "(X['B']>=0)&(X['C']<=0.6610314524532597)&(X['D']==False)",
        'RGDT_Rule_20220201_34': "(X['B']>=4)&(X['C']<=0.9194077847929631)&(X['C']>0.2740656256979518)&(X['D']==False)",
        'RGDT_Rule_20220201_35': "(X['C']<=0.663337811583689)",
        'RGDT_Rule_20220201_36': "(X['C']<=1.1179668118672983)&(X['C']>0.3866744253009945)",
        'RGDT_Rule_20220201_37': "(X['C']<=1.1179668118672983)&(X['C']>0.3866744253009945)",
        'RGDT_Rule_20220201_38': "(X['C']<=0.85767)&(X['C']>0.84748)&(X['D']==True)",
        'RGDT_Rule_20220201_39': "(X['C']<=1.1179668118672983)&(X['C']>0.3866744253009945)"
    }
    steps = [
        ('rg_dt', rg_dt),
        ('ro', ro)
    ]
    lp = LinearPipeline(
        steps=steps,
        use_init_data=['ro']
    )
    # No sample_weight
    lp.fit(X, y)
    assert lp.get_params()['ro']['rule_strings'] == expected_rule_strings
    X_rules = lp.fit_transform(X, y)
    assert lp.get_params()['ro']['rule_strings'] == expected_rule_strings
    assert X_rules.sum().sum() == 871
    assert lp.rules.rule_strings == expected_rule_strings
    # sample_weight provided
    lp.fit(X, y, sample_weight)
    assert lp.get_params()[
        'ro']['rule_strings'] == expected_rule_strings_weights
    X_rules = lp.fit_transform(X, y, sample_weight)
    assert lp.get_params()[
        'ro']['rule_strings'] == expected_rule_strings_weights
    assert X_rules.sum().sum() == 1178
    assert lp.rules.rule_strings == expected_rule_strings_weights
