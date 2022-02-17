"""
Unit tests for methods in _base_generator module that aren't specific to Pandas 
or Koalas
"""
import pytest
from iguanas.metrics.classification import FScore
from iguanas.rule_generation import RuleGeneratorDT
from iguanas.rule_generation.rule_generator_opt import RuleGeneratorOpt
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def rg_instantiated():
    params = {
        'metric': None,
        'n_total_conditions': 4,
        'tree_ensemble': RandomForestClassifier(n_estimators=10, random_state=0),
        'precision_threshold': 0,
        'num_cores': 4
    }
    rg = RuleGeneratorDT(**params)
    rg._today = '20200204'
    return [rg, params]


def test_convert_conditions_to_string(rg_instantiated):
    list_of_conditions = [
        ('A', '>=', 1),
        ('B', '<=', 1.5),
        ('C', '==', 1),
        ('D', '<=', 2.9),
        ('E', '>', 0.5)
    ]
    columns_int = ['B', 'C', 'E']
    columns_cat = ['C', 'E']
    rg, _ = rg_instantiated
    rule_string = rg._convert_conditions_to_string(
        list_of_conditions, columns_int, columns_cat)
    assert rule_string == "(X['A']>=1)&(X['B']<=1)&(X['C']==True)&(X['D']<=2.9)&(X['E']==True)"
    # Test error by removing C from columns_cat
    columns_int = ['B', 'C', 'E']
    columns_cat = ['E']
    with pytest.raises(ValueError, match='Error converting rule conditions - operator should be ">=", ">", "<=", or "<"'):
        rule_string = rg._convert_conditions_to_string(
            list_of_conditions, columns_int, columns_cat
        )


def test_clean_dup_features_from_conditions(rg_instantiated):
    list_of_conditions = [
        ('A', '>=', 1),
        ('A', '>=', 3),
        ('B', '<=', 2),
        ('B', '<=', 1),
        ('C', '>', 0.5),
        ('C', '<=', 1)
    ]
    expected_result = [
        ('A', '>=', 3),
        ('B', '<=', 1),
        ('C', '<=', 1),
        ('C', '>', 0.5)
    ]
    rg, _ = rg_instantiated
    result = rg._clean_dup_features_from_conditions(list_of_conditions)
    assert result == expected_result


def test_generate_rule_name(rg_instantiated):
    # Test with default Rule Gen DT prefix
    rg, _ = rg_instantiated
    rule_name = rg._generate_rule_name()
    assert rule_name == 'RGDT_Rule_20200204_0'
    # Test with default Rule Gen Opt prefix
    f1 = FScore(1)
    rgo = RuleGeneratorOpt(
        metric=f1.fit, n_total_conditions=4, num_rules_keep=10
    )
    rgo._today = '20200204'
    rule_name = rgo._generate_rule_name()
    assert rule_name == 'RGO_Rule_20200204_0'
    # Test with default Rule Gen DT prefix
    rg.rule_name_prefix = 'TEST'
    rule_name = rg._generate_rule_name()
    assert rule_name == 'TEST_1'


def test_remove_misaligned_conditions(rg_instantiated):
    target_feat_corr_types = {
        'PositiveCorr': ['ColA'],
        'NegativeCorr': ['ColB'],
    }
    branch_conditions = [
        ('ColA', '>=', 1),
        ('ColA', '<=', 5),
        ('ColB', '>=', 1),
        ('ColB', '<=', 5),
    ]
    expected_result = [('ColA', '>=', 1), ('ColB', '<=', 5)]
    rg, _ = rg_instantiated
    cleaned_branch_conditions = rg._remove_misaligned_conditions(
        branch_conditions=branch_conditions,
        target_feat_corr_types=target_feat_corr_types
    )
    assert all([a == b for a, b in zip(
        cleaned_branch_conditions, expected_result)])


def test_generate_other_rule_formats(rg_instantiated):
    rg, _ = rg_instantiated
    rg.rule_strings = {'Rule1': "(X['A']>1)"}
    rg._generate_other_rule_formats()
    assert rg.rule_names == ['Rule1']
    assert rg.rule_lambdas['Rule1'](
        **rg.lambda_kwargs['Rule1']) == "(X['A']>1)"
    assert rg.rules.rule_strings == {'Rule1': "(X['A']>1)"}
    assert rg.rules.rule_lambdas['Rule1'](
        **rg.rules.lambda_kwargs['Rule1']) == "(X['A']>1)"
