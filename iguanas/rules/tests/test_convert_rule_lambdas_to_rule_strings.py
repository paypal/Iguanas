import pytest
from iguanas.rules._convert_rule_lambdas_to_rule_strings import _ConvertRuleLambdasToRuleStrings


@pytest.fixture
def _expected_results():
    rule_strings = {
        'Rule1': "(X['A']>1)",
        'Rule2': "(X['A']<10)"
    }
    return rule_strings


def test_convert_with_lambda_kwargs(_expected_results):
    exp_rule_strings = _expected_results
    rule_lambdas = {
        'Rule1': lambda **kwargs: "(X['A']>{A})".format(**kwargs),
        'Rule2': lambda **kwargs: "(X['A']<{A})".format(**kwargs)
    }
    lambda_kwargs = {
        'Rule1': {'A': 1},
        'Rule2': {'A': 10},
    }
    converter = _ConvertRuleLambdasToRuleStrings(rule_lambdas=rule_lambdas,
                                                 lambda_kwargs=lambda_kwargs)
    rule_strings = converter.convert()
    assert rule_strings == exp_rule_strings


def test_convert_with_lambda_args(_expected_results):
    exp_rule_strings = _expected_results
    rule_lambdas = {
        'Rule1': lambda *args: "(X['A']>{})".format(*args),
        'Rule2': lambda *args: "(X['A']<{})".format(*args)
    }
    lambda_args = {
        'Rule1': [1],
        'Rule2': [10],
    }
    converter = _ConvertRuleLambdasToRuleStrings(rule_lambdas=rule_lambdas,
                                                 lambda_args=lambda_args)
    rule_strings = converter.convert()
    assert rule_strings == exp_rule_strings


def test_error():
    with pytest.raises(ValueError, match='Either `lambda_kwargs` or `lambda_args` must be provided'):
        c = _ConvertRuleLambdasToRuleStrings(rule_lambdas={})
