import pytest
import pandas as pd
from iguanas.rule_selection._base_filter import _BaseFilter
from iguanas.rules import Rules


@pytest.fixture
def _create_data():
    X_rules = pd.DataFrame({
        'A': [1, 0, 1],
        'B': [1, 1, 1]
    })
    return X_rules


def test_transform(_create_data):
    X_rules = _create_data
    bf = _BaseFilter(rules_to_keep=['A'], rules=None)
    X_rules_ = bf.transform(X_rules)
    pd.testing.assert_frame_equal(X_rules_, X_rules[['A']])
    # With rules
    rules = Rules(
        rule_strings={
            'A': "X['a']>1",
            'B': "X['b']>1"
        }
    )
    bf = _BaseFilter(rules_to_keep=['A'], rules=rules)
    X_rules_ = bf.transform(X_rules)
    pd.testing.assert_frame_equal(X_rules_, X_rules[['A']])
    assert bf.rules.rule_strings == {'A': "X['a']>1"}


def test_fit_transform(_create_data):
    bf = _BaseFilter(rules_to_keep=['A'], rules=None)
    # Just create dummy fit method for testing
    bf.fit = lambda X_rules, y, sample_weight: None
    X_rules = _create_data
    bf.rules_to_keep = ['A']
    X_rules_ = bf.fit_transform(X_rules)
    pd.testing.assert_frame_equal(X_rules_, X_rules[['A']])
    # With rules
    rules = Rules(
        rule_strings={
            'A': "X['a']>1",
            'B': "X['b']>1"
        }
    )
    bf = _BaseFilter(rules_to_keep=['A'], rules=rules)
    # Just create dummy fit method for testing
    bf.fit = lambda X_rules, y, sample_weight: None
    X_rules_ = bf.fit_transform(X_rules)
    pd.testing.assert_frame_equal(X_rules_, X_rules[['A']])
    assert bf.rules.rule_strings == {'A': "X['a']>1"}
