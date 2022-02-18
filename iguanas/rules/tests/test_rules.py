import pytest
from iguanas.rules import Rules
import pandas as pd
import numpy as np


@pytest.fixture
def _data():
    np.random.seed(0)
    X = pd.DataFrame({
        'ad_price_type': np.random.choice(['FREE', 'NEGOTIATION', 'FOO', 'BAR', np.nan], 1000),
        'billing_country': np.random.choice(['GB', 'US', 'FR', 'GR'], 1000),
        'country_id': np.random.choice(['GB', 'US', 'FR', 'GR'], 1000),
        'forwarder_address': np.random.choice([True, False, np.nan], 1000),
        'ip_address': np.random.choice(['192,168.0.1', '000.000.000', np.nan], 1000),
        'ip_country_iso_code': np.random.choice(['GB', 'US', 'FR', 'GR'], 1000),
        'ip_isp': np.random.choice(['', True, False, np.nan], 1000),
        'is_shipping_billing_address_same': np.random.choice([True, False, np.nan], 1000),
        'method_clean': np.random.choice(['checkout_online', 'bad_checkout', 'phone', 'in person'], 1000),
        'ml_cc_v0': np.random.uniform(0, 1, 1000),
        'num_items': np.random.randint(0, 100, 1000),
        'payer_id_sum_approved_txn_amt_per_paypalid_1day': np.random.uniform(0, 100, 1000),
        'payer_id_sum_approved_txn_amt_per_paypalid_30day': np.random.uniform(0, 300, 1000),
        'payer_id_sum_approved_txn_amt_per_paypalid_7day': np.random.uniform(0, 1000, 1000),
        'kb_distance': np.random.uniform(0, 2e-05, 1000)
    })
    y = pd.Series(np.random.randint(0, 2, 1000))
    return X, y


@pytest.fixture
def _rule_dicts():
    rule_dicts = {
        'Rule1': {'condition': 'AND',
                  'rules': [{'condition': 'OR',
                             'rules': [{'field': 'payer_id_sum_approved_txn_amt_per_paypalid_1day',
                                        'operator': 'greater_or_equal',
                                        'value': 60},
                                       {'field': 'payer_id_sum_approved_txn_amt_per_paypalid_7day',
                                        'operator': 'greater',
                                        'value': 120},
                                       {'field': 'payer_id_sum_approved_txn_amt_per_paypalid_30day',
                                        'operator': 'less_or_equal',
                                        'value': 500}]},
                            {'field': 'num_items', 'operator': 'equal', 'value': 1}]},
        'Rule2': {'condition': 'AND',
                  'rules': [{'field': 'ml_cc_v0', 'operator': 'less', 'value': 0.315},
                            {'condition': 'OR',
                             'rules': [{'field': 'method_clean',
                                        'operator': 'equal',
                                        'value': 'checkout'},
                                       {'field': 'method_clean',
                                        'operator': 'begins_with', 'value': 'checkout'},
                                       {'field': 'method_clean',
                                        'operator': 'ends_with', 'value': 'checkout'},
                                       {'field': 'method_clean',
                                        'operator': 'contains', 'value': 'checkout'},
                                       {'field': 'ip_address',
                                        'operator': 'is_not_null', 'value': None},
                                       {'field': 'ip_isp', 'operator': 'is_not_empty', 'value': None}]}]},
        'Rule3': {'condition': 'AND',
                  'rules': [{'field': 'method_clean',
                             'operator': 'not_begins_with',
                             'value': 'checkout'},
                            {'field': 'method_clean',
                             'operator': 'not_ends_with', 'value': 'checkout'},
                            {'field': 'method_clean',
                             'operator': 'not_contains', 'value': 'checkout'},
                            {'condition': 'OR',
                             'rules': [{'field': 'ip_address', 'operator': 'is_null', 'value': None},
                                       {'field': 'ip_isp', 'operator': 'is_empty', 'value': None}]}]},
        'Rule4': {'condition': 'AND',
                  'rules': [{'field': 'forwarder_address', 'operator': 'equal', 'value': True},
                            {'field': 'is_shipping_billing_address_same',
                             'operator': 'equal',
                             'value': False}]},
        'Rule5': {'condition': 'AND',
                  'rules': [{'field': 'ad_price_type',
                             'operator': 'not_in',
                             'value': ['FREE', 'NEGOTIATION']},
                            {'field': 'ad_price_type', 'operator': 'in', 'value': ['FOO', 'BAR']}]},
        'Rule6': {'condition': 'AND',
                  'rules': [{'field': 'ip_country_iso_code',
                             'operator': 'equal_field',
                             'value': 'billing_country'},
                            {'field': 'country_id',
                             'operator': 'not_equal_field',
                             'value': 'ip_country_iso_code'}]},
        'Rule7': {'condition': 'AND',
                  'rules': [{'field': 'kb_distance',
                             'operator': 'greater',
                             'value': 1e-05}]}
    }
    return rule_dicts


@pytest.fixture
def _rule_strings_pandas():
    rule_strings = {
        'Rule1': "((X['payer_id_sum_approved_txn_amt_per_paypalid_1day']>=60)|(X['payer_id_sum_approved_txn_amt_per_paypalid_7day']>120)|(X['payer_id_sum_approved_txn_amt_per_paypalid_30day']<=500))&(X['num_items']==1)",
        'Rule2': "(X['ml_cc_v0']<0.315)&((X['method_clean']=='checkout')|(X['method_clean'].str.startswith('checkout', na=False))|(X['method_clean'].str.endswith('checkout', na=False))|(X['method_clean'].str.contains('checkout', na=False, regex=False))|(~X['ip_address'].isna())|(X['ip_isp'].fillna('')!=''))",
        'Rule3': "(~X['method_clean'].str.startswith('checkout', na=False))&(~X['method_clean'].str.endswith('checkout', na=False))&(~X['method_clean'].str.contains('checkout', na=False, regex=False))&((X['ip_address'].isna())|(X['ip_isp'].fillna('')==''))",
        'Rule4': "(X['forwarder_address']==True)&(X['is_shipping_billing_address_same']==False)",
        'Rule5': "(~X['ad_price_type'].isin(['FREE', 'NEGOTIATION']))&(X['ad_price_type'].isin(['FOO', 'BAR']))",
        'Rule6': "(X['ip_country_iso_code']==X['billing_country'])&(X['country_id']!=X['ip_country_iso_code'])",
        'Rule7': "(X['kb_distance']>1e-05)"
    }
    return rule_strings


@pytest.fixture
def _rule_strings_numpy():
    rule_strings = {
        'Rule1': "((X['payer_id_sum_approved_txn_amt_per_paypalid_1day'].to_numpy(na_value=np.nan)>=60)|(X['payer_id_sum_approved_txn_amt_per_paypalid_7day'].to_numpy(na_value=np.nan)>120)|(X['payer_id_sum_approved_txn_amt_per_paypalid_30day'].to_numpy(na_value=np.nan)<=500))&(X['num_items'].to_numpy(na_value=np.nan)==1)",
        'Rule2': "(X['ml_cc_v0'].to_numpy(na_value=np.nan)<0.315)&((X['method_clean'].to_numpy(na_value=np.nan)=='checkout')|(X['method_clean'].str.startswith('checkout', na=False))|(X['method_clean'].str.endswith('checkout', na=False))|(X['method_clean'].str.contains('checkout', na=False, regex=False))|(~pd.isna(X['ip_address'].to_numpy(na_value=np.nan)))|(X['ip_isp'].fillna('')!=''))",
        'Rule3': "(~X['method_clean'].str.startswith('checkout', na=False))&(~X['method_clean'].str.endswith('checkout', na=False))&(~X['method_clean'].str.contains('checkout', na=False, regex=False))&((pd.isna(X['ip_address'].to_numpy(na_value=np.nan)))|(X['ip_isp'].fillna('')==''))",
        'Rule4': "(X['forwarder_address'].to_numpy(na_value=np.nan)==True)&(X['is_shipping_billing_address_same'].to_numpy(na_value=np.nan)==False)",
        'Rule5': "(~X['ad_price_type'].isin(['FREE', 'NEGOTIATION']))&(X['ad_price_type'].isin(['FOO', 'BAR']))",
        'Rule6': "(X['ip_country_iso_code'].to_numpy(na_value=np.nan)==X['billing_country'].to_numpy(na_value=np.nan))&(X['country_id'].to_numpy(na_value=np.nan)!=X['ip_country_iso_code'].to_numpy(na_value=np.nan))",
        'Rule7': "(X['kb_distance'].to_numpy(na_value=np.nan)>1e-05)"
    }
    return rule_strings


@pytest.fixture()
def _rule_lambdas_with_kwargs():
    rule_lambdas = {
        'Rule1': lambda **kwargs: "((X['payer_id_sum_approved_txn_amt_per_paypalid_1day']>={payer_id_sum_approved_txn_amt_per_paypalid_1day})|(X['payer_id_sum_approved_txn_amt_per_paypalid_7day']>{payer_id_sum_approved_txn_amt_per_paypalid_7day})|(X['payer_id_sum_approved_txn_amt_per_paypalid_30day']<={payer_id_sum_approved_txn_amt_per_paypalid_30day}))&(X['num_items']=={num_items})".format(**kwargs),
        'Rule2': lambda **kwargs: "(X['ml_cc_v0']<{ml_cc_v0})&((X['method_clean']=='checkout')|(X['method_clean'].str.startswith('checkout', na=False))|(X['method_clean'].str.endswith('checkout', na=False))|(X['method_clean'].str.contains('checkout', na=False, regex=False))|(~X['ip_address'].isna())|(X['ip_isp'].fillna('')!=''))".format(**kwargs),
        'Rule3': lambda **kwargs: "(~X['method_clean'].str.startswith('checkout', na=False))&(~X['method_clean'].str.endswith('checkout', na=False))&(~X['method_clean'].str.contains('checkout', na=False, regex=False))&((X['ip_address'].isna())|(X['ip_isp'].fillna('')==''))".format(**kwargs),
        'Rule4': lambda **kwargs: "(X['forwarder_address']==True)&(X['is_shipping_billing_address_same']==False)".format(**kwargs),
        'Rule5': lambda **kwargs: "(~X['ad_price_type'].isin(['FREE', 'NEGOTIATION']))&(X['ad_price_type'].isin(['FOO', 'BAR']))".format(**kwargs),
        'Rule6': lambda **kwargs: "(X['ip_country_iso_code']==X['billing_country'])&(X['country_id']!=X['ip_country_iso_code'])".format(**kwargs),
        'Rule7': lambda **kwargs: "(X['kb_distance']>{kb_distance}".format(**kwargs)
    }
    lambda_kwargs = {
        'Rule1': {'payer_id_sum_approved_txn_amt_per_paypalid_1day': 60.0, 'payer_id_sum_approved_txn_amt_per_paypalid_7day': 120.0, 'payer_id_sum_approved_txn_amt_per_paypalid_30day': 500, 'num_items': 1.0},
        'Rule2': {'ml_cc_v0': 0.315},
        'Rule3': {},
        'Rule4': {},
        'Rule5': {},
        'Rule6': {},
        'Rule7': {'kb_distance': 1e-05}
    }
    return rule_lambdas, lambda_kwargs


@pytest.fixture()
def _rule_lambdas_with_args():
    rule_lambdas = {
        'Rule1': lambda *args: "((X['payer_id_sum_approved_txn_amt_per_paypalid_1day']>={})|(X['payer_id_sum_approved_txn_amt_per_paypalid_7day']>{})|(X['payer_id_sum_approved_txn_amt_per_paypalid_30day']<={}))&(X['num_items']=={})".format(*args),
        'Rule2': lambda *args: "(X['ml_cc_v0']<{})&((X['method_clean']=='checkout')|(X['method_clean'].str.startswith('checkout', na=False))|(X['method_clean'].str.endswith('checkout', na=False))|(X['method_clean'].str.contains('checkout', na=False, regex=False))|(~X['ip_address'].isna())|(X['ip_isp'].fillna('')!=''))".format(*args),
        'Rule3': lambda *args: "(~X['method_clean'].str.startswith('checkout', na=False))&(~X['method_clean'].str.endswith('checkout', na=False))&(~X['method_clean'].str.contains('checkout', na=False, regex=False))&((X['ip_address'].isna())|(X['ip_isp'].fillna('')==''))".format(*args),
        'Rule4': lambda *args: "(X['forwarder_address']==True)&(X['is_shipping_billing_address_same']==False)".format(*args),
        'Rule5': lambda *args: "(~X['ad_price_type'].isin(['FREE', 'NEGOTIATION']))&(X['ad_price_type'].isin(['FOO', 'BAR']))".format(*args),
        'Rule6': lambda *args: "(X['ip_country_iso_code']==X['billing_country'])&(X['country_id']!=X['ip_country_iso_code'])".format(*args),
        'Rule7': lambda *args: "(X['kb_distance']>{}".format(*args)
    }
    lambda_args = {
        'Rule1': [60.0, 120.0, 500, 1.0],
        'Rule2': [0.315],
        'Rule3': [],
        'Rule4': [],
        'Rule5': [],
        'Rule6': [],
        'Rule7': [1e-05]
    }
    return rule_lambdas, lambda_args


def test_add_and_radd():
    exp_rule_strings = {'Rule1': "X['A']>0", 'Rule2': "X['B']>0"}
    exp_rule_dicts = {
        'Rule1': {
            'condition': 'AND', 'rules': [{
                'field': 'A', 'operator': 'greater', 'value': 0
            }]},
        'Rule2': {
            'condition': 'AND', 'rules': [{
                'field': 'B', 'operator': 'greater', 'value': 0
            }]}
    }
    rules1 = Rules(
        rule_strings={'Rule1': "X['A']>0"},
        rule_dicts={
            'Rule1': {
                'condition': 'AND', 'rules': [{
                    'field': 'A', 'operator': 'greater', 'value': 0
                }]}
        },
        rule_lambdas={'Rule1': lambda **kwargs: "X['A']>{A}".format(**kwargs)},
        lambda_kwargs={'Rule1': {'A': 0}},
    )
    rules2 = Rules(
        rule_strings={'Rule2': "X['B']>0"},
        rule_dicts={
            'Rule2': {
                'condition': 'AND', 'rules': [{
                    'field': 'B', 'operator': 'greater', 'value': 0
                }]}
        },
        rule_lambdas={'Rule2': lambda **kwargs: "X['B']>{B}".format(**kwargs)},
        lambda_kwargs={'Rule2': {'B': 0}},
    )
    # Test __add__
    rules = rules1 + rules2
    assert rules.rule_strings == exp_rule_strings
    assert rules.rule_dicts == exp_rule_dicts
    for r in ['Rule1', 'Rule2']:
        assert rules.rule_lambdas[r](
            **rules.lambda_kwargs[r]) == exp_rule_strings[r]
    # Test __radd__
    rules = sum([rules1, rules2])
    assert rules.rule_strings == exp_rule_strings
    assert rules.rule_dicts == exp_rule_dicts
    for r in ['Rule1', 'Rule2']:
        assert rules.rule_lambdas[r](
            **rules.lambda_kwargs[r]) == exp_rule_strings[r]
    # Test exception
    with pytest.raises(ValueError, match=f"""Attempting to add rule sets with similar keys in `rule_dicts`. Similar keys are 'Rule1'."""):
        rules = rules1 + rules1
        rules = sum([rules1, rules1])
    rules1.rule_dicts = {}
    with pytest.raises(ValueError, match=f"""Attempting to add rule sets with similar keys in `rule_strings`. Similar keys are 'Rule1'."""):
        rules = rules1 + rules1
        rules = sum([rules1, rules1])
    rules1.rule_strings = {}
    with pytest.raises(ValueError, match=f"""Attempting to add rule sets with similar keys in `rule_lambdas`. Similar keys are 'Rule1'."""):
        rules = rules1 + rules1
        rules = sum([rules1, rules1])
    rules1.rule_lambdas = {}
    with pytest.raises(ValueError, match=f"""Attempting to add rule sets with similar keys in `lambda_kwargs`. Similar keys are 'Rule1'."""):
        rules = rules1 + rules1
        rules = sum([rules1, rules1])


def test_repr(_rule_strings_pandas):
    rule_strings_pandas = _rule_strings_pandas
    r = Rules(rule_strings=rule_strings_pandas)
    assert r.__repr__() == 'Rules object containing 7 rules'


def test_as_rule_dicts_starting_with_rule_strings(_rule_strings_pandas, _rule_strings_numpy, _rule_dicts):
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    expected_rule_dicts = _rule_dicts
    r = Rules(rule_strings=rule_strings_pandas)
    rule_dicts = r.as_rule_dicts()
    assert rule_dicts == expected_rule_dicts
    r = Rules(rule_strings=rule_strings_numpy)
    rule_dicts = r.as_rule_dicts()
    assert rule_dicts == expected_rule_dicts


def test_as_rule_dicts_starting_with_rule_lambdas_kwargs(_rule_lambdas_with_kwargs, _rule_dicts):
    rule_lambdas, lambda_kwargs = _rule_lambdas_with_kwargs
    expected_rule_dicts = _rule_dicts
    r = Rules(rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs)
    rule_dicts = r.as_rule_dicts()
    assert rule_dicts == expected_rule_dicts


def test_as_rule_dicts_starting_with_rule_lambdas_args(_rule_lambdas_with_args, _rule_dicts):
    rule_lambdas, lambda_args = _rule_lambdas_with_args
    expected_rule_dicts = _rule_dicts
    r = Rules(rule_lambdas=rule_lambdas, lambda_args=lambda_args)
    rule_dicts = r.as_rule_dicts()
    assert rule_dicts == expected_rule_dicts


def test_as_rule_strings_starting_with_rule_strings(_rule_strings_pandas, _rule_strings_numpy):
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    r = Rules(rule_strings=rule_strings_pandas)
    rule_strings = r.as_rule_strings(as_numpy=False)
    assert rule_strings == rule_strings_pandas
    r = Rules(rule_strings=rule_strings_pandas)
    rule_strings = r.as_rule_strings(as_numpy=True)
    assert rule_strings == rule_strings_numpy


def test_as_rule_strings_starting_with_rule_dicts(_rule_dicts, _rule_strings_pandas, _rule_strings_numpy):
    rule_dicts = _rule_dicts
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    r = Rules(rule_dicts=rule_dicts)
    rule_strings = r.as_rule_strings(as_numpy=False)
    assert rule_strings == rule_strings_pandas
    r = Rules(rule_dicts=rule_dicts)
    rule_strings = r.as_rule_strings(as_numpy=True)
    assert rule_strings == rule_strings_numpy


def test_as_rule_strings_starting_with_rule_lambdas_kwargs(_rule_lambdas_with_kwargs, _rule_strings_pandas, _rule_strings_numpy):
    rule_lambdas, lambda_kwargs = _rule_lambdas_with_kwargs
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    r = Rules(rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs)
    rule_strings = r.as_rule_strings(as_numpy=False)
    assert rule_strings == rule_strings_pandas
    r = Rules(rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs)
    rule_strings = r.as_rule_strings(as_numpy=True)
    assert rule_strings == rule_strings_numpy


def test_as_rule_strings_starting_with_rule_lambdas_args(_rule_lambdas_with_args, _rule_strings_pandas, _rule_strings_numpy):
    rule_lambdas, lambda_args = _rule_lambdas_with_args
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    rule_lambdas, lambda_args = _rule_lambdas_with_args
    r = Rules(rule_lambdas=rule_lambdas, lambda_args=lambda_args)
    rule_strings = r.as_rule_strings(as_numpy=False)
    assert rule_strings == rule_strings_pandas
    r = Rules(rule_lambdas=rule_lambdas, lambda_args=lambda_args)
    rule_strings = r.as_rule_strings(as_numpy=True)
    assert rule_strings == rule_strings_numpy


def test_as_rule_lambdas_starting_with_rule_dicts_with_kwargs_True(_rule_dicts, _rule_strings_pandas, _rule_strings_numpy):
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    rule_dicts = _rule_dicts
    r = Rules(rule_dicts=rule_dicts)
    rule_lambdas = r.as_rule_lambdas(as_numpy=False, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_dicts=rule_dicts)
    rule_lambdas = r.as_rule_lambdas(as_numpy=True, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]


def test_as_rule_lambdas_starting_with_rule_dicts_with_kwargs_False(_rule_dicts, _rule_strings_pandas, _rule_strings_numpy):
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    rule_dicts = _rule_dicts
    r = Rules(rule_dicts=rule_dicts)
    rule_lambdas = r.as_rule_lambdas(as_numpy=False, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_dicts=rule_dicts)
    rule_lambdas = r.as_rule_lambdas(as_numpy=True, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]


def test_as_rule_lambdas_starting_with_rule_strings_with_kwargs_True(_rule_strings_pandas, _rule_strings_numpy):
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    r = Rules(rule_strings=rule_strings_pandas)
    rule_lambdas = r.as_rule_lambdas(as_numpy=False, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_strings=rule_strings_numpy)
    rule_lambdas = r.as_rule_lambdas(as_numpy=True, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]
    r = Rules(rule_strings=rule_strings_numpy)
    rule_lambdas = r.as_rule_lambdas(as_numpy=False, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_strings=rule_strings_numpy)
    rule_lambdas = r.as_rule_lambdas(as_numpy=True, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]


def test_as_rule_lambdas_starting_with_rule_strings_with_kwargs_False(_rule_strings_pandas, _rule_strings_numpy):
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    r = Rules(rule_strings=rule_strings_pandas)
    rule_lambdas = r.as_rule_lambdas(as_numpy=False, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_strings=rule_strings_numpy)
    rule_lambdas = r.as_rule_lambdas(as_numpy=True, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]
    r = Rules(rule_strings=rule_strings_numpy)
    rule_lambdas = r.as_rule_lambdas(as_numpy=False, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_strings=rule_strings_numpy)
    rule_lambdas = r.as_rule_lambdas(as_numpy=True, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]


def test_as_rule_lambdas_starting_with_rule_lambdas_with_kwargs_True(_rule_lambdas_with_kwargs, _rule_lambdas_with_args,
                                                                     _rule_strings_pandas, _rule_strings_numpy):
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    # Starting with lambda_kwargs
    rule_lambdas, lambda_kwargs = _rule_lambdas_with_kwargs
    r = Rules(rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs)
    rule_lambdas_ = r.as_rule_lambdas(as_numpy=False, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas_.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs)
    rule_lambdas_ = r.as_rule_lambdas(as_numpy=True, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas_.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]
    # Starting with lambda_args
    rule_lambdas, lambda_args = _rule_lambdas_with_args
    r = Rules(rule_lambdas=rule_lambdas, lambda_args=lambda_args)
    rule_lambdas_ = r.as_rule_lambdas(as_numpy=False, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas_.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_lambdas=rule_lambdas, lambda_args=lambda_args)
    rule_lambdas_ = r.as_rule_lambdas(as_numpy=True, with_kwargs=True)
    for rule_name, rule_lambda in rule_lambdas_.items():
        rule_string = rule_lambda(**r.lambda_kwargs[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]


def test_as_rule_lambdas_starting_with_rule_lambdas_with_kwargs_False(_rule_lambdas_with_kwargs, _rule_lambdas_with_args,
                                                                      _rule_strings_pandas, _rule_strings_numpy):
    rule_strings_pandas = _rule_strings_pandas
    rule_strings_numpy = _rule_strings_numpy
    # Starting with lambda_kwargs
    rule_lambdas, lambda_kwargs = _rule_lambdas_with_kwargs
    r = Rules(rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs)
    rule_lambdas_ = r.as_rule_lambdas(as_numpy=False, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas_.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs)
    rule_lambdas_ = r.as_rule_lambdas(as_numpy=True, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas_.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]
    # Starting with lambda_args
    rule_lambdas, lambda_args = _rule_lambdas_with_args
    r = Rules(rule_lambdas=rule_lambdas, lambda_args=lambda_args)
    rule_lambdas_ = r.as_rule_lambdas(as_numpy=False, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas_.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_pandas[rule_name]
    r = Rules(rule_lambdas=rule_lambdas, lambda_args=lambda_args)
    rule_lambdas_ = r.as_rule_lambdas(as_numpy=True, with_kwargs=False)
    for rule_name, rule_lambda in rule_lambdas_.items():
        rule_string = rule_lambda(*r.lambda_args[rule_name])
        assert rule_string == rule_strings_numpy[rule_name]


def test_transform(_data, _rule_dicts):
    X, y_ = _data
    rule_dicts = _rule_dicts
    exp_X_rules_sum = np.array([10, 285, 128,  98, 413, 179, 488])
    r = Rules(rule_dicts=rule_dicts)
    X_rules = r.transform(X)
    np.testing.assert_array_equal(X_rules.sum().values, exp_X_rules_sum)


def test_filter_rules(_rule_strings_pandas):
    rule_strings = _rule_strings_pandas
    r = Rules(rule_strings=rule_strings)
    r.as_rule_dicts()
    r.as_rule_lambdas(as_numpy=True, with_kwargs=True)
    inc_rule_names = ['Rule1', 'Rule2', 'Rule3']
    r.filter_rules(include=inc_rule_names)
    assert list(r.rule_strings.keys()) == inc_rule_names
    assert list(r.rule_dicts.keys()) == inc_rule_names
    assert list(r.rule_lambdas.keys()) == inc_rule_names
    assert list(r.lambda_kwargs.keys()) == inc_rule_names
    assert list(r.lambda_args.keys()) == inc_rule_names
    assert list(r.rule_features.keys()) == inc_rule_names
    exc_rule_names = ['Rule2', 'Rule3']
    r.filter_rules(exclude=exc_rule_names)
    assert list(r.rule_strings.keys()) == ['Rule1']
    assert list(r.rule_dicts.keys()) == ['Rule1']
    assert list(r.rule_lambdas.keys()) == ['Rule1']
    assert list(r.lambda_kwargs.keys()) == ['Rule1']
    assert list(r.lambda_args.keys()) == ['Rule1']
    assert list(r.rule_features.keys()) == ['Rule1']


def test_filter_rules_rule_strings_only(_rule_strings_pandas):
    rule_strings = _rule_strings_pandas
    r = Rules(rule_strings=rule_strings)
    inc_rule_names = ['Rule1', 'Rule2', 'Rule3']
    r.filter_rules(include=inc_rule_names)
    assert list(r.rule_strings.keys()) == inc_rule_names
    exc_rule_names = ['Rule2', 'Rule3']
    r.filter_rules(exclude=exc_rule_names)
    assert list(r.rule_strings.keys()) == ['Rule1']


def test_filter_rules_rule_dicts_only(_rule_dicts):
    rule_dicts = _rule_dicts
    r = Rules(rule_dicts=rule_dicts)
    inc_rule_names = ['Rule1', 'Rule2', 'Rule3']
    r.filter_rules(include=inc_rule_names)
    assert list(r.rule_dicts.keys()) == inc_rule_names
    exc_rule_names = ['Rule2', 'Rule3']
    r.filter_rules(exclude=exc_rule_names)
    assert list(r.rule_dicts.keys()) == ['Rule1']


def test_get_rule_features(_rule_dicts):
    exp_result = {
        'Rule1': {
            'num_items', 'payer_id_sum_approved_txn_amt_per_paypalid_1day',
            'payer_id_sum_approved_txn_amt_per_paypalid_30day',
            'payer_id_sum_approved_txn_amt_per_paypalid_7day'
        },
        'Rule2': {
            'ip_address', 'ip_isp', 'method_clean', 'ml_cc_v0'
        },
        'Rule3': {
            'ip_address', 'ip_isp', 'method_clean'
        },
        'Rule4': {
            'forwarder_address', 'is_shipping_billing_address_same'
        },
        'Rule5': {
            'ad_price_type'
        },
        'Rule6': {
            'billing_country', 'country_id', 'ip_country_iso_code'
        },
        'Rule7': {
            'kb_distance'
        }
    }
    rule_dicts = _rule_dicts
    r = Rules(rule_dicts=rule_dicts)
    rule_features = r.get_rule_features()
    assert rule_features == exp_result


def test_errors():
    with pytest.raises(ValueError):
        r = Rules()
    with pytest.raises(ValueError):
        r = Rules(rule_strings="X['A']>2")
        r._rule_dicts_to_rule_strings(as_numpy=False)
    with pytest.raises(ValueError):
        r = Rules(rule_dicts={'ABC': {}})
        r._rule_strings_to_rule_dicts()
    with pytest.raises(ValueError):
        r = Rules(rule_strings={'ABC': {}})
        r._rule_dicts_to_rule_lambdas(as_numpy=True, with_kwargs=True)
    with pytest.raises(Exception):
        r = Rules(rule_strings={'Rule1': {}})
        r.filter_rules(include=['Rule1'], exclude=['Rule1'])
    with pytest.raises(ValueError, match='`lambda_kwargs` or `lambda_args` must be given when `rule_lambdas` is provided'):
        Rules(rule_lambdas={'Rule1': lambda x: None})
    with pytest.raises(ValueError, match='`rule_lambdas` must be given'):
        r = Rules(rule_strings={})
        r._rule_lambdas_to_rule_strings()
    with pytest.raises(ValueError, match='`lambda_kwargs` or `lambda_args` must be given'):
        r = Rules(rule_lambdas={'Rule1': lambda x: None}, lambda_kwargs={})
        r._rule_lambdas_to_rule_strings()
