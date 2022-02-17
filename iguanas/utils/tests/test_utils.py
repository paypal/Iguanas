from iguanas.metrics import FScore, AlertsPerDay
from iguanas.utils.types import PandasDataFrame
import iguanas.utils as utils
import pytest
import numpy as np
import pandas as pd
import json
from sklearn.metrics import fbeta_score, precision_score, recall_score
import string
from tqdm import tqdm


@pytest.fixture
def create_data():
    np.random.seed(0)
    y_preds = pd.DataFrame(np.random.randint(0, 2, size=(1000, 10)), columns=[
                           i for i in string.ascii_letters[:10]])
    y_true = pd.Series(np.random.randint(0, 2, 1000))
    sample_weight = y_true * 10
    return (y_true, y_preds, sample_weight)


def test_concat(create_data):
    y_true, y_preds, _ = create_data
    X = utils.concat([y_preds, y_true], axis=1)
    pd.testing.assert_frame_equal(X, pd.concat([y_preds, y_true], axis=1))


def test_return_columns_types():
    X = pd.DataFrame({
        'A': [2.5, 3.5, 1, 1, 2.5],
        'B': [1, 0, 0, 1, 1],
        'C': [1, 2, 0, 0, 1]
    })
    int_cols, cat_cols, float_cols = utils.return_columns_types(X)
    assert int_cols == ['B', 'C']
    assert cat_cols == ['B']
    assert float_cols == ['A']
    # Test when all ints
    X = pd.DataFrame({
        'A': [1, 0, 0, 1, 1],
        'B': [1, 0, 0, 1, 1],
    })
    int_cols, cat_cols, float_cols = utils.return_columns_types(X)
    assert int_cols == ['A', 'B']
    assert cat_cols == ['A', 'B']
    assert float_cols == []
    # Test when all floats
    X = pd.DataFrame({
        'A': [1.2, 0, 0, 1.1, 1],
        'B': [1.5, 0, 0, 1, 1.1],
    })
    int_cols, cat_cols, float_cols = utils.return_columns_types(X)
    assert int_cols == []
    assert cat_cols == []
    assert float_cols == ['A', 'B']


def test_calc_tps_fps_tns_fns(create_data):
    y_true, y_preds, weights = create_data
    # Without weights
    tps, fps, tns, fns, tps_fps, tps_fns = utils.calc_tps_fps_tns_fns(
        y_true, y_preds, tps=True, fps=True, tns=True, fns=True,
        tps_fps=True, tps_fns=True
    )
    assert all(tps == np.array(
        [262, 264, 267, 270, 268, 270, 246, 270, 268, 254]))
    assert all(fps == np.array(
        [252, 245, 253, 236, 235, 225, 239, 253, 260, 248]))
    assert all(tns == np.array(
        [238, 245, 237, 254, 255, 265, 251, 237, 230, 242]))
    assert all(fns == np.array(
        [248, 246, 243, 240, 242, 240, 264, 240, 242, 256]))
    assert all(tps_fps == np.array(
        [514, 509, 520, 506, 503, 495, 485, 523, 528, 502]))
    assert tps_fns == np.array(y_true.sum()) == np.array(510)
    # With weights
    tps, fps, tns, fns, tps_fps, tps_fns = utils.calc_tps_fps_tns_fns(
        y_true, y_preds, weights, tps=True, fps=True, tns=True, fns=True,
        tps_fps=True, tps_fns=True
    )
    assert all(tps == np.array(
        [2620, 2640, 2670, 2700, 2680, 2700, 2460, 2700, 2680, 2540]))
    assert all(fps == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert all(tns == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert all(fns == np.array(
        [2480, 2460, 2430, 2400, 2420, 2400, 2640, 2400, 2420, 2560]))
    assert all(tps_fps == np.array(
        [2620, 2640, 2670, 2700, 2680, 2700, 2460, 2700, 2680, 2540]))
    assert tps_fns == np.array((y_true * weights).sum()) == np.array(5100)
    # One pred (without weights)
    tps, fps, tns, fns, tps_fps, tps_fns = utils.calc_tps_fps_tns_fns(
        y_true, y_preds['a'], tps=True, fps=True, tns=True, fns=True,
        tps_fps=True, tps_fns=True
    )
    assert (tps, fps, tns, fns, tps_fps, tps_fns) == (
        262, 252, 238, 248, 514, 510)
    # One pred (with weights)
    tps, fps, tns, fns, tps_fps, tps_fns = utils.calc_tps_fps_tns_fns(
        y_true, y_preds['a'], weights, tps=True, fps=True, tns=True, fns=True,
        tps_fps=True, tps_fns=True
    )
    assert (tps, fps, tns, fns, tps_fps, tps_fns) == (
        2620, 0, 0, 2480, 2620, 5100)
    # Error
    with pytest.raises(ValueError, match='One of the parameters `tps`, `fps`, `tns`, `fns`, `tps_fps` or `tps_fns` must be True'):
        utils.calc_tps_fps_tns_fns(
            y_true, y_preds, tps=False, fps=False, tns=False, fns=False,
            tps_fps=False, tps_fns=False
        )


def test_return_binary_pred_perf_of_set(create_data):
    y_true, y_preds, weights = create_data
    f1 = FScore(beta=1)
    # Test multiple preds
    for w in [None, weights]:
        results = utils.return_binary_pred_perf_of_set(
            y_true=y_true, y_preds=y_preds, y_preds_columns=y_preds.columns, sample_weight=w, metric=f1.fit)
        _test_y_preds(y_preds, results, y_true, w)
    # Test one pred
    y_pred = y_preds.iloc[:, 0]
    for w in [None, weights]:
        results = utils.return_binary_pred_perf_of_set(
            y_true=y_true, y_preds=y_pred, y_preds_columns=y_preds.columns, sample_weight=w, metric=f1.fit)
        _test_y_preds(y_pred, results, y_true, w)


def test_return_rule_descriptions_from_X_rules(create_data):
    y_true, y_preds, _ = create_data
    f1 = FScore(beta=1)
    exp_results = pd.DataFrame(
        np.array([[0.50972763, 0.51372549, 0.514, 0.51171875],
                  [0.51866405, 0.51764706, 0.509, 0.51815505],
                  [0.51346154, 0.52352941, 0.52, 0.5184466],
                  [0.53359684, 0.52941176, 0.506, 0.53149606],
                  [0.53280318, 0.5254902, 0.503, 0.52912142],
                  [0.54545455, 0.52941176, 0.495, 0.53731343],
                  [0.50721649, 0.48235294, 0.485, 0.49447236],
                  [0.51625239, 0.52941176, 0.523, 0.52274927],
                  [0.50757576, 0.5254902, 0.528, 0.51637765],
                  [0.5059761, 0.49803922, 0.502, 0.50197628]]),
        columns=['Precision', 'Recall', 'PercDataFlagged', 'Metric'],
        index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    )
    exp_results.index.name = 'Rule'
    rule_descriptions = utils.return_rule_descriptions_from_X_rules(X_rules=y_preds,
                                                                    X_rules_cols=y_preds.columns,
                                                                    y_true=y_true,
                                                                    sample_weight=None,
                                                                    metric=f1.fit)
    assert all(rule_descriptions == exp_results)
    rule_descriptions = utils.return_rule_descriptions_from_X_rules(X_rules=y_preds,
                                                                    X_rules_cols=y_preds.columns,
                                                                    y_true=y_true,
                                                                    sample_weight=None)
    exp_results['Metric'] = None
    assert all(rule_descriptions == exp_results)


def test_return_rule_descriptions_from_X_rules_weighted(create_data):
    y_true, y_preds, weights = create_data
    f1 = FScore(beta=1)
    exp_results = pd.DataFrame(
        np.array([[1., 0.51372549, 0.514, 0.67875648],
                  [1., 0.51764706, 0.509, 0.68217054],
                  [1., 0.52352941, 0.52, 0.68725869],
                  [1., 0.52941176, 0.506, 0.69230769],
                  [1., 0.5254902, 0.503, 0.68894602],
                  [1., 0.52941176, 0.495, 0.69230769],
                  [1., 0.48235294, 0.485, 0.65079365],
                  [1., 0.52941176, 0.523, 0.69230769],
                  [1., 0.5254902, 0.528, 0.68894602],
                  [1., 0.49803922, 0.502, 0.66492147]]),
        columns=['Precision', 'Recall', 'PercDataFlagged', 'Metric'],
        index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    )
    exp_results.index.name = 'Rule'
    rule_descriptions = utils.return_rule_descriptions_from_X_rules(X_rules=y_preds,
                                                                    X_rules_cols=y_preds.columns,
                                                                    y_true=y_true,
                                                                    sample_weight=weights,
                                                                    metric=f1.fit)
    assert all(rule_descriptions == exp_results)
    rule_descriptions = utils.return_rule_descriptions_from_X_rules(X_rules=y_preds,
                                                                    X_rules_cols=y_preds.columns,
                                                                    y_true=y_true,
                                                                    sample_weight=weights)
    exp_results['Metric'] = None
    assert all(rule_descriptions == exp_results)


def test_return_rule_descriptions_from_X_rules_unlabelled(create_data):
    _, y_preds, _ = create_data
    apd = AlertsPerDay(n_alerts_expected_per_day=10, no_of_days_in_file=10)
    exp_results = pd.DataFrame(
        np.array([[5.14000e-01, -1.71396e+03],
                  [5.09000e-01, -1.67281e+03],
                  [5.20000e-01, -1.76400e+03],
                  [5.06000e-01, -1.64836e+03],
                  [5.03000e-01, -1.62409e+03],
                  [4.95000e-01, -1.56025e+03],
                  [4.85000e-01, -1.48225e+03],
                  [5.23000e-01, -1.78929e+03],
                  [5.28000e-01, -1.83184e+03],
                  [5.02000e-01, -1.61604e+03]]),
        columns=['PercDataFlagged', 'Metric'],
        index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    )
    exp_results.index.name = 'Rule'
    rule_descriptions = utils.return_rule_descriptions_from_X_rules(X_rules=y_preds,
                                                                    X_rules_cols=y_preds.columns,
                                                                    y_true=None,
                                                                    sample_weight=None,
                                                                    metric=apd.fit)
    assert all(rule_descriptions == exp_results)


def test_rule_descriptions_from_X_rules(create_data):
    expected_results_wo_weights_metric = np.array([
        [0.5097276264591439, 0.5137254901960784, 0.514, None],
        [0.518664047151277, 0.5176470588235295, 0.509, None],
        [0.5134615384615384, 0.5235294117647059, 0.52, None],
        [0.5335968379446641, 0.5294117647058824, 0.506, None],
        [0.532803180914513, 0.5254901960784314, 0.503, None],
        [0.5454545454545454, 0.5294117647058824, 0.495, None],
        [0.5072164948453608, 0.4823529411764706, 0.485, None],
        [0.5162523900573613, 0.5294117647058824, 0.523, None],
        [0.5075757575757576, 0.5254901960784314, 0.528, None],
        [0.5059760956175299, 0.4980392156862745, 0.502, None]],
        dtype=object)
    expected_results_weights_wo_metric = np.array([
        [1.0, 0.5137254901960784, 0.514, None],
        [1.0, 0.5176470588235295, 0.509, None],
        [1.0, 0.5235294117647059, 0.52, None],
        [1.0, 0.5294117647058824, 0.506, None],
        [1.0, 0.5254901960784314, 0.503, None],
        [1.0, 0.5294117647058824, 0.495, None],
        [1.0, 0.4823529411764706, 0.485, None],
        [1.0, 0.5294117647058824, 0.523, None],
        [1.0, 0.5254901960784314, 0.528, None],
        [1.0, 0.4980392156862745, 0.502, None]],
        dtype=object)
    expected_results_weights_metric = np.array([
        [1., 0.51372549, 0.514, 0.67875648],
        [1., 0.51764706, 0.509, 0.68217054],
        [1., 0.52352941, 0.52, 0.68725869],
        [1., 0.52941176, 0.506, 0.69230769],
        [1., 0.5254902, 0.503, 0.68894602],
        [1., 0.52941176, 0.495, 0.69230769],
        [1., 0.48235294, 0.485, 0.65079365],
        [1., 0.52941176, 0.523, 0.69230769],
        [1., 0.5254902, 0.528, 0.68894602],
        [1., 0.49803922, 0.502, 0.66492147]])
    y_true, y_preds, sample_weight = create_data
    f1 = FScore(1)
    rule_descriptions = utils.return_binary_pred_perf_of_set(
        y_true, y_preds, y_preds_columns=y_preds.columns
    )
    np.testing.assert_array_equal(
        rule_descriptions.values, expected_results_wo_weights_metric)
    rule_descriptions = utils.return_binary_pred_perf_of_set(
        y_true, y_preds, y_preds_columns=y_preds.columns,
        sample_weight=sample_weight
    )
    np.testing.assert_array_equal(
        rule_descriptions.values, expected_results_weights_wo_metric)
    rule_descriptions = utils.return_binary_pred_perf_of_set(
        y_true, y_preds, y_preds_columns=y_preds.columns,
        sample_weight=sample_weight, metric=f1.fit
    )
    np.testing.assert_array_almost_equal(
        rule_descriptions.values, expected_results_weights_metric)


def test_flatten_stringified_json_column():
    X = pd.DataFrame({
        'sim_ll': [
            json.dumps({"A": 10, "B": -1}),
            json.dumps({"A": 10, "C": -2}),
            json.dumps({"B": -1, "D": -1}),
            json.dumps({"A": 10, "B": -1})
        ]
    })
    expected_X = pd.DataFrame({
        'A': [10, 10, np.nan, 10],
        'B': [-1, np.nan, -1, -1],
        'C': [np.nan, -2, np.nan, np.nan],
        'D': [np.nan, np.nan, -1, np.nan]
    })
    X_flattened = utils.flatten_stringified_json_column(X['sim_ll'])
    assert all(X_flattened == expected_X)


def test_count_rule_conditions():
    rule_strings = {
        "(X['A']>=1.0)|(X['A'].isna())": 2,
        "(X['A']>=2)": 1
    }
    for rule_string, expected_num_conditions in rule_strings.items():
        num_conditions = utils.count_rule_conditions(
            rule_string=rule_string)
        assert num_conditions == expected_num_conditions


def test_return_progress_ready_range():
    assert isinstance(utils.return_progress_ready_range(
        verbose=True, range=[1, 2, 3]), tqdm)
    assert isinstance(utils.return_progress_ready_range(
        verbose=False, range=[1, 2, 3]), list)


def test_return_conf_matrix():
    y = pd.Series([1, 0, 1, 0, 0])
    y_pred = pd.Series([1, 0, 0, 1, 0])
    weights = pd.Series([2, 1, 2, 1, 1])
    exp_conf_mat = np.array([
        [1, 1],
        [1, 2]
    ])
    exp_conf_mat_weighted = np.array([
        [2, 1],
        [2, 2]
    ])
    conf_matrix = utils.return_conf_matrix(y, y_pred, None)
    np.testing.assert_array_almost_equal(conf_matrix.values, exp_conf_mat)
    conf_matrix_weighted = utils.return_conf_matrix(y, y_pred, weights)
    np.testing.assert_array_almost_equal(
        conf_matrix_weighted.values, exp_conf_mat_weighted)


def test_check_allowed_types():
    X = pd.DataFrame()
    utils.check_allowed_types(X, 'X', [PandasDataFrame])
    with pytest.raises(TypeError, match="`X` must be a pandas.core.frame.DataFrame. Current type is int."):
        utils.check_allowed_types(2, 'X', [PandasDataFrame])


def _test_y_preds(y_preds, rule_descriptions, y, sample_weight):
    if y_preds.ndim == 1:
        y_preds = pd.DataFrame(y_preds)
    for col in y_preds:
        precision = precision_score(
            y, y_preds[col], sample_weight=sample_weight)
        recall = recall_score(y, y_preds[col], sample_weight=sample_weight)
        perc_data_flagged = y_preds[col].mean()
        opt_metric = fbeta_score(
            y, y_preds[col], beta=1, sample_weight=sample_weight)
        assert precision == rule_descriptions.loc[col, 'Precision']
        assert recall == rule_descriptions.loc[col, 'Recall']
        assert perc_data_flagged == rule_descriptions.loc[col,
                                                          'PercDataFlagged']
        assert opt_metric == rule_descriptions.loc[col, 'Metric']
