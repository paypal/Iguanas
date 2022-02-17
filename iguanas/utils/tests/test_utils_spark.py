from iguanas.metrics import FScore, AlertsPerDay
import iguanas.utils as utils
import pytest
import numpy as np
import pandas as pd
import databricks.koalas as ks
from sklearn.metrics import fbeta_score, precision_score, recall_score
import string


@pytest.fixture
def create_data():
    np.random.seed(0)
    y_preds = ks.DataFrame(np.random.randint(0, 2, size=(100, 10)), columns=[
                           i for i in string.ascii_letters[:10]])
    y_true = ks.Series(np.random.randint(0, 2, 100))
    sample_weight = y_true * 10
    return (y_true, y_preds, sample_weight)


def test_concat(create_data):
    y_true, y_preds, _ = create_data
    X = utils.concat([y_preds, y_true], axis=1)
    with ks.option_context("compute.ops_on_diff_frames", True):
        pd.testing.assert_frame_equal(X.to_pandas(), ks.concat(
            [y_preds, y_true], axis=1).to_pandas())


def test_return_columns_types():
    X = ks.DataFrame({
        'A': [2.5, 3.5, 1, 1, 2.5],
        'B': [1, 0, 0, 1, 1],
        'C': [1, 2, 0, 0, 1]
    })
    int_cols, cat_cols, float_cols = utils.return_columns_types(X)
    assert int_cols == ['B', 'C']
    assert cat_cols == ['B']
    assert float_cols == ['A']
    # Test when all ints
    X = ks.DataFrame({
        'A': [1, 0, 0, 1, 1],
        'B': [1, 0, 0, 1, 1],
    })
    int_cols, cat_cols, float_cols = utils.return_columns_types(X)
    assert int_cols == ['A', 'B']
    assert cat_cols == ['A', 'B']
    assert float_cols == []
    # Test when all floats
    X = ks.DataFrame({
        'A': [1.2, 0, 0, 1.1, 1],
        'B': [1.5, 0, 0, 1, 1.1],
    })
    int_cols, cat_cols, float_cols = utils.return_columns_types(X)
    assert int_cols == []
    assert cat_cols == []
    assert float_cols == ['A', 'B']


def test_create_spark_df(create_data):
    y_true, y_preds, sample_weight = create_data
    spark_df = utils.create_spark_df(y_preds, y_true)
    assert spark_df.columns == ['a', 'b', 'c', 'd',
                                'e', 'f', 'g', 'h', 'i', 'j', 'label_']
    assert spark_df.count() == 100
    spark_df = utils.create_spark_df(y_preds.iloc[:, 0], y_true)
    assert spark_df.columns == ['a', 'label_']
    assert spark_df.count() == 100
    spark_df = utils.create_spark_df(y_preds, y_true, sample_weight)
    assert spark_df.columns == ['a', 'b', 'c', 'd', 'e',
                                'f', 'g', 'h', 'i', 'j', 'label_', 'sample_weight_']
    assert spark_df.count() == 100


def test_return_rule_descriptions_from_X_rules(create_data):
    y_true, y_preds, _ = create_data
    f1 = FScore(beta=1)
    exp_results = pd.DataFrame(
        np.array([
            [0.47169811, 0.54347826, 0.53, 0.50505051],
            [0.42857143, 0.52173913, 0.56, 0.47058824],
            [0.56862745, 0.63043478, 0.51, 0.59793814],
            [0.4375, 0.45652174, 0.48, 0.44680851],
            [0.42857143, 0.39130435, 0.42, 0.40909091],
            [0.45652174, 0.45652174, 0.46, 0.45652174],
            [0.45454545, 0.54347826, 0.55, 0.4950495],
            [0.48, 0.52173913, 0.5, 0.5],
            [0.47368421, 0.58695652, 0.57, 0.52427184],
            [0.47826087, 0.47826087, 0.46, 0.47826087]]),
        columns=['Precision', 'Recall', 'PercDataFlagged', 'Metric'],
        index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    )
    exp_results.index.name = 'Rule'
    rule_descriptions = utils.return_rule_descriptions_from_X_rules(
        X_rules=y_preds, X_rules_cols=y_preds.columns, y_true=y_true,
        sample_weight=None, metric=f1.fit)
    assert all(rule_descriptions == exp_results)
    rule_descriptions = utils.return_rule_descriptions_from_X_rules(
        X_rules=y_preds, X_rules_cols=y_preds.columns, y_true=y_true,
        sample_weight=None)
    exp_results['Metric'] = None
    assert all(rule_descriptions == exp_results)


def test_return_rule_descriptions_from_X_rules_weighted(create_data):
    y_true, y_preds, weights = create_data
    f1 = FScore(beta=1)
    exp_results = pd.DataFrame(
        np.array([
            [1., 0.54347826, 0.53, 0.70422535],
            [1., 0.52173913, 0.56, 0.68571429],
            [1., 0.63043478, 0.51, 0.77333333],
            [1., 0.45652174, 0.48, 0.62686567],
            [1., 0.39130435, 0.42, 0.5625],
            [1., 0.45652174, 0.46, 0.62686567],
            [1., 0.54347826, 0.55, 0.70422535],
            [1., 0.52173913, 0.5, 0.68571429],
            [1., 0.58695652, 0.57, 0.73972603],
            [1., 0.47826087, 0.46, 0.64705882]]),
        columns=['Precision', 'Recall', 'PercDataFlagged', 'Metric'],
        index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    )
    exp_results.index.name = 'Rule'
    rule_descriptions = utils.return_rule_descriptions_from_X_rules(
        X_rules=y_preds, X_rules_cols=y_preds.columns, y_true=y_true,
        sample_weight=weights, metric=f1.fit)
    assert all(rule_descriptions == exp_results)
    rule_descriptions = utils.return_rule_descriptions_from_X_rules(
        X_rules=y_preds, X_rules_cols=y_preds.columns, y_true=y_true,
        sample_weight=weights)
    exp_results['Metric'] = None
    assert all(rule_descriptions == exp_results)


def test_return_rule_descriptions_from_X_rules_unlabelled(create_data):
    _, y_preds, _ = create_data
    apd = AlertsPerDay(n_alerts_expected_per_day=10, no_of_days_in_file=10)
    exp_results = pd.DataFrame(
        np.array([
            [0.53, -22.09],
            [0.56, -19.36],
            [0.51, -24.01],
            [0.48, -27.04],
            [0.42, -33.64],
            [0.46, -29.16],
            [0.55, -20.25],
            [0.5, -25.],
            [0.57, -18.49],
            [0.46, -29.16]]),
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


def test_calc_tps_fps_tns_fns(create_data):
    y_true, y_preds, weights = create_data
    # Without weights
    tps, fps, tns, fns, tps_fps, tps_fns = utils.calc_tps_fps_tns_fns(
        y_true, y_preds, tps=True, fps=True, tns=True, fns=True,
        tps_fps=True, tps_fns=True
    )
    assert all(tps == np.array([25, 24, 29, 21, 18, 21, 25, 24, 27, 22]))
    assert all(fps == np.array([28, 32, 22, 27, 24, 25, 30, 26, 30, 24]))
    assert all(tns == np.array([26, 22, 32, 27, 30, 29, 24, 28, 24, 30]))
    assert all(fns == np.array([21, 22, 17, 25, 28, 25, 21, 22, 19, 24]))
    assert all(tps_fps == np.array([53, 56, 51, 48, 42, 46, 55, 50, 57, 46]))
    assert all(tps_fns == np.array([46]))
    # With weights
    tps, fps, tns, fns, tps_fps, tps_fns = utils.calc_tps_fps_tns_fns(
        y_true, y_preds, weights, tps=True, fps=True, tns=True, fns=True,
        tps_fps=True, tps_fns=True
    )
    assert all(tps == np.array(
        [250, 240, 290, 210, 180, 210, 250, 240, 270, 220]))
    assert all(fps == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert all(tns == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert all(fns == np.array(
        [210, 220, 170, 250, 280, 250, 210, 220, 190, 240]))
    assert all(tps_fps == np.array(
        [250, 240, 290, 210, 180, 210, 250, 240, 270, 220]))
    assert all(tps_fns == np.array([460]))
    # One pred (without weights)
    tps, fps, tns, fns, tps_fps, tps_fns = utils.calc_tps_fps_tns_fns(
        y_true, y_preds['a'], tps=True, fps=True, tns=True, fns=True,
        tps_fps=True, tps_fns=True
    )
    assert (tps, fps, tns, fns, tps_fps, tps_fns) == (
        25, 28, 26, 21, 53, 46)
    # One pred (with weights)
    tps, fps, tns, fns, tps_fps, tps_fns = utils.calc_tps_fps_tns_fns(
        y_true, y_preds['a'], weights, tps=True, fps=True, tns=True, fns=True,
        tps_fps=True, tps_fns=True
    )
    assert (tps, fps, tns, fns, tps_fps, tps_fns) == (
        250, 0, 0, 210, 250, 460)


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


def test_rule_descriptions_from_X_rules(create_data):
    expected_results_wo_weights_metric = np.array([
        [0.4716981132075472, 0.5434782608695652, 0.53, None],
        [0.42857142857142855, 0.5217391304347826, 0.56, None],
        [0.5686274509803921, 0.6304347826086957, 0.51, None],
        [0.4375, 0.45652173913043476, 0.48, None],
        [0.42857142857142855, 0.391304347826087, 0.42, None],
        [0.45652173913043476, 0.45652173913043476, 0.46, None],
        [0.45454545454545453, 0.5434782608695652, 0.55, None],
        [0.48, 0.5217391304347826, 0.5, None],
        [0.47368421052631576, 0.5869565217391305, 0.57, None],
        [0.4782608695652174, 0.4782608695652174, 0.46, None]],
        dtype=object)
    expected_results_weights_wo_metric = np.array([
        [1.0, 0.5434782608695652, 0.53, None],
        [1.0, 0.5217391304347826, 0.56, None],
        [1.0, 0.6304347826086957, 0.51, None],
        [1.0, 0.45652173913043476, 0.48, None],
        [1.0, 0.391304347826087, 0.42, None],
        [1.0, 0.45652173913043476, 0.46, None],
        [1.0, 0.5434782608695652, 0.55, None],
        [1.0, 0.5217391304347826, 0.5, None],
        [1.0, 0.5869565217391305, 0.57, None],
        [1.0, 0.4782608695652174, 0.46, None]],
        dtype=object)
    expected_results_weights_metric = np.array([
        [1., 0.54347826, 0.53, 0.70422535],
        [1., 0.52173913, 0.56, 0.68571429],
        [1., 0.63043478, 0.51, 0.77333333],
        [1., 0.45652174, 0.48, 0.62686567],
        [1., 0.39130435, 0.42, 0.5625],
        [1., 0.45652174, 0.46, 0.62686567],
        [1., 0.54347826, 0.55, 0.70422535],
        [1., 0.52173913, 0.5, 0.68571429],
        [1., 0.58695652, 0.57, 0.73972603],
        [1., 0.47826087, 0.46, 0.64705882]])
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


def test_return_conf_matrix():
    y = ks.Series([1, 0, 1, 0, 0])
    y_pred = ks.Series([1, 0, 0, 1, 0], name='pred')
    weights = ks.Series([2, 1, 2, 1, 1])
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


def _test_y_preds(y_preds, rule_descriptions, y, sample_weight):
    y_preds = y_preds.to_pandas()
    y = y.to_pandas()
    if sample_weight is not None:
        sample_weight = sample_weight.to_pandas()
    if y_preds.ndim == 1:
        y_preds = pd.DataFrame(y_preds)
    for col in y_preds.columns:
        precision = precision_score(
            y, y_preds[col], sample_weight=sample_weight)
        recall = recall_score(y, y_preds[col], sample_weight=sample_weight)
        perc_data_flagged = y_preds[col].mean()
        opt_metric = fbeta_score(
            y, y_preds[col], beta=1, sample_weight=sample_weight)
        assert all(np.array([precision, recall, perc_data_flagged,
                             opt_metric]) == rule_descriptions.loc[col].values)
