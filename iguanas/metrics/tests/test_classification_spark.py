import databricks.koalas as ks
import numpy as np
from iguanas.metrics import Precision, Recall, FScore, Revenue, Bounds
import sklearn.metrics as sklearn_metrics
import pytest


@pytest.fixture
def create_data():
    np.random.seed(0)
    y_pred = np.random.randint(0, 2, 1000)
    y_true = np.random.randint(0, 2, 1000)
    y_preds = np.random.randint(0, 2, (1000, 2))
    weights = (y_true * 10) + 1
    y_true_ks, y_pred_ks, y_preds_ks, weights_ks = ks.Series(y_true, name='label_'), ks.Series(
        y_pred, name='A'), ks.DataFrame(y_preds, columns=['A', 'B']), ks.Series(weights, name='sample_weight_')
    return y_true_ks, y_pred_ks, weights_ks, y_true, y_pred, weights, y_preds_ks, y_preds


def test_Precision(create_data):
    y_true_ks, y_pred_ks, weights_ks, y_true, y_pred, weights, y_preds_ks, y_preds = create_data
    precision = Precision()
    weights_pd = [None, weights]
    for i, w in enumerate([None, weights_ks]):
        prec_calc = precision.fit(y_pred_ks, y_true_ks, w)
        w_pd = weights_pd[i]
        prec_exp = sklearn_metrics.precision_score(
            y_true, y_pred, sample_weight=w_pd)
        assert prec_calc == prec_exp
    for i, w in enumerate([None, weights_ks]):
        prec_calc = precision.fit(y_preds_ks, y_true_ks, w)
        w_pd = weights_pd[i]
        prec_exp = np.array([
            sklearn_metrics.precision_score(
                y_true, y_preds[:, 0], sample_weight=w_pd),
            sklearn_metrics.precision_score(
                y_true, y_preds[:, 1], sample_weight=w_pd)]
        )
        assert all(prec_calc == prec_exp)


def test_Recall(create_data):
    y_true_ks, y_pred_ks, weights_ks, y_true, y_pred, weights, y_preds_ks, y_preds = create_data
    recall = Recall()
    weights_pd = [None, weights]
    for i, w in enumerate([None, weights_ks]):
        recall_calc = recall.fit(y_pred_ks, y_true_ks, w)
        w_pd = weights_pd[i]
        recall_exp = sklearn_metrics.recall_score(
            y_true, y_pred, sample_weight=w_pd)
        assert recall_calc == recall_exp
    for i, w in enumerate([None, weights_ks]):
        recall_calc = recall.fit(y_preds_ks, y_true_ks, w)
        w_pd = weights_pd[i]
        recall_exp = np.array([
            sklearn_metrics.recall_score(
                y_true, y_preds[:, 0], sample_weight=w_pd),
            sklearn_metrics.recall_score(
                y_true, y_preds[:, 1], sample_weight=w_pd)]
        )


def test_FScore(create_data):
    y_true_ks, y_pred_ks, weights_ks, y_true, y_pred, weights, y_preds_ks, y_preds = create_data
    f1 = FScore(1)
    weights_pd = [None, weights]
    for i, w in enumerate([None, weights_ks]):
        f1_calc = f1.fit(y_pred_ks, y_true_ks, w)
        w_pd = weights_pd[i]
        f1_exp = sklearn_metrics.fbeta_score(
            y_true, y_pred, beta=1, sample_weight=w_pd)
        assert f1_calc == f1_exp
    for i, w in enumerate([None, weights_ks]):
        f1_calc = f1.fit(y_preds_ks, y_true_ks, w)
        w_pd = weights_pd[i]
        f1_exp = np.array([
            sklearn_metrics.fbeta_score(
                y_true, y_preds[:, 0], sample_weight=w_pd, beta=1),
            sklearn_metrics.fbeta_score(
                y_true, y_preds[:, 1], sample_weight=w_pd, beta=1)])


def test_Revenue(create_data):
    np.random.seed(0)
    y_true_ks, y_pred_ks, _, _, _, _, y_preds_ks, _ = create_data
    amts = ks.Series(np.random.uniform(0, 1000, 1000))
    r = Revenue(y_type='Fraud', chargeback_multiplier=2)
    rev_calc = r.fit(y_pred_ks, y_true_ks, amts)
    rev_exp = 40092.775872
    assert round(rev_calc, 6) == rev_exp
    rev_calc = r.fit(y_preds_ks, y_true_ks, amts)
    rev_exp = np.array([12718.45497753, 46041.48353109])
    np.testing.assert_array_almost_equal(rev_calc, rev_exp)


def test_Bounds(create_data):
    y_true_ks, y_pred_ks, weights_ks, _, _, _, y_preds_ks, _ = create_data
    p = Precision()
    r = Recall()
    bounds = [
        {
            'metric': p.fit,
            'operator': '>=',
            'threshold': 0.93
        },
        {
            'metric': r.fit,
            'operator': '>=',
            'threshold': 0.53
        }
    ]
    b = Bounds(bounds=bounds)
    result = b.fit(y_pred_ks, y_true_ks)
    assert result == 0.4031633634152096
    result = b.fit(y_preds_ks, y_true_ks)
    np.testing.assert_array_almost_equal(
        result, np.array([0.39481936, 0.40650979])
    )
    result = b.fit(y_pred_ks, y_true_ks, weights_ks)
    assert result == 0.49905339918919434
    result = b.fit(y_preds_ks, y_true_ks, weights_ks)
    np.testing.assert_array_almost_equal(
        result, np.array([0.49371392, 0.50029537])
    )
