import pytest
import numpy as np
import pandas as pd
import databricks.koalas as ks
from iguanas.rule_generation import RuleGeneratorDTSpark
from iguanas.metrics.classification import FScore
from pyspark.ml.classification import RandomForestClassifier as RandomForestClassifierSpark
from pyspark.ml.feature import VectorAssembler
import random


@pytest.fixture
def create_data():
    def return_random_num(y, fraud_min, fraud_max, nonfraud_min, nonfraud_max, rand_func):
        data = [rand_func(fraud_min, fraud_max) if i == 1 else rand_func(
            nonfraud_min, nonfraud_max) for i in y]
        return data

    random.seed(0)
    np.random.seed(0)
    y = pd.Series(data=[0]*95 + [1]*5, index=list(range(0, 100)))
    X = ks.DataFrame(data={
        "num_distinct_txn_per_email_1day": [round(max(i, 0)) for i in return_random_num(y, 2, 1, 1, 2, np.random.normal)],
        "num_distinct_txn_per_email_7day": [round(max(i, 0)) for i in return_random_num(y, 4, 2, 2, 3, np.random.normal)],
        "ip_country_us": [round(min(i, 1)) for i in [max(i, 0) for i in return_random_num(y, 0.3, 0.4, 0.5, 0.5, np.random.normal)]],
        "email_kb_distance": [min(i, 1) for i in [max(i, 0) for i in return_random_num(y, 0.2, 0.5, 0.6, 0.4, np.random.normal)]],
        "email_alpharatio":  [min(i, 1) for i in [max(i, 0) for i in return_random_num(y, 0.33, 0.1, 0.5, 0.2, np.random.normal)]],
    },
        index=list(range(0, 100))
    )
    columns_int = [
        'num_distinct_txn_per_email_1day', 'num_distinct_txn_per_email_7day', 'ip_country_us']
    columns_cat = ['ip_country_us']
    columns_num = ['num_distinct_txn_per_email_1day',
                   'num_distinct_txn_per_email_7day', 'email_kb_distance', 'email_alpharatio']
    weights = ks.Series(y.apply(lambda x: 100 if x == 1 else 1))
    y = ks.from_pandas(y)
    return [X, y, columns_int, columns_cat, columns_num, weights]


@pytest.fixture
def fs_instantiated():
    f = FScore(0.5)
    return f.fit


@pytest.fixture
def rg_instantiated(fs_instantiated):
    f0dot5 = fs_instantiated
    params = {
        'metric': f0dot5,
        'n_total_conditions': 4,
        'tree_ensemble': RandomForestClassifierSpark(bootstrap=False, numTrees=1, impurity='gini'),
        'verbose': 1
    }
    rgs = RuleGeneratorDTSpark(**params)
    rgs._today = '20200204'
    return rgs, params


@pytest.fixture
def create_spark_df(create_data):
    X, y, _, _, _, weights = create_data
    spark_df = X.join(y.rename('label_')).to_spark()
    spark_df_weights = X.join(y.rename('label_')).join(
        weights.rename('sample_weight_')).to_spark()
    vectorAssembler = VectorAssembler(
        inputCols=X.columns.tolist(), outputCol="features")
    spark_df = vectorAssembler.transform(spark_df)
    spark_df_weights = vectorAssembler.transform(spark_df_weights)
    return spark_df, spark_df_weights


@pytest.fixture
def train_rf(create_spark_df):
    spark_df, _ = create_spark_df
    rf = RandomForestClassifierSpark(
        bootstrap=False, numTrees=1, seed=0, labelCol='label_',
        featuresCol='features', impurity='gini'
    )
    trained_rf = rf.fit(spark_df)
    return trained_rf


def test_fit(create_data, rg_instantiated):
    exp_results = [
        np.array([3, 5]),
        np.array([6])
    ]
    X, y, _, _, _, weights = create_data
    rg, params = rg_instantiated
    exp_repr = 'RuleGeneratorDTSpark(metric=<bound method FScore.fit of FScore with beta=0.5>, n_total_conditions=4, tree_ensemble=RandomForestClassifier, precision_threshold=0, target_feat_corr_types=None)'
    assert rg.__repr__() == exp_repr
    # Test without weights
    X_rules = rg.fit(X, y, None)
    np.testing.assert_array_equal(X_rules.sum().to_numpy(), exp_results[0])
    assert rg.rule_strings == rg.rules.rule_strings == {
        'RGDT_Rule_20200204_0': "(X['email_alpharatio']<=0.29912)&(X['email_alpharatio']>0.22241)&(X['num_distinct_txn_per_email_7day']>=4)",
        'RGDT_Rule_20200204_1': "(X['email_alpharatio']<=0.33992)&(X['email_alpharatio']>0.29912)&(X['num_distinct_txn_per_email_7day']>=4)"
    }
    assert rg.rule_names == list(rg.rule_lambdas.keys()) == list(rg.rules.rule_lambdas.keys()) == list(
        rg.lambda_kwargs.keys()) == ['RGDT_Rule_20200204_0', 'RGDT_Rule_20200204_1']
    assert rg.__repr__() == "RuleGeneratorDTSpark object with 2 rules generated"
    # Test with weights
    X_rules = rg.fit(X, y, weights)
    np.testing.assert_array_equal(X_rules.sum().to_numpy(), exp_results[1])
    assert rg.rule_strings == rg.rules.rule_strings == {
        'RGDT_Rule_20200204_2': "(X['email_alpharatio']<=0.32306)&(X['email_alpharatio']>0.22241)&(X['num_distinct_txn_per_email_1day']>=1)&(X['num_distinct_txn_per_email_7day']>=4)"
    }
    assert rg.rule_names == list(rg.rule_lambdas.keys()) == list(rg.rules.rule_lambdas.keys()) == list(
        rg.lambda_kwargs.keys()) == ['RGDT_Rule_20200204_2']
    # Test fit_transform without weights
    X_rules = rg.fit_transform(X, y, None)
    np.testing.assert_array_equal(X_rules.sum().to_numpy(), exp_results[0])
    assert rg.rule_strings == rg.rules.rule_strings == {
        'RGDT_Rule_20200204_3': "(X['email_alpharatio']<=0.29912)&(X['email_alpharatio']>0.22241)&(X['num_distinct_txn_per_email_7day']>=4)",
        'RGDT_Rule_20200204_4': "(X['email_alpharatio']<=0.33992)&(X['email_alpharatio']>0.29912)&(X['num_distinct_txn_per_email_7day']>=4)"
    }
    assert rg.rule_names == list(rg.rule_lambdas.keys()) == list(rg.rules.rule_lambdas.keys()) == list(
        rg.lambda_kwargs.keys()) == ['RGDT_Rule_20200204_3', 'RGDT_Rule_20200204_4']
    # Test fit_transform without weights
    X_rules = rg.fit_transform(X, y, weights)
    np.testing.assert_array_equal(X_rules.sum().to_numpy(), exp_results[1])
    assert rg.rule_strings == rg.rules.rule_strings == {
        'RGDT_Rule_20200204_5': "(X['email_alpharatio']<=0.32306)&(X['email_alpharatio']>0.22241)&(X['num_distinct_txn_per_email_1day']>=1)&(X['num_distinct_txn_per_email_7day']>=4)"
    }
    assert rg.rule_names == list(rg.rule_lambdas.keys()) == list(rg.rules.rule_lambdas.keys()) == list(
        rg.lambda_kwargs.keys()) == ['RGDT_Rule_20200204_5']


def test_fit_target_feat_corr_types_infer(create_data, rg_instantiated):
    exp_results = [
        np.array([5, 8, 25, 13]),
        np.array([5, 23, 10, 12])
    ]
    X, y, _, _, _, weights = create_data
    rg, params = rg_instantiated
    rg.precision_threshold = -1
    rg.target_feat_corr_types = 'Infer'
    # Test without weights
    X_rules = rg.fit(X, y, None)
    np.testing.assert_array_equal(X_rules.sum().to_numpy(), exp_results[0])
    assert rg.rule_strings == rg.rules.rule_strings == {
        'RGDT_Rule_20200204_0': "(X['email_alpharatio']<=0.22241)&(X['num_distinct_txn_per_email_7day']>=4)",
        'RGDT_Rule_20200204_1': "(X['email_alpharatio']<=0.29912)&(X['num_distinct_txn_per_email_7day']>=4)",
        'RGDT_Rule_20200204_2': "(X['email_alpharatio']<=0.33992)",
        'RGDT_Rule_20200204_3': "(X['email_alpharatio']<=0.33992)&(X['num_distinct_txn_per_email_7day']>=4)"
    }
    assert rg.rule_names == list(rg.rule_lambdas.keys()) == list(rg.rules.rule_lambdas.keys()) == list(rg.lambda_kwargs.keys()) == [
        'RGDT_Rule_20200204_0', 'RGDT_Rule_20200204_1', 'RGDT_Rule_20200204_2',
        'RGDT_Rule_20200204_3'
    ]
    # Test with weights
    X_rules = rg.fit(X, y, weights)
    np.testing.assert_array_equal(X_rules.sum().to_numpy(), exp_results[1])
    assert rg.rule_strings == rg.rules.rule_strings == {
        'RGDT_Rule_20200204_4': "(X['email_alpharatio']<=0.22241)&(X['num_distinct_txn_per_email_7day']>=4)",
        'RGDT_Rule_20200204_5': "(X['email_alpharatio']<=0.32306)",
        'RGDT_Rule_20200204_6': "(X['email_alpharatio']<=0.32306)&(X['num_distinct_txn_per_email_1day']>=1)&(X['num_distinct_txn_per_email_7day']>=4)",
        'RGDT_Rule_20200204_7': "(X['email_alpharatio']<=0.32306)&(X['num_distinct_txn_per_email_7day']>=4)"
    }
    assert rg.rule_names == list(rg.rule_lambdas.keys()) == list(rg.rules.rule_lambdas.keys()) == list(rg.lambda_kwargs.keys()) == [
        'RGDT_Rule_20200204_4', 'RGDT_Rule_20200204_5', 'RGDT_Rule_20200204_6',
        'RGDT_Rule_20200204_7'
    ]
    # Test fit_transform without weights
    X_rules = rg.fit_transform(X, y, None)
    np.testing.assert_array_equal(X_rules.sum().to_numpy(), exp_results[0])
    assert rg.rule_strings == rg.rules.rule_strings == {
        'RGDT_Rule_20200204_8': "(X['email_alpharatio']<=0.22241)&(X['num_distinct_txn_per_email_7day']>=4)",
        'RGDT_Rule_20200204_9': "(X['email_alpharatio']<=0.29912)&(X['num_distinct_txn_per_email_7day']>=4)",
        'RGDT_Rule_20200204_10': "(X['email_alpharatio']<=0.33992)",
        'RGDT_Rule_20200204_11': "(X['email_alpharatio']<=0.33992)&(X['num_distinct_txn_per_email_7day']>=4)"
    }
    assert rg.rule_names == list(rg.rule_lambdas.keys()) == list(rg.rules.rule_lambdas.keys()) == list(rg.lambda_kwargs.keys()) == [
        'RGDT_Rule_20200204_8', 'RGDT_Rule_20200204_9', 'RGDT_Rule_20200204_10',
        'RGDT_Rule_20200204_11'
    ]
    # Test fit_transform without weights
    X_rules = rg.fit_transform(X, y, weights)
    np.testing.assert_array_equal(X_rules.sum().to_numpy(), exp_results[1])
    assert rg.rule_strings == rg.rules.rule_strings == {
        'RGDT_Rule_20200204_12': "(X['email_alpharatio']<=0.22241)&(X['num_distinct_txn_per_email_7day']>=4)",
        'RGDT_Rule_20200204_13': "(X['email_alpharatio']<=0.32306)",
        'RGDT_Rule_20200204_14': "(X['email_alpharatio']<=0.32306)&(X['num_distinct_txn_per_email_1day']>=1)&(X['num_distinct_txn_per_email_7day']>=4)",
        'RGDT_Rule_20200204_15': "(X['email_alpharatio']<=0.32306)&(X['num_distinct_txn_per_email_7day']>=4)"
    }
    assert rg.rule_names == list(rg.rule_lambdas.keys()) == list(rg.rules.rule_lambdas.keys()) == list(rg.lambda_kwargs.keys()) == [
        'RGDT_Rule_20200204_12', 'RGDT_Rule_20200204_13', 'RGDT_Rule_20200204_14',
        'RGDT_Rule_20200204_15'
    ]


def test_extract_rules_from_ensemble(create_data, train_rf, rg_instantiated):
    X, y, columns_int, columns_cat, _, _ = create_data
    rf_trained = train_rf
    rg, _ = rg_instantiated
    rg.precision_threshold = -1
    X_rules = rg._extract_rules_from_ensemble(
        X, y, None, rf_trained, columns_int, columns_cat
    )
    assert X_rules.shape == (100, 6)
    assert all(np.sort(X_rules.sum().to_numpy())
               == np.array([2, 3, 3, 5, 12, 75]))


def test_extract_rules_from_ensemble_error(rg_instantiated, train_rf):
    rg, _ = rg_instantiated
    X = ks.DataFrame({
        'A': [0, 0, 0],
        'B': [1, 1, 1]
    })
    y = ks.Series([0, 1, 0])
    spark_df = X.join(y.rename('label_')).to_spark()
    vectorAssembler = VectorAssembler(
        inputCols=X.columns.tolist(), outputCol="features"
    )
    spark_df = vectorAssembler.transform(spark_df)
    rf = RandomForestClassifierSpark(
        bootstrap=False, numTrees=1, seed=0, labelCol='label_',
        featuresCol='features', impurity='gini'
    )
    rf_trained = rf.fit(spark_df)
    with pytest.raises(Exception, match='No rules could be generated. Try changing the class parameters.'):
        with pytest.warns(UserWarning, match='Decision Tree 0 has a depth of zero - skipping'):
            X_rules = rg._extract_rules_from_ensemble(
                X, y, None, rf_trained, ['A', 'B'], ['A', 'B']
            )


def test_extract_rules_from_dt(create_data, train_rf, rg_instantiated):
    expected_results = {
        "(X['email_alpharatio']<=0.22241)&(X['num_distinct_txn_per_email_7day']>=4)",
        "(X['email_alpharatio']<=0.29912)&(X['email_alpharatio']>0.22241)&(X['num_distinct_txn_per_email_7day']>=4)",
        "(X['email_alpharatio']<=0.30749)&(X['email_alpharatio']>0.29912)&(X['num_distinct_txn_per_email_7day']>=4)",
        "(X['email_alpharatio']<=0.33992)&(X['email_alpharatio']>0.30749)&(X['num_distinct_txn_per_email_7day']>=4)",
        "(X['email_alpharatio']<=0.33992)&(X['num_distinct_txn_per_email_7day']<=3)",
        "(X['email_alpharatio']>0.33992)"
    }
    X, _, columns_int, columns_cat, _, _ = create_data
    rf_trained = train_rf
    rg, _ = rg_instantiated
    rg.precision_threshold = -1
    rule_strings = rg._extract_rules_from_dt(
        X.columns.tolist(), rf_trained.trees[0], columns_int, columns_cat)
    assert rule_strings == expected_results
    # Test returns empty set when less that precision_threshold
    rg.precision_threshold = 1
    rule_strings = rg._extract_rules_from_dt(
        X.columns.tolist(), rf_trained.trees[0], columns_int, columns_cat)
    assert rule_strings == set()


def test_create_train_spark_df(create_data, create_spark_df, rg_instantiated):
    X, y, _, _, _, weights = create_data
    expected_spark_df, expected_spark_df_weights = create_spark_df
    rg, _ = rg_instantiated
    spark_df = rg._create_train_spark_df(X=X, y=y, sample_weight=None)
    assert spark_df.columns == expected_spark_df.columns
    spark_df_weights = rg._create_train_spark_df(
        X=X, y=y, sample_weight=weights)
    assert spark_df_weights.columns == expected_spark_df_weights.columns
    assert spark_df.count() == spark_df_weights.count() == 100


def test_get_pyspark_tree_structure(train_rf, rg_instantiated):
    expected_left_children, expected_right_children, expected_features, expected_thresholds, expected_precisions = (
        np.array([1,  2, -1,  4, -1,  6, -1,  8, -1, -1, -1]),
        np.array([10,  3, -1,  5, -1,  7, -1,  9, -1, -1, -1]),
        np.array([4,  1, -2,  4, -2,  4, -2,  4, -2, -2, -2]),
        np.array([0.33992341, 3.5, -2.,  0.22240729, -2.,
                  0.29912349, -2., 0.30748836, -2., -2., -2.]),
        np.array([0.05, 0.2, 0., 0.38461538, 0., 0.625, 1.,
                  0.4, 0., 0.66666667, 0.])
    )
    rf_trained = train_rf
    rg, _ = rg_instantiated
    left_children, right_children, features, thresholds, precisions, tree_prec = rg._get_pyspark_tree_structure(
        rf_trained.trees[0]._call_java('rootNode'))
    assert all(left_children == expected_left_children)
    assert all(right_children == expected_right_children)
    assert all(features == expected_features)
    np.testing.assert_array_almost_equal(thresholds, expected_thresholds)
    np.testing.assert_array_almost_equal(precisions, expected_precisions)
    assert tree_prec == 0.8333333333333334


def test_transform(create_data, rg_instantiated):
    X, y, _, _, _, _ = create_data
    rg, _ = rg_instantiated
    rg.rule_strings = {'Rule1': "(X['num_distinct_txn_per_email_1day']>1)"}
    X_rules = rg.transform(X)
    assert X_rules.sum()[0] == 44


def test_errors(create_data, rg_instantiated):
    X, y, _, _, _, _ = create_data
    rg, _ = rg_instantiated
    with pytest.raises(TypeError, match='`X` must be a databricks.koalas.frame.DataFrame. Current type is list.'):
        rg.fit(X=[], y=y)
    with pytest.raises(TypeError, match='`y` must be a databricks.koalas.series.Series. Current type is list.'):
        rg.fit(X=X, y=[])
    with pytest.raises(TypeError, match='`sample_weight` must be a databricks.koalas.series.Series. Current type is list.'):
        rg.fit(X=X, y=y, sample_weight=[])


# Methods below are part of _base module ------------------------------------


def test_extract_rules_from_tree(train_rf, rg_instantiated, create_data):

    expected_rule_sets = set([
        "(X['email_alpharatio']<=0.29912)&(X['email_alpharatio']>0.22241)&(X['num_distinct_txn_per_email_7day']>=4)",
        "(X['email_alpharatio']<=0.33992)&(X['email_alpharatio']>0.30749)&(X['num_distinct_txn_per_email_7day']>=4)"
    ])
    X, _, columns_int, columns_cat, _, _ = create_data
    rg, _ = rg_instantiated
    rf_trained = train_rf
    left, right, features, thresholds, precisions, _ = rg._get_pyspark_tree_structure(
        rf_trained.trees[0]._call_java('rootNode'))
    rule_sets = rg._extract_rules_from_tree(
        columns=X.columns.tolist(), precision_threshold=rg.precision_threshold,
        columns_int=columns_int, columns_cat=columns_cat, left=left,
        right=right, features=features, thresholds=thresholds,
        precisions=precisions
    )
    assert rule_sets == expected_rule_sets
    rg, _ = rg_instantiated
    rg.precision_threshold = 1
    left, right, features, thresholds, precisions, _ = rg._get_pyspark_tree_structure(
        rf_trained.trees[0]._call_java('rootNode'))
    rule_sets = rg._extract_rules_from_tree(
        columns=X.columns.tolist(), precision_threshold=rg.precision_threshold,
        columns_int=columns_int, columns_cat=columns_cat, left=left,
        right=right, features=features, thresholds=thresholds,
        precisions=precisions
    )
    assert rule_sets == set()


def test_calc_target_ratio_wrt_features(rg_instantiated):
    X = ks.DataFrame({
        'A': [10, 9, 8, 7, 6, 5],
        'B': [10, 10, 0, 0, 0, 0],
        'C': [0, 1, 2, 3, 4, 5],
        'D': [0, 0, 10, 10, 10, 10]
    })
    y = ks.Series([1, 1, 1, 0, 0, 0])
    expected_result = {
        'PositiveCorr': ['A', 'B'],
        'NegativeCorr': ['C', 'D'],
    }
    rg, _ = rg_instantiated
    target_feat_corr_types = rg._calc_target_ratio_wrt_features(X, y)
    assert target_feat_corr_types == expected_result
