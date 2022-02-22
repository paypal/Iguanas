"""Contains functions that are shared across Iguanas modules."""
from iguanas.utils.types import KoalasDataFrame, KoalasSeries, PandasDataFrame, \
    PandasSeries
from iguanas.utils.typing import KoalasDataFrameType, KoalasSeriesType, \
    PandasDataFrameType, PandasSeriesType, PySparkDataFrameType
import pandas as pd
import numpy as np
import json
from typing import List, Tuple, Union, Iterable
from tqdm import tqdm
import sys


def concat(objs: List[Union[PandasDataFrameType, PandasSeriesType, KoalasDataFrameType, KoalasSeriesType]],
           **kwargs) -> Union[PandasDataFrameType, KoalasDataFrameType]:
    """
    Concatenates a set of Pandas Series/DataFrames or a set of Koalas 
    Series/DataFrames.

    Parameters
    ----------
    objs : List[Union[PandasDataFrameType, PandasSeriesType, KoalasDataFrameType, KoalasSeriesType]]
        List of Pandas/Koalas DataFrame to concatenate.

    Raises
    ------
    Exception
        `objs` must be a list of either Pandas objects or Koalas objects.

    Returns
    -------
    Union[PandasDataFrameType, KoalasDataFrameType]
        The concatenated DataFrame.
    """

    check_allowed_types(objs[0], 'objs', [
        PandasSeries, PandasDataFrame, KoalasSeries, KoalasDataFrame
    ])
    if is_type(objs[0], [PandasSeries, PandasDataFrame]):
        return pd.concat(objs, **kwargs)
    if is_type(objs[0], [KoalasSeries, KoalasDataFrame]):
        import databricks.koalas as ks

        with ks.option_context("compute.ops_on_diff_frames", True):
            return ks.concat(objs, **kwargs)


def return_columns_types(X: Union[PandasDataFrameType, KoalasDataFrameType]) -> Tuple[List, List, List]:
    """
    Returns the integer, float and OHE categorical columns for a given dataset.

    Parameters
    ----------
    X : Union[PandasDataFrameType, KoalasDataFrameType])
        Dataset.

    Returns
    -------
    Tuple[List, List, List]
        List of integer columns, list of float columns, list of OHE categorical 
        columns.
    """

    num_cols = X.shape[1]
    int64_cols = list(
        X.dtypes.index[(X.dtypes == 'Int64') | (X.dtypes == 'int64')]
    )
    if len(int64_cols) == num_cols:
        int_cols = int64_cols
        float_cols = []
    elif int64_cols:
        X_no_int64 = X.drop(int64_cols, axis=1)
    else:
        X_no_int64 = X
    if is_type(X, [PandasDataFrame]) and len(int64_cols) < num_cols:
        int_mask = np.sum(X_no_int64.to_numpy() -
                          X_no_int64.to_numpy().round(), axis=0) == 0
    elif is_type(X, [KoalasDataFrame]) and len(int64_cols) < num_cols:
        int_mask = ((X_no_int64 - X_no_int64.round()).sum() == 0).to_numpy()
    if len(int64_cols) < num_cols:
        int_cols = int64_cols + list(X_no_int64.columns[int_mask])
        float_cols = list(X_no_int64.columns[~int_mask])
    if int_cols:
        poss_ohe_cols_mask = (X[int_cols].nunique() == 2)
        poss_ohe_cols = poss_ohe_cols_mask[poss_ohe_cols_mask].index.tolist()
        min_zero_mask = (X[poss_ohe_cols].min() == 0)
        max_one_mask = (X[poss_ohe_cols].max() == 1)
        ohe_mask = min_zero_mask.to_numpy() * max_one_mask.to_numpy()
        ohe_cat_cols = [poss_ohe_cols[i] for i, m in enumerate(ohe_mask) if m]
    else:
        ohe_cat_cols = []
    return int_cols, ohe_cat_cols, float_cols


def create_spark_df(X: KoalasDataFrameType, y: KoalasSeriesType,
                    sample_weight=None) -> PySparkDataFrameType:
    """
    Creates a Spark DataFrame from the features and target given as Koalas 
    objects.

    Parameters
    ----------
    X : KoalasDataFrameType
        The feature set.
    y : KoalasSeriesType
        The target.
    sample_weight : KoalasSeriesType, optional
        Row-wise weights to apply. Defaults to None.

    Returns
    -------
    PySparkDataFrameType
        The Spark DataFrame.
    """
    import databricks.koalas as ks

    if X.ndim == 1:
        X = ks.DataFrame(X)
    if sample_weight is None:
        spark_df = X.join(y.rename('label_')).to_spark()
    else:
        spark_df = X.join(y.rename('label_')).join(
            sample_weight.rename('sample_weight_')).to_spark()
    return spark_df


def calc_tps_fps_tns_fns(y_true: Union[PandasSeriesType, np.ndarray, KoalasSeriesType],
                         y_preds: Union[PandasSeriesType, PandasDataFrameType, np.ndarray,
                                        KoalasSeriesType, KoalasDataFrameType],
                         sample_weight=None, tps=False, fps=False,
                         tns=False, fns=False, tps_fps=False, tps_fns=False) -> Tuple[
        Union[np.ndarray, float],
        Union[np.ndarray, float],
        Union[np.ndarray, float],
        Union[np.ndarray, float],
        Union[np.ndarray, float],
        Union[np.ndarray, float]]:
    """
    Calculates the True Positives, False Positives, True Negatives, False
    Negatives, True Positives + False Positives and True Positives + False 
    Negatives for a set of binary predictors, given a binary target. The 
    option to calculate the True Positives + False Positives or True Positives 
    + False Positives in one sum is given as it's faster to calculate these
    metrics together rather than calculating the individual metrics separately
    and summing them.

    Parameters
    ----------
    y_true : Union[PandasSeriesType, np.ndarray, KoalasSeriesType]
        The binary target.
    y_preds : Union[PandasSeriesType, PandasDataFrameType, np.ndarray, KoalasSeriesType, KoalasDataFrameType]
        The binary predictors.
    sample_weight : Union[np.array, PandasSeriesType, KoalasSeriesType], optional
        Row-wise weights to apply. Defaults to None.
    tps : bool, optional
        If True, the True Positives are calculated. Defaults to False.
    fps : bool, optional
        If True, the False Positives are calculated. Defaults to False.
    tns : bool, optional
        If True, the True Negatives are calculated. Defaults to False.
    fns : bool, optional
        If True, the False Negatives are calculated. Defaults to False.
    tps_fps : bool, optional
        If True, the True Positives + False Positives are calculated. Defaults
        to False.
    tps_fns : bool, optional
        If True, the True Positives + False Negatives are calculated. Defaults
        to False.

    Returns
    -------
    Tuple[Union[np.ndarray, float], Union[np.ndarray, float], Union[np.ndarray, float], Union[np.ndarray, float], Union[np.ndarray, float], Union[np.ndarray, float]]
        The True Positives, False Positives, True Negatives, False Negatives,
        True Positives + False Positives and True Positives + False Negatives.
    """

    def _calc_tps_fps_tns_fns_numpy(y_true: Union[PandasSeriesType, np.ndarray],
                                    y_preds: Union[PandasSeriesType, PandasDataFrameType,
                                                   np.ndarray, KoalasSeriesType,
                                                   KoalasDataFrameType],
                                    sample_weight: Union[PandasSeriesType,
                                                         np.ndarray],
                                    tps: bool, fps: bool, tns: bool,
                                    fns: bool, tps_fps: bool,
                                    tps_fns: bool) -> Tuple[
            Union[np.ndarray, float],
            Union[np.ndarray, float],
            Union[np.ndarray, float],
            Union[np.ndarray, float],
            Union[np.ndarray, float],
            Union[np.ndarray, float]]:
        """
        Calculates the True Positives, False Positives, True Negatives, False
        Negatives, True Positives + False Positives and True Positives + False 
        Negatives for a set of binary predictors, given a binary target, using 
        Numpy.
        """

        # Convert relavent args to numpy arrays
        if is_type(y_true, [PandasSeries]):
            y_true = y_true.to_numpy()
        if is_type(y_preds, [PandasSeries, PandasDataFrame]):
            y_preds = y_preds.to_numpy()
        if is_type(sample_weight, [PandasSeries]) and sample_weight is not None:
            sample_weight = sample_weight.to_numpy()
        # Reshape y_true and sample_weight (if given) into same shape as y_preds
        if y_preds.shape != y_true.shape:
            if sample_weight is not None:
                sample_weight_arr = np.tile(
                    sample_weight, (y_preds.shape[1], 1)).T.astype(int)
            y_true_arr = np.tile(y_true, (y_preds.shape[1], 1)).T.astype(int)
        else:
            y_true_arr = y_true
            if sample_weight is not None:
                sample_weight_arr = sample_weight
        # Calculate TPs, FPs, TNs, FNs, TPs+FPs and TPs+FNs
        tps_sum, fps_sum, tns_sum, fns_sum, tps_fps_sum, tps_fns_sum = None, None, None, None, None, None
        if sample_weight is not None:
            if tps:
                tps_sum = (y_preds * y_true_arr * sample_weight_arr).sum(0)
            if fps:
                fps_sum = (y_preds * (1-y_true_arr) * sample_weight_arr).sum(0)
            if tns:
                tns_sum = ((1-y_preds) * (1-y_true_arr)
                           * sample_weight_arr).sum(0)
            if fns:
                fns_sum = ((1-y_preds) * y_true_arr * sample_weight_arr).sum(0)
            if tps_fps:
                tps_fps_sum = (y_preds * sample_weight_arr).sum(0)
            if tps_fns:
                tps_fns_sum = np.array((y_true * sample_weight).sum(0))
        else:
            if tps:
                tps_sum = (y_preds * y_true_arr).sum(0)
            if fps:
                fps_sum = (y_preds * (1-y_true_arr)).sum(0)
            if tns:
                tns_sum = ((1-y_preds) * (1-y_true_arr)).sum(0)
            if fns:
                fns_sum = ((1-y_preds) * y_true_arr).sum(0)
            if tps_fps:
                tps_fps_sum = (y_preds).sum(0)
            if tps_fns:
                tps_fns_sum = np.array((y_true).sum(0))
        return tps_sum, fps_sum, tns_sum, fns_sum, tps_fps_sum, tps_fns_sum

    def _calc_tps_fps_tns_fns_spark(spark_df: PySparkDataFrameType,
                                    features: List[str], tps: bool, fps: bool,
                                    tns: bool, fns: bool, tps_fps: bool,
                                    tps_fns: bool) -> Tuple[
            Union[np.ndarray, float],
            Union[np.ndarray, float],
            Union[np.ndarray, float],
            Union[np.ndarray, float],
            Union[np.ndarray, float],
            Union[np.ndarray, float]]:
        """
        Calculates the True Positives, False Positives, True Negatives, False
        Negatives, True Positives + False Positives and True Positives + False 
        Negatives for a set of binary predictors, given a binary target, using 
        Spark.
        """
        from pyspark.sql import functions as F

        num_feats = len(features)
        funcs = []
        # Generate spark functions for calculating metrics
        if 'sample_weight_' in spark_df.columns:
            if tps:
                funcs = funcs + [F.sum(F.col(feat) * F.col('label_')
                                       * F.col('sample_weight_')) for feat in features]
            if fps:
                funcs = funcs + [F.sum(F.col(feat) * (1-F.col('label_'))
                                       * F.col('sample_weight_')) for feat in features]
            if tns:
                funcs = funcs + [F.sum((1-F.col(feat)) * (1-F.col('label_'))
                                       * F.col('sample_weight_')) for feat in features]
            if fns:
                funcs = funcs + [F.sum((1-F.col(feat)) * (F.col('label_'))
                                       * F.col('sample_weight_')) for feat in features]
            if tps_fps:
                funcs = funcs + \
                    [F.sum((F.col(feat)) * F.col('sample_weight_'))
                     for feat in features]
            if tps_fns:
                funcs = funcs + \
                    [F.sum((F.col('label_')) * F.col('sample_weight_'))]
        else:
            if tps:
                funcs = funcs + [F.sum(F.col(feat) * F.col('label_'))
                                 for feat in features]
            if fps:
                funcs = funcs + [F.sum(F.col(feat) * (1-F.col('label_')))
                                 for feat in features]
            if tns:
                funcs = funcs + \
                    [F.sum((1-F.col(feat)) * (1-F.col('label_')))
                     for feat in features]
            if fns:
                funcs = funcs + \
                    [F.sum((1-F.col(feat)) * (F.col('label_')))
                     for feat in features]
            if tps_fps:
                funcs = funcs + [F.sum(F.col(feat)) for feat in features]
            if tps_fns:
                funcs = funcs + [F.sum(F.col('label_'))]
        # Run functions on spark dataframe
        all_results = np.array(spark_df.select(funcs).collect()[0])
        split_results = []
        k = 0
        # Extract each result required
        for m in [tps, fps, tns, fns, tps_fps, tps_fns]:
            if m:
                if num_feats == 1:
                    split_results.append(all_results[k:k+num_feats][0])
                else:
                    split_results.append(all_results[k:k+num_feats])
                k += num_feats
            else:
                split_results.append(None)
        return split_results

    if tps == False and fps == False and tns == False and fns == False and tps_fps == False and tps_fns == False:
        raise ValueError(
            'One of the parameters `tps`, `fps`, `tns`, `fns`, `tps_fps` or `tps_fns` must be True')
    if is_type(y_true, [KoalasSeries, KoalasDataFrame]) and is_type(y_preds, [KoalasSeries, KoalasDataFrame]):
        spark_df = create_spark_df(
            X=y_preds, y=y_true, sample_weight=sample_weight)
        features = [y_preds.name] if y_preds.ndim == 1 else y_preds.columns
        return _calc_tps_fps_tns_fns_spark(
            spark_df=spark_df, features=features, tps=tps, fps=fps, tns=tns,
            fns=fns, tps_fps=tps_fps, tps_fns=tps_fns)
    else:
        return _calc_tps_fps_tns_fns_numpy(
            y_true=y_true, y_preds=y_preds, sample_weight=sample_weight,
            tps=tps, fps=fps, tns=tns, fns=fns, tps_fps=tps_fps,
            tps_fns=tps_fns)


def return_binary_pred_perf_of_set(y_true: Union[PandasSeriesType, np.ndarray, KoalasSeriesType],
                                   y_preds: Union[PandasDataFrameType, np.ndarray, KoalasDataFrameType],
                                   y_preds_columns: List[str],
                                   sample_weight=None,
                                   metric=None) -> PandasDataFrameType:
    """
    Calculates the performance of a set of binary predictors given a target 
    column.

    Parameters
    ----------
    y_true : Union[PandasSeriesType, np.ndarray, KoalasSeriesType]
        Binary integer target column.
    y_preds : Union[PandasDataFrameType, np.ndarray, KoalasDataFrameType]
        Set of binary integer predictors. Can also be a single predictor.
    y_preds_columns : List[str]
        Column names for the y_preds array.
    sample_weight : Union[PandasSeriesType, np.ndarray, KoalasSeriesType], optional
        Row-wise sample_weights to apply. Defaults to None.
    metric : Callable, optional
        A function/method which calculates a custom metric (e.g. Fbeta score)
        for each column. Defaults to None.

    Returns
    -------
    PandasDataFrameType
        Dataframe containing the performance metrics for each binary predictor.
    """

    tps_sum, _, _, _, tps_fps_sum, tps_fns_sum = calc_tps_fps_tns_fns(
        y_true=y_true, y_preds=y_preds, sample_weight=sample_weight, tps=True,
        tps_fps=True, tps_fns=True)
    tps_fps_sum = np.where(tps_fps_sum == 0, np.nan, tps_fps_sum)
    precisions = np.nan_to_num(np.divide(tps_sum, tps_fps_sum))
    tps_fns_sum = np.where(tps_fns_sum == 0, np.nan, tps_fns_sum)
    recalls = np.nan_to_num(np.divide(tps_sum, tps_fns_sum))
    if y_preds.ndim == 1:
        perc_data_flagged = y_preds.mean(0)
    else:
        perc_data_flagged = y_preds.mean(0).to_numpy()
    # Calculate opt_metric
    if metric is not None:
        opt_metric_results = metric(
            y_true=y_true, y_preds=y_preds, sample_weight=sample_weight)
    else:
        opt_metric_results = None
    results = pd.DataFrame({
        'Precision': precisions,
        'Recall': recalls,
        'PercDataFlagged': perc_data_flagged,
        'Metric': opt_metric_results,
    }, index=y_preds_columns)

    return results


def return_rule_descriptions_from_X_rules(X_rules: Union[PandasDataFrameType,
                                                         KoalasDataFrameType],
                                          X_rules_cols: List[str],
                                          y_true: None,
                                          sample_weight=None,
                                          metric=None) -> PandasDataFrameType:
    """
    Calculates the performance metrics for the standard `rule_descriptions`
    dataframe, given a set of rule binary columns.

    Parameters
    ----------
    X_rules : Union[PandasDataFrameType, KoalasDataFrameType]
        Set of rule binary columns.
    X_rules_cols : List[str]
        Columns associated with `X_rules`.
    y_true : Union[PandasSeriesType, np.ndarray, KoalasSeriesType], optional
        Binary integer target column. Defaults to None.
    sample_weight : Union[PandasSeriesType, np.ndarray, KoalasSeriesType], optional
        Row-wise sample_weights to apply. Defaults to None.
    metric : Callable, optional
        A function/method which calculates a custom metric (e.g. Fbeta score)
        for each rule. Defaults to None. 

    Returns
    -------
    PandasDataFrameType
        The performance metrics for the standard `rule_descriptions` dataframe.
    """

    if y_true is not None:
        rule_descriptions = return_binary_pred_perf_of_set(
            y_true=y_true, y_preds=X_rules, y_preds_columns=X_rules_cols,
            sample_weight=sample_weight, metric=metric
        )
        rule_descriptions.index.name = 'Rule'
    else:
        opt_metric_results = metric(y_preds=X_rules)
        perc_data_flagged = X_rules.mean(0).to_numpy()
        rule_descriptions = pd.DataFrame(data={
            'PercDataFlagged': perc_data_flagged,
            'Metric': opt_metric_results,
        },
            index=X_rules_cols)
        rule_descriptions.index.name = 'Rule'
    return rule_descriptions


def flatten_stringified_json_column(X_column: PandasSeriesType) -> PandasDataFrameType:
    """
    Flattens JSONs contained in a column to their own columns.

    Parameters
    ----------
    X_column : PandasSeriesType
        Contains the JSONs to be flattened.

    Returns
    -------
    PandasDataFrameType
        Contains a column per key-value pair in the JSONs.
    """

    X_column.fillna('{}', inplace=True)
    X_flattened = pd.DataFrame(
        list(X_column.apply(lambda x: json.loads(x)).values))
    X_flattened.set_index(X_column.index.values, inplace=True)

    return X_flattened


def count_rule_conditions(rule_string: str) -> int:
    """
    Counts the number of conditions in a rule string.

    Parameters
    ----------
    rule_string : str
        The standard Iguanas string representation of the rule.

    Returns
    -------
    int
        Number of conditions in the rule.
    """
    n_conditions = rule_string.count("X['")
    return n_conditions


def return_progress_ready_range(verbose: bool,
                                range: Iterable) -> Union[tqdm, Iterable]:
    """
    Returns a tqdm object for a given iterable, `range`, if `verbose` is True.
    The tqdm object prints the progress of iteration.

    Parameters
    ----------
    verbose : bool
        Dictates whether the tqdm object should be returned. 
    range : Iterable
        The iterable.

    Returns
    -------
    Union[tqdm, Iterable]
        Either the tqdm-version of the iterable, or the original iterable.
    """
    if verbose:
        return tqdm(range, file=sys.stdout)
    else:
        return range


def return_conf_matrix(y_true: Union[PandasSeriesType, np.ndarray, KoalasSeriesType],
                       y_pred: Union[PandasSeriesType, np.ndarray, KoalasSeriesType],
                       sample_weight=None) -> PandasDataFrameType:
    """
    Creates a confusion matrix from a binary target and binary predictor.

    Parameters
    ----------
    y_true : Union[PandasSeriesType, np.ndarray, KoalasSeriesType]
        Binary target.
    y_pred : Union[PandasSeriesType, np.ndarray, KoalasSeriesType]
        Binary predictor.
    sample_weight : Union[PandasSeriesType, np.ndarray, KoalasSeriesType], optional
        Row-wise weights to apply. Defaults to None.          

    Returns
    -------
    PandasDataFrameType
        The confusion matrix (the index shows the predicted class; the column
        shows the actual class).
    """

    tps, fps, tns, fns, _, _ = calc_tps_fps_tns_fns(
        y_true=y_true, y_preds=y_pred, sample_weight=sample_weight, tps=True,
        fps=True, tns=True, fns=True
    )
    conf_matrix = pd.DataFrame([
        [tps, fps],
        [fns, tns]
    ],
        columns=[1, 0], index=[1, 0]
    )
    return conf_matrix


def check_allowed_types(x: object, x_name: str,
                        allowed_types: List[str]) -> None:
    """
    Checks whether the stringified type of `x` is in `allowed_types` - a list
    of stringified types. If not, it raises a TypeError.

    Parameters
    ----------
    x : object
        The object to check the type of.
    x_name : str
        The objects name (used when raising the error).
    allowed_types : List[str]
        The list of allowed types (in string format).

    Raises
    ------
    TypeError
        If str(type(`x`)) is not in `allowed_types`.
    """

    x_type = str(type(x))
    if x_type not in allowed_types:
        allowed_types_str = ' or '.join(
            [allowed_type.split("'")[1] for allowed_type in allowed_types])
        x_type_str = x_type.split("'")[1]
        raise TypeError(
            f'`{x_name}` must be a {allowed_types_str}. Current type is {x_type_str}.')


def is_type(x: object, types: List[str]) -> bool:
    """
    Returns whether the stringified type of `x` is in `types` - a list of 
    stringified types.

    Parameters
    ----------
    x : object
        The object to check the type of.
    types : List[str]
        The list of allowed types (in string format) to check against.

    Returns
    -------
    bool
        If str(type(`x`)) is in `types`.
    """

    x_type = str(type(x))
    return x_type in types


def return_dataset_if_dict(step_tag: str,
                           df: Union[dict, PandasDataFrameType]) -> PandasDataFrameType:
    """
    If `df` is a dict, then the dataframe for the given step in a pipeline is
    extracted from the dict; else, just return `df`.

    Parameters
    ----------
    step_tag : str
        The tag corresponding to the pipeline step.
    df : Union[dict, PandasDataFrameType]
        The dataset or dictionary of datasets.

    Returns
    -------
    PandasDataFrameType
        The dataset for the pipeline step.
    """
    if isinstance(df, dict) and step_tag in df.keys():
        return df[step_tag]
    else:
        return df
