"""Contains classes for calculating classification metrics."""
import iguanas.utils as utils
from iguanas.utils.types import NumpyArray, PandasDataFrame, PandasSeries, \
    KoalasDataFrame, KoalasSeries
from iguanas.utils.typing import NumpyArrayType, PandasDataFrameType, \
    PandasSeriesType, KoalasDataFrameType, KoalasSeriesType
import numpy as np
from typing import Union, List
import math


class Precision:
    """
    Calculates the Precision for either a single or set of binary
    predictors.

    Examples
    --------
    >>> import pandas as pd
    >>> from iguanas.metrics import Precision
    >>> X = pd.DataFrame({
    ...     'A': [1, 0, 1, 0],
    ...     'B': [1, 1, 1, 0]
    ... })
    >>> y = pd.Series([
    ...     1, 0, 1, 0
    ... ])
    >>> p = Precision()
    >>> print(p.fit(y_preds=X, y_true=y))
    [1.         0.66666667]
    """

    def __repr__(self):
        return 'Precision'

    def fit(self,
            y_preds: Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType,
                           PandasDataFrameType, KoalasDataFrameType],
            y_true: Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType],
            sample_weight=None) -> Union[float, NumpyArrayType]:
        """
        Calculates the Precision for either a single or set of binary
        predictors.

        Parameters
        ----------
        y_preds : Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType, PandasDataFrameType, KoalasDataFrameType]
            The binary predictor column(s).
        y_true : Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType]
            The target column.
        sample_weight : Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType], optional
            Row-wise weights to apply. Defaults to None.

        Returns
        -------
        Union[float, NumpyArrayType]
            The Precision score(s).
        """

        utils.check_allowed_types(
            y_true, 'y_true', [
                NumpyArray, PandasSeries,
                KoalasSeries
            ])
        utils.check_allowed_types(
            y_preds, 'y_preds', [
                NumpyArray, PandasSeries,
                PandasDataFrame, KoalasSeries,
                KoalasDataFrame
            ])
        if sample_weight is not None:
            utils.check_allowed_types(
                sample_weight, 'sample_weight', [
                    NumpyArray, PandasSeries,
                    KoalasSeries
                ])
        tps_sum, _, _, _, tps_fps_sum, _ = utils.calc_tps_fps_tns_fns(
            y_true=y_true, y_preds=y_preds, sample_weight=sample_weight, tps=True, tps_fps=True)
        tps_fps_sum = np.where(tps_fps_sum == 0, np.nan, tps_fps_sum)
        precision = np.nan_to_num(np.divide(tps_sum, tps_fps_sum))
        return precision


class Recall:
    """
    Calculates the Recall for either a single or set of binary
    predictors.

    Examples
    --------
    >>> import pandas as pd
    >>> from iguanas.metrics import Recall
    >>> X = pd.DataFrame({
    ...     'A': [1, 0, 1, 0],
    ...     'B': [1, 1, 1, 0]
    ... })
    >>> y = pd.Series([
    ...     1, 0, 1, 0
    ... ])
    >>> r = Recall()
    >>> print(r.fit(y_preds=X, y_true=y))
    [1. 1.]
    """

    def __repr__(self):
        return 'Recall'

    def fit(self,
            y_preds: Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType, PandasDataFrameType, KoalasDataFrameType],
            y_true: Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType],
            sample_weight=None) -> Union[float, NumpyArrayType]:
        """
        Calculates the Recall for either a single or set of binary
        predictors.

        Parameters
        ----------
        y_preds : Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType, PandasDataFrameType, KoalasDataFrameType]
            The binary predictor column(s).
        y_true : Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType]
            The target column.
        sample_weight : Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType], optional
            Row-wise weights to apply. Defaults to None.

        Returns
        -------
        Union[float, NumpyArrayType]
            The Recall score(s).
        """

        utils.check_allowed_types(
            y_true, 'y_true', [
                NumpyArray, PandasSeries,
                KoalasSeries
            ])
        utils.check_allowed_types(
            y_preds, 'y_preds', [
                NumpyArray, PandasSeries,
                PandasDataFrame, KoalasSeries,
                KoalasDataFrame
            ])
        if sample_weight is not None:
            utils.check_allowed_types(
                sample_weight, 'sample_weight', [
                    NumpyArray, PandasSeries,
                    KoalasSeries
                ])
        tps_sum, _, _, _, _, tps_fns_sum = utils.calc_tps_fps_tns_fns(
            y_true=y_true, y_preds=y_preds, sample_weight=sample_weight,
            tps=True, tps_fns=True)
        tps_fns_sum = np.where(tps_fns_sum == 0, np.nan, tps_fns_sum)
        recall = np.nan_to_num(np.divide(tps_sum, tps_fns_sum))
        return recall


class FScore:
    """
    Calculates the Fbeta score for either a single or set of binary
    predictors.

    Parameters
    ----------
    beta : float
        The beta value used to calculate the Fbeta score.   

    Examples
    --------
    >>> import pandas as pd
    >>> from iguanas.metrics import FScore
    >>> X = pd.DataFrame({
    ...     'A': [1, 0, 1, 0],
    ...     'B': [1, 1, 1, 0]
    ... })
    >>> y = pd.Series([
    ...     1, 0, 1, 0
    ... ])
    >>> f1 = FScore(beta=1)
    >>> print(f1.fit(y_preds=X, y_true=y))
    [1.  0.8]
    """

    def __init__(self,
                 beta: float):
        self.beta = beta

    def __repr__(self):
        return f'FScore with beta={self.beta}'

    def fit(self,
            y_preds: Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType, PandasDataFrameType, KoalasDataFrameType],
            y_true: Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType],
            sample_weight=None) -> Union[float, NumpyArrayType]:
        """
        Calculates the Fbeta score for either a single or set of binary
        predictors.

        Parameters
        ----------
        y_preds : Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType, PandasDataFrameType, KoalasDataFrameType]
            The binary predictor column(s).
        y_true : Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType]
            The target column.
        sample_weight : Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType], optional
            Row-wise weights to apply. Defaults to None.

        Returns
        -------
        Union[float, NumpyArrayType]
            The Fbeta score(s).
        """
        def _fscore(p, r, b):
            if p == 0 or r == 0:
                fs = 0
            else:
                fs = (1 + b ** 2) * ((p * r) / ((p * b ** 2) + r))
            return fs

        utils.check_allowed_types(
            y_true, 'y_true', [
                NumpyArray, PandasSeries,
                KoalasSeries
            ])
        utils.check_allowed_types(
            y_preds, 'y_preds', [
                NumpyArray, PandasSeries,
                PandasDataFrame, KoalasSeries,
                KoalasDataFrame
            ])
        if sample_weight is not None:
            utils.check_allowed_types(
                sample_weight, 'sample_weight', [
                    NumpyArray, PandasSeries,
                    KoalasSeries
                ])
        tps_sum, _, _, _, tps_fps_sum, tps_fns_sum = utils.calc_tps_fps_tns_fns(
            y_true=y_true, y_preds=y_preds, sample_weight=sample_weight,
            tps=True, tps_fps=True, tps_fns=True
        )
        tps_fps_sum = np.where(tps_fps_sum == 0, np.nan, tps_fps_sum)
        tps_fns_sum = np.where(tps_fns_sum == 0, np.nan, tps_fns_sum)
        precisions = np.nan_to_num(np.divide(tps_sum, tps_fps_sum))
        recalls = np.nan_to_num(np.divide(tps_sum, tps_fns_sum))

        if utils.is_type(precisions, NumpyArray) and \
                utils.is_type(recalls, NumpyArray):
            fscores = np.array([_fscore(p, r, self.beta)
                               for p, r in zip(precisions, recalls)])
        else:
            fscores = _fscore(precisions, recalls, self.beta)
        return fscores


class Revenue:
    """
    Calculates the revenue for either a single or set of binary
    predictors.

    Parameters
    ----------
    y_type : str
        Dictates whether the binary target column flags fraud (y_type = 
        'Fraud') or non-fraud (y_type = 'NonFraud').
    chargeback_multiplier : int
        Multiplier to apply to chargeback transactions.

    Examples
    --------
    >>> import pandas as pd
    >>> from iguanas.metrics import Revenue
    >>> X = pd.DataFrame({
    ...     'A': [1, 0, 1, 0],
    ...     'B': [1, 1, 1, 0]
    ... })
    >>> y = pd.Series([
    ...     1, 0, 1, 0
    ... ])
    >>> amounts = pd.Series([
    ...     100, 20, 50, 30
    ... ])
    >>> rev = Revenue(
    ...     y_type = 'Fraud',
    ...     chargeback_multiplier=2
    ... )
    >>> print(rev.fit(y_preds=X, y_true=y, sample_weight=amounts))
    [350 310]
    """

    def __init__(self,
                 y_type: str,
                 chargeback_multiplier: int):

        if y_type not in ['Fraud', 'NonFraud']:
            raise ValueError('`y_type` must be either "Fraud" or "NonFraud"')
        self.y_type = y_type
        self.chargeback_multiplier = chargeback_multiplier

    def __repr__(self):
        return f'Revenue with y_type={self.y_type}, chargeback_multiplier={self.chargeback_multiplier}'

    def fit(self,
            y_preds: Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType, PandasDataFrameType, KoalasDataFrameType],
            y_true: Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType],
            sample_weight: Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType]) -> Union[float, NumpyArrayType]:
        """
        Calculates the revenue for either a single or set of binary
        predictors.

        Parameters
        ----------
        y_preds : Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType, PandasDataFrameType, KoalasDataFrameType]
            The binary predictor column.
        y_true : Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType] 
            The target column.
        sample_weight : Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType] 
            Row-wise transaction amounts to apply.

        Returns
        -------
        Union[float, NumpyArrayType]
            Revenue(s).
        """

        utils.check_allowed_types(
            y_true, 'y_true', [
                NumpyArray, PandasSeries,
                KoalasSeries
            ])
        utils.check_allowed_types(
            y_preds, 'y_preds', [
                NumpyArray, PandasSeries,
                PandasDataFrame, KoalasSeries,
                KoalasDataFrame
            ])
        utils.check_allowed_types(
            sample_weight, 'sample_weight', [
                NumpyArray, PandasSeries,
                KoalasSeries
            ])
        tps_sum, fps_sum, tns_sum, fns_sum, _, _ = utils.calc_tps_fps_tns_fns(
            y_true=y_true, y_preds=y_preds, sample_weight=sample_weight,
            tps=True, fps=True, tns=True, fns=True)
        if self.y_type == 'Fraud':
            revenue = self.chargeback_multiplier * \
                (tps_sum - fns_sum) + tns_sum - fps_sum
        elif self.y_type == 'NonFraud':
            revenue = tps_sum - fns_sum + \
                self.chargeback_multiplier * (tns_sum - fps_sum)
        return revenue


class Bounds:
    """
    Calculates whether the predictor(s) are within the set of bounds provided, 
    by applying the following process:

    1. Calculate the value of each `metric` in `bounds`.
    2. Calculate how far this value is from each `threshold` in `bounds`.
    3. For the value that is furthest from its `threshold`, calculate the Sigmoid function using the difference between the value and the `threshold`.

    This means that if the final value returned is >= 0.5, the predictor is 
    within the bounds that have been set.    

    Parameters
    ----------
    bounds : List[dict]
        Each bound to be applied - this should be a dictionary containing the 
        following keys: `metric` - the function to be calculated; `operator` -
        the operator used to calculate whether a result is within a bound; 
        `threshold` - the value corresponding to the boundary.

    Examples
    --------
    >>> import pandas as pd
    >>> from iguanas.metrics import Bounds
    >>> p = Precision()
    >>> r = Recall()
    >>> X = pd.DataFrame({
    ...     'A': [1, 0, 1, 0],
    ...     'B': [1, 1, 1, 0]
    ... })
    >>> y = pd.Series([
    ...     1, 0, 1, 0
    ... ])
    >>> bounds = [
    ...     {
    ...         'metric': p.fit,
    ...         'operator': '>',
    ...         'threshold': 0.7
    ...     },
    ...     {
    ...         'metric': r.fit,
    ...         'operator': '>',
    ...         'threshold': 0.7
    ...     }
    ... ]
    >>> b = Bounds(bounds=bounds)
    >>> print(b.fit(y_preds=X, y_true=y))
    [0.57444252 0.49166744]
    """

    def __init__(self,
                 bounds: List[dict]):

        self.bounds = bounds
        self._comparison_funcs = {
            '>': lambda x, y: x - y,
            '>=': lambda x, y: x - y,
            '<': lambda x, y: x + y,
            '<=': lambda x, y: x + y
        }

    def fit(self,
            y_preds: Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType, PandasDataFrameType, KoalasDataFrameType],
            y_true=None,
            sample_weight=None) -> Union[float, NumpyArrayType]:
        """
        Calculates the Sigmoid function of the difference between the `metric`
        result and its `threshold` (for the result that is furtherest from its 
        `threshold`).

        Parameters
        ----------
        y_preds : Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType, PandasDataFrameType, KoalasDataFrameType]
            The binary predictor column.
        y_true : Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType], optional
            The target column.
        sample_weight : Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType], optional
            Row-wise transaction amounts to apply.

        Returns
        -------
        Union[float, NumpyArrayType]
            Result of the Sigmoid function.
        """

        num_bounds = len(self.bounds)
        if y_preds.ndim == 1:
            result_thr_comp = np.empty(num_bounds)
        else:
            result_thr_comp = np.empty((num_bounds, y_preds.shape[1]))
        for i, bound in enumerate(self.bounds):
            metric = bound['metric']
            operator = bound['operator']
            threshold = bound['threshold']
            result = metric(y_preds, y_true, sample_weight)
            comp_func = self._comparison_funcs[operator]
            result_thr_comp[i] = comp_func(result, threshold)
        result = result_thr_comp.min(axis=0)
        if isinstance(result, np.ndarray):
            return np.array([self._sigmoid(r) for r in result])
        else:
            return self._sigmoid(result)

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Calculates the Sigmoid function for a given input `x`"""

        return 1/(1+math.e**-x)
