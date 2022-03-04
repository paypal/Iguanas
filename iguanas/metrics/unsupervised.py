"""Contains classes for calculating unsupervised metrics."""
from typing import Union
import iguanas.utils as utils
from iguanas.utils.types import NumpyArray, PandasDataFrame, PandasSeries, \
    KoalasDataFrame, KoalasSeries
from iguanas.utils.typing import NumpyArrayType, PandasDataFrameType, \
    PandasSeriesType, KoalasDataFrameType, KoalasSeriesType


class AlertsPerDay:
    """
    Calculates the negative squared difference between the number of alerts per
    day in the binary predictor(s) vs the expected.

    Parameters
    ----------
    n_alerts_expected_per_day : int
        Expected number of alerts per day for the given rule.
    no_of_days_in_file : int
        Number of days of data provided in the file.

    Examples
    --------
    >>> import pandas as pd
    >>> from iguanas.metrics import AlertsPerDay
    >>> X = pd.DataFrame({
    ...     'A': [1, 0, 1, 0],
    ...     'B': [1, 1, 1, 0]
    ... })
    >>> apd = AlertsPerDay(
    ...    n_alerts_expected_per_day=3,
    ...    no_of_days_in_file=1
    ... )
    >>> print(apd.fit(y_preds=X))
    [-1. -0.]
    """

    def __init__(self, n_alerts_expected_per_day: int,
                 no_of_days_in_file: int):

        self.n_alerts_expected_per_day = n_alerts_expected_per_day
        self.no_of_days_in_file = no_of_days_in_file

    def __repr__(self):
        return f'AlertsPerDay with n_alerts_expected_per_day={self.n_alerts_expected_per_day}, no_of_days_in_file={self.no_of_days_in_file}'

    def fit(self,
            y_preds: Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType, PandasDataFrameType, KoalasDataFrameType],
            y_true=None,
            sample_weight=None) -> Union[float, NumpyArrayType]:
        """
        Calculates the negative squared difference between the number of alerts
        per day in the binary predictor(s) vs the expected.

        Parameters
        ----------
        y_preds : Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType, PandasDataFrameType, KoalasDataFrameType]
            The binary predictor column(s).
        y_true : None
            Ignored.
        sample_weight : None
            Ignored.

        Returns
        -------
        Union[float, NumpyArrayType]
            The negative squared difference(s).
        """

        utils.check_allowed_types(
            y_preds, 'y_preds', [
                NumpyArray, PandasSeries,
                PandasDataFrame, KoalasSeries,
                KoalasDataFrame
            ])
        if utils.is_type(
            y_preds, [PandasSeries, PandasDataFrame]
        ):
            y_preds = y_preds.to_numpy()
        num_flagged = y_preds.sum(0)
        if utils.is_type(num_flagged, [KoalasSeries]):
            num_flagged = num_flagged.to_numpy()
        n_alerts_per_day = num_flagged/self.no_of_days_in_file
        f_min = (n_alerts_per_day-self.n_alerts_expected_per_day) ** 2
        return -f_min


class PercVolume:
    """
    Calculates the negative squared difference(s) between the percentage of the
    overall volume that the binary predictor(s) vs the expected.

    Parameters
    ----------
    perc_vol_expected : float
        Expected percentage of the overall volume that the binary predictor
        should flag.

    Examples
    --------
    >>> import pandas as pd
    >>> from iguanas.metrics import PercVolume
    >>> X = pd.DataFrame({
    ...     'A': [1, 0, 1, 0],
    ...     'B': [1, 1, 1, 0]
    ... })
    >>> apd = PercVolume(
    ...    perc_vol_expected=0.5,
    ... )
    >>> print(apd.fit(y_preds=X))
    [-0.     -0.0625]
    """

    def __init__(self, perc_vol_expected: float):
        self.perc_vol_expected = perc_vol_expected

    def __repr__(self):
        return f'PercVolume with perc_vol_expected={self.perc_vol_expected}'

    def fit(self,
            y_preds: Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType, PandasDataFrameType, KoalasDataFrameType],
            y_true=None,
            sample_weight=None) -> Union[float, NumpyArrayType]:
        """
        Calculates the negative squared difference(s) between the percentage of
        the overall volume that the binary predictor(s) vs the expected.

        Parameters
        ----------
        y_preds : Union[NumpyArrayType, PandasSeriesType, KoalasSeriesType, PandasDataFrameType, KoalasDataFrameType]
            The binary predictor column(s).
        y_true : None
            Ignored.
        sample_weight : None
            Ignored.

        Returns
        -------
        Union[float, NumpyArrayType]
            The negative squared difference(s).
        """

        utils.check_allowed_types(
            y_preds, 'y_preds', [
                NumpyArray, PandasSeries,
                PandasDataFrame, KoalasSeries,
                KoalasDataFrame
            ])
        if utils.is_type(
            y_preds, [PandasSeries, PandasDataFrame]
        ):
            y_preds = y_preds.to_numpy()
        perc_flagged = y_preds.mean(0)
        if utils.is_type(perc_flagged, [KoalasSeries]):
            perc_flagged = perc_flagged.to_numpy()
        f_min = (perc_flagged-self.perc_vol_expected) ** 2
        return -f_min
