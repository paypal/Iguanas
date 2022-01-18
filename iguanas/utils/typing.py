"""Custom typing objects. Used so packages don't need to be imported."""
from typing import TypeVar
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     import numpy
#     import pandas
#     import databricks
#     import pyspark

NumpyArrayType = TypeVar('numpy.ndarray')
PandasDataFrameType = TypeVar('pandas.core.frame.DataFrame')
PandasSeriesType = TypeVar('pandas.core.series.Series')
KoalasDataFrameType = TypeVar('databricks.koalas.frame.DataFrame')
KoalasSeriesType = TypeVar('databricks.koalas.series.Series')
PySparkDataFrameType = TypeVar('pyspark.sql.dataframe.DataFrame')

# NumpyArrayType = 'numpy.ndarray'
# PandasDataFrameType = 'pandas.core.frame.DataFrame'
# PandasSeriesType = 'pandas.core.series.Series'
# KoalasDataFrameType = 'databricks.koalas.frame.DataFrame'
# KoalasSeriesType = 'databricks.koalas.series.Series'
# PySparkDataFrameType = 'pyspark.sql.dataframe.DataFrame'
