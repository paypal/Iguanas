"""Custom typing objects. Used so packages don't need to be imported."""
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy
    import pandas
    import databricks
    import pyspark

NumpyArrayType = 'numpy.ndarray'
PandasDataFrameType = 'pandas.core.frame.DataFrame'
PandasSeriesType = 'pandas.core.series.Series'
KoalasDataFrameType = 'databricks.koalas.frame.DataFrame'
KoalasSeriesType = 'databricks.koalas.series.Series'
PySparkDataFrameType = 'pyspark.sql.dataframe.DataFrame'
