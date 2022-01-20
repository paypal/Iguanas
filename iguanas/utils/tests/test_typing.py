from iguanas.utils.typing import NumpyArrayType, PandasDataFrameType, \
    PandasSeriesType, KoalasDataFrameType, KoalasSeriesType, \
    PySparkDataFrameType


def test_typing():
    assert NumpyArrayType == 'numpy.ndarray'
    assert PandasDataFrameType == 'pandas.core.frame.DataFrame'
    assert PandasSeriesType == 'pandas.core.series.Series'
    assert KoalasDataFrameType == 'databricks.koalas.frame.DataFrame'
    assert KoalasSeriesType == 'databricks.koalas.series.Series'
    assert PySparkDataFrameType == 'pyspark.sql.dataframe.DataFrame'
