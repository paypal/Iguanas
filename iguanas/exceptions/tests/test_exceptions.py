from iguanas.exceptions import DataFrameSizeError


def test_exceptions():
    assert issubclass(DataFrameSizeError, Exception)
