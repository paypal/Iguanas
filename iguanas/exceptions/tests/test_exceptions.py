from iguanas.exceptions import DataFrameSizeError, NoRulesError


def test_exceptions():
    assert issubclass(DataFrameSizeError, Exception)
    assert issubclass(NoRulesError, Exception)
