import pytest
from iguanas.exceptions import DataFrameSizeError, NoRulesError


def test_DataFrameSizeError():
    with pytest.raises(DataFrameSizeError, match="DataFrameSizeError"):
        raise DataFrameSizeError('DataFrameSizeError')


def test_NoRulesError():
    with pytest.raises(NoRulesError, match="NoRulesError"):
        raise NoRulesError('NoRulesError')
