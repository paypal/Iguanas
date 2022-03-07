import pytest
from iguanas.exceptions import DataFrameSizeError, NoRulesError, RulesNotOptimisedError


def test_DataFrameSizeError():
    with pytest.raises(DataFrameSizeError, match="DataFrameSizeError"):
        raise DataFrameSizeError('DataFrameSizeError')


def test_NoRulesError():
    with pytest.raises(NoRulesError, match="NoRulesError"):
        raise NoRulesError('NoRulesError')


def test_RulesNotOptimisedError():
    with pytest.raises(RulesNotOptimisedError, match="RulesNotOptimisedError"):
        raise RulesNotOptimisedError('RulesNotOptimisedError')
