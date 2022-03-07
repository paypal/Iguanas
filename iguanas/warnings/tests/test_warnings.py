import pytest
import warnings
from iguanas.warnings import DataFrameSizeWarning, NoRulesWarning, RulesNotOptimisedWarning


def test_DataFrameSizeWarning():
    with pytest.warns(DataFrameSizeWarning, match="DataFrameSizeWarning"):
        warnings.warn('DataFrameSizeWarning', DataFrameSizeWarning)


def test_NoRulesWarning():
    with pytest.warns(NoRulesWarning, match="NoRulesWarning"):
        warnings.warn('NoRulesWarning', NoRulesWarning)


def test_RulesNotOptimisedWarning():
    with pytest.warns(RulesNotOptimisedWarning, match="RulesNotOptimisedWarning"):
        warnings.warn('RulesNotOptimisedWarning', RulesNotOptimisedWarning)
