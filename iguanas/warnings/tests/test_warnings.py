import pytest
import warnings
from iguanas.warnings import DataFrameSizeWarning, NoRulesWarning


def test_DataFrameSizeWarning():
    with pytest.warns(DataFrameSizeWarning, match="DataFrameSizeWarning"):
        warnings.warn('DataFrameSizeWarning', DataFrameSizeWarning)


def test_NoRulesWarning():
    with pytest.warns(NoRulesWarning, match="NoRulesWarning"):
        warnings.warn('NoRulesWarning', NoRulesWarning)
