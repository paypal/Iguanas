class DataFrameSizeError(Exception):
    """
    Custom exception for when `X` has no columns.
    """
    pass


class NoRulesError(Exception):
    """
    Custom exception for no rules can be generated.
    """
    pass
