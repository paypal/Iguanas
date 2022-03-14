class DataFrameSizeError(Exception):
    """
    Custom exception for when `X` has no columns.
    """
    pass


class NoRulesError(Exception):
    """
    Custom exception for when rules cannot be generated.
    """
    pass


class RulesNotOptimisedError(Exception):
    """
    Custom exception for when rules cannot be optimised.
    """
    pass
