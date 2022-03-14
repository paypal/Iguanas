class DataFrameSizeWarning(Warning):
    """
    Custom warning for when `X` has no columns.
    """
    pass


class NoRulesWarning(Warning):
    """
    Custom warning for when rules cannot be generated.
    """
    pass


class RulesNotOptimisedWarning(Warning):
    """
    Custom warning for when rules cannot be optimised.
    """
    pass
