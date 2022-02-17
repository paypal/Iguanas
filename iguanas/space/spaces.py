"""Classes for defining search spaces."""
from hyperopt import hp


class UniformInteger:
    """
    Returns a random integer in the range [`self.low`, `self.high`].

    Parameters
    ----------
    low : int
        Low limit.
    high : int
        High limit.
    """

    def __init__(self, low: int, high: int) -> None:
        self.low = low
        self.high = high

    def transform(self, label: str) -> hp.quniform:
        """
        Return hp.quniform(`label`, `self.low`, `self.high`, q=1) function.

        Parameters
        ----------
        label : str
            Function label.

        Returns
        -------
        hp.quniform
            Hyperopt space function which returns an integer between `self.low`
            and `self.high`.
        """

        return hp.quniform(label, self.low, self.high, q=1)


class UniformFloat:
    """
    Returns a random float in the range [`self.low`, `self.high`].

    Parameters
    ----------
    low : int
        Low limit.
    high : int
        High limit.
    """

    def __init__(self, low: int, high: int) -> None:
        self.low = low
        self.high = high

    def transform(self, label: str) -> hp.uniform:
        """
        Return hp.uniform(`label`, `self.low`, `self.high`) function.

        Parameters
        ----------
        label : str
            Function label.

        Returns
        -------
        hp.uniform
            Hyperopt space function which returns a float between `self.low`
            and `self.high`.
        """

        return hp.uniform(label, self.low, self.high)


class Choice:
    """
    Returns a random choice from `self.options`.

    Parameters
    ----------
    options : List
        List of choices to choose from.
    """

    def __init__(self, options) -> None:
        self.options = options

    def transform(self, label: str) -> hp.choice:
        """
        Return hp.choice(`label`, `self.options`) function.

        Parameters
        ----------
        label : str
            Function label.

        Returns
        -------
        hp.choice
            Hyperopt space function which returns a choice from `self.options`.
        """

        return hp.choice(label, self.options)
