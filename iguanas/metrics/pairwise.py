"""Contains classes for calculating pairwise metrics."""
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import iguanas.utils as utils
from iguanas.utils.types import PandasDataFrame
from iguanas.utils.typing import PandasDataFrameType


class CosineSimilarity:
    """
    Computes the cosine similarity between columns in X.

    Parameters
    ----------
    **kwargs : dict
        Any keyword arguments to be used in the sklearn `cosine_similarity`
        function.        

    Examples
    --------
    >>> import pandas as pd
    >>> from iguanas.metrics import CosineSimilarity
    >>> cs = CosineSimilarity()
    >>> X = pd.DataFrame({
    ...     'A': [1, 0, 1, 0],
    ...     'B': [1, 1, 1, 0]
    ... })
    >>> y = pd.Series([
    ...     1, 0, 1, 0
    ... ])
    >>> print(cs.fit(X=X))
              A         B
    A  1.000000  0.816497
    B  0.816497  1.000000
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __repr__(self):
        return 'CosineSimilarity'

    def fit(self, X: PandasDataFrameType) -> PandasDataFrameType:
        """
        Computes the cosine similarity between columns in X.

        Parameters
        ----------
        X : PandasDataFrameType
            Dataframe containing binary columns.

        Returns
        -------
        PandasDataFrameType
            Dataframe containing the pairwise cosine similarities.
        """

        utils.check_allowed_types(X, 'X', [PandasDataFrame])
        cos_sim_matrix = 1 - \
            pairwise_distances(X=X.values.T, metric='cosine', **self.kwargs)
        return pd.DataFrame(cos_sim_matrix, index=X.columns, columns=X.columns)


class JaccardSimilarity:
    """ 
    Computes the Jaccard similarity between columns in X.

    Parameters
    ----------
    **kwargs : dict
        Any keyword arguments to be used in the sklearn `pairwise_distances`
        function.     

    Examples
    --------
    >>> import pandas as pd
    >>> from iguanas.metrics import JaccardSimilarity
    >>> js = JaccardSimilarity()
    >>> X = pd.DataFrame({
    ...     'A': [1, 0, 1, 0],
    ...     'B': [1, 1, 1, 0]
    ... })
    >>> y = pd.Series([
    ...     1, 0, 1, 0
    ... ])
    >>> print(js.fit(X=X))   
              A         B
    A  1.000000  0.666667
    B  0.666667  1.000000
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __repr__(self):
        return 'JaccardSimilarity'

    def fit(self, X: PandasDataFrameType) -> PandasDataFrameType:
        """
        Computes the Jaccard similarity between columns in X.

        Parameters
        ----------
        X : PandasDataFrameType
            Dataframe containing binary columns.

        Returns
        -------
        PandasDataFrameType
            Dataframe containing the pairwise Jaccard similarities.
        """

        utils.check_allowed_types(X, 'X', [PandasDataFrame])
        jaccard_matrix = 1 - \
            pairwise_distances(X=X.values.T.astype(
                bool), metric="jaccard", **self.kwargs)
        return pd.DataFrame(jaccard_matrix, index=X.columns, columns=X.columns)
