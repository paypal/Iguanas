"""Class for creating a Linear Pipeline."""
from copy import deepcopy
from typing import List, Tuple
from iguanas.pipeline._base_pipeline import _BasePipeline
from iguanas.utils.typing import PandasDataFrameType, PandasSeriesType


class LinearPipeline(_BasePipeline):
    """
    Generates a pipeline, which is a sequence of steps which are applied 
    sequentially to a dataset. Each step should be an instantiated class with 
    both `fit` and `transform` methods. The final step should be an 
    instantiated class with both `fit` and `predict` methods.

    Parameters
    ----------
    steps : List[Tuple[str, object]]
        The steps to be applied as part of the pipeline. Each element of the
        list corresponds to a single step. Each step should be a tuple of two
        elements - the first element should be a string which refers to the 
        step; the second element should be the instantiated class which is run
        as part of the step. 
    """

    def __init__(self, steps: List[Tuple[str, object]]):
        _BasePipeline.__init__(self, steps=steps)

    def fit(self,
            X: PandasDataFrameType,
            y: PandasSeriesType,
            sample_weight=None) -> None:
        """
        Sequentially runs the `fit_transform` methods of each step in the 
        pipeline, except for the last step, where the `fit` method is run.

        Parameters
        ----------
        X : PandasDataFrameType
            The dataset.
        y : PandasSeriesType
            The binary target column.
        sample_weight : PandasSeriesType, optional
            Row-wise weights to apply. Defaults to None.
        """

        self.steps_ = deepcopy(self.steps)
        for step_tag, step in self.steps_[:-1]:
            X = self._pipeline_fit_transform(
                step_tag=step_tag, step=step, X=X, y=y,
                sample_weight=sample_weight
            )
        final_step_tag = self.steps_[-1][0]
        final_step = self.steps_[-1][1]
        self._pipeline_fit(
            step_tag=final_step_tag, step=final_step, X=X, y=y,
            sample_weight=sample_weight
        )

    def predict(self, X: PandasDataFrameType) -> PandasSeriesType:
        """
        Sequentially runs the `transform` methods of each step in the pipeline,
        except for the last step, where the `predict` method is run. Note that
        before using this method, you should first run either the `fit` or 
        `fit_predict` methods.

        Parameters
        ----------
        X : PandasDataFrameType
            The dataset.

        Returns
        -------
        PandasSeriesType
            The prediction of the final step.
        """

        for (step_tag, step) in self.steps_[:-1]:
            X = self._pipeline_transform(step_tag=step_tag, step=step, X=X)
        final_step = self.steps_[-1][1]
        return self._pipeline_predict(step=final_step, X=X)

    def transform(self,
                  X):

        for step_tag, step in self.steps_:
            X = self._pipeline_transform(step_tag=step_tag, step=step, X=X)
        return X

    def fit_transform(self,
                      X: PandasDataFrameType,
                      y: PandasSeriesType,
                      sample_weight=None) -> PandasDataFrameType:
        """
        Sequentially runs the `fit_transform` methods of each step in the 
        pipeline.

        X : PandasDataFrameType
            The dataset.
        y : PandasSeriesType
            The binary target column.
        sample_weight : PandasSeriesType, optional
            Row-wise weights to apply. Defaults to None.

        Returns
        -------
        PandasDataFrameType
            The transformed dataset.
        """

        self.fit(X=X, y=y, sample_weight=sample_weight)
        return self.transform(X=X)

    def fit_predict(self,
                    X: PandasDataFrameType,
                    y: PandasSeriesType,
                    sample_weight=None) -> PandasSeriesType:
        """
        Sequentially runs the `fit_transform` methods of each step in the 
        pipeline, except for the last step, where the `fit_predict` method is 
        run.

        Parameters
        ----------
        X : PandasDataFrameType
            The dataset.
        y : PandasSeriesType
            The binary target column.
        sample_weight : PandasSeriesType, optional
            Row-wise weights to apply. Defaults to None.

        Returns
        -------
        PandasSeriesType
            The prediction of the final step.
        """

        self.fit(X=X, y=y, sample_weight=sample_weight)
        return self.predict(X=X)
