"""Class for creating a Linear Pipeline."""
from copy import deepcopy
from random import sample
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
        for step_idx, (step_tag, step) in enumerate(self.steps_[:-1]):
            # step = self._check_accessor(step)
            # X = step.fit_transform(X, y, sample_weight)
            # self._exception_if_no_cols_in_X(X, step_tag)
            X = self._pipeline_fit_transform(
                step_idx=step_idx, step_tag=step_tag, step=step, X=X, y=y,
                sample_weight=sample_weight
            )
        final_step, y, sample_weight = self._prepare_final_step(
            y=y, sample_weight=sample_weight
        )
        final_step.fit(X, y, sample_weight)

    def transform(self,
                  X):

        for step_idx, (step_tag, step) in enumerate(self.steps_):
            X = step.transform(X)
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

        self.steps_ = deepcopy(self.steps)
        for step_idx, (step_tag, step) in enumerate(self.steps_):
            # step = self._check_accessor(step)
            # X = step.fit_transform(X, y, sample_weight)
            # self._exception_if_no_cols_in_X(X, step_tag)
            X = self._pipeline_fit_transform(
                step_idx=step_idx, step_tag=step_tag, step=step, X=X, y=y,
                sample_weight=sample_weight
            )
        return X

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

        self.steps_ = deepcopy(self.steps)
        for step_idx, (step_tag, step) in enumerate(self.steps_[:-1]):
            # step = self._check_accessor(step)
            # X = step.fit_transform(X, y, sample_weight)
            # self._exception_if_no_cols_in_X(X, step_tag)
            # print(step_idx, step_tag, step)
            X = self._pipeline_fit_transform(
                step_idx=step_idx, step_tag=step_tag, step=step, X=X, y=y,
                sample_weight=sample_weight
            )
        # final_step = self.steps_[-1][1]
        # final_step = self._check_accessor(final_step)
        final_step, y, sample_weight = self._prepare_final_step(
            y=y, sample_weight=sample_weight
        )
        return final_step.fit_predict(X, y, sample_weight)

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
            X = step.transform(X)
            self._exception_if_no_cols_in_X(X, step_tag)
        final_step = self.steps_[-1][1]
        return final_step.predict(X)

    # def get_params(self) -> dict:
    #     """
    #     Returns the parameters of each step in the pipeline.

    #     Returns
    #     -------
    #     dict
    #         The parameters of each step in the pipeline.
    #     """

    #     pipeline_params = self.__dict__
    #     steps_ = self.steps if self.steps_ is None else self.steps_
    #     for step_tag, step in steps_:
    #         step_param_dict = {
    #             f'{step_tag}__{param}': value for param,
    #             value in step.__dict__.items()
    #         }
    #         pipeline_params.update(step_param_dict)
    #     return pipeline_params

    # @staticmethod
    # def _check_accessor(step: object,
    #                     steps: List[Tuple[str, object]]) -> object:
    #     """
    #     Checks whether the any of the parameters in the given `step` is of type
    #     ClassAccessor. If so, then it runs the ClassAccessor's `get` method,
    #     which extracts the given attribute from the given step in the pipeline,
    #     and injects it into the parameter.
    #     """

    #     step_param_dict = step.__dict__
    #     for param, value in step_param_dict.items():
    #         if isinstance(value, ClassAccessor):
    #             step.__dict__[param] = value.get(steps)
    #     return step

#     @staticmethod
#     def _exception_if_no_cols_in_X(X: PandasDataFrameType, step_tag: str):
#         """Raises an exception if `X` has no columns."""
#         if X.shape[1] == 0:
#             raise DataFrameSizeError(
#                 f'`X` has been reduced to zero columns after the `{step_tag}` step in the pipeline.'
#             )


# class DataFrameSizeError(Exception):
#     """
#     Custom exception for when `X` has no columns.
#     """
#     pass
