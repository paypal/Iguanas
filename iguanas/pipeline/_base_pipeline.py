from copy import deepcopy
from iguanas.pipeline.class_accessor import ClassAccessor
from iguanas.utils.typing import PandasDataFrameType
import iguanas.utils.utils as utils
from iguanas.exceptions import DataFrameSizeError


class _BasePipeline:
    def __init__(self, steps) -> None:
        self.steps = steps
        self.steps_ = None

    def get_params(self) -> dict:
        """
        Returns the parameters of each step in the pipeline.

        Returns
        -------
        dict
            The parameters of each step in the pipeline.
        """

        pipeline_params = {}
        steps_ = self.steps if self.steps_ is None else self.steps_
        for step_tag, step in steps_:
            # If step inherits from _BasePipeline, call its get_params
            if issubclass(step.__class__, _BasePipeline):
                step_param_dict = step.get_params()
                pipeline_params.update(step_param_dict)
            else:
                step_param_dict = deepcopy(step.__dict__)
                pipeline_params[step_tag] = step_param_dict
        return pipeline_params

    def _update_kwargs(self, params):
        for step_tag, step in self.steps:
            # If step inherits from _BasePipeline, call its _update_kwargs
            if issubclass(step.__class__, _BasePipeline):
                step._update_kwargs(params)
            if step_tag in params.keys():
                step.__dict__.update(params[step_tag])

    def _pipeline_fit(self, step_tag, step, X, y, sample_weight):
        step = self._check_accessor(step)
        X, y, sample_weight = [
            utils.return_dataset_if_dict(
                step_tag=step_tag, df=df
            ) for df in (X, y, sample_weight)
        ]
        step.fit(X, y, sample_weight)

    def _pipeline_transform(self, step_tag, step, X):
        step = self._check_accessor(step)
        X = utils.return_dataset_if_dict(step_tag=step_tag, df=X)
        X = step.transform(X)
        self._exception_if_no_cols_in_X(X, step_tag)
        return X

    def _pipeline_predict(self, step, X):
        step = self._check_accessor(step)
        return step.predict(X)

    def _pipeline_fit_transform(self, step_tag, step, X, y, sample_weight):
        step = self._check_accessor(step)
        X, y, sample_weight = [
            utils.return_dataset_if_dict(
                step_tag=step_tag, df=df
            ) for df in (X, y, sample_weight)
        ]
        X = step.fit_transform(X, y, sample_weight)
        self._exception_if_no_cols_in_X(X, step_tag)
        return X

    def _check_accessor(self,
                        step: object,
                        ) -> object:
        """
        Checks whether the any of the parameters in the given `step` is of type
        ClassAccessor. If so, then it runs the ClassAccessor's `get` method,
        which extracts the given attribute from the given step in the pipeline,
        and injects it into the parameter.
        """

        step_param_dict = step.__dict__
        for param, value in step_param_dict.items():
            if isinstance(value, ClassAccessor):
                pipeline_params = self.get_params()
                step.__dict__[param] = value.get(pipeline_params)
        return step

    @ staticmethod
    def _exception_if_no_cols_in_X(X: PandasDataFrameType, step_tag: str):
        """Raises an exception if `X` has no columns."""
        if X.shape[1] == 0:
            raise DataFrameSizeError(
                f'`X` has been reduced to zero columns after the `{step_tag}` step in the pipeline.'
            )
