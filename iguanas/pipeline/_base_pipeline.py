from copy import deepcopy
from random import sample
from typing import List, Tuple, Dict
from iguanas.pipeline.class_accessor import ClassAccessor
from iguanas.utils.typing import PandasDataFrameType, PandasSeriesType


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

        # pipeline_params = deepcopy(self.__dict__)
        pipeline_params = {}
        steps_ = self.steps if self.steps_ is None else self.steps_
        for step_tag, step in steps_:
            if issubclass(step.__class__, _BasePipeline):
                step_param_dict = step.get_params()
                pipeline_params.update(step_param_dict)
                # pipeline_params[step_tag] = step.get_params()
            else:
                # step_param_dict = {
                #     f'{step_tag}__{param}': value for param,
                #     value in step.__dict__.items()
                # }
                step_param_dict = deepcopy(step.__dict__)
            # pipeline_params.update(step_param_dict)
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
            self._return_dataset_if_dict(
                step_tag=step_tag, df=df
            ) for df in (X, y, sample_weight)
        ]
        step.fit(X, y, sample_weight)
        # return step

    def _pipeline_transform(self, step_tag, step, X):
        step = self._check_accessor(step)
        X = self._return_dataset_if_dict(step_tag=step_tag, df=X)
        X = step.transform(X)
        self._exception_if_no_cols_in_X(X, step_tag)
        return X

    def _pipeline_predict(self, step, X):
        step = self._check_accessor(step)
        return step.predict(X)

    def _pipeline_fit_transform(self, step_tag, step, X, y, sample_weight):
        step = self._check_accessor(step)
        X, y, sample_weight = [
            self._return_dataset_if_dict(
                step_tag=step_tag, df=df
            ) for df in (X, y, sample_weight)
        ]
        X = step.fit_transform(X, y, sample_weight)
        self._exception_if_no_cols_in_X(X, step_tag)
        return X

    def _check_accessor(self,
                        step: object,
                        # steps: List[Tuple[str, object]]
                        # pipeline_params: Dict[str, dict]
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
                # step.__dict__[param] = value.get(steps)
                pipeline_params = self.get_params()
                step.__dict__[param] = value.get(pipeline_params)
        return step

    # @ staticmethod
    # def _return_datasets_if_dict(step_tag, X, y, sample_weight):
    #     # def _return_dataset_if_dict(step_tag, df):
    #     #     if isinstance(df, dict) and step_tag in df.keys():
    #     #         return df[step_tag]
    #     #     else:
    #     #         return df

    #     # X, y, sample_weight = [
    #     #     _return_dataset_if_dict(step_tag, df) for df in (X, y, sample_weight)
    #     # ]
    #     # return X, y, sample_weight
    #     df_list = []
    #     for df in X, y, sample_weight:
    #         if isinstance(df, dict) and step_tag in df.keys():
    #             df_list.append(df[step_tag])
    #         else:
    #             df_list.append(df)
    #     return df_list

    @ staticmethod
    def _return_dataset_if_dict(step_tag, df):
        if isinstance(df, dict) and step_tag in df.keys():
            return df[step_tag]
        else:
            return df

    @ staticmethod
    def _exception_if_no_cols_in_X(X: PandasDataFrameType, step_tag: str):
        """Raises an exception if `X` has no columns."""
        if X.shape[1] == 0:
            raise DataFrameSizeError(
                f'`X` has been reduced to zero columns after the `{step_tag}` step in the pipeline.'
            )


class DataFrameSizeError(Exception):
    """
    Custom exception for when `X` has no columns.
    """
    pass
