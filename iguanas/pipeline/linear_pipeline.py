"""Class for creating a Linear Pipeline."""
from copy import deepcopy
from typing import List, Tuple, Union
from iguanas.pipeline._base_pipeline import _BasePipeline
from iguanas.utils.typing import PandasDataFrameType, PandasSeriesType
from iguanas.utils.types import PandasDataFrame, PandasSeries, Dictionary, ClassList, ClassNoneType
import iguanas.utils.utils as utils


class LinearPipeline(_BasePipeline):
    """
    Generates a linear pipeline, which is a sequence of steps which are applied 
    sequentially to a dataset. Each step should be an instantiated class with 
    both `fit` and `transform` methods. The final step should be an 
    instantiated class with either `fit` and `tranform` methods or `fit` and
    `predict` methods.

    Parameters
    ----------
    steps : List[Tuple[str, object]]
        The steps to be applied as part of the pipeline. Each element of the
        list corresponds to a single step. Each step should be a tuple of two
        elements - the first element should be a string which refers to the 
        step; the second element should be the instantiated class which is run
        as part of the step. 
    use_init_data : List[str], optional
        Use to denote steps that come after the first step which require the 
        initial dataset `X` to be used as the input to the `fit` method. For 
        example, if a rule optimiser class is placed after a rule generator 
        class in the pipeline, the tag for the rule optimiser should be added 
        to the `use_init_data` list, as the rule optimiser requires the initial
        dataset to optimise the rules. Defaults to None.

    Attributes
    ----------
    steps_ : List[Tuple[str, object]]
        The steps corresponding to the fitted pipeline.
    rules : Rules
        The Rules object containing the rules produced from fitting the 
        pipeline.
    """

    def __init__(self,
                 steps: List[Tuple[str, object]],
                 use_init_data=None):
        _BasePipeline.__init__(self, steps=steps)
        utils.check_allowed_types(
            use_init_data, 'use_init_data', [ClassList, ClassNoneType]
        )
        self.use_init_data = use_init_data

    def fit(self,
            X: Union[PandasDataFrameType, dict],
            y: Union[PandasSeriesType, dict],
            sample_weight=None) -> None:
        """
        Sequentially runs the `fit_transform` method of each step in the 
        pipeline, except for the last step, where the `fit` method is run.

        Parameters
        ----------
        X : Union[PandasDataFrameType, dict]
            The dataset or dictionary of datasets for each pipeline step.
        y : Union[PandasSeriesType, dict]
            The binary target column or dictionary of binary target columns
            for each pipeline step.
        sample_weight : Union[PandasSeriesType, dict], optional
            Row-wise weights or dictionary of row-wise weights for each
            pipeline step. Defaults to None.
        """

        utils.check_allowed_types(X, 'X', [PandasDataFrame, Dictionary])
        utils.check_allowed_types(y, 'y', [PandasSeries, Dictionary])
        if sample_weight is not None:
            utils.check_allowed_types(
                sample_weight, 'sample_weight', [PandasSeries, Dictionary])
        X_init = self._copy_X_if_use_init_data(X)
        self.steps_ = deepcopy(self.steps)
        for step_tag, step in self.steps_[:-1]:
            X = self._return_X(step_tag=step_tag, X=X, X_init=X_init)
            X = self._pipeline_fit_transform(
                step_tag=step_tag, step=step, X=X, y=y,
                sample_weight=sample_weight
            )
        final_step_tag = self.steps_[-1][0]
        final_step = self.steps_[-1][1]
        X = self._return_X(step_tag=final_step_tag, X=X, X_init=X_init)
        self._pipeline_fit(
            step_tag=final_step_tag, step=final_step, X=X, y=y,
            sample_weight=sample_weight
        )
        self.rules = final_step.rules

    def predict(self, X: Union[PandasDataFrameType, dict]) -> PandasSeriesType:
        """
        Sequentially runs the `transform` method of each step in the pipeline,
        except for the last step, where the `predict` method is run. Note that
        before using this method, you should first run either the `fit` or 
        `fit_predict` methods.

        Parameters
        ----------
        X : Union[PandasDataFrameType, dict]
            The dataset or dictionary of datasets for each pipeline step.

        Returns
        -------
        PandasSeriesType
            The prediction of the final step.
        """

        utils.check_allowed_types(X, 'X', [PandasDataFrame, Dictionary])
        X_init = self._copy_X_if_use_init_data(X)
        for (step_tag, step) in self.steps_[:-1]:
            X = self._return_X(step_tag=step_tag, X=X, X_init=X_init)
            X = self._pipeline_transform(step_tag=step_tag, step=step, X=X)
        final_step_tag = self.steps_[-1][0]
        final_step = self.steps_[-1][1]
        X = self._return_X(step_tag=final_step_tag, X=X, X_init=X_init)
        return self._pipeline_predict(step=final_step, X=X)

    def transform(self,
                  X: Union[PandasDataFrameType, dict]) -> PandasDataFrame:
        """
        Sequentially runs the `transform` method of each step in the pipeline.
        Note that before using this method, you should first run either the 
        `fit` or `fit_transform` methods.

        Parameters
        ----------
        X : Union[PandasDataFrameType, dict]
            The dataset or dictionary of datasets for each pipeline step.

        Returns
        -------
        PandasDataFrame
            The transformed dataset.
        """

        utils.check_allowed_types(X, 'X', [PandasDataFrame, Dictionary])
        X_init = self._copy_X_if_use_init_data(X)
        for step_tag, step in self.steps_:
            X = self._return_X(step_tag=step_tag, X=X, X_init=X_init)
            X = self._pipeline_transform(step_tag=step_tag, step=step, X=X)
        return X

    def fit_transform(self,
                      X: Union[PandasDataFrameType, dict],
                      y: Union[PandasSeriesType, dict],
                      sample_weight=None) -> PandasDataFrameType:
        """
        Sequentially runs the `fit_transform` method of each step in the 
        pipeline.

        Parameters
        ----------
        X : Union[PandasDataFrameType, dict]
            The dataset or dictionary of datasets for each pipeline step.
        y : Union[PandasSeriesType, dict]
            The binary target column or dictionary of binary target columns
            for each pipeline step.
        sample_weight : Union[PandasSeriesType, dict], optional
            Row-wise weights or dictionary of row-wise weights for each
            pipeline step. Defaults to None.

        Returns
        -------
        PandasDataFrameType
            The transformed dataset.
        """

        self.fit(X=X, y=y, sample_weight=sample_weight)
        return self.transform(X=X)

    def fit_predict(self,
                    X: Union[PandasDataFrameType, dict],
                    y: Union[PandasSeriesType, dict],
                    sample_weight=None) -> PandasSeriesType:
        """
        Sequentially runs the `fit_transform` method of each step in the 
        pipeline, except for the last step, where the `fit_predict` method is 
        run.

        Parameters
        ----------
        X : Union[PandasDataFrameType, dict]
            The dataset or dictionary of datasets for each pipeline step.
        y : Union[PandasSeriesType, dict]
            The binary target column or dictionary of binary target columns
            for each pipeline step.
        sample_weight : Union[PandasSeriesType, dict], optional
            Row-wise weights or dictionary of row-wise weights for each
            pipeline step. Defaults to None.

        Returns
        -------
        PandasSeriesType
            The prediction of the final step.
        """

        self.fit(X=X, y=y, sample_weight=sample_weight)
        return self.predict(X=X)

    def _return_X(self,
                  step_tag: str,
                  X: Union[PandasDataFrameType, dict],
                  X_init: Union[PandasDataFrameType, dict, None]) -> Union[
                      PandasDataFrameType, dict]:
        """
        If `use_init_data` is populated and `step_tag` is in `use_init_data`,
        returns `X_init` (i.e. the initial dataset provided); else returns
        `X`.
        """

        if self.use_init_data is not None and step_tag in self.use_init_data:
            return X_init
        else:
            return X

    def _copy_X_if_use_init_data(self,
                                 X: Union[PandasDataFrameType, dict]) -> Union[
                                     PandasDataFrameType, dict, None]:
        """
        If `use_init_data` is populated, copies `X` and returns it; else 
        returns None/
        """

        if self.use_init_data:
            return X.copy()
        else:
            return None
