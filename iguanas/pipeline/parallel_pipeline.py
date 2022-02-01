from copy import deepcopy
from typing import List, Tuple, Union
from iguanas.pipeline._base_pipeline import _BasePipeline
from iguanas.utils.typing import PandasDataFrameType, PandasSeriesType
import pandas as pd


class ParallelPipeline(_BasePipeline):
    """
    Generates a parallel pipeline, which is a set of steps which run
    independently - their outputs are then combined and returned. Each step 
    should be an instantiated class with both `fit` and `transform` methods.

    Parameters
    ----------
    steps : List[Tuple[str, object]]
        The steps to be applied as part of the pipeline. Each element of the
        list corresponds to a single step. Each step should be a tuple of two
        elements - the first element should be a string which refers to the 
        step; the second element should be the instantiated class which is run
        as part of the step. 

    Attributes
    ----------
    steps_ : List[Tuple[str, object]]
        The steps corresponding to the fitted pipeline.
    """

    def __init__(self,
                 steps: List[Tuple[str, object]]) -> None:
        _BasePipeline.__init__(self, steps=steps)

    def fit_transform(self,
                      X: Union[PandasDataFrameType, dict],
                      y: Union[PandasSeriesType, dict],
                      sample_weight=None) -> PandasDataFrameType:
        """
        Independently runs the `fit_transform` method of each step in the 
        pipeline, then concatenates the output of each step column-wise.        

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

        self.steps_ = deepcopy(self.steps)
        X_rules_list = []
        for step_tag, step in self.steps_:
            X_rules_list.append(
                self._pipeline_fit_transform(
                    step_tag, step, X, y, sample_weight
                )
            )
        X_rules = pd.concat(X_rules_list, axis=1)
        return X_rules

    def transform(self,
                  X: Union[PandasDataFrameType, dict]) -> PandasDataFrameType:
        """
        Independently runs the `transform` method of each step in the pipeline,
        then concatenates the output of each step column-wise. Note that before
        using this method, you should first run the `fit_transform` method.     

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

        X_rules_list = []
        for step_tag, step in self.steps_:
            X_rules_list.append(
                self._pipeline_transform(
                    step_tag, step, X
                )
            )
        X_rules = pd.concat(X_rules_list, axis=1)
        return X_rules
