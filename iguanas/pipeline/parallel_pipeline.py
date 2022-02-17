"""Class for creating a Parallel Pipeline."""
from copy import deepcopy
from typing import List, Tuple, Union
from iguanas.pipeline._base_pipeline import _BasePipeline
from iguanas.utils.typing import PandasDataFrameType, PandasSeriesType
from iguanas.utils.types import PandasDataFrame, PandasSeries, Dictionary
import iguanas.utils.utils as utils
from iguanas.rules import Rules
import pandas as pd


class ParallelPipeline(_BasePipeline):
    """
    Generates a parallel pipeline, which is a set of steps which run
    independently - their outputs are then concatenated and returned. Each step 
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
    rule_names : List[str]
        The names of the rules in the concatenated output.
    rules : Rules
        The Rules object containing the rules produced from fitting the 
        pipeline.
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

        utils.check_allowed_types(X, 'X', [PandasDataFrame, Dictionary])
        utils.check_allowed_types(y, 'y', [PandasSeries, Dictionary])
        if sample_weight is not None:
            utils.check_allowed_types(
                sample_weight, 'sample_weight', [PandasSeries, Dictionary])
        self.steps_ = deepcopy(self.steps)
        X_rules_list = []
        rules_list = []
        for step_tag, step in self.steps_:
            X_rules_list.append(
                self._pipeline_fit_transform(
                    step_tag, step, X, y, sample_weight
                )
            )
            rules_list.append(step.rules)
        X_rules = pd.concat(X_rules_list, axis=1)
        self.rules = self._concat_rules(rules_list)
        self.rule_names = X_rules.columns.tolist()
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

        utils.check_allowed_types(X, 'X', [PandasDataFrame, Dictionary])
        X_rules_list = []
        for step_tag, step in self.steps_:
            X_rules_list.append(
                self._pipeline_transform(
                    step_tag, step, X
                )
            )
        X_rules = pd.concat(X_rules_list, axis=1)
        self.rule_names = X_rules.columns.tolist()
        return X_rules

    @staticmethod
    def _concat_rules(rules_list: List[Rules]) -> Rules:
        """
        Returns the combined rule set given a list of individual rule sets. If
        `rules_list` is all None, returns None. If elements in `rules_list` are
        None, raises an exception.
        """

        if all([rule is None for rule in rules_list]):
            return None
        elif None in rules_list:
            raise TypeError(
                """
                One or more of the classes in the pipeline has `None` assigned to 
                the `rules` parameter, whereas other classes in the pipeline have 
                the `rules` parameter populated. Either set all to `None` or 
                provide the `rules` parameter for all classes.
                """
            )
        else:
            return sum(rules_list)
