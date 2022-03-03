"""Class for creating a Parallel Pipeline."""
from copy import deepcopy
from typing import List, Tuple, Union
from iguanas.exceptions.exceptions import DataFrameSizeError, NoRulesError
from iguanas.pipeline._base_pipeline import _BasePipeline
from iguanas.utils.typing import PandasDataFrameType, PandasSeriesType
from iguanas.utils.types import PandasDataFrame, PandasSeries, Dictionary
import iguanas.utils.utils as utils
from iguanas.rules import Rules
import pandas as pd
import warnings


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
    verbose : int, optional
        Controls the verbosity - the higher, the more messages. >0 : gives
        the overall progress of the training of the pipeline; >1 : shows the 
        current step being trained.

    Attributes
    ----------
    steps_ : List[Tuple[str, object]]
        The steps corresponding to the fitted pipeline.
    rule_names : List[str]
        The names of the rules in the concatenated output.
    rules : Rules
        The Rules object containing the rules produced from fitting the 
        pipeline.

    Examples
    --------
    >>> from iguanas.pipeline import ParallelPipeline
    >>> from iguanas.rbs import RBSOptimiser, RBSPipeline
    >>> from iguanas.rule_generation import RuleGeneratorDT, RuleGeneratorOpt
    >>> from iguanas.metrics import FScore
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> import pandas as pd
    >>> f1 = FScore(beta=1)
    >>> rg_dt = RuleGeneratorDT(
    ...     metric=f1.fit, 
    ...     n_total_conditions=2, 
    ...     tree_ensemble=RandomForestClassifier(random_state=0), 
    ...     rule_name_prefix='RuleGenDT'
    ... )
    >>> rg_opt = RuleGeneratorOpt(
    ...     metric=f1.fit, 
    ...     n_total_conditions=2, 
    ...     num_rules_keep=10,
    ...     rule_name_prefix='RuleGenOpt'
    ... )
    >>> pp = ParallelPipeline(
    ...     steps=[
    ...         ('rg_dt', rg_dt), 
    ...         ('rg_opt', rg_opt)
    ...     ]
    ... )
    >>> X = pd.DataFrame({
    ...     'A': [1, 0, 1, 0],
    ...     'B': [1, 1, 1, 0]
    ... })
    >>> y = pd.Series([
    ...     1, 0, 1, 0
    ... ])
    >>> X_rules = pp.fit_transform(X=X, y=y)
    >>> print(X_rules)
       RuleGenDT_0  RuleGenDT_1  RuleGenDT_2  RuleGenOpt_0
    0            1            1            1             1
    1            0            0            1             0
    2            1            1            1             1
    3            0            0            0             0
    >>> X_rules = pp.transform(X=X)
    >>> print(X_rules)
       RuleGenDT_0  RuleGenDT_1  RuleGenDT_2  RuleGenOpt_0
    0            1            1            1             1
    1            0            0            1             0
    2            1            1            1             1
    3            0            0            0             0
    """

    def __init__(self,
                 steps: List[Tuple[str, object]],
                 verbose=0) -> None:
        _BasePipeline.__init__(self, steps=steps, verbose=verbose)
        self.rules = Rules()

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
        steps_ = utils.return_progress_ready_range(
            verbose=self.verbose == 1, range=self.steps_
        )
        for step_tag, step in steps_:
            if self.verbose > 1:
                print(
                    f'--- Applying `fit_transform` method for step `{step_tag}` ---'
                )
            # Try applying fit_transform for `step`
            try:
                X_rules_list.append(
                    self._pipeline_fit_transform(
                        step_tag, step, X, y, sample_weight
                    )
                )
                rules_list.append(step.rules)
            # If no rules generated/remain, raise warning and skip `step`
            except (DataFrameSizeError, NoRulesError) as e:
                warnings.warn(
                    f'No rules remain in step `{step_tag}` as it raised the following error: "{e}"'
                )
                X_rules_list.append(pd.DataFrame())
                rules_list.append(Rules())
        X_rules = pd.concat(X_rules_list, axis=1)
        self.rules = sum(rules_list)
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
            # Try applying transform for `step`
            try:
                X_rules_list.append(
                    self._pipeline_transform(
                        step_tag, step, X
                    )
                )
            # If no rules present, raise warning and skip `step`; else raise
            # exception
            except Exception as e:
                if str(e) == '`rule_dicts` must be given':
                    warnings.warn(
                        f'No rules present in step `{step_tag}` - `transform` method cannot be applied for this step.'
                    )
                    X_rules_list.append(pd.DataFrame())
                else:
                    raise e
        X_rules = pd.concat(X_rules_list, axis=1)
        self.rule_names = X_rules.columns.tolist()
        return X_rules
