"""Optimises a Rules-Based System (RBS) pipeline."""
from typing import Callable, List, Tuple
from hyperopt import hp, tpe, fmin
import numpy as np
from copy import deepcopy
from iguanas.rbs import RBSPipeline
import iguanas.utils as utils
from iguanas.utils.types import PandasDataFrame, PandasSeries
from iguanas.utils.typing import PandasDataFrameType, PandasSeriesType


class RBSOptimiser(RBSPipeline):
    """
    Optimises the rules within an RBS Pipeline based on an optimisation 
    function. If the `config` parameter is an empty list, then the pipeline
    configuration is optimised from scratch; else, the rules included within
    the existing pipeline configuration are optimised.

    Parameters
    ----------
    pipeline : RBSPipeline
        The RBS Pipeline to optimise.
    metric : Callable
        The method/function used to optimise the pipeline (e.g. F1 score).
    n_iter : int
        The number of iterations that the optimiser should perform.
    algorithm : Callable, optional
        The algorithm leveraged by hyperopt's `fmin` function, which optimises
        the rules. Defaults to tpe.suggest, which corresponds to 
        Tree-of-Parzen-Estimator.
    pos_pred_rules : List[str], optional
        The list of rules that predict a positive case (i.e. 1). This parameter 
        and/or `neg_pred_rules` must be given when the `config` parameter in 
        the `pipeline` is an empty list. Defaults to None.
    neg_pred_rules : List[str], optional
        The list of rules that predict a negative case (i.e. 0). This parameter 
        and/or `pos_pred_rules` must be given when the `config` parameter in 
        the `pipeline` is an empty list. Defaults to None.    
    rules : Rules, optional
        An Iguanas `Rules` object containing the rules that need to be 
        filtered. If provided, the rules within the object will be filtered. 
        Defaults to None.
    verbose : int, optional
        Controls the verbosity - the higher, the more messages. >0 : shows the 
        overall progress of the optimisation process. Defaults to 0.

    Raises
    ------
    ValueError
        If `config` not provided in `pipeline`, then one or both of 
        `pos_pred_rules` and `neg_pred_rules` must be given.

    Attributes
    ----------
    config : List[Tuple(int, list)] 
        The optimised pipeline configuration, where each element aligns to a
        stage in the pipeline. Each element is a tuple with 2 elements: the 
        first element corresponds to the decision made at that stage (either 0
        or 1); the second element is a list of the rules that must trigger to 
        give that decision.
    rules_to_keep: List[str]
        The rules used in the optimised pipeline.    
    rules : Rules
        The Rules object containing the rules remaining after optimisation.
    """

    def __init__(self,
                 pipeline: RBSPipeline,
                 metric: Callable,
                 n_iter: int,
                 algorithm=tpe.suggest,
                 pos_pred_rules=None,
                 neg_pred_rules=None,
                 rules=None,
                 verbose=0,
                 **kwargs) -> None:
        RBSPipeline.__init__(
            self,
            config=deepcopy(pipeline.config),
            final_decision=pipeline.final_decision,
        )
        self.metric = metric
        self.n_iter = n_iter
        self.algorithm = algorithm
        self.pos_pred_rules = [] if pos_pred_rules is None else pos_pred_rules
        self.neg_pred_rules = [] if neg_pred_rules is None else neg_pred_rules
        self.rules = deepcopy(rules)
        self.verbose = verbose
        self.kwargs = kwargs
        self.orig_config = deepcopy(pipeline.config)
        self.config_given = self.orig_config != []
        if not self.config_given and self.pos_pred_rules == [] and self.neg_pred_rules == []:
            raise ValueError(
                'If `config` not provided in `pipeline`, then one or both of `pos_pred_rules` and `neg_pred_rules` must be given.')

    def fit(self, X_rules: PandasDataFrameType, y: PandasSeriesType,
            sample_weight=None) -> None:
        """
        Optimises the pipeline for the given dataset.

        Parameters
        ----------
        X_rules : PandasDataFrameType
            Dataset of each applied rule.
        y : PandasSeriesType
            The target.
        sample_weight : PandasSeriesType, optional
            Record-wise weights to apply. Defaults to None. Defaults to 
            None.
        """

        utils.check_allowed_types(X_rules, 'X_rules', [PandasDataFrame])
        utils.check_allowed_types(y, 'y', [PandasSeries])
        if sample_weight is not None:
            utils.check_allowed_types(
                sample_weight, 'sample_weight', [PandasSeries])
        # Get space functions
        space_funcs = self._get_space_funcs(X_rules)
        # Optimise pipeline
        opt_thresholds = self._optimise_pipeline(
            X_rules, y, sample_weight, space_funcs
        )
        # Generate config
        self._generate_config(opt_thresholds)
        # Create list of final rules present in pipeline
        self.rules_to_keep = [
            rule for stage in self.config for rule in stage[1]
        ]
        # If `rules` given, filter it
        if self.rules is not None:
            self.rules.filter_rules(include=self.rules_to_keep)

    def fit_predict(self, X_rules: PandasDataFrameType, y: PandasSeriesType,
                    sample_weight=None) -> PandasSeriesType:
        """
        Optimises the pipeline for the given dataset and applies the pipeline 
        to the dataset.

        Parameters
        ----------
        X_rules : PandasDataFrameType
            Dataset of each applied rule.
        y : PandasSeriesType
            The target.
        sample_weight : PandasSeriesType, optional
            Record-wise weights to apply. Defaults to None. Defaults to 
            None.

        Returns
        -------
        PandasSeriesType
            The prediction of the pipeline.
        """

        self.fit(X_rules, y, sample_weight)
        return self.predict(X_rules)

    def _get_space_funcs(self, X_rules: PandasDataFrameType) -> dict:
        """Returns the space functions for each rule."""

        if self.config_given:
            space_funcs = {
                rule: hp.choice(rule, [0, 1]) for rule in X_rules.columns
            }
        else:
            space_funcs = {
                rule: (hp.choice(f'{rule}%activate', [0, 1]), hp.choice(f'{rule}%stage', list(range(0, X_rules.shape[1])))) for rule in X_rules.columns
            }
        return space_funcs

    def _optimise_pipeline(self,
                           X_rules: PandasDataFrameType,
                           y: PandasSeriesType,
                           sample_weight: PandasSeriesType,
                           space_funcs: dict) -> dict:
        """Calculates the optimal pipeline configuration"""

        def _objective_update_config(space_funcs: dict) -> float:
            """Evaluates the optimisation metric for the updated pipeline"""
            rules_for_iter = [
                rule for rule, keep in space_funcs.items() if keep == 1
            ]
            self.config = self._update_config(
                space_funcs, deepcopy(self.orig_config)
            )
            y_pred = self.predict(X_rules[rules_for_iter])
            score = self.metric(y_pred, y, sample_weight)
            return -score

        def _objective_generate_config(space_funcs: dict) -> float:
            """Evaluates the optimisation metric for the generated pipeline"""
            rules_for_iter = [
                rule for rule, (keep, _) in space_funcs.items() if keep == 1
            ]
            self.config = self._create_config(space_funcs)
            y_pred = self.predict(X_rules[rules_for_iter])
            score = self.metric(y_pred, y, sample_weight)
            return -score

        if self.config_given:
            _objective = _objective_update_config
        else:
            _objective = _objective_generate_config

        opt_thresholds = fmin(
            fn=_objective,
            space=space_funcs,
            algo=self.algorithm,
            max_evals=self.n_iter,
            verbose=self.verbose > 0,
            rstate=np.random.RandomState(0),
            **self.kwargs
        )
        return opt_thresholds

    def _generate_config(self, opt_thresholds: dict) -> None:
        """Generates final pipeline config based on optimisation"""

        if self.config_given:
            self.config = self._update_config(opt_thresholds, self.config)
        else:
            opt_thresholds = self._convert_opt_thr(opt_thresholds)
            self.config = self._create_config(opt_thresholds)

    def _create_config(self, space_funcs: dict) -> dict:
        """Creates pipeline config from space functions"""

        config = []
        rule_positions = {
            rule: position for rule, (keep, position) in space_funcs.items() if keep == 1
        }
        rules_ordered = [
            rule for rule, _ in sorted(rule_positions.items(), key=lambda item: item[1])
        ]
        rules_decisions = [
            (rule, 0) if rule in self.neg_pred_rules else (rule, 1) for rule in rules_ordered
        ]
        for i, (rule, decision) in enumerate(rules_decisions):
            if i == 0:
                config.append((decision, [rule]))
            # If decision for iter is same as previous iter, then append rule
            # to the existing decision's list of rules
            elif decision == rules_decisions[i-1][1]:
                config[-1][1].append(rule)
            else:
                config.append((decision, [rule]))
        return config

    @staticmethod
    def _update_config(space_funcs: dict,
                       config: List[Tuple[int, list]]) -> List[dict]:
        """Updates existing config from space functions"""

        for rule, to_keep in space_funcs.items():
            if to_keep == 0:
                for stage in config:
                    rules = stage[1]
                    if rule in rules:
                        rules.remove(rule)
        return config

    @staticmethod
    def _convert_opt_thr(opt_thresholds: dict) -> dict:
        """Converts output of optimiser into space function format"""

        opt_thresholds_ = {rule_label.split(
            '%')[0]: [None, None] for rule_label in opt_thresholds.keys()}
        for rule_label, value in opt_thresholds.items():
            rule = rule_label.split('%')[0]
            value_type = rule_label.split('%')[1]
            if value_type == 'activate':
                opt_thresholds_[rule][0] = value
            elif value_type == 'stage':
                opt_thresholds_[rule][1] = value
        return opt_thresholds_
