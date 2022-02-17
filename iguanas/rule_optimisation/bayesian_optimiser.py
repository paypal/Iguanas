"""Optimises a set of rules using Bayesian Optimisation."""
from iguanas.rules import Rules
import iguanas.utils as utils
from iguanas.utils.types import NumpyArray, PandasDataFrame, PandasSeries
from iguanas.utils.typing import PandasDataFrameType, PandasSeriesType
from iguanas.rule_optimisation._base_optimiser import _BaseOptimiser
import pandas as pd
from typing import Callable, Dict, List
from hyperopt import hp, tpe, fmin
from hyperopt.pyll import scope
import numpy as np
from joblib import Parallel, delayed


class BayesianOptimiser(_BaseOptimiser):
    """
    Optimises a set of rules (given in the standard Iguanas lambda expression
    format) using Bayesian Optimisation.

    Parameters
    ----------
    rule_lambdas : Dict[str, Callable[[Dict], str]]
        Set of rules defined using the standard Iguanas lambda expression 
        format (values) and their names (keys).
    lambda_kwargs : Dict[str, Dict[str, float]]
        For each rule (keys), a dictionary containing the features used in the 
        rule (keys) and the current values (values).
    metric : Callable
        The optimisation function used to calculate the metric which the rules
        are optimised for (e.g. F1 score).
    n_iter : int
        The number of iterations that the optimiser should perform.
    algorithm : Callable, optional
        The algorithm leveraged by hyperopt's `fmin` function, which 
        optimises the rules. Defaults to tpe.suggest, which corresponds to 
        Tree-of-Parzen-Estimator.
    num_cores : int, optional
        The number of cores to use when optimising the rule thresholds. 
        Defaults to 1.
    verbose : int, optional 
        Controls the verbosity - the higher, the more messages. >0 : shows 
        the overall progress of the optimisation process; >1 : shows the 
        progress of the optimisation of each rule, as well as the overall 
        optimisation process. Note that setting `verbose` > 1 only works 
        when `num_cores` = 1. Defaults to 0.
    **kwargs : tuple , optional
        Any additional keyword arguments to pass to hyperopt's `fmin` 
        function.

    Attributes
    ----------
    rule_strings : Dict[str, str]
        The optimised rules, defined using the standard Iguanas string 
        format (values) and their names (keys).   
    rule_lambdas : Dict[str, object]
        The optimised rules, defined using the standard Iguanas lambda 
        expression format (values) and their names (keys).   
    lambda_kwargs : Dict[str, object]
        The keyword arguments for the optimised rules defined using the 
        standard Iguanas lambda expression format.
    rules : Rules
        The Rules object containing the optimised rules.
    rule_names_missing_features : List[str]
        Names of rules which use features that are not present in the dataset 
        (and therefore can't be optimised or applied).
    rule_names_no_opt_conditions : List[str]
        Names of rules which have no optimisable conditions (e.g. rules that 
        only contain string-based conditions).
    rule_names_zero_var_features : List[str]
        Names of rules which exclusively contain zero variance features (based
        on `X`), so cannot be optimised.
    opt_rule_performances : Dict[str, float]
        The optimisation metric (values) calculated for each optimised rule
        (keys).
    orig_rule_performances : Dict[str, float]
        The optimisation metric (values) calculated for each original rule 
        (keys).
    non_optimisable_rules : Rules
        A `Rules` object containing the rules which could not be optimised.    
    """

    def __init__(self, rule_lambdas: Dict[str, Callable],
                 lambda_kwargs: Dict[str, Dict[str, float]],
                 metric: Callable,
                 n_iter: int, algorithm=tpe.suggest, num_cores=1, verbose=0,
                 **kwargs):
        _BaseOptimiser.__init__(
            self, rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs,
            metric=metric
        )
        self.n_iter = n_iter
        self.algorithm = algorithm
        self.verbose = verbose
        self.num_cores = num_cores
        self.kwargs = kwargs
        self.rule_strings = {}
        self.rule_names = []

    def __repr__(self):
        if self.rule_strings == {}:
            return f'BayesianOptimiser object with {len(self.orig_rule_lambdas)} rules to optimise'
        else:
            return f'BayesianOptimiser object with {len(self.rule_strings)} rules optimised'

    def fit(self, X: PandasDataFrameType, y=None, sample_weight=None) -> PandasDataFrameType:
        """
        Optimises a set of rules (given in the standard Iguanas lambda expression
        format) using Bayesian Optimisation.

        Parameters
        ----------
        X : PandasDataFrameType
            The feature set.
        y : PandasSeriesType, optional
            The binary target column. Not required if optimising rules on 
            unlabelled data. Defaults to None.
        sample_weight : PandasSeriesType, optional
            Record-wise weights to apply. Defaults to None.

        Returns
        -------
        PandasDataFrameType
            The binary columns of the optimised rules on the fitted dataset.
        """

        utils.check_allowed_types(X, 'X', [PandasDataFrame])
        if y is not None:
            utils.check_allowed_types(y, 'y', [PandasSeries])
        if sample_weight is not None:
            utils.check_allowed_types(
                sample_weight, 'sample_weight', [PandasSeries])
        self.orig_rules = Rules(
            rule_lambdas=self.orig_rule_lambdas.copy(),
            lambda_kwargs=self.orig_lambda_kwargs.copy(),
        )
        _ = self.orig_rules.as_rule_strings(as_numpy=False)
        if self.verbose > 0:
            print(
                '--- Checking for rules with features that are missing in `X` ---')
        self.rule_names_missing_features, rule_features_in_X = self._return_rules_missing_features(
            rules=self.orig_rules,
            columns=X.columns,
            verbose=self.verbose)
        if self.rule_names_missing_features:
            self.orig_rules.filter_rules(
                exclude=self.rule_names_missing_features)
        X = X[rule_features_in_X]
        if self.verbose > 0:
            print(
                '--- Checking for rules that exclusively contain non-optimisable conditions ---')
        all_rule_features, self.rule_names_no_opt_conditions = self._return_all_optimisable_rule_features(
            lambda_kwargs=self.orig_rules.lambda_kwargs, X=X, verbose=self.verbose
        )
        X_min, X_max = self._return_X_min_max(X, all_rule_features)
        int_cols = self._return_int_cols(X=X)
        all_space_funcs = self._return_all_space_funcs(
            all_rule_features=all_rule_features, X_min=X_min, X_max=X_max,
            int_cols=int_cols
        )
        if self.verbose > 0:
            print(
                '--- Checking for rules that exclusively contain zero-variance features ---')
        self.rule_names_zero_var_features = self._return_rules_with_zero_var_features(
            lambda_kwargs=self.orig_rules.lambda_kwargs, X_min=X_min, X_max=X_max,
            rule_names_no_opt_conditions=self.rule_names_no_opt_conditions, verbose=self.verbose
        )
        optimisable_rules, self.non_optimisable_rules = self._return_optimisable_rules(
            rules=self.orig_rules, rule_names_no_opt_conditions=self.rule_names_no_opt_conditions,
            rule_names_zero_var_features=self.rule_names_zero_var_features
        )
        if not optimisable_rules.rule_lambdas:
            raise Exception('There are no optimisable rules in the set')
        self.optimisable_rules = optimisable_rules
        orig_X_rules = optimisable_rules.transform(
            X=X
        )
        self.orig_rule_performances = dict(
            zip(
                orig_X_rules.columns.tolist(),
                self.metric(orig_X_rules, y, sample_weight)
            )
        )
        if self.verbose > 0:
            print('--- Optimising rules ---')
        opt_rule_strings = self._optimise_rules(
            rule_lambdas=optimisable_rules.rule_lambdas,
            lambda_kwargs=optimisable_rules.lambda_kwargs,
            X=X, y=y, sample_weight=sample_weight, int_cols=int_cols,
            all_space_funcs=all_space_funcs)
        opt_rules = Rules(
            rule_strings=opt_rule_strings,
        )
        opt_X_rules = opt_rules.transform(
            X=X
        )
        self.opt_rule_performances = dict(
            zip(
                opt_X_rules.columns.tolist(),
                self.metric(opt_X_rules, y, sample_weight)
            )
        )
        self.rule_strings, self.opt_rule_performances, X_rules = self._return_orig_rule_if_better_perf(
            orig_rule_performances=self.orig_rule_performances,
            opt_rule_performances=self.opt_rule_performances,
            orig_rule_strings=optimisable_rules.rule_strings,
            opt_rule_strings=opt_rules.rule_strings,
            orig_X_rules=orig_X_rules,
            opt_X_rules=opt_X_rules
        )
        self._generate_other_rule_formats()
        return X_rules

    def _optimise_rules(self, rule_lambdas: Dict[str, Callable[[Dict], str]],
                        lambda_kwargs: Dict[str, Dict[str, float]], X: PandasDataFrameType,
                        y: PandasSeriesType, sample_weight: PandasSeriesType,
                        int_cols: list, all_space_funcs: dict) -> Dict[str, str]:
        """Optimises each rule in the set"""

        opt_rule_strings = {}
        if self.verbose == 1:
            rule_lambdas_items = utils.return_progress_ready_range(
                verbose=self.verbose, range=rule_lambdas.items())
        else:
            rule_lambdas_items = rule_lambdas.items()
        with Parallel(n_jobs=self.num_cores) as parallel:
            opt_rule_strings_list = parallel(delayed(self._optimise_single_rule)(
                rule_name, rule_lambda, lambda_kwargs, X, y, sample_weight,
                int_cols, all_space_funcs
            ) for rule_name, rule_lambda in rule_lambdas_items
            )
        opt_rule_strings = dict(
            zip(rule_lambdas.keys(), opt_rule_strings_list))
        return opt_rule_strings

    def _optimise_single_rule(self, rule_name, rule_lambda, lambda_kwargs, X, y,
                              sample_weight, int_cols, all_space_funcs):
        """Optimises a single rule"""

        if self.verbose > 1:
            print(f'Optimising rule `{rule_name}`')
        rule_lambda_kwargs = lambda_kwargs[rule_name]
        rule_features = list(rule_lambda_kwargs.keys())
        rule_space_funcs = self._return_rule_space_funcs(
            all_space_funcs=all_space_funcs, rule_features=rule_features)
        opt_thresholds = self._optimise_rule_thresholds(
            rule_lambda=rule_lambda, rule_space_funcs=rule_space_funcs, X_=X,
            y=y, sample_weight=sample_weight, metric=self.metric, n_iter=self.n_iter,
            algorithm=self.algorithm, verbose=self.verbose, kwargs=self.kwargs)
        opt_thresholds = self._convert_opt_int_values(
            opt_thresholds=opt_thresholds, int_cols=int_cols)
        return rule_lambda(**opt_thresholds)

    @staticmethod
    def _return_int_cols(X: PandasDataFrameType) -> List[str]:
        """Returns the list of integer columns"""

        int_cols = X.select_dtypes(include=np.int).columns.tolist()
        float_cols = X.select_dtypes(include=np.float).columns.tolist()
        for float_col in float_cols:
            if abs(X[float_col] - X[float_col].round()).sum() == 0:
                int_cols.append(float_col)
        return int_cols

    @staticmethod
    def _return_all_space_funcs(all_rule_features: List[str],
                                X_min: PandasSeriesType,
                                X_max: PandasSeriesType,
                                int_cols: List[str]) -> Dict[str, hp.uniform]:
        """
        Returns a dictionary of the space function (used in the optimiser) for 
        each feature in the dataset
        """
        space_funcs = {}
        for feature in all_rule_features:
            # If features contains %, means that there's more than one
            # occurance of the feature in the rule. To get the column, we need
            # to get the string precending the % symbol.
            col = feature.split('%')[0]
            col_min = X_min[col]
            col_max = X_max[col]
            # If column is zero variance (excl. nulls), then set the space
            # function to the minimum value
            if col_min == col_max:
                space_funcs[feature] = col_min
                continue
            if col in int_cols:
                space_func = scope.int(
                    hp.uniform(feature, col_min, col_max))
            else:
                space_func = hp.uniform(feature, col_min, col_max)
            space_funcs[feature] = space_func
        return space_funcs

    @staticmethod
    def _return_rule_space_funcs(all_space_funcs: Dict[str, hp.uniform],
                                 rule_features: List[str]) -> Dict[str, hp.uniform]:
        """
        Returns a dictionary of the space function for each feature in 
        the rule.
        """

        rule_space_funcs = dict((rule_feature, all_space_funcs[rule_feature])
                                for rule_feature in rule_features)
        return rule_space_funcs

    @staticmethod
    def _optimise_rule_thresholds(rule_lambda: Callable[[Dict], str],
                                  rule_space_funcs: Dict[str, hp.uniform],
                                  X_: PandasDataFrameType, y: PandasSeriesType,
                                  sample_weight: PandasSeriesType,
                                  metric: Callable,
                                  algorithm: Callable,
                                  n_iter: int,
                                  verbose: int,
                                  kwargs: dict) -> Dict[str, float]:
        """Calculates the optimal rule thresholds"""

        def _objective(rule_space_funcs: Dict[str, hp.uniform]) -> float:
            """
            Evaluates the optimisation metric for each
            iteration in the optimisation process.
            """
            # Bring X_ into local scope (for eval() function)
            X = X_
            rule_string = rule_lambda(**rule_space_funcs)
            y_pred = eval(rule_string)
            # If evaluated rule is PandasSeriesType, replace pd.NA with False
            # (since pd.NA used in any condition returns pd.NA, not False as with
            # numpy)
            if utils.is_type(y_pred, [PandasSeries]):
                y_pred = y_pred.fillna(False).astype(int)
            if utils.is_type(y_pred, [NumpyArray]):
                y_pred = y_pred.astype(int)
            if y is not None:
                result = metric(
                    y_true=y, y_preds=y_pred, sample_weight=sample_weight)
            else:
                result = metric(y_preds=y_pred)

            return -result

        opt_thresholds = fmin(
            fn=_objective,
            space=rule_space_funcs,
            algo=algorithm,
            max_evals=n_iter,
            verbose=verbose > 1,
            rstate=np.random.RandomState(0),
            **kwargs
        )

        # If rule_space_funcs contained constant values (due to min/max of
        # feature being equal in the dataset), then add those values back into
        # the optimised_thresholds dictionary
        if len(opt_thresholds) < len(rule_space_funcs):
            for feature, space_func in rule_space_funcs.items():
                if feature not in opt_thresholds.keys():
                    opt_thresholds[feature] = space_func
        return opt_thresholds

    @staticmethod
    def _convert_opt_int_values(opt_thresholds: Dict[str, float],
                                int_cols: List[str]) -> Dict[str, float]:
        """
        Converts threshold values based on integer columns into integer 
        format.
        """
        for feature, value in opt_thresholds.items():
            col = feature.split('%')[0]
            if col in int_cols:
                opt_thresholds[feature] = int(value)
        return opt_thresholds
