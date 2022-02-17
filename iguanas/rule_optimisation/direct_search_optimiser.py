"""Optimises a set of rules using Direct Search algorithms."""
from iguanas.rules import Rules
from iguanas.rule_optimisation._base_optimiser import _BaseOptimiser
import iguanas.utils as utils
from iguanas.utils.types import NumpyArray, PandasDataFrame, PandasSeries
from iguanas.utils.typing import PandasDataFrameType, PandasSeriesType
import pandas as pd
from typing import Callable, Dict, List, Union
import numpy as np
from scipy.optimize import minimize
import warnings


class DirectSearchOptimiser(_BaseOptimiser):
    """
    Optimises a set of rules (given in the standard Iguanas lambda expression
    format) using Direct Search-type algorithms.

    Parameters
    ----------
    rule_lambdas : Dict[str, Callable]
        Set of rules defined using the standard Iguanas lambda expression 
        format (values) and their names (keys).
    lambda_kwargs : Dict[str, Dict[str, float]]
        For each rule (keys), a dictionary containing the features used in
        the rule (keys) and the current values (values).
    metric : Callable
        The optimisation function used to calculate the metric which the 
        rules are optimised for (e.g. F1 score).
    x0 : dict, optional
        Dictionary of the initial guess (values) for each rule (keys). If
        None, defaults to the current values used in each rule (taken from
        the `lambda_kwargs` parameter). See scipy.optimize.minimize()
        documentation for more information. Defaults to None.
    method : str, optional
        Type of solver. See scipy.optimize.minimize() documentation for 
        more information. Defaults to None.
    jac : dict, optional
        Dictionary of the method for computing the gradient vector (values)
        for each rule (keys). See scipy.optimize.minimize() documentation 
        for more information. Defaults to None.
    hess : dict, optional
        Dictionary of the method for computing the Hessian matrix (values)
        for each rule (keys). See scipy.optimize.minimize() documentation 
        for more information. Defaults to None.
    hessp : dict, optional
        Dictionary of the Hessian of objective function times an arbitrary
        vector p (values) for each rule (keys). See 
        scipy.optimize.minimize() documentation for more information. 
        Defaults to None.
    bounds : dict, optional
        Dictionary of the bounds on variables (values) for each rule 
        (keys). See scipy.optimize.minimize() documentation for more 
        information. Defaults to None.
    constraints : dict, optional
        Dictionary of the constraints definition (values) for each rule
        (keys). See scipy.optimize.minimize() documentation for more 
        information. Defaults to None.
    tol : dict, optional
        Dictionary of the tolerance for termination (values) for each rule
        (keys). See scipy.optimize.minimize() documentation for more 
        information. Defaults to None.
    callback : dict, optional
        Dictionary of the callbacks (values) for each rule (keys). See
        scipy.optimize.minimize() documentation for more information. 
        Defaults to None.
    options : dict, optional
        Dictionary of the solver options (values) for each rule (keys). See
        scipy.optimize.minimize() documentation for more information. 
        Defaults to None.
    verbose : int, optional
        Controls the verbosity - the higher, the more messages. >0 : shows
        the overall progress of the optimisation process. Defaults to 0.

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
    rule_names : List[str]
        The names of the optimised rules.
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

    def __init__(self,
                 rule_lambdas: Dict[str, Callable],
                 lambda_kwargs: Dict[str, Dict[str, float]],
                 metric: Callable,
                 x0=None,
                 method=None,
                 jac=None,
                 hess=None,
                 hessp=None,
                 bounds=None,
                 constraints=None,
                 tol=None,
                 callback=None,
                 options=None,
                 verbose=0):

        _BaseOptimiser.__init__(
            self, rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs,
            metric=metric
        )
        self.x0 = x0
        self.method = method
        self.jac = jac
        self.hess = hess
        self.hessp = hessp
        self.bounds = bounds
        self.constraints = constraints
        self.tol = tol
        self.callback = callback
        self.options = options
        self.verbose = verbose
        self.rule_strings = {}
        self.rule_names = []

    def __repr__(self):
        if self.rule_strings == {}:
            return f'DirectSearchOptimiser object with {len(self.orig_rule_lambdas)} rules to optimise'
        else:
            return f'DirectSearchOptimiser object with {len(self.rule_strings)} rules optimised'

    def fit(self, X: PandasDataFrameType, y=None, sample_weight=None) -> PandasDataFrameType:
        """
        Optimises a set of rules (given in the standard Iguanas lambda expression
        format) using Direct Search-type algorithms.

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
        orig_X_rules = optimisable_rules.transform(X=X)
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
            X=X, y=y, sample_weight=sample_weight
        )
        opt_rules = Rules(rule_strings=opt_rule_strings)
        opt_X_rules = opt_rules.transform(X=X)
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

    @classmethod
    def create_bounds(cls, X: PandasDataFrameType,
                      lambda_kwargs: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Creates the `bounds` parameter using the min and max of each feature in
        each rule.

        Parameters
        ----------
        X : PandasDataFrameType
            The feature set.
        lambda_kwargs : Dict[str, Dict[str, float]]
            For each rule (keys), a dictionary containing the features used in
            the rule (keys) and the current values (values).

        Returns
        -------
        Dict[str, np.ndarray]
            The bounds for each feature (values) in each rule (keys).
        """

        bounds = cls._param_base_calc(
            X=X, lambda_kwargs=lambda_kwargs, param='bounds',
            func=lambda X_min, X_max: list(zip(X_min, X_max))
        )
        return bounds

    @classmethod
    def create_x0(cls, X: PandasDataFrameType,
                  lambda_kwargs: Dict[str, dict]) -> Dict[str, np.ndarray]:
        """
        Creates the `x0` parameter using the mid-range value of each feature in
        each rule.

        Parameters
        ----------        
        X : PandasDataFrameType
            The feature set.
        lambda_kwargs : Dict[str, Dict[str, float]]
            For each rule (keys), a dictionary containing the features used in
            the rule (keys) and the current values (values).

        Returns
        -------
        Dict[str, np.ndarray]
            The x0 for each feature (values) in each rule (keys).
        """

        x0 = cls._param_base_calc(
            X=X, lambda_kwargs=lambda_kwargs, param='x0',
            func=lambda X_min, X_max: (
                (X_max-X_min)/2).astype(float)
        )
        return x0

    @classmethod
    def create_initial_simplexes(cls, X: PandasDataFrameType,
                                 lambda_kwargs: Dict[str, dict],
                                 shape: str) -> Dict[str, np.ndarray]:
        """
        Creates the `initial_simplex` parameter for each rule.

        Parameters
        ----------
        X : PandasDataFrameType
            The feature set.
        lambda_kwargs : Dict[str, dict]
            For each rule (keys), a dictionary
            containing the features used in the rule (keys) and the current
            values (values).
        shape : str
            Name of specified simplex structure. Can be
            'Origin-based' (simplex begins at origin and extends to feature
            maximums), 'Minimum-based' (simplex begins at feature minimums
            and extends to feature maximums) or 'Random-based' (randomly
            assigned simplex between feature minimums and feature maximums).

        Returns
        -------
        Dict[str, np.ndarray]
            The initial simplex (values) for each rule (keys).
        """

        def _create_origin_based(X_min, X_max):
            num_features = len(X_min)
            initial_simplex = np.vstack(
                (X_min, np.multiply(np.identity(num_features), X_max)))
            initial_simplex = initial_simplex.astype(float)
            return initial_simplex

        def _create_minimum_based(X_min, X_max):
            num_features = len(X_min)
            simplex = np.empty((num_features, num_features))
            for i in range(num_features):
                dropped = X_min[i]
                X_min[i] = X_max[i]
                simplex[i, :] = X_min
                X_min[i] = dropped
            initial_simplex = np.vstack((X_min, simplex))
            initial_simplex = initial_simplex.astype(float)
            return initial_simplex

        def _create_random_based(X_min, X_max):
            num_features = len(X_min)
            np.random.seed(0)
            initial_simplex = np.empty(
                (num_features, num_features+1))
            for i in range(0, num_features):
                feature_min = X_min[i]
                feature_max = X_max[i]
                if np.isnan(feature_min) and np.isnan(feature_max):
                    feature_vertices = np.array(
                        [np.nan] * (num_features + 1))
                else:
                    feature_vertices = np.random.uniform(
                        feature_min, feature_max+((feature_max-feature_min)/1000), num_features+1)
                initial_simplex[i] = feature_vertices
            initial_simplex = initial_simplex.T
            return initial_simplex

        if shape not in ["Origin-based", "Minimum-based", "Random-based"]:
            raise ValueError(
                '`shape` must be either "Origin-based", "Minimum-based" or "Random-based"')
        if shape == 'Origin-based':
            initial_simplexes = cls._param_base_calc(
                X=X, lambda_kwargs=lambda_kwargs, param='initial_simplex',
                func=_create_origin_based
            )
        elif shape == 'Minimum-based':
            initial_simplexes = cls._param_base_calc(
                X=X, lambda_kwargs=lambda_kwargs, param='initial_simplex',
                func=_create_minimum_based
            )
        elif shape == 'Random-based':
            initial_simplexes = cls._param_base_calc(
                X=X, lambda_kwargs=lambda_kwargs, param='initial_simplex',
                func=_create_random_based
            )
        initial_simplexes = {
            rule_name: {'initial_simplex': simplex}
            for rule_name, simplex in initial_simplexes.items()
        }
        return initial_simplexes

    def _optimise_rules(self,
                        rule_lambdas: Dict[str, Callable[[Dict], str]],
                        lambda_kwargs: Dict[str, Dict[str, float]],
                        X: PandasDataFrameType,
                        y: PandasSeriesType,
                        sample_weight: PandasSeriesType) -> Dict[dict, dict]:
        """Optimise each rule in the set"""

        def _objective(rule_vals: List,
                       rule_lambda: dict,
                       rule_features: List,
                       X: PandasDataFrameType,
                       y: PandasSeriesType,
                       sample_weight: PandasSeriesType) -> np.float:
            """
            Evaluates the optimisation metric for each
            iteration in the optimisation process.
            """

            lambda_kwargs = dict(zip(rule_features, rule_vals))
            y_pred = eval(rule_lambda(**lambda_kwargs))
            # If evaluated rule is PandasSeriesType, replace pd.NA with False
            # (since pd.NA used in any condition returns pd.NA, not False as with
            # numpy)
            if utils.is_type(y_pred, [PandasSeries]):
                y_pred = y_pred.fillna(False).astype(int)
            if utils.is_type(y_pred, [NumpyArray]):
                y_pred = y_pred.astype(int)
            if y is not None:
                result = self.metric(
                    y_true=y, y_preds=y_pred, sample_weight=sample_weight)
            else:
                result = self.metric(y_preds=y_pred)
            return -result

        opt_rule_strings = {}
        if self.verbose == 1:
            rule_lambdas_items = utils.return_progress_ready_range(
                verbose=self.verbose, range=rule_lambdas.items())
        else:
            rule_lambdas_items = rule_lambdas.items()
        for rule_name, rule_lambda in rule_lambdas_items:
            minimize_kwargs = self._return_kwargs_for_minimize(
                rule_name=rule_name)
            rule_features = list(lambda_kwargs[rule_name].keys())
            opt_val = minimize(
                fun=_objective,
                args=(rule_lambda, rule_features, X, y, sample_weight),
                method=self.method, **minimize_kwargs
            )
            lambda_kwargs_opt = dict(zip(rule_features, opt_val.x))
            opt_rule_strings[rule_name] = rule_lambda(**lambda_kwargs_opt)
        return opt_rule_strings

    def _return_kwargs_for_minimize(self, rule_name: str) -> dict:
        """
        Returns the keyword-arguments to inject into the minimize() function
        for the given rule.
        """

        kwargs_dicts = {
            'x0': self.x0,
            'jac': self.jac,
            'hess': self.hess,
            'hessp': self.hessp,
            'bounds': self.bounds,
            'constraints': self.constraints,
            'tol': self.tol,
            'callback': self.callback,
            'options': self.options
        }
        minimize_kwargs = {}
        for kwarg_name, kwarg_dict in kwargs_dicts.items():
            minimize_kwargs[kwarg_name] = self._return_opt_param_for_rule(
                param_name=kwarg_name, param_dict=kwarg_dict,
                rule_name=rule_name
            )
        return minimize_kwargs

    def _return_opt_param_for_rule(self, param_name: str, param_dict: dict,
                                   rule_name: str) -> Union[str, float, dict]:
        """Returns the keyword-argument for the given parameter and rule."""

        if param_name == 'constraints' and param_dict is None:
            return ()
        elif param_name == 'x0' and param_dict is None:
            return np.array(list(self.orig_lambda_kwargs[rule_name].values()))
        elif param_dict is None:
            return None
        elif isinstance(param_dict, dict):
            return param_dict[rule_name]
        else:
            raise TypeError(
                f'`{param_name}` must be a dictionary with each element aligning with a rule.')

    @staticmethod
    def _param_base_calc(X: PandasDataFrameType,
                         lambda_kwargs: Dict[str, Dict[str, float]],
                         param: str, func: Callable) -> np.ndarray:
        """Base calculator for input parameters"""

        results = {}
        X_min = X.min()
        X_max = X.max()
        non_opt_rules = []
        missing_feat_rules = []
        for rule_name, lambda_kwarg in lambda_kwargs.items():
            if not lambda_kwarg:
                non_opt_rules.append(rule_name)
                continue
            cols = [feat.split('%')[0] for feat in lambda_kwarg.keys()]
            cols_missing = [col not in X.columns for col in cols]
            if sum(cols_missing) > 0:
                missing_feat_rules.append(rule_name)
                continue
            results[rule_name] = func(
                X_min.loc[cols].to_numpy(), X_max.loc[cols].to_numpy()
            )
        if non_opt_rules:
            warnings.warn(
                f'Rules `{"`, `".join(non_opt_rules)}` have no optimisable conditions - unable to calculate `{param}` for these rules')
        if missing_feat_rules:
            warnings.warn(
                f'Rules `{"`, `".join(missing_feat_rules)}` use features that are missing from `X` - unable to calculate `{param}` for these rules')
        return results
