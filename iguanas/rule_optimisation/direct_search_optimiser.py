"""Optimises a set of rules using Direct Search algorithms."""
from iguanas.rule_optimisation._base_optimiser import _BaseOptimiser
import iguanas.utils as utils
from iguanas.utils.types import NumpyArray, PandasDataFrame, PandasSeries
from iguanas.utils.typing import PandasDataFrameType, PandasSeriesType
from iguanas.warnings import RulesNotOptimisedWarning
import pandas as pd
from typing import Callable, Dict, List, Union, Tuple
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
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
    num_cores : int, optional
        The number of cores to use when optimising the rule thresholds.
        Defaults to 1.
    verbose : int, optional
        Controls the verbosity - the higher, the more messages. >0 : shows
        the overall progress of the optimisation process. Defaults to 0.

    Attributes
    ----------
    rule_strings : Dict[str, str]
        The optimised + unoptimisable (but applicable) rules, defined using the
        standard Iguanas string format (values) and their names (keys).
    rule_lambdas : Dict[str, object]
        The optimised rules + unoptimisable (but applicable), defined using the
        standard Iguanas lambda expression format (values) and their names
        (keys).
    lambda_kwargs : Dict[str, object]
        The keyword arguments for the optimised + unoptimisable (but
        applicable) rules defined using the standard Iguanas lambda expression
        format.
    rules : Rules
        The Rules object containing the optimised + unoptimisable (but
        applicable) rules.
    rule_names : List[str]
        The names of the optimised + unoptimisable (but applicable) rules.
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
        A `Rules` object containing the rules which contained exclusively
        non-optimisable conditions.
    zero_varaince_rules : Rules
        A `Rules` object containing the rules which contained exclusively zero
        variance features.

    Examples
    --------
    >>> from iguanas.rule_optimisation import DirectSearchOptimiser
    >>> from iguanas.rules import Rules
    >>> from iguanas.metrics import FScore
    >>> import pandas as pd
    >>> X = pd.DataFrame({
    ...     'A': [0.9, 0.2, 0.1, 0.3],
    ...     'B': [0.01, 0.2, 0.5, 0.1]
    ... })
    >>> y = pd.Series([
    ...     1, 0, 1, 0
    ... ])
    >>> f1 = FScore(beta=1)
    >>> rule_strings = {
    ...     'Rule1': "X['A']>0",
    ...     'Rule2': "X['B']>1",
    ...     'Rule3': "(X['A']>0)|(X['B']>0)"
    ... }
    >>> rules = Rules(rule_strings=rule_strings)
    >>> rule_lambdas = rules.as_rule_lambdas(
    ...     as_numpy=False, 
    ...     with_kwargs=True
    ... )
    >>> ds = DirectSearchOptimiser(
    ...     rule_lambdas=rule_lambdas, 
    ...     lambda_kwargs=rules.lambda_kwargs, 
    ...     metric=f1.fit, 
    ...     method='Nelder-Mead'
    ... )
    >>> x0 = ds.create_x0(X=X, lambda_kwargs=rules.lambda_kwargs)
    >>> ds.x0 = x0
    >>> X_rules = ds.fit(X=X, y=y)
    >>> print(X_rules)
       Rule1  Rule2  Rule3
    0      1      0      1
    1      1      0      0
    2      1      1      1
    3      1      0      0
    >>> print(ds.rule_strings)
    {'Rule1': "(X['A']>0)", 'Rule2': "(X['B']>0.255)", 'Rule3': "(X['A']>0.5)|(X['B']>0.255)"}
    >>> print(ds.opt_rule_performances)
    {'Rule1': 0.6666666666666666, 'Rule2': 0.6666666666666666, 'Rule3': 1.0}
    >>> X_rules = ds.transform(X=X)
    >>> print(X_rules)
       Rule1  Rule2  Rule3
    0      1      0      1
    1      1      0      0
    2      1      1      1
    3      1      0      0
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
                 num_cores=1,
                 verbose=0):

        _BaseOptimiser.__init__(
            self, rule_lambdas=rule_lambdas, lambda_kwargs=lambda_kwargs,
            metric=metric, num_cores=num_cores, verbose=verbose
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
        self.rule_strings = {}
        self.rule_names = []

    def __repr__(self):
        if self.rule_strings == {}:
            return f'DirectSearchOptimiser object with {len(self.orig_rule_lambdas)} rules to optimise'
        else:
            return f'DirectSearchOptimiser object with {len(self.optimisable_rules.rule_strings)} optimised rules and {len(self.non_optimisable_rules.rule_strings)} unoptimisable rules'

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
            The binary columns of the optimised + unoptimisable (but
            applicable) rules on the fitted dataset.
        """

        _, _, orig_X_rules = self._prepare_rules_for_opt(
            X=X,
            y=y,
            sample_weight=sample_weight
        )
        opt_rule_strings = self._optimise_rules(
            rule_lambdas=self.optimisable_rules.rule_lambdas,
            lambda_kwargs=self.optimisable_rules.lambda_kwargs,
            X=X,
            y=y,
            sample_weight=sample_weight
        )
        X_rules = self._return_final_rule_set(
            X=X,
            y=y,
            sample_weight=sample_weight,
            opt_rule_strings=opt_rule_strings,
            orig_X_rules=orig_X_rules
        )
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
            func=lambda X_min, X_max: ((X_max+X_min)/2).astype(float)
        )
        return x0

    @ classmethod
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
                feature_vertices = np.random.uniform(
                    feature_min, feature_max +
                    ((feature_max-feature_min)/1000), num_features+1
                )
                initial_simplex[i] = feature_vertices
            initial_simplex = initial_simplex.T
            return initial_simplex

        if shape not in ["Origin-based", "Minimum-based", "Random-based"]:
            raise ValueError(
                '`shape` must be either "Origin-based", "Minimum-based" or "Random-based"'
            )
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

        rule_lambdas_items = utils.return_progress_ready_range(
            verbose=self.verbose == 1, range=rule_lambdas.items()
        )
        with Parallel(n_jobs=self.num_cores) as parallel:
            opt_rule_strings_list = parallel(delayed(self._optimise_single_rule)(
                rule_name, rule_lambda, lambda_kwargs, X, y, sample_weight
            ) for rule_name, rule_lambda in rule_lambdas_items
            )
        opt_rule_strings = dict(opt_rule_strings_list)
        return opt_rule_strings

    def _optimise_single_rule(self,
                              rule_name: str,
                              rule_lambda: object,
                              lambda_kwargs: Dict[str, Dict[str, float]],
                              X: PandasDataFrameType,
                              y: PandasSeriesType,
                              sample_weight: PandasSeriesType) -> Tuple[str, str]:
        """Optimises a single rule"""

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

        minimize_kwargs = self._return_kwargs_for_minimize(rule_name=rule_name)
        rule_features = list(lambda_kwargs[rule_name].keys())
        opt_val = minimize(
            fun=_objective,
            args=(rule_lambda, rule_features, X, y, sample_weight),
            method=self.method, **minimize_kwargs
        )
        lambda_kwargs_opt = dict(zip(rule_features, opt_val.x))
        return rule_name, rule_lambda(**lambda_kwargs_opt)

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
                f'`{param_name}` must be a dictionary with each element aligning with a rule.'
            )

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
            # Get min/max of features in rule, convert to np array
            X_min_rule_feats = X_min.loc[cols].to_numpy(dtype=np.number)
            X_max_rule_feats = X_max.loc[cols].to_numpy(dtype=np.number)
            # If nans, convert to 0
            X_min_rule_feats = np.where(
                np.isnan(X_min_rule_feats), 0, X_min_rule_feats
            )
            X_max_rule_feats = np.where(
                np.isnan(X_max_rule_feats), 0, X_max_rule_feats
            )
            # Apply function
            results[rule_name] = func(
                X_min_rule_feats,
                X_max_rule_feats
            )
        if non_opt_rules:
            warnings.warn(
                f'Rules `{"`, `".join(non_opt_rules)}` have no optimisable conditions - unable to calculate `{param}` for these rules'
            )
        if missing_feat_rules:
            warnings.warn(
                f'Rules `{"`, `".join(missing_feat_rules)}` use features that are missing from `X` - unable to calculate `{param}` for these rules',
            )
        return results
