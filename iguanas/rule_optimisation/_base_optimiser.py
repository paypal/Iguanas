"""
Base rule optimiser class. Main rule optimisers classes inherit from this one.
"""
from iguanas.rules import Rules
import iguanas.utils as utils
from iguanas.utils.typing import PandasDataFrameType, PandasSeriesType
from iguanas.utils.types import NumpyArray, PandasDataFrame, PandasSeries
from typing import Callable, Dict, List, Set, Tuple
import pandas as pd
import numpy as np
import warnings
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns


class _BaseOptimiser(Rules):
    """
    Base rule optimiser class. Main rule optimiser classes inherit from this 
    one.

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
    """

    def __init__(self, rule_lambdas: Dict[str, Callable[[Dict], str]],
                 lambda_kwargs: Dict[str, Dict[str, float]],
                 metric: Callable):
        Rules.__init__(self, rule_strings={})
        self.orig_rule_lambdas = rule_lambdas
        self.orig_lambda_kwargs = lambda_kwargs
        self.metric = metric

    def fit_transform(self,
                      X: PandasDataFrameType,
                      y=None,
                      sample_weight=None) -> PandasDataFrameType:
        """
        Same as `.fit()` method - ensures rule optimiser conforms to 
        fit/transform methodology.        

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

        return self.fit(X=X, y=y, sample_weight=sample_weight)

    @classmethod
    def plot_performance_uplift(self, orig_rule_performances: Dict[str, float],
                                opt_rule_performances: Dict[str, float],
                                figsize=(20, 10)) -> sns.scatterplot:
        """
        Generates a scatterplot showing the performance of each rule before
        and after optimisation.

        Parameters
        ----------
        orig_rule_performances : Dict[str, float]
            The performance metric of each rule prior to optimisation.
        opt_rule_performances : Dict[str, float]
            The performance metric of each rule after optimisation.
        figsize : tuple, optional
            The width and height of the scatterplot. Defaults to (20, 10).

        Returns
        -------
        sns.scatterplot
            Compares the performance of each rule before and after optimisation.
        """
        performance_comp, _ = self._calculate_performance_comparison(
            orig_rule_performances=orig_rule_performances,
            opt_rule_performances=opt_rule_performances
        )
        sns.set_style("whitegrid")
        plt.figure(figsize=figsize)
        sns.scatterplot(x=list(performance_comp.index),
                        y=performance_comp['OriginalRule'], color='blue', label='Original rule')
        sns.scatterplot(x=list(performance_comp.index),
                        y=performance_comp['OptimisedRule'], color='red', label='Optimised rule')
        plt.title(
            'Performance comparison of original rules vs optimised rules')
        plt.xticks(rotation=90)
        plt.ylabel('Performance (of the provided optimisation metric)')
        plt.show()

    @classmethod
    def plot_performance_uplift_distribution(self,
                                             orig_rule_performances: Dict[str, float],
                                             opt_rule_performances: Dict[str, float],
                                             figsize=(8, 10)) -> sns.boxplot:
        """
        Generates a boxplot showing the distribution of performance uplifts
        (original rules vs optimised rules).

        Parameters
        ----------
        orig_rule_performances : Dict[str, float]
            The performance metric of each rule prior to optimisation.
        opt_rule_performances : Dict[str, float]
            The performance metric of each rule after optimisation.
        figsize : tuple, optional
            The width and height of the boxplot. Defaults to (20, 10).

        Returns
        -------
        sns.boxplot
            Shows the distribution of performance uplifts (original rules vs optimised rules).
        """

        _, performance_difference = self._calculate_performance_comparison(
            orig_rule_performances=orig_rule_performances,
            opt_rule_performances=opt_rule_performances
        )
        sns.set_style("whitegrid")
        plt.figure(figsize=figsize)
        sns.boxplot(y=performance_difference)
        plt.title(
            'Distribution of performance uplift, original rules vs optimised rules')
        plt.xticks(rotation=90)
        plt.ylabel(
            'Performance uplift (of the provided optimisation metric)')
        plt.show()

    def _prepare_rules_for_opt(self,
                               X: PandasDataFrameType,
                               y: PandasSeriesType,
                               sample_weight: PandasSeriesType) -> Tuple[
                                   PandasSeriesType,
                                   PandasSeriesType,
                                   PandasDataFrameType]:
        """
        Performs the following before rule optimisation can take place:

            1. Checks if any rules contain features missing in `X` - if so,
            these rules are dropped.
            2. Checks for rules that exclusively contain non-optimisable 
            conditions - if so, these rules are not optimised (but are added
            to the final rule set).
            3. Checks for rules that exclusively contain zero variance features 
            - if so, these rules are not optimised (but are added to the final
            rule set).
            4. Creates the `Rules` object `optimisable_rules` - these are the
            rules that are used in the optimisation process.            
        """

        utils.check_allowed_types(X, 'X', [PandasDataFrame])
        if y is not None:
            utils.check_allowed_types(y, 'y', [PandasSeries])
        if sample_weight is not None:
            utils.check_allowed_types(
                sample_weight, 'sample_weight', [PandasSeries])
        utils.check_duplicate_cols(X, 'X')
        self.orig_rules = Rules(
            rule_lambdas=self.orig_rule_lambdas.copy(),
            lambda_kwargs=self.orig_lambda_kwargs.copy(),
        )
        _ = self.orig_rules.as_rule_strings(as_numpy=False)
        if self.verbose > 0:
            print(
                '--- Checking for rules with features that are missing in `X` ---'
            )
        self.rule_names_missing_features, rule_features_in_X = self._return_rules_missing_features(
            rules=self.orig_rules,
            columns=X.columns,
            verbose=self.verbose
        )
        # If there are rules with missing features in `X`, drop these rules
        if self.rule_names_missing_features:
            self.orig_rules.filter_rules(
                exclude=self.rule_names_missing_features
            )
        # Filter `X` to rule features
        X = X[rule_features_in_X]
        if self.verbose > 0:
            print(
                '--- Checking for rules that exclusively contain non-optimisable conditions ---'
            )
        # Return rules with no optimisable conditions (e.g. categorical)
        self.rule_names_no_opt_conditions = self._return_all_optimisable_rule_features(
            lambda_kwargs=self.orig_rules.lambda_kwargs,
            verbose=self.verbose
        )
        # Get set of features (values) for each rule (keys)
        rule_features = self.orig_rules.get_rule_features()
        # Get set of features used in whole rule set
        rule_features_set = set().union(*self.orig_rules.get_rule_features().values())
        # Get min, max of `X`
        X_min, X_max = self._return_X_min_max(
            X=X,
            cols=rule_features_set
        )
        if self.verbose > 0:
            print(
                '--- Checking for rules that exclusively contain zero-variance features ---'
            )
        # Return rules with exclusively zero variance features
        self.rule_names_zero_var_features = self._return_rules_with_zero_var_features(
            rule_features=rule_features,
            rule_names=list(self.orig_rules.rule_lambdas.keys()),
            X_min=X_min,
            X_max=X_max,
            rule_names_no_opt_conditions=self.rule_names_no_opt_conditions,
            verbose=self.verbose
        )
        # Generate optimisable, non-optimisable and zero-variance rule sets
        self.optimisable_rules, self.non_optimisable_rules, self.zero_variance_rules = self._return_optimisable_rules(
            rules=self.orig_rules,
            rule_names_no_opt_conditions=self.rule_names_no_opt_conditions,
            rule_names_zero_var_features=self.rule_names_zero_var_features
        )
        if not self.optimisable_rules.rule_lambdas:
            raise Exception('There are no optimisable rules in the set')
        # Get performance of original, optimisable rules
        orig_X_rules = self.optimisable_rules.transform(X=X)
        self.orig_rule_performances = dict(
            zip(
                orig_X_rules.columns.tolist(),
                self.metric(orig_X_rules, y, sample_weight)
            )
        )
        if self.verbose > 0:
            print('--- Optimising rules ---')
        return X_min, X_max, orig_X_rules

    def _return_final_rule_set(self,
                               X: PandasDataFrameType,
                               y: PandasSeriesType,
                               sample_weight: PandasSeriesType,
                               opt_rule_strings: Dict[str, str],
                               orig_X_rules: PandasDataFrameType) -> PandasDataFrameType:
        """
        Performs the following before generating the final rule set:

            1. Calculates the performance of the optimised rules.
            2. Compares the performance of the optimised rules to the original
            rules - if the original rule is better performing, it's added to 
            the final rule set; else the optimised rule is added.
            3. Any rules that exclusively contain non-optimisable conditions
            are added to the final rule set.            
        """

        # Get performance of optimised rules
        opt_rules = Rules(rule_strings=opt_rule_strings)
        opt_X_rules = opt_rules.transform(X=X)
        self.opt_rule_performances = dict(
            zip(
                opt_X_rules.columns.tolist(),
                self.metric(opt_X_rules, y, sample_weight)
            )
        )
        # Compare original to optimised rules and return original if better
        # performing
        opt_rule_strings, self.opt_rule_performances, X_rules = self._return_orig_rule_if_better_perf(
            orig_rule_performances=self.orig_rule_performances,
            opt_rule_performances=self.opt_rule_performances,
            orig_rule_strings=self.optimisable_rules.rule_strings,
            opt_rule_strings=opt_rules.rule_strings,
            orig_X_rules=orig_X_rules,
            opt_X_rules=opt_X_rules
        )
        # Combine optimised rules with non-optimised rules (so both can be
        # applied)
        self.rule_strings = {
            **opt_rule_strings, **self.non_optimisable_rules.rule_strings
        }
        # If non-optimisable rules present, apply and combine with `X_rules`
        # (this reduces runtime by not applying the full rule set again)
        if self.non_optimisable_rules.rule_strings:
            X_rules = pd.concat(
                [X_rules, self.non_optimisable_rules.transform(X)], axis=1
            )
        self._generate_other_rule_formats()
        return X_rules

    @staticmethod
    def _calculate_performance_comparison(orig_rule_performances: Dict[str, float],
                                          opt_rule_performances: Dict[str, float]) -> Tuple[PandasDataFrameType, PandasSeriesType]:
        """
        Generates two dataframe - one showing the performance of the original 
        rules and the optimised rules, the other showing the difference in 
        performance per rule.
        """

        performance_comp = pd.concat([pd.Series(
            orig_rule_performances), pd.Series(opt_rule_performances)], axis=1)
        performance_comp.columns = ['OriginalRule', 'OptimisedRule']
        performance_difference = performance_comp['OptimisedRule'] - \
            performance_comp['OriginalRule']
        return performance_comp, performance_difference

    @staticmethod
    def _return_X_min_max(X: PandasDataFrameType,
                          cols: List[str]) -> Tuple[PandasSeriesType, PandasSeriesType]:
        """Returns the min and max of columns provided"""

        X_min = X[cols].min()
        X_max = X[cols].max()
        return X_min, X_max

    @staticmethod
    def _return_rules_missing_features(rules: Rules,
                                       columns: List[str],
                                       verbose: int) -> Tuple[List, Set]:
        """
        Returns the names of rules that contain features missing from `X`.
        """

        rule_features = rules.get_rule_features()
        rule_names_missing_features = []
        rule_features_set = set()
        rule_features_items = utils.return_progress_ready_range(
            verbose=verbose, range=rule_features.items())
        for rule_name, feature_set in rule_features_items:
            missing_features = [
                feature for feature in feature_set if feature not in columns]
            [rule_features_set.add(feature)
             for feature in feature_set if feature in columns]
            if missing_features:
                rule_names_missing_features.append(rule_name)
        if rule_names_missing_features:
            warnings.warn(
                f'Rules `{"`, `".join(rule_names_missing_features)}` use features that are missing from `X` - unable to optimise or apply these rules')
        return rule_names_missing_features, rule_features_set

    @staticmethod
    def _return_all_optimisable_rule_features(lambda_kwargs: Dict[str, Dict[str, float]],
                                              verbose: int) -> Tuple[List[str], List[str]]:
        """
        Returns a list of all of the features used in each optimisable rule
        within the set.
        """
        rule_names_no_opt_conditions = []
        lambda_kwargs_items = utils.return_progress_ready_range(
            verbose=verbose, range=lambda_kwargs.items()
        )
        for rule_name, lambda_kwarg in lambda_kwargs_items:
            if lambda_kwarg == {}:
                rule_names_no_opt_conditions.append(rule_name)
        if rule_names_no_opt_conditions:
            warnings.warn(
                f'Rules `{"`, `".join(rule_names_no_opt_conditions)}` have no optimisable conditions - unable to optimise these rules'
            )
        return rule_names_no_opt_conditions

    @staticmethod
    def _return_rules_with_zero_var_features(rule_features: List[str],
                                             rule_names: List[str],
                                             X_min: Dict[str, float],
                                             X_max: Dict[str, float],
                                             rule_names_no_opt_conditions: List[str],
                                             verbose: int) -> List[str]:
        """
        Returns list of rule names that have all zero variance features, so
        cannot be optimised.
        """

        # Get zero var features (including np.nan)
        zero_var_features = X_min.index[
            X_min.replace(np.nan, 'np.nan') == X_max.replace(np.nan, 'np.nan')
        ].tolist()
        # Get rules that exclusively contain zero var features
        rule_names_all_zero_var = []
        rule_names = utils.return_progress_ready_range(
            verbose=verbose, range=rule_names
        )
        for rule_name in rule_names:
            # If rule has no optimisable conditions, skip
            if rule_name in rule_names_no_opt_conditions:
                continue
            rule_is_all_zero_var = all(
                [rule_feature in zero_var_features for rule_feature in rule_features[rule_name]]
            )
            # If all rule features are zero var, add rule to rule_names_all_zero_var
            if rule_is_all_zero_var:
                rule_names_all_zero_var.append(rule_name)
        if rule_names_all_zero_var:
            warnings.warn(
                f'Rules `{"`, `".join(rule_names_all_zero_var)}` have all zero variance features based on the dataset `X` - unable to optimise these rules')
        return rule_names_all_zero_var

    @staticmethod
    def _return_optimisable_rules(rules: Rules,
                                  rule_names_no_opt_conditions: List[str],
                                  rule_names_zero_var_features: List[str]) -> Tuple[Rules, Rules]:
        """
        Copies the Rules class and filters out rules which cannot be 
        optimised from the original Rules class. Then filters to only those
        un-optimisable rules in the copied Rules class, and returns both
        """

        rule_names_to_exclude = rule_names_no_opt_conditions + rule_names_zero_var_features
        non_optimisable_rules = deepcopy(rules)
        zero_variance_rules = deepcopy(rules)
        rules.filter_rules(exclude=rule_names_to_exclude)
        non_optimisable_rules.filter_rules(
            include=rule_names_no_opt_conditions
        )
        zero_variance_rules.filter_rules(
            include=rule_names_zero_var_features
        )
        return rules, non_optimisable_rules, zero_variance_rules

    @staticmethod
    def _return_orig_rule_if_better_perf(orig_rule_performances: Dict[str, float],
                                         opt_rule_performances: Dict[str, float],
                                         orig_rule_strings: Dict[str, str],
                                         opt_rule_strings: Dict[str, str],
                                         orig_X_rules: PandasDataFrameType,
                                         opt_X_rules: PandasDataFrameType) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Overwrites the optimised rule string with the original if the original 
        is better performing. Also update the performance dictionary with the 
        original if this is the case.
        """

        for rule_name in opt_rule_strings.keys():
            if orig_rule_performances[rule_name] >= opt_rule_performances[rule_name]:
                opt_rule_strings[rule_name] = orig_rule_strings[rule_name]
                opt_rule_performances[rule_name] = orig_rule_performances[rule_name]
                opt_X_rules[rule_name] = orig_X_rules[rule_name]
        return opt_rule_strings, opt_rule_performances, opt_X_rules

    def _generate_other_rule_formats(self) -> None:
        """Generates other rule formats from `self.rule_strings`"""

        # Generate rule names
        self.rule_names = list(self.rule_strings.keys())
        # Convert generated rules into lambda format. Set rule_lambdas to an
        # empty dict first, prevents errors when running fit more than once.
        self.rule_lambdas = {}
        self.rule_lambdas = self.as_rule_lambdas(
            as_numpy=False, with_kwargs=True
        )
        # Generate rules object
        self.rules = Rules(
            rule_strings=self.rule_strings, rule_lambdas=self.rule_lambdas,
            lambda_kwargs=self.lambda_kwargs
        )
