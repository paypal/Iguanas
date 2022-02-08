"""
Base rule optimiser class. Main rule optimisers classes inherit from this one.
"""
from typing import Callable, Dict, List, Set, Tuple
import pandas as pd
from iguanas.rules import Rules
import iguanas.utils as utils
from iguanas.utils.typing import PandasDataFrameType, PandasSeriesType
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
            The binary columns of the optimised rules on the fitted dataset.
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
        cols = list(set([col.split('%')[0] for col in cols]))
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
                                              X: PandasDataFrameType,
                                              verbose: int) -> Tuple[List[str], List[str]]:
        """
        Returns a list of all of the features used in each optimisable rule
        within the set.
        """
        X_isna_means = X.isna().mean()
        cols_all_null = X_isna_means[X_isna_means == 1].index.tolist()
        all_rule_features = set()
        rule_names_no_opt_conditions = []
        lambda_kwargs_items = utils.return_progress_ready_range(
            verbose=verbose, range=lambda_kwargs.items()
        )
        for rule_name, lambda_kwarg in lambda_kwargs_items:
            if lambda_kwarg == {}:
                rule_names_no_opt_conditions.append(rule_name)
                continue
            rule_features = list(lambda_kwarg.keys())
            for feature in rule_features:
                if feature.split('%')[0] in cols_all_null:
                    rule_names_no_opt_conditions.append(rule_name)
                    break
                else:
                    all_rule_features.add(feature)
        if rule_names_no_opt_conditions:
            warnings.warn(
                f'Rules `{"`, `".join(rule_names_no_opt_conditions)}` have no optimisable conditions - unable to optimise these rules')
        all_rule_features = list(all_rule_features)
        return all_rule_features, rule_names_no_opt_conditions

    @staticmethod
    def _return_rules_with_zero_var_features(lambda_kwargs: Dict[str, Dict[str, float]],
                                             X_min: Dict[str, float],
                                             X_max: Dict[str, float],
                                             rule_names_no_opt_conditions: List[str],
                                             verbose: int) -> List[str]:
        """
        Returns list of rule names that have all zero variance features,
        so cannot be optimised
        """

        zero_var_cols = X_min.index[X_min == X_max].tolist()
        rule_names_zero_var_features = []
        lambda_kwargs_items = utils.return_progress_ready_range(
            verbose=verbose, range=lambda_kwargs.items()
        )
        for rule_name, rule_lambda_kwargs in lambda_kwargs_items:
            if rule_name in rule_names_no_opt_conditions:
                continue
            rule_features = list(rule_lambda_kwargs.keys())
            if all([rule_feature in zero_var_cols for rule_feature in rule_features]):
                rule_names_zero_var_features.append(rule_name)
                continue
        if rule_names_zero_var_features:
            warnings.warn(
                f'Rules `{"`, `".join(rule_names_zero_var_features)}` have all zero variance features based on the dataset `X` - unable to optimise these rules')
        return rule_names_zero_var_features

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
        rules.filter_rules(exclude=rule_names_to_exclude)
        non_optimisable_rules.filter_rules(
            include=rule_names_to_exclude)
        return rules, non_optimisable_rules

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
