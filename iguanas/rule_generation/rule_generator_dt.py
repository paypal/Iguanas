"""Generates rules using decision trees."""
from joblib import Parallel, delayed
import numpy as np
import iguanas.utils as utils
from iguanas.rule_generation._base_generator import _BaseGenerator
from iguanas.utils.types import PandasDataFrame, PandasSeries
from iguanas.utils.typing import PandasSeriesType, PandasDataFrameType
from iguanas.exceptions import NoRulesError
from typing import Union, Callable, List, Set, Tuple
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import sys


class RuleGeneratorDT(_BaseGenerator):
    """
    Generate rules by extracting the highest performing branches from a
    tree ensemble model.

    Parameters
    ----------
    metric : Callable
        A function/method which calculates the desired performance metric
        (e.g. Fbeta score).
    n_total_conditions : int
        The maximum number of conditions per generated rule.
    tree_ensemble : Union[RandomForestClassifier, ExtraTreesClassifier]
        Instantiated Sklearn tree ensemble classifier object used to generated
        rules.
    precision_threshold : float, optional
        Precision threshold for the tree/branch to be used to create rules.
        If the overall precision of the tree/branch is less than or equal
        to this value, it will not be used in rule generation. Note that if
        `bootstrap` == True in the tree_ensemble class, the precision will
        be based on the bootstrapped sample used to create the tree.
        Defaults to 0.
    num_cores : int, optional
        The number of cores to use when iterating through the ensemble to
        generate rules. Defaults to 1.
    target_feat_corr_types : Union[Dict[str, List[str]], str], optional
        Limits the conditions of the rules based on the target-feature
        correlation (e.g. if a feature has a positive correlation with
        respect to the target, then only greater than operators are used
        for conditions that utilise that feature). Can be either a
        dictionary specifying the list of positively correlated features
        wrt the target (under the key `PositiveCorr`) and negatively
        correlated features wrt the target (under the key `NegativeCorr`),
        or 'Infer' (where each target-feature correlation type is inferred
        from the data). Defaults to None.
    verbose : int, optional
        Controls the verbosity - the higher, the more messages. >0 : gives
        the overall progress of the training of the ensemble model and the
        extraction of the rules from the trees. Defaults to 0.
    rule_name_prefix : str, optional
        Prefix to use for each rule. Defaults to 'RGDT_Rule'.

    Attributes
    ----------
    rule_strings : Dict[str, str]
        The generated rules, defined using the standard Iguanas string 
        format (values) and their names (keys).   
    rule_lambdas : Dict[str, object]
        The generated rules, defined using the standard Iguanas lambda 
        expression format (values) and their names (keys).   
    lambda_kwargs : Dict[str, object]
        The keyword arguments for the generated rules defined using the 
        standard Iguanas lambda expression format.
    rules : Rules
        The Rules object containing the generated rules.
    rule_names : List[str]
        The names of the generated rules.
    """

    def __init__(self, metric: Callable,
                 n_total_conditions: int,
                 tree_ensemble: Union[RandomForestClassifier, ExtraTreesClassifier],
                 precision_threshold=0,
                 num_cores=1,
                 target_feat_corr_types=None,
                 verbose=0,
                 rule_name_prefix='RGDT_Rule'):

        _BaseGenerator.__init__(
            self,
            metric=metric,
            target_feat_corr_types=target_feat_corr_types,
            rule_name_prefix=rule_name_prefix,
        )
        self.tree_ensemble = tree_ensemble
        self.n_total_conditions = n_total_conditions
        self.precision_threshold = precision_threshold
        self.num_cores = num_cores
        self.verbose = verbose
        self.rule_strings = {}
        self.rule_names = []

    def __repr__(self):
        if self.rule_strings:
            return f'RuleGeneratorDT object with {len(self.rule_strings)} rules generated'
        else:
            return f'RuleGeneratorDT(metric={self.metric}, n_total_conditions={self.n_total_conditions}, tree_ensemble={self.tree_ensemble}, precision_threshold={self.precision_threshold}, num_cores={self.num_cores}, target_feat_corr_types={self.target_feat_corr_types})'

    def fit(self, X: PandasDataFrameType, y: PandasSeriesType,
            sample_weight=None) -> PandasDataFrameType:
        """
        Generates rules by extracting the highest performing branches in a tree
        ensemble model.

        Parameters
        ----------
        X : PandasDataFrameType
            The feature set used for training the model.
        y : PandasSeriesType
            The target column.
        sample_weight : PandasSeriesType, optional
            Record-wise weights to apply. Defaults to None.

        Returns
        -------
        PandasDataFrameType
            The binary columns of the generated rules.
        """

        utils.check_allowed_types(X, 'X', [PandasDataFrame])
        utils.check_allowed_types(y, 'y', [PandasSeries])
        if sample_weight is not None:
            utils.check_allowed_types(
                sample_weight, 'sample_weight', [PandasSeries])
        # Ensures rule names are the same when fit run without reinstantiating
        self._rule_name_counter = 0
        # Ensures max_depth updates if n_total_conditions changed in
        # already instantiated class
        self.tree_ensemble.max_depth = self.n_total_conditions
        self.tree_ensemble.random_state = 0
        if self.target_feat_corr_types == 'Infer':
            if self.verbose:
                print(
                    '--- Calculating correlation of features with respect to the target ---')
            self.target_feat_corr_types = self._calc_target_ratio_wrt_features(
                X=X, y=y
            )
        if self.verbose:
            print('--- Returning column datatypes ---')
        columns_int, columns_cat, _ = utils.return_columns_types(X)
        if self.verbose:
            print('--- Training tree ensemble ---')
        trained_tree_ensemble = self._train_ensemble(
            X=X, y=y, tree_ensemble=self.tree_ensemble,
            sample_weight=sample_weight, verbose=self.verbose)
        if self.verbose:
            print('--- Extracting rules from tree ensemble ---')
        X_rules = self._extract_rules_from_ensemble(
            X=X,
            y=y,
            num_cores=self.num_cores,
            tree_ensemble=trained_tree_ensemble,
            columns_int=columns_int,
            columns_cat=columns_cat,
            sample_weight=sample_weight
        )
        self._generate_other_rule_formats()
        return X_rules

    def _extract_rules_from_ensemble(self, X: PandasDataFrameType, y: PandasSeriesType,
                                     tree_ensemble: Union[RandomForestClassifier, ExtraTreesClassifier],
                                     num_cores: int,
                                     sample_weight: PandasSeriesType,
                                     columns_int: List[str],
                                     columns_cat: List[str]) -> PandasDataFrameType:
        """
        Method for returning all of the rules from the ensemble tree-based
        model.
        """

        decision_trees = utils.return_progress_ready_range(
            verbose=self.verbose, range=tree_ensemble.estimators_)
        with Parallel(n_jobs=num_cores) as parallel:
            list_of_rule_string_sets = parallel(delayed(self._extract_rules_from_dt)(
                X.columns.tolist(), decision_tree, columns_int, columns_cat
            ) for decision_tree in decision_trees
            )
        rule_strings_set = sorted(set().union(*list_of_rule_string_sets))
        self.rule_strings = dict(
            (self._generate_rule_name(), rule_string)
            for rule_string in rule_strings_set
        )
        if not self.rule_strings:
            raise NoRulesError(
                'No rules could be generated. Try changing the class parameters.')
        X_rules = self.transform(X=X)
        return X_rules

    def _extract_rules_from_dt(self, columns: List[str],
                               decision_tree: DecisionTreeClassifier,
                               columns_int: List[str], columns_cat: List[str]) -> Set[str]:
        """
        Removes low precision DTs and returns the rules from the DT.
        """

        left, right, features, thresholds, precisions, tree_prec = self._get_dt_attributes(
            decision_tree
        )
        if tree_prec <= self.precision_threshold:
            return set()
        else:
            return self._extract_rules_from_tree(
                columns=columns, precision_threshold=self.precision_threshold,
                columns_int=columns_int,
                columns_cat=columns_cat, left=left, right=right,
                features=features, thresholds=thresholds, precisions=precisions
            )

    @staticmethod
    def _train_ensemble(X: PandasDataFrameType, y: PandasSeriesType,
                        tree_ensemble: Union[RandomForestClassifier,
                                             ExtraTreesClassifier],
                        sample_weight: PandasSeriesType,
                        verbose: int) -> Union[RandomForestClassifier,
                                               ExtraTreesClassifier]:
        """Method for running ML model"""

        def _switch_stderr_stdout(verbose: int):
            """Switches stderr and stdout, if verbose > 0"""
            if verbose > 0:
                sys.stdout, sys.stderr = sys.stderr, sys.stdout

        _switch_stderr_stdout(verbose)
        tree_ensemble.fit(X=X, y=y, sample_weight=sample_weight)
        _switch_stderr_stdout(verbose)
        return tree_ensemble

    @staticmethod
    def _get_dt_attributes(decision_tree: DecisionTreeClassifier) -> Tuple[
            np.ndarray]:
        """Returns the attributes associated with a given DT"""

        left = decision_tree.tree_.children_left
        right = decision_tree.tree_.children_right
        thresholds = decision_tree.tree_.threshold
        node_splits = decision_tree.tree_.value
        features = decision_tree.tree_.feature
        node_precs = np.empty(len(node_splits))
        tps_l, tps_fps_l = [], []
        for i, node_split in enumerate(node_splits):
            node_precision = node_split[0][1]/np.sum(node_split[0])
            node_pred = np.argmax(node_split[0])
            node_precs[i] = node_precision
            if left[i] == -1 and node_pred == 1:
                tps_l.append(node_splits[i][0][1])
                tps_fps_l.append(sum(node_splits[i][0]))
        tps = sum(tps_l)
        tps_fps = sum(tps_fps_l)
        tree_prec = 0 if tps_fps == 0 else tps/tps_fps
        return left, right, features, thresholds, node_precs, tree_prec
