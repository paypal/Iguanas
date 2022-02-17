"""
Generates rules using decision trees (applied to Koalas/Spark 
dataframes).
"""
import numpy as np
import iguanas.utils as utils
from iguanas.rule_generation._base_generator import _BaseGenerator
from iguanas.utils.types import KoalasDataFrame, KoalasSeries
from iguanas.utils.typing import KoalasDataFrameType, KoalasSeriesType
from typing import Callable, List, Set, Tuple
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassificationModel
from pyspark.sql import DataFrame
import warnings


class RuleGeneratorDTSpark(_BaseGenerator):
    """
    Generate rules by extracting the highest performing branches from a 
    tree ensemble model trained on a Spark DataFrame.

    Parameters
    ----------
    metric : Callable
        A function/method which calculates the desired performance metric 
        (e.g. Fbeta score).
    n_total_conditions : int
        The maximum number of conditions per generated rule.
    tree_ensemble : pyspark.ml.classification.RandomForestClassifier
        Pyspark tree ensemble classifier object used to generated rules.
    precision_threshold : float, optional
        Precision threshold for the tree/branch to be used to create rules.
        If the overall precision of the tree/branch is less than or equal 
        to this value, it will not be used in rule generation. Note that if 
        `bootstrap` == True in the tree_ensemble class, the precision will
        be based on the bootstrapped sample used to create the tree. 
        Defaults to 0.            
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
        extraction of the rules from the trees.
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
                 tree_ensemble: RandomForestClassifier,
                 precision_threshold=0,
                 target_feat_corr_types=None,
                 verbose=0, rule_name_prefix='RGDT_Rule'):

        _BaseGenerator.__init__(
            self,
            metric=metric,
            target_feat_corr_types=target_feat_corr_types,
            rule_name_prefix=rule_name_prefix,
        )
        self.orig_tree_ensemble = tree_ensemble
        self.orig_tree_ensemble.setMaxDepth(n_total_conditions)
        self.orig_tree_ensemble.setSeed(0)
        self.orig_tree_ensemble.setLabelCol('label_')
        self.precision_threshold = precision_threshold
        self.verbose = verbose
        self.rule_strings = {}
        self.rule_names = []

    def __repr__(self):
        if self.rule_strings:
            return f'RuleGeneratorDTSpark object with {len(self.rule_strings)} rules generated'
        else:
            return f'RuleGeneratorDTSpark(metric={self.metric}, n_total_conditions={self.orig_tree_ensemble.getMaxDepth()}, tree_ensemble={self.orig_tree_ensemble.__repr__().split("_")[0]}, precision_threshold={self.precision_threshold}, target_feat_corr_types={self.target_feat_corr_types})'

    def fit(self, X: KoalasDataFrameType, y: KoalasSeriesType,
            sample_weight=None) -> KoalasDataFrameType:
        """
        Generates rules by extracting the highest performing branches in a tree
        ensemble model.

        Parameters
        ----------
        X : KoalasDataFrameType 
            The feature set used for training the model.
        y : KoalasSeriesType 
            The target column.            
        sample_weight : KoalasSeriesType, optional 
            Record-wise weights to apply. Defaults to None.

        Returns
        -------
        KoalasDataFrameType
            The binary columns of the generated rules. 
        """

        utils.check_allowed_types(X, 'X', [KoalasDataFrame])
        utils.check_allowed_types(y, 'y', [KoalasSeries])
        if sample_weight is not None:
            utils.check_allowed_types(
                sample_weight, 'sample_weight', [KoalasSeries])
        # Copy spark tree ensemble - ensures repeatable results when class
        # not instantiated but fit method run for different inputs
        self.tree_ensemble = self.orig_tree_ensemble.copy()
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
            print('--- Creating Spark DataFrame for training ---')
        spark_df = self._create_train_spark_df(
            X=X, y=y, sample_weight=sample_weight)
        if self.verbose:
            print('--- Training tree ensemble ---')
        if sample_weight is not None:
            self.tree_ensemble.setWeightCol('sample_weight_')
        trained_tree_ensemble = self.tree_ensemble.fit(spark_df)
        if self.verbose:
            print('--- Extracting rules from tree ensemble ---')
        X_rules = self._extract_rules_from_ensemble(
            X=X, y=y, sample_weight=sample_weight,
            tree_ensemble=trained_tree_ensemble, columns_int=columns_int,
            columns_cat=columns_cat,
        )
        self._generate_other_rule_formats()
        return X_rules

    def _extract_rules_from_ensemble(self, X: KoalasDataFrameType, y: KoalasSeriesType,
                                     sample_weight: KoalasSeriesType,
                                     tree_ensemble: RandomForestClassifier,
                                     columns_int: List[str],
                                     columns_cat: List[str]) -> KoalasDataFrameType:
        """
        Method for returning all of the rules from the ensemble tree-based 
        model.
        """

        list_of_rule_string_sets = []
        for i, decision_tree in enumerate(tree_ensemble.trees):
            if decision_tree.depth == 0:
                warnings.warn(
                    f'Decision Tree {i} has a depth of zero - skipping')
                continue
            dt_rule_strings_set = self._extract_rules_from_dt(
                columns=X.columns.tolist(), decision_tree=decision_tree,
                columns_cat=columns_cat, columns_int=columns_int
            )
            list_of_rule_string_sets.append(dt_rule_strings_set)
        rule_strings_set = sorted(set().union(*list_of_rule_string_sets))
        self.rule_strings = dict(
            (self._generate_rule_name(), rule_string)
            for rule_string in rule_strings_set
        )
        if not self.rule_strings:
            raise Exception(
                'No rules could be generated. Try changing the class parameters.')
        X_rules = self.transform(X=X)
        return X_rules

    def _extract_rules_from_dt(self, columns: List[str],
                               decision_tree: DecisionTreeClassificationModel,
                               columns_int: List[str],
                               columns_cat: List[str]) -> Set[str]:
        """
        Removes low precision DTs and returns the rules from the DT.
        """

        left, right, features, thresholds, precisions, tree_prec = self._get_pyspark_tree_structure(
            decision_tree._call_java('rootNode')
        )
        if tree_prec <= self.precision_threshold:
            return set()
        else:
            return self._extract_rules_from_tree(
                columns=columns, precision_threshold=self.precision_threshold,
                columns_int=columns_int, columns_cat=columns_cat,
                left=left, right=right, features=features,
                thresholds=thresholds, precisions=precisions
            )

    @staticmethod
    def _create_train_spark_df(X: KoalasDataFrameType, y: KoalasSeriesType,
                               sample_weight: KoalasSeriesType) -> DataFrame:
        """
        Creates a Spark DataFrame from `X`, `y` and `sample_weight` (if 
        provided) for training the pyspark tree ensemble.
        """

        spark_df = utils.create_spark_df(
            X=X, y=y, sample_weight=sample_weight)
        vectorAssembler = VectorAssembler(
            inputCols=X.columns.tolist(), outputCol="features")
        spark_df = vectorAssembler.transform(spark_df)
        return spark_df

    @staticmethod
    def _get_pyspark_tree_structure(node) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Extracts the left_children, right_children, features, thresholds, 
        node precisions and overall precision of the given DT (equivalent to 
        those attributes seen in Sklearn's DecisionTreeClassifier class)
        """

        def _get_node_info(node):
            """
            Appends info to the left_children, right_children, features, 
            thresholds, node precisions and node predictions arrays for each 
            node
            """

            network[k[0]-1] = {}
            k[0] = k[0] + 1
            num_desc_left = node.leftChild().numDescendants()
            num_desc_right = node.rightChild().numDescendants()
            features.append(node.split().featureIndex())
            thresholds.append(node.split().threshold())
            node_splits.append(list(node.impurityStats().stats()))
            if num_desc_left == 0:
                network[k[0]-2]['Left'] = k[0]-1
                t = k[0]-2
                k[0] = k[0] + 1
                left_children.append(-1)
                node_splits.append(
                    list(node.leftChild().impurityStats().stats()))
                features.append(-2)
                thresholds.append(-2)
                node_preds.append(node.leftChild().prediction())
            else:
                network[k[0]-2]['Left'] = k[0]-1
                t = k[0]-2
                left_children.append(k[0])
                node_preds.append(None)
                _get_node_info(node.leftChild())
            if num_desc_right == 0:
                network[t]['Right'] = k[0]-1
                k[0] = k[0] + 1
                left_children.append(-1)
                node_splits.append(
                    list(node.rightChild().impurityStats().stats()))
                features.append(-2)
                thresholds.append(-2)
                node_preds.append(node.rightChild().prediction())
            else:
                network[t]['Right'] = k[0]-1
                left_children.append(k[0])
                node_preds.append(None)
                _get_node_info(node.rightChild())
        k = [1]
        left_children = [k[0]]
        features = []
        thresholds = []
        node_preds = [None]
        node_splits = []
        node_precs = []
        network = {}
        _get_node_info(node)
        left_children, features, thresholds = np.array(
            left_children), np.array(features), np.array(thresholds)
        right_children = np.array([network[i]['Right'] if i in network.keys(
        ) else -1 for i in range(len(left_children))])
        node_precs = np.empty(len(node_splits))
        tps_l, tps_fps_l = [], []
        for i, lc in enumerate(left_children):
            node_precs[i] = node_splits[i][1]/sum(node_splits[i])
            if lc == -1 and node_preds[i] == 1:
                tps_l.append(node_splits[i][1])
                tps_fps_l.append(sum(node_splits[i]))
        tps = sum(tps_l)
        tps_fps = sum(tps_fps_l)
        tree_prec = 0 if tps_fps == 0 else tps/tps_fps
        return left_children, right_children, features, thresholds, node_precs, tree_prec
