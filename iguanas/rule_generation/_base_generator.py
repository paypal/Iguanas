"""
Base rule generator class. Main rule generator classes inherit from this one.
"""
from typing import List, Set, Tuple, Dict, Union
import numpy as np
import math
from datetime import date
from iguanas.rules.rules import Rules
import iguanas.utils as utils
from iguanas.utils.types import KoalasSeries, KoalasDataFrame
from iguanas.utils.typing import KoalasDataFrameType, KoalasSeriesType,\
    PandasDataFrameType, PandasSeriesType


class _BaseGenerator(Rules):

    """
    Base rule generator class. Main rule generator classes inherit from this 
    one.

    Parameters
    ----------
    metric : Callable
        A function/method which calculates the desired performance metric 
        (e.g. Fbeta score).
    target_feat_corr_types : Union[Dict[str, List[str]], str]
        Limits the conditions of the rules based on the target-feature
        correlation (e.g. if a feature has a positive correlation with 
        respect to the target, then only greater than operators are used 
        for conditions that utilise that feature). Can be either a 
        dictionary specifying the list of positively correlated features 
        wrt the target (under the key `PositiveCorr`) and negatively 
        correlated features wrt the target (under the key `NegativeCorr`),
        or 'Infer' (where each target-feature correlation type is inferred 
        from the data). Defaults to None.
    rule_name_prefix : str
        Prefix to use for each rule name. If None, the standard prefix is 
        used. Defaults to None.
    """

    def __init__(self,
                 metric,
                 target_feat_corr_types,
                 rule_name_prefix) -> None:
        self.metric = metric
        self.target_feat_corr_types = target_feat_corr_types
        self.rule_name_prefix = rule_name_prefix
        self._rule_name_counter = 0
        _today = date.today()
        self._today = _today.strftime("%Y%m%d")
        Rules.__init__(self, rule_strings={})

    def fit_transform(self,
                      X: Union[PandasDataFrameType, KoalasDataFrameType],
                      y: Union[PandasSeriesType, KoalasSeriesType],
                      sample_weight=None) -> Union[PandasDataFrameType, KoalasDataFrameType]:
        """
        Same as `.fit()` method - ensures rule generator conforms to 
        fit/transform methodology.        

        Parameters
        ----------
        X : Union[PandasDataFrameType, KoalasDataFrameType]
            The feature set used for training the model.
        y : Union[PandasSeriesType, KoalasSeriesType]
            The target column.            
        sample_weight : Union[PandasSeriesType, KoalasSeriesType], optional
            Record-wise weights to apply. Defaults to None.

        Returns
        -------
        Union[PandasDataFrameType, KoalasDataFrameType]
            The binary columns of the generated rules. 
        """
        return self.fit(X=X, y=y, sample_weight=sample_weight)

    def _extract_rules_from_tree(self, columns: List[str],
                                 precision_threshold: float,
                                 columns_int: List[str],
                                 columns_cat: List[str],
                                 left: np.ndarray, right: np.ndarray,
                                 features: np.ndarray, thresholds: np.ndarray,
                                 precisions: np.ndarray) -> Set[str]:
        """
        Method for returning the rules of all the leaves of the decision 
        tree passed.
        """

        leaf_nodes = np.argwhere(left == -1)[:, 0]

        def recurse_rule(left: np.ndarray, right: np.ndarray, child: int,
                         rule=None) -> Tuple[str, str, float]:
            """
            IDs each leaf node, then iterates through up to the parent, noting 
            the conditions at each node.
            """
            if rule is None:
                rule = []
            if child in left:
                parent = np.where(left == child)[0].item()
                split = '<='
            else:
                parent = np.where(right == child)[0].item()
                split = '>'
            rule.append((columns[features[parent]], split,
                         round(thresholds[parent], 5)))
            if parent == 0:
                rule.reverse()
                return rule
            else:
                return recurse_rule(left, right, parent, rule)

        if not leaf_nodes.any():
            return set()
        rule_strings_set = set()
        for child in leaf_nodes:
            child_precision = precisions[child]
            if child_precision <= precision_threshold:
                continue
            branch_conditions = recurse_rule(left, right, child)
            branch_conditions = self._clean_dup_features_from_conditions(
                branch_conditions)
            if self.target_feat_corr_types is not None:
                branch_conditions = self._remove_misaligned_conditions(
                    branch_conditions=branch_conditions,
                    target_feat_corr_types=self.target_feat_corr_types
                )
                # If no conditions left after removing misaligned conditions,
                # continue to next branch
                if not branch_conditions:
                    continue
            rule_logic = self._convert_conditions_to_string(
                list_of_conditions=branch_conditions, columns_int=columns_int,
                columns_cat=columns_cat)
            rule_strings_set.add(rule_logic)
        return rule_strings_set

    def _generate_rule_name(self) -> str:
        """Generates rule name"""

        if self.rule_name_prefix == 'RGDT_Rule' or self.rule_name_prefix == 'RGO_Rule':
            rule_name = f'{self.rule_name_prefix}_{self._today}_{self._rule_name_counter}'
        else:
            rule_name = f'{self.rule_name_prefix}_{self._rule_name_counter}'
        self._rule_name_counter += 1
        return rule_name

    @staticmethod
    def _convert_conditions_to_string(list_of_conditions: List[Tuple[str, str, float]],
                                      columns_int: List[str],
                                      columns_cat: List[str]) -> str:
        """
        Converts a list of conditions to the standard Iguanas string format.

        Parameters
        ----------
        list_of_conditions : List[Tuple[str, str, float]] 
            Each element contains a tuple of the feature (str), operator (str) 
            and value (numeric) for each condition in the rule.
        columns_int : List[str] 
            List of integer columns.
        columns_cat : List[str] 
            List of OHE categorical columns.

        Returns
        -------
        str
            The Iguanas-readable rule name.
        """

        def convert_values_for_columns_int(feature: str, operator: str,
                                           value: float) -> Tuple[str, str, float]:
            """
            Method for converting a condition containing an integer value from 
            float to int
            """
            if operator in ['>=', '>']:
                return feature, '>=', math.ceil(value)
            elif operator in ['<=', '<']:
                return feature, '<=', math.floor(value)
            else:
                raise ValueError(
                    'Error converting rule conditions - operator should be ">=", ">", "<=", or "<"'
                )

        conditions = []
        for feature, operator, value in list_of_conditions:
            if feature in columns_cat:
                if (operator == '<=' and value < 1) or (operator == '==' and value == 0):
                    condition = f"(X['{feature}']==False)"
                elif (operator == '>' and value >= 0) or (operator == '==' and value == 1):
                    condition = f"(X['{feature}']==True)"
            # If feature is an int, round the value
            elif feature in columns_int:
                feature, operator, value = convert_values_for_columns_int(
                    feature, operator, value)
                condition = f"(X['{feature}']{operator}{value})"
            else:
                condition = f"(X['{feature}']{operator}{value})"
            conditions.append(condition)
        conditions.sort()
        name = '&'.join(conditions)
        return name

    @staticmethod
    def _clean_dup_features_from_conditions(list_of_conditions: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
        """
        Removes unnecessary conditions from a rule (e.g. for a branch in a tree, 
        the same feature and condition can be referenced, the threshold value is 
        different. This method just takes the relevant threshold value).

        Parameters
        ----------
        list_of_conditions (List[Tuple[str, str, float]]): Each element 
            contains a tuple of the feature, operator and value for each 
            condition in the rule.

        Returns
        -------
        List[Tuple[str, str, float]]
            Cleaned list of conditions.
        """

        def dedupe_conditions(feature: str, operator: str) -> Tuple[str, str, float]:
            list_of_values = [
                val for feat, op, val in list_of_conditions if feat == feature and op == operator]
            if operator in ['<', '<=']:
                return feature, operator, min(list_of_values)
            if operator in ['>', '>=']:
                return feature, operator, max(list_of_values)

        unique_feat_op_list = {(feat, op)
                               for feat, op, _ in list_of_conditions}
        list_of_conditions_cleaned = [dedupe_conditions(
            unique_feat, unique_op) for unique_feat, unique_op in unique_feat_op_list]
        list_of_conditions_cleaned.sort()
        return list_of_conditions_cleaned

    @staticmethod
    def _remove_misaligned_conditions(branch_conditions: List[Tuple[str, str, float]],
                                      target_feat_corr_types: Dict[str, List[str]]):
        """
        Removes conditions which contain operators that do not align to
        the type of correlation seen between the feature and the target column
        """

        cleaned_branch_conditions = branch_conditions.copy()
        for feature, operator, value in branch_conditions:
            if (feature in target_feat_corr_types['PositiveCorr'] and operator in ['<', '<=']) or \
                    (feature in target_feat_corr_types['NegativeCorr'] and operator in ['>', '>=']):
                cleaned_branch_conditions.remove((feature, operator, value))
        return cleaned_branch_conditions

    @staticmethod
    def _calc_target_ratio_wrt_features(X: Union[PandasDataFrameType,
                                                 KoalasDataFrameType],
                                        y: Union[PandasSeriesType,
                                                 KoalasSeriesType]) -> Dict[str, List[str]]:
        """
        Calculates the ratio of the target column for the bottom 50% and top 
        50% of each feature in `X`, then assigns the feature to the 
        `PositiveCorr` key in the output dictionary (if higher values are 
        indicative of the target) or to the `NegativeCorr` key (if lower values 
        are indicative of the target).

        Parameters
        ----------
        X : PandasDataFrameType
            Feature set.
        y : PandasSeriesType
            Target.

        Returns
        -------
        Dict[str, List[str]]
            Contains the list of features that are positively correlated to the 
            target (under the key 'PositiveCorr') and the list of features that
            are negatively correlated to the target (under the key 
            'NegativeCorr').
        """

        def _calc_target_ratios_numpy(X: PandasDataFrameType,
                                      y: PandasSeriesType) -> Dict[str, List[str]]:
            """
            Calculates the ratio of the target column for the bottom 50% and top 
            50% of each feature in `X`, using Koalas.
            """

            feature_medians = X.median()
            for feature, median in feature_medians.iteritems():
                if median == X[feature].min():
                    target_ratio_positive = (y[X[feature] > median]).mean()
                    target_ratio_negative = (y[X[feature] <= median]).mean()
                else:
                    target_ratio_positive = (y[X[feature] >= median]).mean()
                    target_ratio_negative = (y[X[feature] < median]).mean()
                if target_ratio_positive >= target_ratio_negative:
                    target_feat_corr_types['PositiveCorr'].append(feature)
                elif target_ratio_positive < target_ratio_negative:
                    target_feat_corr_types['NegativeCorr'].append(feature)
            return target_feat_corr_types

        def _calc_target_ratios_koalas(X: KoalasDataFrameType,
                                       y: KoalasSeriesType) -> Dict[str, List[str]]:
            """
            Calculates the ratio of the target column for the bottom 50% and top 
            50% of each feature in `X`, using Koalas.
            """
            # Import Koalas in function - keeps spark/non-spark functionality
            # separate
            import databricks.koalas as ks

            df = X.join(y.rename('label_'))
            features = X.columns.tolist()
            feature_medians = X.median().to_numpy()
            feature_mins = X.min().to_numpy()
            ratios_sql_list = []
            for feature, f_median, f_min in zip(features, feature_medians, feature_mins):
                if f_median == f_min:
                    pos_op = '>'
                    neg_op = '<='
                else:
                    pos_op = '>='
                    neg_op = '<'
                pos_sql = f'sum(case when label_==1 and {feature} {pos_op} {f_median} then 1 else 0 end)/sum(case when {feature} {pos_op} {f_median} then 1 else 0 end) as target_ratio_positive_{feature},'
                neg_sql = f'sum(case when label_==1 and {feature} {neg_op} {f_median} then 1 else 0 end)/sum(case when {feature} {neg_op} {f_median} then 1 else 0 end) as target_ratio_negative_{feature}'
                ratios_sql = pos_sql + neg_sql
                ratios_sql_list.append(ratios_sql)
            ratios_sql = ','.join(ratios_sql_list)
            sql_query = f'select {ratios_sql} from '
            ratios = ks.sql(sql_query + '{df}').to_numpy()[0]
            for feat_idx, ratio_idx in list(zip(range(0, len(features)), range(0, 2*len(features)+1, 2))):
                if ratios[ratio_idx] >= ratios[ratio_idx+1]:
                    target_feat_corr_types['PositiveCorr'].append(
                        features[feat_idx])
                elif ratios[ratio_idx] < ratios[ratio_idx+1]:
                    target_feat_corr_types['NegativeCorr'].append(
                        features[feat_idx])
            return target_feat_corr_types

        target_feat_corr_types = {
            'PositiveCorr': [],
            'NegativeCorr': []
        }
        if utils.is_type(y, [KoalasSeries]):
            return _calc_target_ratios_koalas(X, y)
        else:
            return _calc_target_ratios_numpy(X, y)

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
