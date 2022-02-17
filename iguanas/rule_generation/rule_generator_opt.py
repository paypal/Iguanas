"""
Generates rules by optimising the thresholds of each feature individually, then
combining them.
"""
import pandas as pd
import numpy as np
import math
from itertools import combinations
from iguanas.correlation_reduction.agglomerative_clustering_reducer import AgglomerativeClusteringReducer
import iguanas.utils as utils
from iguanas.rule_application import RuleApplier
from iguanas.rule_generation._base_generator import _BaseGenerator
from iguanas.metrics.pairwise import CosineSimilarity
from iguanas.metrics.classification import FScore, Precision
from iguanas.utils.types import PandasDataFrame, PandasSeries
from iguanas.utils.typing import PandasDataFrameType, PandasSeriesType
from typing import Callable, List, Tuple, Dict
import warnings
f1 = FScore(1)


class RuleGeneratorOpt(_BaseGenerator):

    """
    Generate rules by optimising the thresholds of single features and
    combining these one condition rules with AND conditions to create more
    complex rules.

    Parameters
    ----------
    metric : Callable 
        A function/method which calculates the desired performance metric 
        (e.g. Fbeta score). Note that the module will assume higher values
        correspond to better performing rules.
    n_total_conditions : int 
        The maximum number of conditions per generated rule.
    num_rules_keep : int 
        The top number of rules (by Fbeta score) to keep at the end of each 
        stage of rule combination. Reducing this number will improve the 
        runtime, but may result in useful rules being removed.
    n_points : int, optional 
        Number of points to split a numeric feature's range into when
        generating the numeric one condition rules. A larger number will result
        in better optimised one condition rules, but will take longer to 
        calculate. Defaults to 10.
    ratio_window : int, optional 
        Factor which determines the optimisation range for numeric features 
        (e.g. if a numeric field has range of 1 to 11 and ratio_window = 3, the
        optimisation range for the <= operator will be from 1 to (11-1)/3 = 
        3.33; the optimisation range for the >= operator will be from 
        11-((11-1)/3)=7.67 to 11). A larger number (greater than 1) will result
        in a smaller range being used for optimisation of one condition rules; 
        set to 1 if you want to optimise the one condition rules across the 
        full range of the numeric feature. Defaults to 2.
    one_cond_rule_opt_metric : Callable, optional 
        The method/function used for optimising the one condition rules. Note 
        that the module will assume higher values correspond to better 
        performing rules. Defaults to the method used for calculating the F1 
        score.
    remove_corr_rules : bool, optional 
        Dictates whether correlated rules should be removed at the end of each
        pairwise combination. Defaults to True.
    target_feat_corr_types : Union[Dict[str, List[str]], str], optional 
        Limits the conditions of the rules based on the target-feature
        correlation (e.g. if a feature has a positive correlation with respect
        to the target, then only greater than operators are used for conditions
        that utilise that feature). Can be either a dictionary specifying the 
        list of positively correlated features wrt the target (under the key 
        `PositiveCorr`) and negatively correlated features wrt the target 
        (under the key `NegativeCorr`), or 'Infer' (where each target-feature 
        correlation type is inferred from the data). Defaults to None.
    verbose : int, optional 
        Controls the verbosity - the higher, the more messages. >0 : gives the
        progress of the training of the rules. Defaults to 0.
    rule_name_prefix : str, optional 
        Prefix to use for each rule. Defaults to 'RGO_Rule'.

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
                 n_total_conditions: int, num_rules_keep: int, n_points=10,
                 ratio_window=2, one_cond_rule_opt_metric=f1.fit,
                 remove_corr_rules=True, target_feat_corr_types=None,
                 verbose=0, rule_name_prefix='RGO_Rule'):

        _BaseGenerator.__init__(
            self,
            metric=metric,
            target_feat_corr_types=target_feat_corr_types,
            rule_name_prefix=rule_name_prefix,
        )
        self.n_total_conditions = n_total_conditions
        self.num_rules_keep = num_rules_keep
        self.n_points = n_points
        self.ratio_window = ratio_window
        self.one_cond_rule_opt_metric = one_cond_rule_opt_metric
        self.remove_corr_rules = remove_corr_rules
        self.verbose = verbose
        self.rule_strings = {}
        self.rule_names = []

    def __repr__(self):
        if self.rule_strings:
            return f'RuleGeneratorOpt object with {len(self.rule_strings)} rules generated'
        else:
            return f'RuleGeneratorOpt(metric={self.metric}, n_total_conditions={self.n_total_conditions}, num_rules_keep={self.num_rules_keep}, n_points={self.n_points}, ratio_window={self.ratio_window}, one_cond_rule_opt_metric={self.one_cond_rule_opt_metric}, remove_corr_rules={self.remove_corr_rules}, target_feat_corr_types={self.target_feat_corr_types})'

    def fit(self, X: PandasDataFrameType, y: PandasSeriesType,
            sample_weight=None) -> PandasDataFrameType:
        """
        Generate rules by optimising the thresholds of single features and
        combining these one condition rules with AND conditions to create more
        complex rules.

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
        if self.target_feat_corr_types == 'Infer':
            self.target_feat_corr_types = self._calc_target_ratio_wrt_features(
                X=X, y=y
            )
        X_rules = pd.DataFrame()
        rule_strings = {}
        columns_int, columns_cat, columns_float = utils.return_columns_types(
            X)
        columns_num = [
            col for col in columns_int if col not in columns_cat] + columns_float
        if columns_num:
            if self.verbose > 0:
                print(
                    '--- Generating one condition rules for numeric features ---')
            rule_strings_num, X_rules_num = self._generate_numeric_one_condition_rules(
                X, y, columns_num, columns_int, sample_weight
            )
            X_rules = pd.concat([X_rules, X_rules_num], axis=1)
            rule_strings.update(rule_strings_num)
        if columns_cat:
            if self.verbose > 0:
                print(
                    '--- Generating one condition rules for OHE categorical features ---')
            rule_strings_cat, X_rules_cat = self._generate_categorical_one_condition_rules(
                X, y, columns_cat, sample_weight
            )
            X_rules = pd.concat([X_rules, X_rules_cat], axis=1)
            rule_strings.update(rule_strings_cat)
        if self.verbose > 0:
            print('--- Generating pairwise rules ---')
        self.rule_strings, X_rules = self._generate_n_order_pairwise_rules(
            X_rules, y, rule_strings, self.remove_corr_rules, sample_weight
        )
        self._generate_other_rule_formats()
        return X_rules

    def _generate_numeric_one_condition_rules(self, X: PandasDataFrameType,
                                              y: PandasSeriesType,
                                              columns_num: List[str],
                                              columns_int: List[str],
                                              sample_weight: PandasSeriesType) -> Tuple[PandasDataFrameType, PandasDataFrameType]:
        """
        Generates one condition rules for numeric columns by optimising the 
        threshold of each column based on `metric`.
        """

        rule_strings = {}
        if self.target_feat_corr_types is not None:
            pos_corr_num_feats = [
                col for col in columns_num if col in self.target_feat_corr_types['PositiveCorr']]
            neg_corr_num_feats = [
                col for col in columns_num if col in self.target_feat_corr_types['NegativeCorr']]
            cols_and_operators = list(zip(pos_corr_num_feats, ['>='] * len(pos_corr_num_feats))) + list(
                zip(neg_corr_num_feats, ['<='] * len(neg_corr_num_feats)))
        else:
            cols_and_operators = list(
                zip(columns_num * 2, [">="] * len(columns_num) + ["<="] * len(columns_num)))
        cols_and_operators = utils.return_progress_ready_range(
            verbose=self.verbose, range=cols_and_operators)
        for column, operator in cols_and_operators:
            X_col = X[column].values
            if np.std(X_col) == 0:
                continue
            x_min, x_max = self._set_iteration_range(
                X_col=X_col, column=column, operator=operator, n_points=self.n_points,
                ratio_window=self.ratio_window, columns_int=columns_int)
            x_iter = self._set_iteration_array(
                column, columns_int, x_min, x_max, self.n_points)
            # Optimise threshold using self.one_cond_rule_beta
            opt_metric_iter = self._calculate_opt_metric_across_range(
                x_iter=x_iter, operator=operator, X_col=X_col, y=y,
                metric=self.one_cond_rule_opt_metric, sample_weight=sample_weight)
            x_max_opt_metric = self._return_x_of_max_opt_metric(
                opt_metric_iter, operator, x_iter)
            rule_logic = f"(X['{column}']{operator}{x_max_opt_metric})"
            rule_name = self._generate_rule_name()
            rule_strings[rule_name] = rule_logic
        if not rule_strings:
            warnings.warn('No numeric one condition rules could be created.')
            return pd.DataFrame(), pd.DataFrame()
        ara = RuleApplier(rule_strings=rule_strings)
        X_rules = ara.transform(X=X)
        rule_strings, X_rules = self._drop_zero_var_and_precision_rules(
            X_rules=X_rules, y=y, rule_strings=rule_strings
        )
        return rule_strings, X_rules

    def _generate_categorical_one_condition_rules(self,
                                                  X: PandasDataFrameType,
                                                  y: PandasSeriesType,
                                                  columns_cat: List[str],
                                                  sample_weight: PandasDataFrameType) -> Tuple[PandasDataFrameType, PandasDataFrameType]:
        """Generates one condition rules for OHE categorical columns"""

        def _gen_rules_from_target_feat_corr_types(X: PandasDataFrameType, y: PandasSeriesType,
                                                   columns_cat: List[str],
                                                   sample_weight: PandasSeriesType) -> Tuple[PandasDataFrameType, PandasDataFrameType]:
            """
            Generates rules using the target-feature correlation types given
            in `target_feat_corr_types`.
            """

            pos_corr_cat_feats = [
                col for col in columns_cat if col in self.target_feat_corr_types['PositiveCorr']]
            neg_corr_cat_feats = [
                col for col in columns_cat if col in self.target_feat_corr_types['NegativeCorr']]
            cols_and_operators = list(zip(pos_corr_cat_feats, ['True'] * len(pos_corr_cat_feats))) + list(
                zip(neg_corr_cat_feats, ['False'] * len(neg_corr_cat_feats)))
            rule_strings = {
                self._generate_rule_name(): f"(X['{col}']=={operator})"
                for col, operator in cols_and_operators
            }
            ara = RuleApplier(
                rule_strings=rule_strings
            )
            X_rules = ara.transform(X=X)
            return rule_strings, X_rules

        def _gen_rules_best_perf_bool_option(X: PandasDataFrameType, y: PandasSeriesType,
                                             columns_cat: List[str],
                                             sample_weight: PandasSeriesType) -> Tuple[PandasDataFrameType, PandasDataFrameType]:
            """
            Generates rules by keeping only the best performing boolean value 
            per OHE column.
            """

            X_rules_list = []
            rule_strings = {}
            for col in columns_cat:
                X_rules_col_list = []
                for value in ['True', 'False']:
                    rule_name = self._generate_rule_name()
                    rule_logic = f"(X['{col}']=={value})"
                    rule_string = {rule_name: rule_logic}
                    ara = RuleApplier(rule_strings=rule_string)
                    X_rule = ara.transform(X=X)
                    rule_strings.update(rule_string)
                    X_rules_col_list.append(X_rule)
                X_rules_col = pd.concat(X_rules_col_list, axis=1)
                # Keep only best performing condition per feature
                metrics = self.one_cond_rule_opt_metric(
                    X_rules_col, y, sample_weight
                )
                X_rules_col = X_rules_col.iloc[:, np.argmax(metrics)]
                X_rules_list.append(X_rules_col)
            X_rules = pd.concat(X_rules_list, axis=1)
            rule_strings = {
                rule_name: rule_logic for rule_name, rule_logic in rule_strings.items() if rule_name in X_rules.columns
            }
            return rule_strings, X_rules

        columns_cat = utils.return_progress_ready_range(
            verbose=self.verbose, range=columns_cat
        )
        if self.target_feat_corr_types is not None:
            rule_strings, X_rules = _gen_rules_from_target_feat_corr_types(
                X=X, y=y, columns_cat=columns_cat, sample_weight=sample_weight
            )
        else:
            rule_strings, X_rules = _gen_rules_best_perf_bool_option(
                X=X, y=y, columns_cat=columns_cat, sample_weight=sample_weight
            )
        rule_strings, X_rules = self._drop_zero_var_and_precision_rules(
            X_rules=X_rules, y=y, rule_strings=rule_strings
        )
        if X_rules.empty:
            warnings.warn(
                'No categorical one condition rules could be created.'
            )
        return rule_strings, X_rules

    def _generate_pairwise_rules(self,
                                 X_rules: PandasDataFrameType, y: PandasSeriesType,
                                 rules_combinations: List[Tuple[Tuple[str, str], Tuple[str, str]]],
                                 sample_weight: PandasSeriesType) -> Tuple[PandasDataFrameType, PandasDataFrameType, Dict[str, list]]:
        """Combines binary columns of rules using AND conditions"""

        pairwise_info_dict = self._return_pairwise_information(
            rules_combinations)
        pairwise_logics = list(pairwise_info_dict.keys())
        pairwise_info_list = list(pairwise_info_dict.values())
        rules_names_1, rules_names_2, pairwise_names = [], [], []
        for info_dict in pairwise_info_list:
            rules_names_1.append(info_dict['RuleName1'])
            rules_names_2.append(info_dict['RuleName2'])
            pairwise_names.append(info_dict['PairwiseRuleName'])
        X_rules_pairwise_df = self._generate_pairwise_df(
            X_rules, rules_names_1, rules_names_2, pairwise_names)
        pairwise_descriptions = utils.return_binary_pred_perf_of_set(
            y_true=y, y_preds=X_rules_pairwise_df, y_preds_columns=pairwise_names,
            sample_weight=sample_weight, metric=self.metric)
        pairwise_descriptions.index.name = 'Rule'
        pairwise_descriptions['Logic'] = pairwise_logics
        pairwise_descriptions['nConditions'] = pairwise_descriptions['Logic'].apply(
            utils.count_rule_conditions)
        pairwise_descriptions = pairwise_descriptions.reindex(
            ['Logic', 'Precision', 'Recall', 'nConditions', 'PercDataFlagged', 'Metric'], axis=1)
        pairwise_components = dict((info_dict['PairwiseRuleName'], info_dict['PairwiseComponents'])
                                   for _, info_dict in pairwise_info_dict.items())
        return pairwise_descriptions, X_rules_pairwise_df, pairwise_components

    def _drop_unnecessary_pairwise_rules(self,
                                         pairwise_descriptions: PandasDataFrameType,
                                         X_rules_pairwise_df: PandasDataFrameType,
                                         y: PandasSeriesType,
                                         pairwise_to_orig_lookup: Dict[str, Tuple[str, str]],
                                         rule_descriptions: PandasDataFrameType) -> Tuple[PandasDataFrameType, PandasDataFrameType]:
        """
        Drops pairwise rules with precision == 0 or that have a precision less 
        than one of their component rules.
        """

        zero_var_rules = self._return_zero_variance_rules(
            X_rules=X_rules_pairwise_df
        )
        zero_prec_rules = self._return_zero_precision_rules(
            X_rules=X_rules_pairwise_df, y=y
        )
        # Get rules with precision less than either of the individual rules
        pw_rules_less_prec = self._return_pairwise_rules_to_drop(
            pairwise_descriptions, pairwise_to_orig_lookup, rule_descriptions
        )
        rules_to_drop = list(
            set(zero_var_rules + zero_prec_rules + pw_rules_less_prec))
        pairwise_descriptions = pairwise_descriptions.drop(
            rules_to_drop, axis=0)
        X_rules_pairwise_df = X_rules_pairwise_df.drop(
            rules_to_drop, axis=1)

        return pairwise_descriptions, X_rules_pairwise_df

    def _generate_n_order_pairwise_rules(self,
                                         X_rules: PandasDataFrameType,
                                         y: PandasSeriesType,
                                         rule_strings: Dict[str, str],
                                         remove_corr_rules: bool,
                                         sample_weight: PandasSeriesType) -> Tuple[PandasDataFrameType, PandasDataFrameType]:
        """
        Loops through ruleset (starting with one condition rules) and combines 
        them pairwise to a given order.
        """

        n_loops = int(
            math.log(2 ** math.ceil(math.log(self.n_total_conditions, 2)), 2))
        loop_range = utils.return_progress_ready_range(
            verbose=self.verbose, range=range(1, n_loops + 1))
        rule_descriptions = self._generate_rule_descriptions(
            X_rules=X_rules, y=y, sample_weight=sample_weight,
            rule_strings=rule_strings, metric=self.metric
        )
        for n_loop in loop_range:
            if remove_corr_rules:
                rule_descriptions, X_rules = self._remove_corr_rules(
                    X_rules=X_rules, y=y, sample_weight=sample_weight,
                    rule_descriptions=rule_descriptions, metric=self.metric
                )
            rules_combinations = self._get_rule_combinations_for_loop(
                rule_descriptions, n_loop, self.num_rules_keep)
            if len(rules_combinations) == 0:
                break
            rule_descriptions_pairwise, X_rules_pairwise, pairwise_components = self._generate_pairwise_rules(
                X_rules, y, rules_combinations, sample_weight)
            rule_descriptions_pairwise, X_rules_pairwise = self._drop_unnecessary_pairwise_rules(
                rule_descriptions_pairwise, X_rules_pairwise, y, pairwise_components, rule_descriptions
            )
            X_rules = pd.concat(
                [X_rules, X_rules_pairwise], axis=1)
            rule_descriptions = pd.concat(
                [rule_descriptions, rule_descriptions_pairwise], axis=0)
        rule_descriptions = rule_descriptions[rule_descriptions['nConditions']
                                              <= self.n_total_conditions]
        X_rules = X_rules[rule_descriptions.index.tolist()]
        rule_descriptions, X_rules = self._sort_rule_dfs_by_opt_metric(
            rule_descriptions, X_rules
        )
        rule_strings = rule_descriptions['Logic'].to_dict()
        return rule_strings, X_rules

    def _return_pairwise_information(self,
                                     rules_combinations: List[Tuple[Tuple[str, str], Tuple[str, str]]]) -> Dict[str, dict]:
        """
        Returns a dict of the pairwise rule logic and associated 
        information
        """

        def clean_rule_logic(rule_name: str) -> str:
            """Removes duplicate columns in combined rule logic"""
            rule_name_list = rule_name.split("&")
            rule_name_set = sorted(list(set(rule_name_list)))
            rule_name = '&'.join(rule_name_set)
            return rule_name

        pairwise_info_dict = {}
        rule_logics_list = []
        # Loop through rule combinations and calculate pairwise logic. Then
        # link the component rule names, logic and distinct components to the
        # pairwise logic
        for (rule_name_1, rule_name_2), (rule_logic_1, rule_logic_2) in rules_combinations:
            pairwise_rule_logic = clean_rule_logic(
                f'{rule_logic_1}&{rule_logic_2}')
            if rule_logics_list.count(pairwise_rule_logic) == 0:
                pairwise_rule_name = self._generate_rule_name()
                pairwise_info_dict[pairwise_rule_logic] = {
                    'RuleName1': rule_name_1,
                    'RuleName2': rule_name_2,
                    'PairwiseRuleName': pairwise_rule_name,
                    'PairwiseComponents': [rule_name_1, rule_name_2]
                }
                rule_logics_list.append(pairwise_rule_logic)
            else:
                pairwise_info_dict[pairwise_rule_logic]['PairwiseComponents'].extend(
                    [rule_name_1, rule_name_2])
                pairwise_info_dict[pairwise_rule_logic]['PairwiseComponents'] = list(set(
                    pairwise_info_dict[pairwise_rule_logic]['PairwiseComponents']))
        return pairwise_info_dict

    def _drop_zero_var_and_precision_rules(self,
                                           X_rules: PandasDataFrameType,
                                           y: PandasSeriesType,
                                           rule_strings: Dict[str, str]) -> Tuple[PandasDataFrameType, PandasDataFrameType]:
        """
        Drops zero variance and zero precisions rules from rule_descriptions 
        and X_rules
        """

        zero_var_rules = self._return_zero_variance_rules(X_rules=X_rules)
        zero_precision_rules = self._return_zero_precision_rules(
            X_rules=X_rules, y=y)
        rules_to_drop = list(set(zero_var_rules + zero_precision_rules))
        X_rules = X_rules.drop(rules_to_drop, axis=1)
        rule_strings = {rule_name: rule_logic for rule_name,
                        rule_logic in rule_strings.items() if rule_name not in rules_to_drop}
        return rule_strings, X_rules

    @staticmethod
    def _generate_rule_descriptions(X_rules: PandasDataFrameType,
                                    y: PandasSeriesType,
                                    sample_weight: PandasSeriesType,
                                    rule_strings: Dict[str, str],
                                    metric: Callable) -> PandasDataFrameType:
        """
        Generates the rule_descriptions dataframe for the set of rules 
        provided.
        """

        p = Precision()
        n_conditions = {
            rule_name: utils.count_rule_conditions(rule_logic) for rule_name, rule_logic in rule_strings.items()
        }
        precisions = pd.Series(
            data=p.fit(X_rules, y, sample_weight),
            index=list(rule_strings.keys())
        )
        metrics = pd.Series(
            data=metric(X_rules, y, sample_weight),
            index=list(rule_strings.keys())
        )
        rule_descriptions = pd.DataFrame([
            rule_strings,
            n_conditions,
            precisions,
            metrics
        ]).T
        rule_descriptions.columns = [
            'Logic', 'nConditions', 'Precision', 'Metric']
        rule_descriptions = rule_descriptions.astype({
            'Logic': object, 'nConditions': int, 'Precision': float, 'Metric': float
        })
        return rule_descriptions

    @staticmethod
    def _remove_corr_rules(X_rules: PandasDataFrameType,
                           y: PandasSeriesType,
                           sample_weight: PandasSeriesType,
                           rule_descriptions: PandasDataFrameType,
                           metric: Callable,) -> Tuple[PandasDataFrameType, PandasDataFrameType]:
        """
        Remove correlated rules using the AgglomerativeClusteringReducer class
        """

        cs = CosineSimilarity()
        rcr = AgglomerativeClusteringReducer(
            threshold=0.75,
            metric=metric,
            strategy='bottom_up', similarity_function=cs.fit)
        X_rules = rcr.fit_transform(X_rules, y, sample_weight)
        rule_descriptions = rule_descriptions.loc[X_rules.columns]
        return rule_descriptions, X_rules

    @staticmethod
    def _set_iteration_range(X_col: np.array, column: str, operator: str,
                             n_points: int, ratio_window: int,
                             columns_int: List[str]) -> Tuple[float, float]:
        """Sets the iteration range for a given column"""

        X_col_max = max(X_col)
        X_col_min = min(X_col)
        if column in columns_int and n_points > abs(X_col_max - X_col_min):
            x_min = X_col_min
            x_max = X_col_max
        elif operator == "<=":
            x_min = X_col_min
            x_max = x_min + (X_col_max - x_min) / ratio_window
        elif operator == ">=":
            x_max = X_col_max
            x_min = x_max - (x_max - X_col_min) / ratio_window
        return (x_min, x_max)

    @staticmethod
    def _set_iteration_array(column: str, columns_int: List[str], x_min: float,
                             x_max: float, n_points: int) -> np.array:
        """Returns the iteration array for a given column"""

        def _round_to_n_sf(x: float, n_sf: int) -> float:
            """Method for rounding a float to n significant figures"""

            if x == 0:
                return 0
            return round(x, -int(math.floor(math.log10(abs(x)))) + (n_sf - 1))

        if column in columns_int:
            x_min, x_max = int(x_min), int(x_max)
            if abs(x_min - x_max) < n_points:
                x_iter = np.array(range(x_min, x_max + 1))
            else:
                x_iter = np.ceil(np.linspace(x_min, x_max, n_points))
        else:
            x_iter = np.linspace(x_min, x_max, n_points)
            x_iter = np.array([_round_to_n_sf(x, 2) for x in x_iter])
        return x_iter

    @staticmethod
    def _calculate_opt_metric_across_range(x_iter: np.array, operator: str,
                                           X_col: np.array, y: np.array,
                                           metric: Callable[[PandasSeriesType, PandasSeriesType, PandasSeriesType], PandasSeriesType],
                                           sample_weight: np.array) -> np.array:
        """
        Calculates the optimisation function at each point in the x_iter 
        range
        """

        opt_metric_iter = np.zeros(len(x_iter))
        for i, x in enumerate(x_iter):
            X_rule = eval(f'X_col {operator} {x}').astype(int)
            opt_metric_iter[i] = metric(
                y_true=y, y_preds=X_rule, sample_weight=sample_weight)
        return opt_metric_iter

    @staticmethod
    def _return_x_of_max_opt_metric(opt_metric_iter: np.array, operator: str,
                                    x_iter: np.array) -> float:
        """Returns the threshold value which maximises the FBeta score"""

        max_opt_metric = np.nanmax(opt_metric_iter)
        if max_opt_metric == 0:
            return None
        if operator == "<=":
            idx_max_opt_metric = min([i for i, j in enumerate(
                opt_metric_iter) if j == max_opt_metric])
        elif operator == ">=":
            idx_max_opt_metric = max([i for i, j in enumerate(
                opt_metric_iter) if j == max_opt_metric])
        return x_iter[idx_max_opt_metric]

    @staticmethod
    def _get_rule_combinations_for_loop(rule_descriptions: PandasDataFrameType,
                                        n_loop: int,
                                        num_rules_keep: int) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """Generates pairwise combinations of rules"""

        # At beginning of each loop, filter list of rules to include only those
        # needed for pairwise calculation
        rules_n_conditions = rule_descriptions[(
            rule_descriptions['nConditions'] == 2 ** (n_loop - 1))]
        # Then sort resulting ruleset by Metric and take top num_rules_keep
        # rules for pairwise calculation
        rules_n_conditions = rules_n_conditions.sort_values(
            by='Metric', ascending=False)
        rules_n_conditions = rules_n_conditions.iloc[:num_rules_keep]
        # Get the rule names and their logic
        rule_names = rules_n_conditions.index.values
        rule_logic = rules_n_conditions['Logic'].values
        # Calculate distinct combinations of both the rule names and their
        # logic
        rules_logic_combinations = list(combinations(rule_logic, r=2))
        rules_name_combinations = list(combinations(rule_names, r=2))
        # Combine these into a list
        rules_combinations = list(
            zip(rules_name_combinations, rules_logic_combinations))
        return rules_combinations

    @staticmethod
    def _generate_pairwise_df(X_rules: PandasDataFrameType, rules_names_1: List[str],
                              rules_names_2: List[str],
                              pairwise_names: List[str]) -> PandasDataFrameType:
        """
        Multiplies the component rules together to give the pairwise dataframe
        """

        X_rules_pairwise_arr = X_rules[rules_names_1].values * \
            X_rules[rules_names_2].values
        X_rules_pairwise_df = pd.DataFrame(
            X_rules_pairwise_arr, columns=pairwise_names, index=X_rules.index)
        return X_rules_pairwise_df

    @staticmethod
    def _return_pairwise_rules_to_drop(pairwise_descriptions: PandasDataFrameType,
                                       pairwise_to_orig_lookup: Dict[str, List[str]],
                                       rule_descriptions: PandasDataFrameType) -> List[str]:
        """
        Drops pairwise rule if its precision is less than or equal to the 
        precision of one of its component rules
        """

        rules_to_drop = []
        for idx, row in pairwise_descriptions.iterrows():
            orig_rules = pairwise_to_orig_lookup[idx]
            max_orig_prec = rule_descriptions.loc[orig_rules, 'Precision'].max(
            )
            if row['Precision'] <= max_orig_prec:
                rules_to_drop.append(idx)
        return rules_to_drop

    @staticmethod
    def _return_zero_variance_rules(X_rules: PandasDataFrameType) -> List[str]:
        """Returns list of zero variance rules"""

        X_rules_std = X_rules.to_numpy().std(axis=0)
        mask = X_rules_std == 0
        zero_var_rules = X_rules.columns[mask].tolist()
        return zero_var_rules

    @staticmethod
    def _return_zero_precision_rules(X_rules: PandasDataFrameType,
                                     y: PandasSeriesType) -> List[str]:
        """Returns list of zero precision rules"""

        p = Precision()
        precisions = p.fit(X_rules, y)
        zero_precision_rules = X_rules.columns[precisions == 0].tolist()
        return zero_precision_rules

    @staticmethod
    def _sort_rule_dfs_by_opt_metric(rule_descriptions: PandasDataFrameType,
                                     X_rules: PandasDataFrameType) -> Tuple[PandasDataFrameType, PandasDataFrameType]:
        """
        Method for sorting and reindexing `rule_descriptions` and `X_rules` by
        the 'Metric' column.
        """

        rule_descriptions.sort_values(
            by=['Metric'], ascending=False, inplace=True)
        X_rules = X_rules.reindex(rule_descriptions.index.tolist(), axis=1)
        return rule_descriptions, X_rules
