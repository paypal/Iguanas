"""Optimises the parameters of a pipeline using Bayesian Optimisation."""
from typing import Callable, Tuple, Dict, List, Union
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from hyperopt import tpe, fmin
import numpy as np
from copy import deepcopy
import pandas as pd
import warnings
from iguanas.pipeline import LinearPipeline
from iguanas.exceptions import DataFrameSizeError, NoRulesError
from iguanas.utils.typing import PandasDataFrameType, PandasSeriesType
from iguanas.utils.types import PandasDataFrame, PandasSeries, Dictionary
import iguanas.utils as utils
from iguanas.space import Choice


class BayesSearchCV:
    """
    Optimises the parameters of a pipeline using a user-defined set of
    search spaces in conjuncion with Bayesian Optimisation.

    The data is first split into cross validation datasets. For each fold, the
    Bayesian optimiser chooses a set of parameters (from the ranges provided),
    applies the pipeline's `fit` method to the training set, then applies the
    `predict` method to the validation set. The pipeline's prediction is scored
    using the `metric` function, and these scores are averaged across the
    folds. New parameter sets are chosen and applied until `n_iter` is reached.
    The parameter setwith the highest mean score is deemed to be the best
    performing.

    Parameters
    ----------
    pipeline : LinearPipeline
        The pipeline to be optimised. Note that the final step in the pipeline
        must include a `predict` method, which utilises a set of rules to make
        a prediction on a binary target.
    search_spaces : Dict[str, dict]
        The search spaces for each relevant parameter of each step in the
        pipeline. Each key should correspond to the tag used for the relevant
        pipeline step; each value should be a dictionary of the parameters
        (keys) and their search spaces (values). Search spaces should be
        defined using the classes in `iguanas.space`.
    metric : Callable
        The metric used to optimise the pipeline.
    cv : int
        The number of splits for cross-validation.
    n_iter : int
        The number of iterations that the optimiser should perform.
    refit : bool, optional
        Refit the best pipeline using the entire dataset. Must be set to True
        if predictions need to be made using the best pipeline. Defaults to
        True.
    algorithm : Callable, optional
        The algorithm leveraged by hyperopt's `fmin` function, which optimises
        the rules. Defaults to tpe.suggest, which corresponds to
        Tree-of-Parzen-Estimator.
    error_score : Union[str, float], optional
        Value to assign to the score of a validation fold if an error occurs
        in the pipeline fitting. If set to ‘raise’, the error is raised. If a
        numeric value is given, a warning is raised. This parameter does not
        affect the refit step, which will always raise the error. Defaults to
        'raise'.
    sample_weight_in_val : bool, optional
        Whether the `sample_weight` should be used when calculating the 
        `metric` on the validation fold. If True, the `sample_weight` is used.
        Defaults to False.
    num_cores : int, optional
        Number of cores to use when fitting a given parameter set. Should be
        set to <= `cv`. Defaults to 1.    
    verbose : int, optional
        Controls the verbosity - the higher, the more messages. >0 : shows the
        overall progress of the optimisation process. Defaults to 0.

    Attributes
    ----------
    cv_results : PandasDataFrameType
        Shows the scores per fold, mean score and standard deviation of the
        score for each trialled parameter set.
    best_score : float
        The best mean score achieved.
    best_index : int
        The parameter set index that produced the best mean score.
    best_params : dict
        The parameter set that produced the best mean score.
    pipeline_ : LinearPipeline
        The optimised LinearPipeline.
    """

    def __init__(self,
                 pipeline: LinearPipeline,
                 search_spaces: Dict[str, dict],
                 metric: Callable,
                 cv: int,
                 n_iter: int,
                 refit=True,
                 algorithm=tpe.suggest,
                 error_score='raise',
                 sample_weight_in_val=False,
                 num_cores=1,
                 verbose=0,
                 **kwargs) -> None:

        utils.check_allowed_types(pipeline, 'pipeline', [
            "<class 'iguanas.pipeline.linear_pipeline.LinearPipeline'>"
        ])
        self._check_search_spaces_type(search_spaces=search_spaces)
        self.pipeline = pipeline
        self.search_spaces = search_spaces
        self.metric = metric
        self.cv = cv
        self.n_iter = n_iter
        self.refit = refit
        self.algorithm = algorithm
        self.error_score = error_score
        self.sample_weight_in_val = sample_weight_in_val
        self.num_cores = num_cores
        self.verbose = verbose
        self.kwargs = kwargs

    def fit(self,
            X: Union[PandasDataFrameType, dict],
            y: Union[PandasSeriesType, dict],
            sample_weight=None) -> None:
        """
        Optimises the parameters of the given pipeline.

        Parameters
        ----------
        X : Union[PandasDataFrameType, dict]
            The dataset or dictionary of datasets for each pipeline step.
        y : Union[PandasSeriesType, dict]
            The binary target column or dictionary of binary target columns
            for each pipeline step.
        sample_weight : Union[PandasSeriesType, dict], optional
            Row-wise weights or dictionary of row-wise weights for each
            pipeline step.. Defaults to None.
        """

        utils.check_allowed_types(X, 'X', [PandasDataFrame, Dictionary])
        utils.check_allowed_types(y, 'y', [PandasSeries, Dictionary])
        if sample_weight is not None:
            utils.check_allowed_types(
                sample_weight, 'sample_weight', [PandasSeries, Dictionary])
        # Copy original pipeline
        self.pipeline_ = deepcopy(self.pipeline)
        # Generate CV datasets
        cv_datasets = self._generate_cv_datasets(
            X=X, y=y, sample_weight=sample_weight, cv=self.cv
        )
        # Convert values of `search_spaces` into hyperopt search functions
        search_spaces_ = self._convert_search_spaces_to_hyperopt(
            search_spaces=self.search_spaces
        )
        # Optimise pipeline parameters
        if self.verbose > 0:
            print('--- Optimising pipeline parameters ---')
        self.best_params, self.cv_results = self._optimise_params(
            cv_datasets=cv_datasets, pipeline=self.pipeline_,
            search_spaces=search_spaces_
        )
        # Reformat hyperopt output
        self.best_params = self._reformat_best_params(
            best_params=self.best_params, search_spaces=self.search_spaces
        )
        # Format CV results
        self.cv_results = self._format_cv_results(cv_results=self.cv_results)
        self.best_score = self.cv_results['MeanScore'].max()
        self.best_index = self.cv_results['MeanScore'].idxmax()
        # If `refit`==True, fit best pipeline on entire dataset
        if self.refit:
            if self.verbose > 0:
                print('--- Refitting on entire dataset with best pipeline ---')
            self.pipeline_._update_kwargs(params=self.best_params)
            self.pipeline_.fit(X, y, sample_weight)

    def predict(self, X: Union[PandasDataFrameType, dict]) -> PandasSeriesType:
        """
        Predict using the optimised pipeline.

        Parameters
        ----------
        X : Union[PandasDataFrameType, dict]
            The dataset or dictionary of datasets for each pipeline step.

        Returns
        -------
        PandasSeriesType
            The prediction of the pipeline.
        """

        utils.check_allowed_types(X, 'X', [PandasDataFrame, Dictionary])
        return self.pipeline_.predict(X)

    def fit_predict(self,
                    X: Union[PandasDataFrameType, dict],
                    y: Union[PandasSeriesType, dict],
                    sample_weight=None) -> PandasSeriesType:
        """
        Optimises the parameters of the given pipeline, then generates the
        optimised pipeline's prediction on the dataset.

        Parameters
        ----------
        X : Union[PandasDataFrameType, dict]
            The dataset or dictionary of datasets for each pipeline step.
        y : Union[PandasSeriesType, dict]
            The binary target column or dictionary of binary target columns
            for each pipeline step.
        sample_weight : Union[PandasSeriesType, dict], optional
            Row-wise weights or dictionary of row-wise weights for each
            pipeline step.. Defaults to None.

        Returns
        -------
        PandasSeriesType
            The prediction of the pipeline.
        """

        self.fit(X=X, y=y, sample_weight=sample_weight)
        return self.predict(X=X)

    def _optimise_params(self,
                         cv_datasets: Dict[int, list],
                         pipeline: LinearPipeline,
                         search_spaces: Dict[str, dict]) -> Tuple[dict, dict]:
        """Optimises the parameters of the given pipeline."""

        self.cv_results = []
        objective_inputs = (search_spaces, pipeline, cv_datasets)
        best_params = fmin(
            fn=self._objective,
            space=objective_inputs,
            algo=self.algorithm,
            max_evals=self.n_iter,
            verbose=self.verbose,
            rstate=np.random.RandomState(0),
            **self.kwargs
        )
        return best_params, self.cv_results

    def _objective(self,
                   objective_inputs: Tuple[dict, LinearPipeline, dict]) -> float:
        """
        Objective function for hyperopt's fmin function. Returns the mean score
        (across the CV datasets) for the given parameter set.
        """

        params_iter, pipeline, cv_datasets = objective_inputs
        pipeline._update_kwargs(params=params_iter)
        # Fit/predict/score on each fold
        with Parallel(n_jobs=self.num_cores) as parallel:
            scores_over_folds = parallel(delayed(self._fit_predict_on_fold)(
                self.metric, self.error_score, datasets, pipeline, params_iter,
                fold_idx, self.sample_weight_in_val) for fold_idx, datasets in cv_datasets.items()
            )
        scores_over_folds = np.array(scores_over_folds)
        mean_score = scores_over_folds.mean()
        std_dev_score = scores_over_folds.std()
        self.cv_results = self._update_cv_results(
            cv_results=self.cv_results, params_iter=params_iter,
            fold_idxs=list(cv_datasets.keys()),
            scores_over_folds=scores_over_folds, mean_score=mean_score,
            std_dev_score=std_dev_score
        )
        return -mean_score.mean()

    @staticmethod
    def _check_search_spaces_type(search_spaces: dict) -> None:
        """
        Checks that values of search_spaces are the correct type -
        UniformInteger, UniformFloat or Choice
        """

        for _, step_search_spaces in search_spaces.items():
            for param, search_space in step_search_spaces.items():
                utils.check_allowed_types(search_space, param, [
                    "<class 'iguanas.space.spaces.UniformFloat'>",
                    "<class 'iguanas.space.spaces.UniformInteger'>",
                    "<class 'iguanas.space.spaces.Choice'>"
                ])

    def _generate_cv_datasets(self,
                              X: Union[PandasDataFrameType, dict],
                              y: Union[PandasSeriesType, dict],
                              sample_weight: Union[PandasSeriesType, dict],
                              cv: int) -> dict:
        """Generates the cross validation datasets for each fold."""

        # If X or y are dicts, use first dataset in dict to calc folds
        X_ = list(X.values())[0] if isinstance(X, dict) else X
        y_ = list(y.values())[0] if isinstance(y, dict) else y
        cv_datasets = {}
        skf = StratifiedKFold(
            n_splits=cv,
            random_state=0,
            shuffle=True
        )
        skf.get_n_splits(X_, y_)
        folds = {
            fold_idx: (train_idxs, val_idxs) for fold_idx, (train_idxs, val_idxs) in enumerate(skf.split(X_, y_))
        }
        for fold_idx, (train_idxs, val_idxs) in folds.items():
            X_train, X_val = self._split_df_into_train_and_val(
                df=X, train_idxs=train_idxs, val_idxs=val_idxs
            )
            y_train, y_val = self._split_df_into_train_and_val(
                df=y, train_idxs=train_idxs, val_idxs=val_idxs
            )
            if sample_weight is None:
                sample_weight_train = None
                sample_weight_val = None
            else:
                sample_weight_train, sample_weight_val = self._split_df_into_train_and_val(
                    df=sample_weight, train_idxs=train_idxs, val_idxs=val_idxs
                )
            cv_datasets[fold_idx] = X_train, X_val, y_train, y_val, sample_weight_train, sample_weight_val
        return cv_datasets

    @staticmethod
    def _split_df_into_train_and_val(df: Union[PandasSeriesType,
                                               PandasDataFrameType, dict],
                                     train_idxs: np.ndarray,
                                     val_idxs: np.ndarray) -> Tuple[Union[
                                         PandasSeriesType,
                                         PandasDataFrameType, dict]]:
        """
        Splits a dataset or dictionary of datasets into training and validation
        sets.
        """

        def _splitter(df, idxs):
            # If the data is a Pandas data object, return the filtered object
            if isinstance(df, (pd.Series, pd.DataFrame)):
                return df.iloc[idxs]
            # If the data is a dict, loop through the dict and filter the
            # Pandas data objects
            elif isinstance(df, dict):
                df_dict = {}
                for step_tag, dataset in df.items():
                    if dataset is None:
                        df_dict[step_tag] = None
                    else:
                        df_dict[step_tag] = dataset.iloc[idxs]
                return df_dict
            else:
                raise TypeError(
                    '`df` must be a Pandas Series/DataFrame or a dict')
        df_train, df_val = (
            _splitter(df, idxs) for idxs in [train_idxs, val_idxs]
        )
        return df_train, df_val

    @ staticmethod
    def _convert_search_spaces_to_hyperopt(search_spaces: dict) -> dict:
        """
        Converts ranges in the search_spaces that are stored using
        `iguanas.space.spaces` types into hyperopt's stochastic expressions.
        """

        search_spaces_ = deepcopy(search_spaces)
        for step_tag, params in search_spaces_.items():
            for param, value in params.items():
                search_spaces_[step_tag][param] = value.transform(
                    label=f'{step_tag}__{param}'
                )
        return search_spaces_

    @ staticmethod
    def _fit_predict_on_fold(metric: Callable,
                             error_score: Union[str, float],
                             datasets: list,
                             pipeline: LinearPipeline,
                             params_iter: dict,
                             fold_idx: int,
                             sample_weight_in_val: bool) -> float:
        """
        Tries to to fit the pipeline (using a given parameter set) on the
        training set, then apply it to the validation set. If no rules remain
        after any of the stages of the pipeline, an error is thrown (if
        `self.error_score` == 'raise') or the score for the pipeline for that
        validation set is set to `self.error_score`.
        """

        try:
            X_train, X_val, y_train, y_val, sample_weight_train, sample_weight_val = datasets
            pipeline.fit(X_train, y_train, sample_weight_train)
            y_pred_val = pipeline.predict(X_val)
            # If y_val or sample_weight_val are dicts, extract the dataset
            # corresponding to the final pipeline step, so the score of the
            # pipeline predictor can be calculated
            y_val = utils.return_dataset_if_dict(
                step_tag=pipeline.steps_[-1][0], df=y_val
            )
            sample_weight_val = utils.return_dataset_if_dict(
                step_tag=pipeline.steps_[-1][0], df=sample_weight_val
            )
            # If sample_weight_in_val is True, use the sample_weight_val in the
            # metric calculation
            if sample_weight_in_val:
                fold_score = metric(y_pred_val, y_val, sample_weight_val)
            else:
                fold_score = metric(y_pred_val, y_val)
        except (DataFrameSizeError, NoRulesError):
            if error_score == 'raise':
                raise Exception(
                    f"""No rules remaining for: Pipeline parameter set = {params_iter}; Fold index = {fold_idx}."""
                )
            else:
                warnings.warn(
                    f"""No rules remaining for: Pipeline parameter set = {params_iter}; Fold index = {fold_idx}. The metric score for this parameter set & fold will be set to {error_score}"""
                )
                fold_score = error_score
        return fold_score

    @ staticmethod
    def _update_cv_results(cv_results: dict, params_iter: dict,
                           fold_idxs: List[int], scores_over_folds: np.ndarray,
                           mean_score: float, std_dev_score: float) -> dict:
        """
        Updates the cv_results dictionary with the results for the given
        parameter set.
        """

        flattened_params = {
            f'{step_tag}__{param}': value for step_tag,
            step_params in params_iter.items() for param, value in step_params.items()
        }
        cv_results.append({
            'Params': params_iter,
            **flattened_params,
            'FoldIdx': fold_idxs,
            'Scores': scores_over_folds,
            'MeanScore': mean_score,
            'StdDevScore': std_dev_score
        })
        return cv_results

    @ staticmethod
    def _reformat_best_params(best_params: dict, search_spaces: dict) -> dict:
        """
        Reformats the output of hyperopt's fmin function into the same
        dictionary format that is used to define the search_spaces. This allows
        the best parameters to be injected back into the pipeline.
        """

        # Convert back into original search_spaces format (i.e. a dictionary
        # whose keys are the steps in the pipeline, and whose values are
        # dictionaries of the parameters for that step).
        best_params_ = {
            param_tag.split("__")[0]: {} for param_tag in best_params.keys()
        }
        for param_tag, param_value in best_params.items():
            stage_tag = param_tag.split("__")[0]
            param = param_tag.split("__")[1]
            best_params_[stage_tag][param] = param_value
        # If value for param in search_spaces was a iguanas.space.Choice type,
        # then the outputted opt value will be the index of the best choice. So
        # need to return the best choice option from the original list.
        for step_tag, step_params in search_spaces.items():
            for param, value in step_params.items():
                if isinstance(value, Choice):
                    best_choice_idx = best_params_[step_tag][param]
                    best_params_[
                        step_tag][param] = value.options[best_choice_idx]
        return best_params_

    @ staticmethod
    def _format_cv_results(cv_results: dict) -> PandasDataFrameType:
        """
        Formats the cv_results dictionary into a Pandas Dataframe, and sorts by
        MeanScore descending, StdDevScore ascending.
        """

        cv_results = pd.DataFrame(cv_results)
        cv_results.sort_values(
            by=['MeanScore', 'StdDevScore'], ascending=[False, True],
            inplace=True
        )
        return cv_results
