from copy import deepcopy
from joblib import Parallel, delayed
from typing import List, Tuple
from iguanas.pipeline._base_pipeline import _BasePipeline
from iguanas.pipeline import LinearPipeline
from iguanas.utils.typing import PandasDataFrameType, PandasSeriesType
import pandas as pd


class ParallelPipeline(_BasePipeline):
    def __init__(self,
                 steps: List[Tuple[str, object]],
                 num_cores=1) -> None:
        _BasePipeline.__init__(self, steps=steps)
        self.num_cores = num_cores

    def fit_transform(self,
                      X: PandasDataFrameType,
                      y: PandasSeriesType,
                      sample_weight=None):

        self.steps_ = deepcopy(self.steps)
        with Parallel(n_jobs=self.num_cores) as parallel:
            X_rules_list = parallel(delayed(self._pipeline_fit_transform)(
                step_tag, step, X, y, sample_weight
            ) for step_tag, step in self.steps_
            )
        X_rules = pd.concat(X_rules_list, axis=1)
        return X_rules

    def transform(self, X):
        with Parallel(n_jobs=self.num_cores) as parallel:
            X_rules_list = parallel(delayed(self._pipeline_transform)(
                step_tag, step, X
            ) for step_tag, step in self.steps_
            )
        X_rules = pd.concat(X_rules_list, axis=1)
        return X_rules
