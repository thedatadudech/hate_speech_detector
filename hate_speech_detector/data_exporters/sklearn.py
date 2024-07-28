from typing import Callable, Dict, Tuple, Union

from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator

from hate_speech_detector.utils.models.sklearn import load_class, train_model

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def register(
    settings: Tuple[
        Dict[str, Union[bool, float, int, str]],
        csr_matrix,
        Series,
        BaseEstimator,
    ],
    **kwargs,
) -> Tuple[BaseEstimator, Dict[str, str]]:
    X, y, model = settings

    model.fit(X, y)

    return model
