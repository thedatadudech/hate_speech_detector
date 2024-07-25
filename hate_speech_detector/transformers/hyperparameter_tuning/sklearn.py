from typing import Callable, Dict, Tuple, Union

from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator


from hate_speech_detector.utils.models.sklearn import (
    load_class,
    tune_hyperparameters,
)
from hate_speech_detector.utils.logging import track_experiment

if "transformer" not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def hyperparameter_tuning(
    training_set: Dict[str, Union[Series, csr_matrix]],
    model_class_name: str,
    *args,
    **kwargs,
) -> Tuple[
    Dict[str, Union[bool, float, int, str]],
    csr_matrix,
    Series,
    Callable[..., BaseEstimator],
]:

    
    print (training_set)
   
    X, y, X_train, y_train, X_test, y_test, _ = training_set["build"]

    model_class = load_class(model_class_name[0][0])

    

    hyperparameters = tune_hyperparameters(
        model_class,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        callback=lambda **opts: track_experiment(**{**opts, **kwargs}),
        max_evaluations=kwargs.get("max_evaluations"),
        random_state=kwargs.get("random_state"),
    )

    return hyperparameters, X, y, dict(cls=model_class, name=model_class_name)
