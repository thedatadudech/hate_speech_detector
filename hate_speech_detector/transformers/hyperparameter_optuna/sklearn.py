from typing import Dict, Union

from pandas import Series
from scipy.sparse._csr import csr_matrix


from hate_speech_detector.utils.models.sklearn import (
    tune_hyperparameters_optuna,
)
from hate_speech_detector.utils.logging import launch_objective

if "transformer" not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def hyperparameter_tuning(
    training_set: Dict[str, Union[Series, csr_matrix]],
    *args,
    **kwargs,
):

    X, y, X_train, y_train, X_test, y_test, _ = training_set["build"]

    objective = launch_objective(X_train, y_train, X_test, y_test)
    best_model = tune_hyperparameters_optuna(objective)

    return X, y, best_model
