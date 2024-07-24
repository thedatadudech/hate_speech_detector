from typing import Callable, Dict, List, Tuple, Union

from hyperopt import hp, tpe
from hyperopt.pyll import scope

from sklearn.tree import DecisionTreeClassifier


def build_hyperparameters_space(
    model_class: Callable[
        ...,
        Union[
            DecisionTreeClassifier
        ],
    ],
    random_state: int = 42,
    **kwargs,
) -> Tuple[Dict, Dict[str, List]]:
    params = {}
    choices = {}

    if DecisionTreeClassifier is model_class:
        params = dict(     
                max_depth=scope.int(hp.quniform('max_depth', 5, 30, 5)),
                min_samples_leaf=scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
                min_samples_split=scope.int(hp.quniform('min_samples_split', 2, 20, 2)),               
                random_state=random_state,
            )
        choices["criterion"] = ["gini", "entropy"]
        choices["splitter"]  = ["best", "random"]
        choices["max_features"]  = ["sqrt", "log2", None]


    for key, value in choices.items():
        params[key] = hp.choice(key, value)

    if kwargs:
        for key, value in kwargs.items():
            if value is not None:
                kwargs[key] = value

    return params, choices