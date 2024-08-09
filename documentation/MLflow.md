### Step 1: Setting up MLflow Tracking Server

Use `Dockerfile.mlflow` to build the MLflow tracking server:

#### Dockerfile.mlflow

```Dockerfile
FROM python:3.11.4-slim

ARG MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
ARG DEFAULT_ARTIFACT_ROOT=$DEFAULT_ARTIFACT_ROOT

ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
ENV DEFAULT_ARTIFACT_ROOT=${DEFAULT_ARTIFACT_ROOT}


RUN pip install mlflow==2.14.1 psycopg2-binary


EXPOSE 5001

CMD ["sh", "-c", "mlflow server..."]


```

### Step 2: Docker Compose for MLflow Tracking Server

Use the `docker-compose.yml` file to set up the MLflow tracking server or better use the
`make mlflow-build` command from the:

#### docker-compose.yml

```yaml
version: "3.8"
services:
  mlflow:
    env_file:
      - .env.mlflow.dev
    build:
      context: .
      dockerfile: Dockerfile.mlflow
    ports:
      - "5001:5001"
    volumes:
      - ./db:/db
      - ./data/artifacts:/data/artifacts
    networks:
```

### Step 3: Running the MLflow Tracking Server

Run the following command to start the MLflow tracking server:

```sh
docker-compose up --build mlflow (only MLflow)
docker-compose up --build (all)

make mlflow-build
make compose-all-build

```

### Step 4: Parameter Tuning with Optuna and Logging with MLflow

#### .transformers.sklearn_voting.py

This script launches parameter tuning with Optuna.

```python
from typing import Dict, Union

from pandas import Series, DataFrame


from hate_speech_detector.utils.models.sklearn import (
    tune_hyperparameters_optuna,
)
from hate_speech_detector.utils.logging_voting import launch_objective

if "transformer" not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def hyperparameter_tuning(
    training_set: Dict[str, Union[DataFrame, Series]],
    *args,
    **kwargs,
):

    X, y, X_train, y_train, X_test, y_test = training_set["build2"]

    objective = launch_objective(X_train, y_train, X_test, y_test)
    best_model = tune_hyperparameters_optuna(objective)

    return X, y, best_model

```

#### .utils.models.sklearn.py

This script contains the tuning function or the optuna study

```python
from typing import Callable, Dict, Optional, Tuple, Union

import sklearn
import optuna
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score


def load_class(module_and_class_name: str) -> BaseEstimator:
    """
    module_and_class_name:
            tree.DecisionTreeClassifier
    """
    parts = module_and_class_name.split(".")
    cls = sklearn
    for part in parts:
        cls = getattr(cls, part)

    return cls


def tune_hyperparameters_optuna(objective):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    # Print the best parameters
    print("Best parameters: ", study.best_params)

    best_trial = study.best_trial
    best_classifier = best_trial.user_attrs["classifier"]
    print("Best Trial Params:", best_trial.params)
    print("Best Trial Accuracy:", best_trial.value)
    return best_classifier

```

#### .utils.logging_voting.py

This file contains the logging for the parameter tuning to MLflow

```python
import os
from typing import Optional, Tuple
import pandas as pd

import mlflow
from mlflow.sklearn import log_model as log_model_sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)


DEFAULT_DEVELOPER = os.getenv("EXPERIMENTS_DEVELOPER", "mager")
DEFAULT_EXPERIMENT_NAME = "hate_speech_voting"
DEFAULT_TRACKING_URI = (
    "postgresql+psycopg2://"
    "hatespeechadmin:admin0815!"
    "@hate-speech-pg.postgres.database.azure.com:5432/mlflow"
)

DEFAULT_ARTIFACT_LOCATION = os.getenv(
    "DEFAULT_ARTIFACT_LOCATION", "/data/artifacts"
)
# Testing the parameters on testset


def setup_experiment(
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> Tuple[mlflow.MlflowClient, str]:
    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    experiment_name = experiment_name or DEFAULT_EXPERIMENT_NAME

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=DEFAULT_ARTIFACT_LOCATION + f"/{experiment_name}",
        )

    return client, experiment_id


def launch_objective(X_train, y_train, X_test, y_test):
    client, experiment_id = setup_experiment(
        experiment_name=DEFAULT_EXPERIMENT_NAME,
        tracking_uri=DEFAULT_TRACKING_URI,
    )

    def objective(trial):
        # Suggest values for hyperparameters

        # DecisionTree
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        splitter = trial.suggest_categorical("splitter", ["best", "random"])
        max_depth = trial.suggest_int("max_depth", 1, 50)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 100)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 50)
        max_features = trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        )
        max_leaf_nodes = (
            trial.suggest_int("max_leaf_nodes", 2, 200)
            if trial.suggest_categorical("max_leaf_nodes_enable", [True, False])
            else None
        )

        # Initialize the classifier
        dt_clf = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=42,
        )

        # KNN
        knn_n_neighbors = trial.suggest_int("n_neighbors", 1, 3)
        knn_weights = trial.suggest_categorical(
            "weights", ["uniform", "distance"]
        )
        knn_p = trial.suggest_int("p", 1, 2)

        nn_clf = KNeighborsClassifier(
            n_neighbors=knn_n_neighbors, weights=knn_weights, p=knn_p, n_jobs=-1
        )

        # RF
        rf_n_estimators = trial.suggest_int("n_estimators", 10, 200)
        rf_max_depth = trial.suggest_int("max_depth", 2, 32)
        rf_min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        rf_min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        rf_bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        rf_clf = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_split=rf_min_samples_split,
            min_samples_leaf=rf_min_samples_leaf,
            bootstrap=rf_bootstrap,
            n_jobs=-1,
        )

        # ADA
        ada_n_estimators = trial.suggest_int("n_estimators", 10, 200)
        ada_learning_rate = trial.suggest_float(
            "learning_rate", 0.01, 1.0, log=True
        )

        ada_clf = AdaBoostClassifier(
            n_estimators=ada_n_estimators,
            learning_rate=ada_learning_rate,
            random_state=42,
        )

        classifiers = {
            "KNeighborsClassifier": nn_clf,
            "DecisionTreeClassifier": dt_clf,
            "RandomForestClassifier": rf_clf,
            "AdaBoostClasifier": ada_clf,
        }

        voting_clf = VotingClassifier(
            estimators=list(classifiers.items()), voting="hard"
        )

        # Fit the VotingClassifier
        voting_clf.fit(X_train, y_train)
        voting_predictions = voting_clf.predict(X_test)
        accuracy_test = voting_clf.score(X_test, y_test)

        individual_predictions = {
            name: clf.predict(X_test)
            for name, clf in voting_clf.named_estimators_.items()
        }

        # Combine predictions into a DataFrame for easier viewing
        predictions_df = pd.DataFrame(individual_predictions)
        predictions_df["VotingClassifier"] = voting_predictions
        predictions_df["True Labels"] = y_test

        # Cross-validation
        scores = cross_val_score(voting_clf, X_train, y_train, cv=5)
        accuracy_train = scores.mean()

        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_params(
                {
                    "dt.criterion": criterion,
                    "dt.splitter": splitter,
                    "dt.max_depth": max_depth,
                    "dt.min_samples_split": min_samples_split,
                    "dt.min_samples_leaf": min_samples_leaf,
                    "dt.max_features": max_features,
                    "dt.max_leaf_nodes": max_leaf_nodes,
                    "knn.n_neighbors": knn_n_neighbors,
                    "knn.weights": knn_weights,
                    "knn.p": knn_p,
                    "rf.n_estimators": rf_n_estimators,
                    "rf.max_depth": rf_max_depth,
                    "rf.min_samples_split": rf_min_samples_split,
                    "rf.min_samples_leaf": rf_min_samples_leaf,
                    "rf.boostrap": rf_bootstrap,
                    "ada.n_estimators": ada_n_estimators,
                    "ada.learning_rate": ada_learning_rate,
                }
            )
            mlflow.log_metric("accuracy_train", accuracy_train)
            mlflow.log_metric("accuracy_test", accuracy_test)
            log_model_sklearn(voting_clf, "model")

            trial.set_user_attr("classifier", voting_clf)
        return accuracy_train

    return objective


```

### Step 5: Registering the Best Model

#### .data_loaders.register_model_voting.py

This script registers the best model in MLflow.

```python
import mlflow
import os
from mlflow.tracking import MlflowClient

if "data_loader" not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test

DEFAULT_EXPERIMENT_NAME = os.getenv(
    "DEFAULT_EXPERIMENT_NAME", "hate_speech_voting"
)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@data_loader
def load_data(*args, **kwargs):

    # Set the tracking URI to PostgreSQL (replace with your actual URI)

    # Fetch all runs in the experiment
    experiment_name = DEFAULT_EXPERIMENT_NAME

    print("Experiment with name:", experiment_name)

    print("Connecting with MlFlowclient on : ", MLFLOW_TRACKING_URI)
    client = MlflowClient(MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy_train DESC"],
    )

    # Find the best run based on accuracy
    best_run = runs[0] if runs else None

    if best_run:
        best_run_id = best_run.info.run_id
        best_run_accuracy = best_run.data.metrics["accuracy_train"]
        print(f"Best run ID: {best_run_id}")
        print(f"Best run accuracy: {best_run_accuracy}")

        # Check if the model with this run ID is already registered
        model_name = "hatespeech_classifier_voting"
        registered_models = client.search_registered_models()

        is_registered = False
        for model in registered_models:
            for version in model.latest_versions:
                if version.run_id == best_run_id:
                    is_registered = True
                    break
            if is_registered:
                break

        if is_registered:
            best_model_uri = f"runs:/{best_run_id}/model"
            print(
                f"""A model with run ID {str(best_model_uri)} is already
                  registered as {str(model_name)}"""
            )
        else:
            # Register the best model
            best_model_uri = f"runs:/{best_run_id}/model"
            mlflow.register_model(model_uri=best_model_uri, name=model_name)
            print(
                f"""Registered the model '{model_name}' \
                    with accuracy {best_run_accuracy}."""
            )
    else:
        print("No runs found in the experiment.")
        return False

    return True, best_model_uri, True


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, "The output is undefined"

```

### Step 6: Downloading the Best Model Artifact

#### .data_exporters.download_artifact.py

This script downloads the best model artifact and stores it on Azure Blob Storage that is mounted by the infrastructure creation

```python
import mlflow
import joblib
import os

DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
DESTINATION_PATH_BEST_MODEL = os.getenv(
    "DESTINATION_PATH_BEST_MODEL", "/data/best_model/"
)

mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)

if "data_exporter" not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(response, training_set, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    best_exist, best_model_uri, voting = response

    if best_exist:
        X, y, _, _, _, _ = training_set["build2"]

        mlflow.artifacts.download_artifacts(
            artifact_uri=best_model_uri, dst_path=DESTINATION_PATH_BEST_MODEL
        )

        model = mlflow.sklearn.load_model(
            DESTINATION_PATH_BEST_MODEL + "/model"
        )

        model.fit(X, y)
        joblib.dump(
            model, DESTINATION_PATH_BEST_MODEL + "/model/best_model_fittedX.pkl"
        )

    return model
    # Specify your data exporting logic here

```

### Conclusion

This guide provided a comprehensive setup of the experiment tracking and model registry using MLflow, along with parameter tuning using Optuna, logging of parameters and artifacts, registering the best model, and downloading the best model artifact to Azure Blob Storage. Modify the configurations and scripts as needed to fit your specific requirements.
