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

#### .transformers.hyperparameter_optuna.sklearn.py

This script launches parameter tuning with Optuna.

```python
from hate_speech_detector.utils.models.sklearn import (
    tune_hyperparameters_optuna,
)
from hate_speech_detector.utils.logging import launch_objective

...

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
```

#### .utils.models.sklearn.py

This script contains the tuning function or the optuna study

```python
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import sklearn
import optuna
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.early_stop import no_progress_loss
from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

...


def tune_hyperparameters_optuna(objective):
 study = optuna.create_study(direction="maximize")
 study.optimize(objective, n_trials=3)
 # Print the best parameters
 print("Best parameters: ", study.best_params)

 best_trial = study.best_trial
 best_classifier = best_trial.user_attrs["classifier"]
 print("Best Trial Params:", best_trial.params)
 print("Best Trial Accuracy:", best_trial.value)
 return best_classifier
```

#### .utils.logging.py

This file contains the logging for the parameter tuning to MLflow

```python
import os
from typing import Dict, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from mlflow.data import from_numpy, from_pandas
from mlflow.entities import DatasetInput, InputTag, Run
from mlflow.models import infer_signature
from mlflow.sklearn import log_model as log_model_sklearn
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

...

def launch_objective(X_train, y_train, X_test, y_test):
    client, experiment_id = setup_experiment(
        experiment_name="hatespeech_clf_tuning_optuna",
        tracking_uri=DEFAULT_TRACKING_URI,
    )

    def objective(trial):
        # Suggest values for hyperparameters
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
        clf = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=42,
        )

        # Cross-validation
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        accuracy_train = scores.mean()

        clf.fit(X_train, y_train)
        accuracy_test = clf.score(X_test, y_test)

        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_params(
                {
                    "criterion": criterion,
                    "splitter": splitter,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features,
                    "max_leaf_nodes": max_leaf_nodes,
                }
            )
            mlflow.log_metric("accuracy_train", accuracy_train)
            mlflow.log_metric("accuracy_test", accuracy_test)
            log_model_sklearn(clf, "model")

            trial.set_user_attr("classifier", clf)
        return accuracy_train

    return objective

```

### Step 5: Registering the Best Model

#### .data_loaders.register_best_model.py

This script registers the best model in MLflow.

```python
import mlflow
import os
from mlflow.tracking import MlflowClient

...

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
        model_name = "model_hatespeech_classifier"
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
```

### Step 6: Downloading the Best Model Artifact

#### .data_exporters.download_artifact.py

This script downloads the best model artifact and stores it on Azure Blob Storage that is mounted by the infrastructure creation

```python
import mlflow
import joblib
import os

...

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
    best_exist, best_model_uri = response

    if best_exist:
        X, y, _, _, _, _, cv = training_set["build"]
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
        joblib.dump(cv, DESTINATION_PATH_BEST_CV)

    return True
```

### Conclusion

This guide provided a comprehensive setup of the experiment tracking and model registry using MLflow, along with parameter tuning using Optuna, logging of parameters and artifacts, registering the best model, and downloading the best model artifact to Azure Blob Storage. Modify the configurations and scripts as needed to fit your specific requirements.
