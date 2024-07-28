import os
from typing import Dict, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from mlflow.data import from_numpy, from_pandas
from mlflow.entities import DatasetInput, InputTag, Run
from mlflow.models import infer_signature, signature
from mlflow.sklearn import log_model as log_model_sklearn
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

DEFAULT_DEVELOPER = os.getenv("EXPERIMENTS_DEVELOPER", "mager")
DEFAULT_EXPERIMENT_NAME = "hate_speech_mage"
DEFAULT_TRACKING_URI = "postgresql+psycopg2://hatespeechadmin:admin0815!@hate-speech-pg.postgres.database.azure.com:5432/mlflow"
DEFAULT_ARTIFACT_LOCATION = os.getenv("DEFAULT_ARTIFACT_LOCATION", "/data/artifacts")
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
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
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


def track_experiment(
    experiment_name: Optional[str] = None,
    block_uuid: Optional[str] = None,
    developer: Optional[str] = None,
    hyperparameters: Dict[str, Union[float, int, str]] = {},
    metrics: Dict[str, float] = {},
    model: Optional[Union[BaseEstimator]] = None,
    partition: Optional[str] = None,
    pipeline_uuid: Optional[str] = None,
    predictions: Optional[np.ndarray] = None,
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    training_set: Optional[pd.DataFrame] = None,
    training_targets: Optional[pd.Series] = None,
    track_datasets: bool = False,
    validation_set: Optional[pd.DataFrame] = None,
    validation_targets: Optional[pd.Series] = None,
    verbosity: Union[
        bool, int
    ] = True,  # False by default or else it creates too many logs
    **kwargs,
) -> Run:
    experiment_name = experiment_name or DEFAULT_EXPERIMENT_NAME
    tracking_uri = tracking_uri or DEFAULT_TRACKING_URI

    client, experiment_id = setup_experiment(experiment_name, tracking_uri)

    if not run_name:
        run_name = ":".join(
            [str(s) for s in [pipeline_uuid, partition, block_uuid] if s]
        )

    with mlflow.start_run(experiment_id, run_name=run_name or None) as run:
        print("storing run_id: ", run.info.run_id)

        for key, value in [
            ("developer", developer or DEFAULT_DEVELOPER),
            ("model", model.__class__.__name__),
        ]:
            if value is not None:
                mlflow.set_tag(key, value)

        for key, value in [
            ("block_uuid", block_uuid),
            ("partition", partition),
            ("pipeline_uuid", pipeline_uuid),
        ]:
            if value is not None:
                mlflow.log_param(key, value)

        for key, value in hyperparameters.items():
            mlflow.log_param(key, value)
            if verbosity:
                print(f"Logged hyperparameter {key}: {value}.")

        for key, value in metrics.items():
            mlflow.log_metric(key, value)
            if verbosity:
                print(f"Logged metric {key}: {value}.")

        dataset_inputs = []

        # This increases memory too much.
        if track_datasets:
            for dataset_name, dataset, tags in [
                ("dataset", training_set, dict(context="training")),
                (
                    "targets",
                    (
                        training_targets.to_numpy()
                        if training_targets is not None
                        else None
                    ),
                    dict(context="training"),
                ),
                ("dataset", validation_set, dict(context="validation")),
                (
                    "targets",
                    (
                        validation_targets.to_numpy()
                        if validation_targets is not None
                        else None
                    ),
                    dict(context="validation"),
                ),
                ("predictions", predictions, dict(context="training")),
            ]:
                if dataset is None:
                    continue

                dataset_from = None
                if isinstance(dataset, pd.DataFrame):
                    dataset_from = from_pandas
                elif isinstance(dataset, np.ndarray):
                    dataset_from = from_numpy

                if dataset_from:
                    ds = dataset_from(dataset, name=dataset_name)._to_mlflow_entity()
                    ds_input = DatasetInput(
                        ds, tags=[InputTag(k, v) for k, v in tags.items()]
                    )
                    dataset_inputs.append(ds_input)

                if verbosity:
                    context = tags["context"]
                    if dataset_from:
                        print(f"Logged input for {context} {dataset_name}.")
                    else:
                        print(
                            f"Unable to log input for {context} {dataset_name}, "
                            f"{type(dataset)} not registered."
                        )

            if len(dataset_inputs) >= 1:
                mlflow.log_inputs(dataset_inputs)

        if model:
            log_model = None

            if isinstance(model, BaseEstimator):
                log_model = log_model_sklearn

            if log_model:
                opts = dict(artifact_path="models", input_example=None)

                if training_set is not None and predictions is not None:
                    opts["signature"] = infer_signature(training_set, predictions)

                log_model(model, **opts)
                if verbosity:
                    print(f"Logged model {model.__class__.__name__}.")

    return run
