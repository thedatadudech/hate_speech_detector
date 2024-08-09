import os
from typing import Dict, Optional, Tuple
import pandas as pd

import mlflow
from mlflow.data import from_numpy, from_pandas
from mlflow.models import infer_signature
from mlflow.sklearn import log_model as log_model_sklearn
from sklearn.base import BaseEstimator
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