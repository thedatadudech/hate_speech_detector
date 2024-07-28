import mlflow
from utils import generate_testdata
import mlflow.sklearn
from config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def launch_objective(X, y):
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

        # Load your dataset here
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        generate_testdata(X_test, y_test, "data/model_data/test_data_opt.pkl")

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
        accuracy = scores.mean()
        clf.fit(X_train, y_train)
        # Log parameters and metrics to MLflow

        with mlflow.start_run():
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
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(clf, "model")

        return accuracy

    return objective
