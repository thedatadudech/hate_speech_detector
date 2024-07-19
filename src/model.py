import joblib
import optuna
import mlflow
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from mlflow.entities import ViewType


from optimization import launch_objective

from feature_engineering import vectorize_tweets
from config import MODEL_NAME, EXPERIMENT_NAME
from utils import generate_testdata, load_mlflow_client


client = load_mlflow_client()


def train_model(X, y, opt):
    # Optimize the hyperparameters
    if opt:
        if not check_experiment_name(EXPERIMENT_NAME, client=client):
            print("create experiment")
            client.create_experiment(EXPERIMENT_NAME)

        objective = launch_objective(X, y)
        tune_hyperparameters(objective)

        experiment_id = get_experiment_id(
            experiment_name=EXPERIMENT_NAME, client=client
        )

        run_id_best = get_best_run_id(experiment_id, client=client)

        if check_model_registered(client=client, model_name=MODEL_NAME):
            run_id_model_latest, _ = get_run_id_from_latest_model_version(
                model_name=MODEL_NAME, client=client
            )
        else:
            run_id_model_latest = None

        if run_id_best != run_id_model_latest:
            register_model_run(run_id_best)
            model = load_model_from_run_id(run_id=run_id_best)
            save_optimized_model(model, path=f"models/{MODEL_NAME}_opt.pkl")
        else:
            model = load_latest_best_model(model_name=MODEL_NAME, client=client)
    else:
        model = DecisionTreeClassifier()
        model.fit(X, y)
        save_optimized_model(model, f"models/{MODEL_NAME}.pkl")
    return model


def check_model_registered(client, model_name):
    try:
        client.get_registered_model(name=model_name)
    except:
        return False
    return True


def check_experiment_name(name, client):
    return client.get_experiment_by_name(name)


def save_optimized_model(model, path):
    joblib.dump(model, path)


def register_model_run(run_id_best, model_name=MODEL_NAME):
    model_uri = f"runs:/{run_id_best}/model"
    mlflow.register_model(model_uri=model_uri, name=model_name)


def get_experiment_id(experiment_name, client=None):
    return client.get_experiment_by_name(experiment_name).experiment_id


def tune_hyperparameters(objective):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)
    # Print the best parameters
    print("Best parameters: ", study.best_params)


def get_best_run_id(experiment_id=None, client=None):
    runs = client.search_runs(
        experiment_ids=experiment_id,
        filter_string="metrics.accuracy > 0.77",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=2,
        order_by=["metrics.accuracy DESC"],
    )
    run_id = runs[0].info.run_id
    return run_id


def get_run_id_from_latest_model_version(model_name, client=None):

    # Get the latest version of the registered model
    latest_models = client.get_registered_model(name=model_name)

    if not latest_models:
        raise ValueError(f"No versions found for model '{model_name}'")
    # Assume the latest version is the one with the highest version number
    latest_models = max(
        latest_models.latest_versions, key=lambda version: int(version.version)
    )
    # Extract the run_id from the latest version
    return latest_models.run_id, latest_models.version


def load_latest_best_model(model_name, client=None):
    run_id_best, _ = get_run_id_from_latest_model_version(
        model_name=model_name, client=client
    )
    model = load_model_from_run_id(run_id_best)
    return model


def load_model_from_run_id(run_id):
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)
    return model


if __name__ == "__main__":

    X, y = vectorize_tweets()
    mlflow.end_run()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    generate_testdata(X_test, y_test, "data/model_data/test_data.pkl")

    model = train_model(X_train, y_train, opt=True)
    mlflow.end_run()
    print("model saved")
