import mlflow
import joblib
import os

DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
DESTINATION_PATH_BEST_MODEL = os.getenv(
    "DESTINATION_PATH_BEST_MODEL", "/data/best_model/"
)
DESTINATION_PATH_BEST_CV = os.getenv(
    "DESTINATION_PATH_BEST_CV", "/data/cv/cv_best_model.pkl"
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
    # Specify your data exporting logic here
