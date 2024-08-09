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