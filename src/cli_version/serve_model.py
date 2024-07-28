from model import get_run_id_from_latest_model_version, load_mlflow_client
from config import MODEL_NAME

client = load_mlflow_client()


if __name__ == "__main__":
    run_id, _ = get_run_id_from_latest_model_version(MODEL_NAME, client=client)
    client.download_artifacts(
        run_id=run_id, path=".", dst_path="models/best_model"
    )
