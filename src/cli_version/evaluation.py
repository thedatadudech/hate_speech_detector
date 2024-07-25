import pandas
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.sklearn
from utils import load_test_data, load_mlflow_client
from model import load_latest_best_model
from config import MODEL_NAME


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    mlflow.log_param("model_type", "Decision Tree model")
    mlflow.log_metric("accuracy", accuracy_score(y_test, predictions))
    mlflow.sklearn.log_model(model, "model")
    print(report)
    return report


if __name__ == "__main__":
    X_test, y_test = load_test_data("data/model_data/test_data_opt.pkl")
    model = load_latest_best_model(MODEL_NAME, client=load_mlflow_client())
    evaluate_model(model, X_test, y_test)
