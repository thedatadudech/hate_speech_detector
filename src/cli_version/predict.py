from model import load_latest_best_model
from config import MODEL_NAME
from utils import load_mlflow_client
from feature_engineering import load_count_vectorizer
import argparse


def predict(text):
    client = load_mlflow_client()
    cv = load_count_vectorizer()
    model = load_latest_best_model(model_name=MODEL_NAME, client=client)
    input = cv.transform([text]).toarray()
    prediction = model.predict(input)
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict hate speech from input text"
    )
    parser.add_argument(
        "--text", type=str, default="We have Trump in the bullseye"
    )
    args = parser.parse_args()
    prediction = predict(args.text)
    print(f"The text {args.text} is predicted: {prediction}")
