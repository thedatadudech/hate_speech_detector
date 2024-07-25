import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") or "http://0.0.0.0:5001/"
EXPERIMENT_NAME = "hate_speech"
MODEL_NAME = "hate_speech_classifier"
RAW_DATA_PATH = "https://github.com/amankharwal/Website-data/raw/master/twitter.csv"
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH") or "models/count_vectorizer.pkl"
TRAIN_RATIO = 0.80
TEST_RATIO = 1 - TRAIN_RATIO
