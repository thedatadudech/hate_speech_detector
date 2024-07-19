import pandas as pd
from config import RAW_DATA_PATH
from feature_engineering import extract_features
from utils import clean


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Example preprocessing steps

    df["labels"] = df["class"].map(
        {0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"}
    )
    df = df[["tweet", "labels"]]
    df = extract_features(df, column="tweet")
    df["tweet"] = df["tweet"].apply(clean)
    return df


if __name__ == "__main__":
    print("Loading raw data...")
    df = load_data(RAW_DATA_PATH)
    print("preprocessing data")
    df = preprocess_data(df)
    print("saving processed data")
    df.to_csv("data/processed/hate_speech_data_processed.csv", index=True)
    print("data has been processed")
