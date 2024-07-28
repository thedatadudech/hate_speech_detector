import pandas as pd
import numpy as np
import emoji
import joblib
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
from src.cli_version.config import VECTORIZER_PATH

import mlflow


def extract_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
    features_df = (
        df[column].apply(extract_tweet_features).apply(pd.Series).fillna(0)
    )
    df = pd.concat([df, features_df], axis=1)
    return df


def vectorize_tweets(
    df=None,
    path="data/processed/features.csv",
    tweet_column="tweet",
    label_column="labels",
):
    cv = initialize_count_vectorizer()
    if not df:
        try:
            print("Reading tweets from path:", path)
            df = pd.read_csv(path)
            print("Tweets succesfully read")
        except Exception as e:
            print(f"Unexpected error {e} , could not read tweets from path")
    df = df.dropna()
    x = np.array(df[tweet_column])
    X = cv.fit_transform(x)
    y = np.array(df[label_column])
    save_count_vectorizer(cv, VECTORIZER_PATH)
    try:
        print("logging artifact")
        mlflow.log_artifact(VECTORIZER_PATH)
    except Exception as e:
        print(f"Error {e} :, artifact could not be logged")
    return X, y


def save_count_vectorizer(cv, path):
    try:
        print("saving vectorizer to path:", path)
        with open(path, "wb") as file:
            cv_dict = dict()
            cv_dict["cv"] = cv
            joblib.dump(cv_dict, file)
            print("vectorizer saved succesfully")
    except Exception as e:
        print(f"Error {e} Could not save vectorizer")
        return False
    return True


def load_count_vectorizer(path=VECTORIZER_PATH):
    with open(path, "rb") as file:
        cv = joblib.load(file)
    return cv["cv"]


def initialize_count_vectorizer():
    cv = CountVectorizer()
    return cv


def extract_tweet_features(tweet: str) -> dict:
    # Initialize stopwords and stemmer

    sentiment_analyzer = SentimentIntensityAnalyzer()

    features = {}

    # Text Length
    features["ft_charcount"] = len(tweet)
    features["ft_wordcount"] = len(tweet.split())

    # Number of Hashtags
    features["ft_hashtagcount"] = len(re.findall(r"#(\w+)", tweet))

    # Number of Mentions
    features["ft_mentioncount"] = len(re.findall(r"@(\w+)", tweet))

    # Number of URLs
    features["ft_urlcount"] = len(re.findall(r"http\S+|www\.\S+", tweet))

    # Number of Emojis
    features["ft_emojicount"] = len([c for c in tweet if c in emoji.EMOJI_DATA])

    # Sentiment Score
    sentiment_scores = sentiment_analyzer.polarity_scores(tweet)
    features["ft_sent_compound"] = sentiment_scores["compound"]
    features["ft_sent_positive"] = sentiment_scores["pos"]
    features["ft_sent_neutral"] = sentiment_scores["neu"]
    features["ft_sent_negative"] = sentiment_scores["neg"]

    # Part-of-Speech (POS) Tags
    # words = nltk.word_tokenize(tweet)
    # pos_tags = nltk.pos_tag(words)
    # pos_counts = Counter(tag for word, tag in pos_tags)
    # features.update(pos_counts)

    # Presence of Specific Words
    # keywords = ["Hate", "Kill", "Shoot"]
    # for keyword in keywords:
    #     features[f"ft_cont_{keyword}"] = keyword.lower() in tweet.lower()

    # Number of Stopwords
    # features["stopword_count"] = len(
    #     [word for word in words if word.lower() in stopword]
    # )

    # Word Frequency
    # word_freq = Counter(words)
    # features.update(word_freq)

    # Presence of Hate Speech Words
    # hate_speech_count = sum(word.lower() in hate_speech_words
    #                                               for word in words)
    # features["hate_speech_word_count"] = hate_speech_count

    return features


if __name__ == "__main__":
    df = pd.read_csv("data/processed/hate_speech_data_processed.csv")
    features = df[["tweet", "labels"]]
    features.to_csv("data/processed/features.csv", index=True)
    print("features generated")
