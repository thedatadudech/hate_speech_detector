import re
from string import punctuation
import joblib
import mlflow
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from config import MLFLOW_TRACKING_URI


def clean(text):
    stopword = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")
    text = str(text).lower()
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    text = [word for word in text.split(" ") if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(" ")]
    text = " ".join(text)
    return text


hate_speech_words = [
    "abuse",
    "bigot",
    "bully",
    "discriminate",
    "extremist",
    "hate",
    "homophobe",
    "incite",
    "intolerant",
    "racist",
    "supremacist",
    "terrorist",
    "violence",
    "xenophobe",
]


def load_binary(path):
    return joblib.load(path)


def load_test_data(path):
    testdata = load_binary(path)
    return testdata["X_test"], testdata["y_test"]


def load_mlflow_client():
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    return client


def generate_testdata(X_test, y_test, path):
    testdata = dict()
    testdata["X_test"] = X_test
    testdata["y_test"] = y_test
    with open(path, "wb") as file:
        joblib.dump(testdata, file)
