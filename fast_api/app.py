import requests
import joblib
import os
import re
import emoji
from pandas import DataFrame
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from nltk.sentiment import SentimentIntensityAnalyzer


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

    # Presence of Specific Words
    keywords = [
        "Hate",
        "Kill",
        "Shoot",
        "Fuck",
        "Motherfucker",
        "Scumbag",
        "Filthy",
        "Morone",
        "Sucker",
        "Bitch",
        "Cunt",
        "Slut",
        "Monkey",
        "Butthole",
        "Asshole",
        "Dushbag",
        "Terrorist",
        "Nazi",
    ]
    for keyword in keywords:
        features[f"ft_cont_{keyword}"] = keyword.lower() in tweet.lower()

    return features


# Modell- und Vektorisierer-Pfade
bestmodel_path = os.getenv(
    "BESTMODEL_PATH",
    "https://hatespeechstorage.blob.core.windows.net/hatespeech-data/mlmodel/"
    "hate_speech_detector/best_model_fittedX.pkl"
    "?sp=r&st=2024-08-09T19:59:28Z&se=2024-08-23T03:59:28Z"
    "&spr=https&sv=2022-11-02&sr="
    "b&sig=s091yRCR%2BgO25Z%2FMG2dLl2XbAsYX74Qslk6KHpHTlmA%3D",
)


# function for loading models


def load_model(url_blob, local_filename):
    with requests.get(url_blob, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    model = joblib.load(local_filename)
    os.remove(local_filename)
    return model


# FastAPI-Initialisierung
app = FastAPI()

# Cross-Origin Resource Sharing (CORS) Konfiguration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update mit spezifischen Ursprüngen bei Bedarf
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lade das Modell und den Vektorisierer
model = load_model(bestmodel_path, "bestmodel.pkl")


# Pydantic-Modell für die Vorhersageanforderung
class PredictionRequest(BaseModel):
    inputs: str = Field(..., example="We should put Trump on the bullseye")


@app.get("/")
def read_root():
    """Redirect to Swagger UI"""
    return RedirectResponse(url="/docs")


@app.post("/predict", response_model=dict)
def predict(request: PredictionRequest):
    text = request.inputs
    X = extract_tweet_features(text)
    df = DataFrame([X])
    prediction = model.predict(df)[0]
    return {"received_text": text, "prediction": str(prediction)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=80)
