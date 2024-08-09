import re
from string import punctuation
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import emoji


def clean(text):
    stopword = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = [word for word in text.split(" ") if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(" ")]
    text = " ".join(text)
    return text


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
