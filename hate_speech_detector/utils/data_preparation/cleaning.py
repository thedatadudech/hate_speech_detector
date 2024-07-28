import re
from string import punctuation
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")


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
