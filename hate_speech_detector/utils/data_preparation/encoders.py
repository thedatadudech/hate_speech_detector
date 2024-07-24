
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def initialize_count_vectorizer():
    cv = CountVectorizer()
    return cv


def vectorize_tweets(
    df=None,
    tweet_column="tweet",
    label_column="labels",
):
    cv = initialize_count_vectorizer()
    df = df.dropna()
    x = np.array(df[tweet_column])
    X = cv.fit_transform(x)
    y =df[label_column] 

  

    return X,y, cv

