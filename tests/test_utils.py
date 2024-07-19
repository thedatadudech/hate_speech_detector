import os, sys

sys.path.append(os.getenv("SRC_PATH"))

from utils import clean


def test_clean():
    text = "This is a sample text with some stopwords and punctuation!"
    expected_result = "sampl text stopword punctuat"
    assert clean(text) == expected_result
