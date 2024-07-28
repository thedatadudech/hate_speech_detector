import os
import sys
import pandas as pd
from src.cli_version.data_preparation import preprocess_data

sys.path.append(os.getenv("SRC_PATH"))


def test_preprocess_data():
    # Create a sample DataFrame for testing
    df = pd.DataFrame(
        {"tweet": ["This is a tweet", "Another tweet"], "class": [0, 1]}
    )

    # Call the preprocess_data function
    processed_df = preprocess_data(df)

    # Assert the expected output
    expected_df = pd.DataFrame(
        {
            "tweet": ["tweet", "anoth tweet"],
            "labels": ["Hate Speech", "Offensive Language"],
            "ft_charcount": [15.0, 13.0],
            "ft_wordcount": [4.0, 2.0],
            "ft_hashtagcount": [0.0, 0.0],
            "ft_mentioncount": [0.0, 0.0],
            "ft_urlcount": [0.0, 0.0],
            "ft_emojicount": [0.0, 0.0],
            "ft_sent_compound": [0.0, 0.0],
            "ft_sent_positive": [0.0, 0.0],
            "ft_sent_neutral": [1.0, 1.0],
            "ft_sent_negative": [0.0, 0.0],
        }
    )
    pd.testing.assert_frame_equal(processed_df, expected_df)
