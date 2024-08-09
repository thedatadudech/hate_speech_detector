import nltk
import pandas as pd
from hate_speech_detector.utils.data_preparation.cleaning import extract_tweet_features


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    features_df = (
        data["tweet"].apply(extract_tweet_features).apply(pd.Series).fillna(0)
    )
    
    features_df = features_df.applymap(lambda x: int(x) if isinstance(x, bool) else x)
    data = pd.concat([data, features_df], axis=1)


    return data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'