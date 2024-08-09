import joblib
from pandas import DataFrame, Series

from hate_speech_detector.utils.data_preparation.cleaning import extract_tweet_features

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


DEFAULT_INPUTS = [
    {
        "We have to put Trump on the bullseye",
    },
    {
        "Romeo must die",
    },
    {
        "Its an inanimate fucking object",
    },
]



@data_loader
def load_data(*args, **kwargs):
    """
    Template code for loading data from any source.

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    inputs = kwargs.get('inputs', DEFAULT_INPUTS)

    input_df = DataFrame(inputs, columns=['tweet'])
    
    X_test = input_df["tweet"].apply(extract_tweet_features).apply(Series).fillna(0)


    bestmodel_path = "/data/best_model/model/best_model_fittedX.pkl"

    model = joblib.load(bestmodel_path)
    prediction = DataFrame(model.predict(X_test), columns=['predictions'])


    return model, input_df, prediction

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'