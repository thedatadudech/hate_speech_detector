from typing import Dict, Tuple
from pandas import Series
from sklearn.base import BaseEstimator
from scipy.sparse._csr import csr_matrix

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from hate_speech_detector.utils.data_preparation.cleaning import clean


DEFAULT_INPUTS = [
    {
      "Trump is a bitch ass motherfucker",
    },
    {
      "What is the fucking mather you shit ass bastard",
    },
    {
      "I love eating pizza",
    },
]

@custom
def predict(model_sklearn : Tuple[BaseEstimator, Dict[str, str]],
training_set: Tuple[
    csr_matrix,
    csr_matrix,
    csr_matrix,
    Series,
    Series,
    Series,
    BaseEstimator,
],
**kwargs
):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    inputs : List[str] = kwargs.get('inputs', DEFAULT_INPUTS)

    
    # Specify your custom logic here
    cls, info_dict = model_sklearn["sklearn"]
    _,_,_,_,_,_,cv = training_set['build']

    output1 = list(map(clean, inputs))

    output2 = cv.transform(output1)

    prediction= list(cls.predict(output2))

    
    print(inputs)
    print(prediction)
    return cls, info_dict, inputs, output1,  output2, prediction


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
