from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


if "data_exporter" not in globals():
    from mage_ai.data_preparation.decorators import data_exporter



@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    X = data.filter(like='ft_')
    y = data["labels"]


    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)


    

    
    X_train, X_test, y_train, y_test = train_test_split(
      X_resampled, y_resampled, test_size=0.2, random_state=42)
    

    # Specify your data exporting logic here
    return X_resampled, y_resampled, X_train, y_train, X_test, y_test
    
  