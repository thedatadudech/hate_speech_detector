from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import numpy as np
import pandas as pd

np.seterr(divide="ignore", invalid="ignore")


if "data_exporter" not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def monitor_model(training_data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    _, y, _, _, _, _ = training_data["build2"]
    y_df = pd.DataFrame({"target": y})
    data_drift_report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    data_drift_report.run(
        current_data=y_df.iloc[:10000],
        reference_data=y_df.iloc[10000:],
        column_mapping=None,
    )
    data_drift_report
    data_drift_report.save_html("/data/artifacts/drift_report.html")
    print("Drift report saved")
    return True
