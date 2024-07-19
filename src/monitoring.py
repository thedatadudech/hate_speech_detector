from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def monitor_model(df):
    data_drift_report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )
    df = df.filter(regex=r"^(ft_|labels)", axis=1)
    data_drift_report.run(
        current_data=df.iloc[:10000],
        reference_data=df.iloc[10000:],
        column_mapping=None,
    )
    data_drift_report
    data_drift_report.save_html("monitoring/drift_report.html")


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/processed/hate_speech_data_processed.csv")
    monitor_model(df)
