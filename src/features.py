from __future__ import annotations

import pandas as pd

FAILURES = [
    ("2020-04-18 00:00", "2020-04-18 23:59"),
    ("2020-05-29 23:30", "2020-05-30 06:00"),
    ("2020-06-05 10:00", "2020-06-07 14:30"),
    ("2020-07-15 14:30", "2020-07-15 19:00"),
]
ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]


def make_features(data: pd.DataFrame, prediction_horizon: str = "24h") -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Aggregate 1 Hz telemetry and label pre-failure windows without future leakage."""
    analog = [column for column in ANALOG if column in data.columns]
    means = data[analog].resample("10min").mean().add_suffix("__mean")
    stds = data[analog].resample("10min").std().add_suffix("__std")
    features = pd.concat([means, stds], axis=1).interpolate(limit=3).dropna()
    for column in means.columns:
        features[f"{column}__delta_1h"] = features[column].diff(6)
    features = features.dropna()
    target = pd.Series(0, index=features.index, dtype=int, name="failure_within_24h")
    event_id = pd.Series(0, index=features.index, dtype=int, name="event_id")
    horizon = pd.Timedelta(prediction_horizon)
    for number, (start_text, end_text) in enumerate(FAILURES, start=1):
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        target.loc[(target.index >= start - horizon) & (target.index < start)] = 1
        event_id.loc[(event_id.index >= start - horizon) & (event_id.index <= end)] = number
        features = features.loc[~((features.index >= start) & (features.index <= end))]
        target = target.reindex(features.index)
        event_id = event_id.reindex(features.index)
    return features, target.astype(int), event_id.astype(int)
