import pandas as pd
from src.features import make_features

def test_pre_failure_labels_do_not_include_failure_period():
    index = pd.date_range("2020-04-16", "2020-04-19", freq="1min")
    data = pd.DataFrame({name: 1.0 for name in ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]}, index=index)
    X, y, events = make_features(data)
    assert y.loc["2020-04-17 12:00"] == 1
    assert not ((X.index >= "2020-04-18") & (X.index <= "2020-04-18 23:59")).any()
    assert events.loc["2020-04-17 12:00"] == 1
