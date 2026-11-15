from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
CSV_NAME = "MetroPT3(AirCompressor).csv"


def download_dataset(destination: Path, url: str = DATA_URL) -> Path:
    """Download MetroPT-3 from UCI and extract its CSV without persisting the ZIP."""
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / CSV_NAME
    if output.exists():
        return output
    with urllib.request.urlopen(url, timeout=90) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise RuntimeError(f"Expected one CSV in UCI archive, found {csv_members}")
        output.write_bytes(archive.read(csv_members[0]))
    return output


def load_dataset(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    expected = {"timestamp", "TP2", "TP3", "Oil_temperature", "Motor_current"}
    missing = expected.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    return data.sort_values("timestamp").set_index("timestamp")
