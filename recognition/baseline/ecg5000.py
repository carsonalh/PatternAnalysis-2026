from typing import Literal
from urllib.request import urlopen
from pathlib import Path
import logging
import torch
import numpy as np
from zipfile import ZipFile

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_URL = "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip"
FILENAME = "ECG5000.zip"


class ECG5000Dataset(Dataset):
    def __init__(self, path: Path = DATA_DIR / FILENAME, cache: bool = True,
                 split: Literal["train", "test"] = "train") -> None:
        if not cache or not path.is_file():
            logger.info("downloading ECG5000 dataset")
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as file, urlopen(DATA_URL) as remote:
                file.write(remote.read())
            logger.info("ECG5000 dataset downloaded successfully")

        with ZipFile(path) as archive:
            with archive.open(f"ECG5000_{split.upper()}.txt") as file:
                rows = np.loadtxt(file, dtype=np.float32, ndmin=2)
        self.series = torch.from_numpy(rows[:, 1:].copy()).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.series[index]



class ECG5000WindowedDataset(Dataset):
    def __init__(self, path: Path = DATA_DIR / FILENAME, cache: bool = True, split: Literal["train", "test"] = "train", window_size: int = 30) -> None:
        self._dataset = ECG5000Dataset(path=path, cache=cache, split=split)
        self._window_size = window_size

    def __len__(self) -> int:
        return len(self._dataset) * self._pairs_per_time_series

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        time_series_index = index // self._pairs_per_time_series
        time_series_offset = index % self._pairs_per_time_series
        time_series = self._dataset[time_series_index]
        window = time_series[time_series_offset:time_series_offset + self._window_size]
        label = time_series[time_series_offset + 1:time_series_offset + self._window_size + 1]
        return window, label

    @property
    def _pairs_per_time_series(self) -> int:
        """How many window-label pairs per time series from the original dataset."""
        return len(self._dataset[0]) - self._window_size
