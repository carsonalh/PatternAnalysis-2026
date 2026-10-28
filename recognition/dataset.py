from pathlib import Path
from urllib.request import urlopen
import pandas as pd
import logging
from zipfile import ZipFile
import torch
from torch.utils.data import Dataset
from collections.abc import Sequence
from typing import Final
import numpy as np

logger = logging.getLogger(__name__)


_ZIP_URL = "https://php.lobsterdata.com/info/sample/LOBSTER_SampleFile_AMZN_2012-06-21_10.zip"
_ZIP_FILENAME = "LOBSTER_SampleFile_AMZN_2012-06-21_10.zip" 
_ORDER_BOOK_LEVELS = 10
FEATURE_DIMS = 4 * _ORDER_BOOK_LEVELS
ORDER_BOOK_COLUMNS = tuple(
    f"{side}{field}{level}"
    for level in range(1, _ORDER_BOOK_LEVELS + 1)
    for side, field in (("ASK", "p"), ("ASK", "s"), ("BID", "p"), ("BID", "s"))
)
_MESSAGE_BOOK_COLUMN_NAMES = ("time" , "type" , "order_id" , "size" , "price" , "trade_direction")

def download_zip(
    url: str =_ZIP_URL,
    dir: Path = Path(__file__).resolve().parent / "data",
    filename =_ZIP_FILENAME,
    cache: bool = True,
) -> Path:
    path = dir / filename
    if cache and path.is_file():
        return path
    dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading file URL %s to local path %s", url, str(path))
    with open(path, "wb") as local_file, urlopen(url) as remote_file:
        local_file.write(remote_file.read())
    logger.info("Download complete")
    return path

def load_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = download_zip()
    with ZipFile(path) as archive:
        files = archive.namelist()
        book_file = next(name for name in files if name.endswith('_orderbook_10.csv'))
        message_file = next(name for name in files if name.endswith('_message_10.csv'))
        books = pd.read_csv(archive.open(book_file), header=None, names=ORDER_BOOK_COLUMNS)
        messages = pd.read_csv(archive.open(message_file), header=None, names=_MESSAGE_BOOK_COLUMN_NAMES)
    return books, messages


def positive_transform(values):
    """Log coordinates for strictly positive physical features."""
    return np.log(np.maximum(values, 1e-8))


def positive_inverse(values: torch.Tensor) -> torch.Tensor:
    # Bound only numerical extremes before exponentiating.
    upper = np.log(torch.finfo(values.dtype).max) - 1
    return values.clamp(min=np.log(1e-8), max=upper).exp()


class LOBSTERLevel10Dataset(Dataset, Sequence):
    _VALIDATION_RATIO: Final[float] = 0.15

    def __init__(self, train: bool = True) -> None:
        books, messages = load_dfs()
        validation_begin = round((1 - LOBSTERLevel10Dataset._VALIDATION_RATIO) * len(books))

        ask_gaps = [positive_transform(books[f"ASKp{i}"] - books[f"ASKp{i - 1}"]).to_numpy() for i in range(2, 11)]
        bid_gaps = [positive_transform(books[f"BIDp{i - 1}"] - books[f"BIDp{i}"]).to_numpy() for i in range(2, 11)]
        spreads = positive_transform(books["ASKp1"] - books["BIDp1"]).to_numpy()
        midpoints = np.log(0.5 * (books["ASKp1"] + books["BIDp1"])).to_numpy()

        self.data = torch.tensor(np.array([midpoints, spreads, *ask_gaps, *bid_gaps])).T
        self.data = torch.cat((self.data, torch.from_numpy(positive_transform(books[[c for c in ORDER_BOOK_COLUMNS if "s" in c]].to_numpy(dtype=float)))), dim=1)

        # Use log returns on midpoint prices, instead of the log prices themselves
        returns = self.data[1:, 0] - self.data[:-1, 0]
        self.data = self.data[1:].clone()
        self.data[:, 0] = returns
        validation_begin -= 1

        train_data = self.data[:validation_begin]
        train_mean, train_std = train_data.mean(0), train_data.std(0).clamp_min(1e-8)
        self.mean, self.std = train_mean, train_std

        if train:
            self.data = (self.data[:validation_begin] - train_mean) / train_std
        else:
            self.data = (self.data[validation_begin:] - train_mean) / train_std

    def __len__(self) -> None:
        return len(self.data)

    def __getitem__(self, index: int | range) -> None:
        return self.data[index]


class WindowDataset(Dataset):
    def __init__(self, data: Sequence[torch.Tensor], window_size: int = 64, stride: int = 1):
        if data[0].ndim != 1 or window_size < 2 or stride < 1:
            raise ValueError()
        self.data = data
        self.window_size = window_size
        self.stride = stride

    def __len__(self) -> int:
        return max(0, (len(self.data) - self.window_size + self.stride - 1) // self.stride)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)

        index *= self.stride
        return self.data[index:index + self.window_size], self.data[index + self.window_size]
