from pathlib import Path
from urllib.request import urlopen
import pandas as pd
import logging
from zipfile import ZipFile
import torch
from torch.utils.data import Dataset
from typing import ClassVar
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


class LOBSTERLevel10Dataset(Dataset):
    """Event-time book features, optionally standardized using the training split."""

    FEATURE_DIMS: ClassVar = FEATURE_DIMS
    TICK_SIZE: ClassVar = 100  # One cent in LOBSTER's 1/10000-dollar price units.
    FEATURE_NAMES: ClassVar = (
        ("log_midpoint", "log1p_spread")
        + tuple(f"log1p_ask_gap_{i}" for i in range(2, _ORDER_BOOK_LEVELS + 1))
        + tuple(f"log1p_bid_gap_{i}" for i in range(2, _ORDER_BOOK_LEVELS + 1))
        + tuple(f"log1p_ask_size_{i}" for i in range(1, _ORDER_BOOK_LEVELS + 1))
        + tuple(f"log1p_bid_size_{i}" for i in range(1, _ORDER_BOOK_LEVELS + 1))
    )

    def __init__(self, train: bool = True, normalize: bool = True) -> None:
        books, _messages = load_dfs()
        self.normalize = normalize
        VALIDATION_RATIO = 0.15
        validation_begin = round((1 - VALIDATION_RATIO) * len(books))

        asks = books[[f"ASKp{i}" for i in range(1, _ORDER_BOOK_LEVELS + 1)]].to_numpy()
        bids = books[[f"BIDp{i}" for i in range(1, _ORDER_BOOK_LEVELS + 1)]].to_numpy()
        ask_sizes = books[[f"ASKs{i}" for i in range(1, _ORDER_BOOK_LEVELS + 1)]].to_numpy()
        bid_sizes = books[[f"BIDs{i}" for i in range(1, _ORDER_BOOK_LEVELS + 1)]].to_numpy()
        midpoints = 0.5 * (asks[:, 0] + bids[:, 0])
        features = np.column_stack((
            np.log(midpoints),
            np.log1p((asks[:, 0] - bids[:, 0]) / self.TICK_SIZE),
            np.log1p(np.diff(asks, axis=1) / self.TICK_SIZE),
            np.log1p(-np.diff(bids, axis=1) / self.TICK_SIZE),
            np.log1p(ask_sizes),
            np.log1p(bid_sizes),
        ))
        self.feature_mean = features[:validation_begin].mean(axis=0)
        self.feature_std = features[:validation_begin].std(axis=0)
        self.feature_std[self.feature_std == 0] = 1

        begin, end = (0, validation_begin) if train else (validation_begin, len(books))
        self.books = books.iloc[begin:end]
        features = features[begin:end]
        if self.normalize:
            features = (features - self.feature_mean) / self.feature_std
        self.features = torch.from_numpy(features.astype(np.float32))

    def __len__(self) -> int:
        return len(self.books)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.features[index]

    def feature_sequence_to_df(
        self,
        features: torch.Tensor | np.ndarray,
    ) -> pd.DataFrame:
        """Reconstruct a (T, 40) feature array in original LOBSTER units.

        Normalization is undone automatically; each row decodes independently.
        Prices are rounded to ticks; spread, gaps, and sizes are at least one.
        The returned DataFrame has a fresh RangeIndex.
        """
        return features_to_order_book(
            features,
            self.feature_mean if self.normalize else None,
            self.feature_std if self.normalize else None,
        )


class WindowDataset(Dataset):
    """Contiguous views into one already-split (events, features) tensor."""

    def __init__(self, features: torch.Tensor, sequence_length: int = 64, stride: int = 1):
        if features.ndim != 2 or sequence_length < 2 or stride < 1:
            raise ValueError("Expected 2D features, sequence_length >= 2, and stride >= 1")
        self.features = features
        self.sequence_length = sequence_length
        self.stride = stride

    def __len__(self) -> int:
        return max(0, (len(self.features) - self.sequence_length) // self.stride + 1)

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        begin = index * self.stride
        return self.features[begin:begin + self.sequence_length]


def features_to_order_book(
    features: torch.Tensor | np.ndarray,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> pd.DataFrame:
    """Decode (T, 40) features without loading data, e.g. from a saved model.

    Provide both training statistics for standardized inputs, or neither for
    unstandardized log features. Tick rounding and book constraints are applied
    only here, outside the differentiable recovery network.
    """
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIMS:
        raise ValueError(f"Expected a (T, {FEATURE_DIMS}) feature sequence")
    if not np.isfinite(values).all():
        raise ValueError("Features must be finite")
    if (feature_mean is None) != (feature_std is None):
        raise ValueError("Provide both normalization statistics or neither")
    if feature_mean is not None:
        values = values * feature_std + feature_mean

    tick_size = LOBSTERLevel10Dataset.TICK_SIZE
    with np.errstate(over='raise', invalid='raise'):
        midpoints = np.exp(values[:, 0])
        counts = np.maximum(1, np.rint(np.expm1(np.maximum(values[:, 1:], 0))))
    levels = _ORDER_BOOK_LEVELS
    spread = counts[:, 0]
    ask_gaps = counts[:, 1:levels]
    bid_gaps = counts[:, levels:2 * levels - 1]
    ask_sizes = counts[:, 2 * levels - 1:3 * levels - 1]
    bid_sizes = counts[:, 3 * levels - 1:4 * levels - 1]
    ask_offsets = np.column_stack((np.zeros(len(values)), np.cumsum(ask_gaps, axis=1)))
    bid_offsets = np.column_stack((np.zeros(len(values)), np.cumsum(bid_gaps, axis=1)))
    # Round the best bid, then add the integer spread so both quotes stay on ticks.
    best_bid = np.rint(midpoints / tick_size - 0.5 * spread)
    best_bid = np.maximum(best_bid, bid_offsets[:, -1] + 1)
    asks = (best_bid[:, None] + spread[:, None] + ask_offsets) * tick_size
    bids = (best_bid[:, None] - bid_offsets) * tick_size
    book = np.stack((asks, ask_sizes, bids, bid_sizes), axis=-1).reshape(-1, FEATURE_DIMS)
    if not np.isfinite(book).all() or (book >= 2**63).any():
        raise ValueError("Decoded book is outside the int64 range")
    return pd.DataFrame(book.astype(np.int64), columns=ORDER_BOOK_COLUMNS)
