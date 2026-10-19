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


def sample_order_book(books: pd.DataFrame, messages: pd.DataFrame,
                      interval_seconds: float) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Previous-tick snapshots on a clock grid; never select a future event.

    Empty intervals carry the latest book forward. For simultaneous messages,
    the last row is the book after all events at that timestamp.
    """
    if not np.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("Sampling interval must be finite and positive")
    if len(books) != len(messages) or len(books) < 2 or "time" not in messages:
        raise ValueError("Sampling requires aligned books and event timestamps")
    times = messages["time"].to_numpy(dtype=float)
    if not np.isfinite(times).all() or (np.diff(times) < 0).any():
        raise ValueError("Event timestamps must be finite and nondecreasing")
    first = int(np.ceil(times[0] / interval_seconds))
    last = int(np.floor(times[-1] / interval_seconds))
    grid = np.arange(first, last + 1) * interval_seconds
    grid = grid[(grid >= times[0]) & (grid <= times[-1])]
    if len(grid) < 2:
        raise ValueError("Sampling must leave at least two snapshots")
    source_indices = np.searchsorted(times, grid, side="right") - 1
    return books.iloc[source_indices].reset_index(drop=True), grid, source_indices


class LOBSTERLevel10Dataset(Dataset):
    """Midpoint log returns and book structure, scaled using training rows only."""

    FEATURE_DIMS: ClassVar = FEATURE_DIMS
    TICK_SIZE: ClassVar = 100  # One cent in LOBSTER's 1/10000-dollar price units.
    FEATURE_NAMES: ClassVar = (
        ("log_midpoint_return", "log1p_spread")
        + tuple(f"log1p_ask_gap_{i}" for i in range(2, _ORDER_BOOK_LEVELS + 1))
        + tuple(f"log1p_bid_gap_{i}" for i in range(2, _ORDER_BOOK_LEVELS + 1))
        + tuple(f"log1p_ask_size_{i}" for i in range(1, _ORDER_BOOK_LEVELS + 1))
        + tuple(f"log1p_bid_size_{i}" for i in range(1, _ORDER_BOOK_LEVELS + 1))
    )

    def __init__(self, train: bool = True, normalize: bool = True,
                 *, sampling_interval_seconds: float | None = None) -> None:
        books, messages = load_dfs()
        self.normalize = normalize
        self.sampling_interval_seconds = sampling_interval_seconds
        self.data_config = {"sampling_interval_seconds": sampling_interval_seconds,
                            "split_policy": "original_event_cutoff"}
        VALIDATION_RATIO = 0.15
        validation_begin = round((1 - VALIDATION_RATIO) * len(books))
        self.original_event_count = len(books)
        times = (messages["time"].to_numpy(dtype=float)
                 if "time" in messages and len(messages) == len(books) else None)
        self.split_time_seconds = float(times[validation_begin]) if times is not None else None
        source_indices = np.arange(len(books))
        if sampling_interval_seconds is not None:
            books, snapshot_times, source_indices = sample_order_book(
                books, messages, sampling_interval_seconds)
            # Preserve the old chronological boundary, not 85% of the new row count.
            validation_begin = int(np.searchsorted(snapshot_times, self.split_time_seconds))
        else:
            snapshot_times = times
        if not 0 < validation_begin < len(books):
            raise ValueError("Sampling must leave nonempty training and validation splits")
        self.snapshot_count = len(books)

        asks = books[[f"ASKp{i}" for i in range(1, _ORDER_BOOK_LEVELS + 1)]].to_numpy()
        bids = books[[f"BIDp{i}" for i in range(1, _ORDER_BOOK_LEVELS + 1)]].to_numpy()
        ask_sizes = books[[f"ASKs{i}" for i in range(1, _ORDER_BOOK_LEVELS + 1)]].to_numpy()
        bid_sizes = books[[f"BIDs{i}" for i in range(1, _ORDER_BOOK_LEVELS + 1)]].to_numpy()
        midpoints = 0.5 * (asks[:, 0] + bids[:, 0])
        # r_t = log(m_t / m_{t-1}). Compute before splitting so the first
        # validation return uses the last training price, never a future event.
        log_returns = np.r_[0.0, np.diff(np.log(midpoints))]
        features = np.column_stack((
            log_returns,
            np.log1p((asks[:, 0] - bids[:, 0]) / self.TICK_SIZE),
            np.log1p(np.diff(asks, axis=1) / self.TICK_SIZE),
            np.log1p(-np.diff(bids, axis=1) / self.TICK_SIZE),
            np.log1p(ask_sizes),
            np.log1p(bid_sizes),
        ))
        self.reference_midpoint = float(np.median(midpoints[:validation_begin]))
        self.feature_mean = features[:validation_begin].mean(axis=0)
        self.feature_std = features[:validation_begin].std(axis=0)
        self.feature_std[self.feature_std == 0] = 1

        begin, end = (0, validation_begin) if train else (validation_begin, len(books))
        # Anchor for reconstructing this split; the first day event has return zero.
        self.initial_midpoint = float(midpoints[max(0, begin - 1)])
        self.books = books.iloc[begin:end]
        self.timestamps = snapshot_times[begin:end] if snapshot_times is not None else None
        self.source_indices = source_indices[begin:end]
        self.source_timestamps = times[self.source_indices] if times is not None else None
        features = features[begin:end]
        if self.normalize:
            features = (features - self.feature_mean) / self.feature_std
        self.features = torch.from_numpy(features.astype(np.float32))

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.FEATURE_NAMES

    def __len__(self) -> int:
        return len(self.books)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.features[index]

    def feature_sequence_to_df(
        self,
        features: torch.Tensor | np.ndarray,
        *, initial_midpoint: float | None = None,
    ) -> pd.DataFrame:
        """Reconstruct a (T, 40) feature array in original LOBSTER units.

        Normalization is undone automatically. Midpoint log returns recover
        m_t = m_anchor * exp(cumsum(r)_t).
        initial_midpoint is in LOBSTER price units; standalone log-return
        windows default to the training median price.
        Prices are rounded to ticks; spread, gaps, and sizes are at least one.
        The returned DataFrame has a fresh RangeIndex.
        """
        return features_to_order_book(
            features,
            self.feature_mean if self.normalize else None,
            self.feature_std if self.normalize else None,
            initial_midpoint=self.reference_midpoint if initial_midpoint is None else initial_midpoint,
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
    *, initial_midpoint: float | None = None,
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
    if initial_midpoint is None or not np.isfinite(initial_midpoint) or initial_midpoint <= 0:
        raise ValueError("Decoding log returns requires a positive initial midpoint in LOBSTER units")
    values = values.copy()
    values[:, 0] = np.log(initial_midpoint) + np.cumsum(values[:, 0])

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
