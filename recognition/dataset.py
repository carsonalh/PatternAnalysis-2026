from functools import cache, partial
from pathlib import Path
from urllib.request import urlopen
import pandas as pd
import logging
from zipfile import ZipFile
import torch
from torch.utils.data import Dataset
from typing import Callable
import numpy as np

logger = logging.getLogger(__name__)


_ZIP_URL = "https://php.lobsterdata.com/info/sample/LOBSTER_SampleFile_AMZN_2012-06-21_10.zip"
_ZIP_FILENAME = "LOBSTER_SampleFile_AMZN_2012-06-21_10.zip" 
_ORDER_BOOK_LEVELS = 10
_MESSAGE_BOOK_COLUMN_NAMES = ("time" , "type" , "order_id" , "size" , "price" , "trade_direction")

def download_zip(
    url: str =_ZIP_URL,
    dir: Path = Path("data/"),
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
        order_book_names = []
        for i in range(1, _ORDER_BOOK_LEVELS + 1):
            order_book_names.append(f"ASKp{i}")
            order_book_names.append(f"ASKs{i}")
            order_book_names.append(f"BIDp{i}")
            order_book_names.append(f"BIDs{i}")
        books = pd.read_csv(archive.open(book_file), header=None, names=order_book_names)
        messages = pd.read_csv(archive.open(message_file), header=None, names=_MESSAGE_BOOK_COLUMN_NAMES)
    return books, messages


class LOBSTERLevel10Dataset(Dataset):
    def __init__(self, train: bool = True) -> None:
        books, _messages = load_dfs()
        VALIDATION_RATIO = 0.15
        validation_begin = round((1 - VALIDATION_RATIO) * len(books))
        if train:
            self.books = books[:validation_begin]
        else:
            self.books = books[validation_begin:]

    def __len__(self) -> int:
        return len(self.books)

    @cache
    def __getitem__(self, index: int) -> torch.Tensor:
        prev_index = max(index - 1, 0)
        prev_row = self.books.loc[prev_index]
        row = self.books.loc[index]

        midpoint = lambda r: 0.5 * (r['ASKp1'] + r['BIDp1'])
        spread = lambda r: r['ASKp1'] - r['BIDp1']

        ask_price_diff = lambda n, r: r[f'ASKp{n+1}'] - r[f'ASKp{n}']
        bid_price_diff = lambda n, r: r[f'BIDp{n}'] - r[f'BIDp{n+1}']
        volume = lambda k, n, r: r[f'{k}s{n}']

        return torch.Tensor([
            LOBSTERLevel10Dataset._log_return(midpoint, row, prev_row),
            LOBSTERLevel10Dataset._log_return(spread, row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(ask_price_diff, 1), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(ask_price_diff, 2), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(ask_price_diff, 3), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(ask_price_diff, 4), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(ask_price_diff, 5), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(ask_price_diff, 6), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(ask_price_diff, 7), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(ask_price_diff, 8), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(ask_price_diff, 9), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(bid_price_diff, 1), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(bid_price_diff, 2), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(bid_price_diff, 3), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(bid_price_diff, 4), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(bid_price_diff, 5), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(bid_price_diff, 6), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(bid_price_diff, 7), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(bid_price_diff, 8), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(bid_price_diff, 9), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'ASK',  1), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'ASK',  2), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'ASK',  3), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'ASK',  4), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'ASK',  5), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'ASK',  6), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'ASK',  7), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'ASK',  8), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'ASK',  9), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'ASK', 10), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'BID',  1), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'BID',  2), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'BID',  3), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'BID',  4), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'BID',  5), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'BID',  6), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'BID',  7), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'BID',  8), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'BID',  9), row, prev_row),
            LOBSTERLevel10Dataset._log_return(partial(volume, 'BID', 10), row, prev_row),
        ])

    @staticmethod
    def feature_sequence_to_df(tensor: torch.Tensor, initial_book: pd.Series) -> pd.DataFrame:
        match tensor.shape:
            case (_outer, 40):
                pass
            case _:
                raise AssertionError('expected tensor to be of shape (n, 40)')

        # Tensor was in log-returns space, move to returns space
        # Then with cumulative product we can reconstruct the full sequence
        tensor = torch.exp(tensor)

        cum_midpoint_returns = torch.cumprod(tensor[:, 0], dim=0)
        cum_spread_returns = torch.cumprod(tensor[:, 1], dim=0)

        ask_diff_returns = torch.cumprod(tensor[:, 2:11], dim=0)

        initial_ask_diffs = torch.as_tensor(
            [initial_book[f'ASKp{i+1}'] - initial_book[f'ASKp{i}'] for i in range(1, 10)],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        ask_diffs = initial_ask_diffs[None, :] * ask_diff_returns

        bid_diff_returns = torch.cumprod(tensor[:, 11:20], dim=0)

        initial_bid_diffs = torch.as_tensor(
            [initial_book[f'BIDp{i}'] - initial_book[f'BIDp{i+1}'] for i in range(1, 10)],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        bid_diffs = initial_bid_diffs[None, :] * bid_diff_returns

        initial_midpoint = 0.5 * (initial_book['ASKp1'] + initial_book['BIDp1'])
        initial_spread = initial_book['ASKp1'] - initial_book['BIDp1']

        ask_volume_returns = torch.cumprod(tensor[:, 20:30], dim=0)
        bid_volume_returns = torch.cumprod(tensor[:, 30:40], dim=0)

        initial_ask_volumes = torch.as_tensor(
            [initial_book[f'ASKs{i}'] for i in range(1, 11)],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        initial_bid_volumes = torch.as_tensor(
            [initial_book[f'BIDs{i}'] for i in range(1, 11)],
            dtype=tensor.dtype,
            device=tensor.device,
        )

        ask_volumes = initial_ask_volumes[None, :] * ask_volume_returns
        bid_volumes = initial_bid_volumes[None, :] * bid_volume_returns

        midpoints = initial_midpoint * cum_midpoint_returns
        spreads = initial_spread * cum_spread_returns

        asks_p1 = midpoints + 0.5 * spreads
        bids_p1 = midpoints - 0.5 * spreads

        asks = torch.cat((asks_p1[:, None], ask_diffs), dim=1)
        asks = torch.cumsum(asks, dim=1)

        bids = torch.cat((bids_p1[:, None], bid_diffs), dim=1)
        bids = torch.cumsum(torch.cat((bids[:, 0, None], -bids[:, 1:]), dim=1), dim=1)

        columns = {}
        for i in range(1, 11):
            columns[f'ASKp{i}'] = asks[:, i - 1].detach().cpu().numpy()
            columns[f'ASKs{i}'] = ask_volumes[:, i - 1].detach().cpu().numpy()
            columns[f'BIDp{i}'] = bids[:, i - 1].detach().cpu().numpy()
            columns[f'BIDs{i}'] = bid_volumes[:, i - 1].detach().cpu().numpy()

        return pd.DataFrame(columns)


    @staticmethod
    def _log_return(fn: Callable[[object], float], current: object, previous: object) -> float:
        current_value = fn(current)
        previous_value = fn(previous)
        epsilon = 1e-12
        if abs(previous_value) < epsilon:
            previous_value = (np.sign(previous_value) or 1.0) * epsilon
        return np.log(current_value / previous_value)
