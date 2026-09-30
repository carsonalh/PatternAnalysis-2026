from functools import cache, partial
from pathlib import Path
from urllib.request import urlopen
import pandas as pd
import logging
from zipfile import ZipFile
import torch
from torch.utils.data import Dataset
from typing import Callable, ClassVar
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
    # Pre-computed from training data set
    LOG_MIDPOINT_MEAN: ClassVar = 14.617649918846451
    LOG_MIDPOINT_STD: ClassVar = 0.005719877315309554

    def __init__(self, train: bool = True, normalize: bool = False) -> None:
        books, _messages = load_dfs()
        self.normalize = normalize
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
        row = self.books.loc[index]
        midpoint = np.log(0.5 * (row['ASKp1'] + row['BIDp1']))
        midpoint_z = (midpoint - self.LOG_MIDPOINT_MEAN) / self.LOG_MIDPOINT_STD
        return torch.Tensor([midpoint_z])
