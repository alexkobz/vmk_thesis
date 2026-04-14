import pandas as pd
from config import config
from config import columns
from loguru import logger


class DataReader:
    def __init__(self, filter_secids_size: int = 5000):
        self.filter_secids_size = filter_secids_size

    def _read(self, cols):
        raise NotImplementedError()

    def read_data(self) -> pd.DataFrame:
        logger.info(f"Start reading data")
        df = self._read()
        logger.info(f"Shape: {df.shape}")
        return df

class CsvReader(DataReader):

    def _read(self, cols: list[str] = None) -> pd.DataFrame:
        if cols is None:
            cols = list(columns.dtype_dict.keys())
        df = pd.read_csv(
            config.DATA_DIR / 'processed' / 'dataset.csv',
            sep='\t',
            dtype=columns.dtype_dict,
            usecols=cols,
        )
        df['tradedate'] = pd.to_datetime(df['tradedate'])
        df = df.set_index(columns.index).sort_index()
        secid_counts = df.index.get_level_values(0).value_counts()
        df = df[df.index.get_level_values(0).isin(
            secid_counts[secid_counts > self.filter_secids_size].index
        )]
        return df

class BaseCsvReader(CsvReader):

    def _read(self, cols: list[str] = None) -> pd.DataFrame:
        df = super()._read(cols=columns.base_cols)
        return df