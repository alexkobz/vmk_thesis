import pandas as pd
from config import config
from config import columns
from loguru import logger


class DataReader:
    def __init__(self, filter_secids_size: int = 5000):
        self.filter_secids_size = filter_secids_size

    def _read(self):
        raise NotImplementedError()

    def _transform(self, df):
        raise NotImplementedError()

    def _filter_cols(self, df, cols: list[str]) -> pd.DataFrame:
        raise NotImplementedError()

    def read_data(self) -> pd.DataFrame:
        logger.info(f"Start reading data")
        df = self._read()
        df = self._transform(df)
        df = self._filter_cols(df)
        logger.info(f"Shape: {df.shape}")
        return df


class CsvReader(DataReader):

    def __init__(self, filter_secids_size: int = 5000, cols: list[str] = None):
        super().__init__(filter_secids_size)
        if cols is None:
            cols = columns.base_cols
        self.cols = cols


    def _read(self) -> pd.DataFrame:
        df = pd.read_csv(
            config.DATA_DIR / 'processed' / 'dataset.csv',
            sep='\t',
            dtype=columns.dtype_dict,
            usecols=list(columns.dtype_dict.keys()),
        )
        return df

    def _transform(self, df):
        df['tradedate'] = pd.to_datetime(df['tradedate'])
        df = df.set_index(columns.index).sort_index()
        secid_counts = df.index.get_level_values(0).value_counts()
        df = df[df.index.get_level_values(0).isin(
            secid_counts[secid_counts > self.filter_secids_size].index
        )]
        return df

    def _filter_cols(self, df, cols: list[str] = None) -> pd.DataFrame:
        if cols is None:
            return df
        return df[cols]

    def read_data(self) -> pd.DataFrame:
        df = super().read_data()
        df = self._transform(df)
        return df



class BaseCsvReader(CsvReader):


    def read_data(self) -> pd.DataFrame:
        df = super().read_data()
        df = self._filter_cols(df, columns.base_cols)
        return df

class SelectedCsvReader(CsvReader):


    def read_data(self) -> pd.DataFrame:
        df = super().read_data()
        df = self._filter_cols(df, columns.feat_cols)
        return df
