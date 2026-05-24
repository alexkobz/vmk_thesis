import pandas as pd

from config import DATA_DIR
from logs.logger import logger
from src.utils.read_yaml import (dtype_dict, dtype_dict_raw,
                                 processed_dataset_name, raw_dataset_name,
                                 tradedate, index_cols, y_log)

SEP = '\t'


class DataReader:
    def __init__(self, cols: list[str] | None = None):
        self.df = None
        self.cols = cols

    def _read(self) -> pd.DataFrame:
        raise NotImplementedError()

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[tradedate] = pd.to_datetime(df[tradedate])
        return df

    def read(self) -> pd.DataFrame:
        logger.info("Start reading data")
        df = self._read()
        df = self._transform(df)
        logger.info(f"Shape: {df.shape}")
        return df


class CsvReader(DataReader):
    def __init__(self, cols: list[str] | None = None):
        super().__init__(cols=cols)


class RawCsvReader(DataReader):
    def __init__(self, cols: list[str] | None = None):
        super().__init__(cols=cols)

    def _read(self) -> pd.DataFrame:
        df = pd.DataFrame(pd.read_csv(
            DATA_DIR / "raw" / raw_dataset_name,
            sep=SEP,
            dtype=dtype_dict_raw,
        ))
        return df


class ProcessedCsvReader(DataReader):
    def __init__(self, cols: list[str] | None = None):
        super().__init__(cols=cols)

    def _read(self) -> pd.DataFrame:
        df = pd.DataFrame(pd.read_csv(
            DATA_DIR / "processed" / processed_dataset_name,
            sep=SEP,
            dtype=dtype_dict,
        ))
        return df

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = super()._transform(df)
        df.set_index(index_cols, inplace=True)
        df = df[df[y_log].notna()]
        df.fillna(0, inplace=True)
        return df

def save_processed_csv(df: pd.DataFrame, filename: str = processed_dataset_name) -> None:
    df.to_csv(
        DATA_DIR / "processed" / filename,
        sep=SEP,
    )
