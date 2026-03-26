# Читаемая последовательность очистки и импутации
import numpy as np

def normalize_and_filter(df, y_name):
    df = df.copy()
    df['type'] = df['type'].fillna('common_share').astype('category')
    df = df[df['type'].isin(['common_share', 'preferred_share'])]
    df['sector'] = df['sector'].astype('category')
    df['sector'].replace('', np.nan, inplace=True)
    df[y_name].replace(0, np.nan, inplace=True)
    df['close'].replace(0, np.nan, inplace=True)
    df['volume'].replace(0, np.nan, inplace=True)
    df['outstanding_shares'] = df[y_name] / df['close']
    df.sort_values(['secid', 'tradedate', 'boardid'], inplace=True)
    return df

def impute_target(df, y_name):
    df = df.copy()
    df[y_name] = df.groupby('secid')[y_name].ffill()
    df[y_name] = df.groupby('secid')[y_name].bfill()
    return df

def impute_outstanding_shares(df, y_name):
    df = df.copy()
    df['outstanding_shares'] = df.groupby('secid')['outstanding_shares'].ffill()
    df['outstanding_shares'] = df.groupby('secid')['outstanding_shares'].bfill()
    return df

def impute_close(df):
    df = df.copy()
    df['close'] = df.groupby('secid')['close'].ffill()
    df['close'] = df.groupby('secid')['close'].bfill()
    return df

def apply_issuesize(df):
    df = df.copy()
    size = df['issuesize'] > 0
    df.loc[size, 'outstanding_shares'] = df.loc[size, 'outstanding_shares'].combine_first(df.loc[size, 'issuesize'])
    return df

def impute_volume(df):
    df = df.copy()
    df['volume'] = df.groupby('secid')['volume'].ffill()
    df['volume'] = df.groupby('secid')['volume'].bfill()
    return df

def fill_remaining_cap(df, y_name):
    df = df.copy()
    mask = df[y_name].isna() & df['close'].notna()
    df.loc[mask, 'outstanding_shares'] = df['outstanding_shares'].mean()
    mask = df[y_name].isna() & df['close'].notna() & df['outstanding_shares'].notna()
    df.loc[mask, y_name] = df.loc[mask, 'close'] * df.loc[mask, 'outstanding_shares']
    return df

def filter_zero_target(df, y_name):
    df = df.copy()
    df = df[df[y_name] > 1e-6]
    return df
