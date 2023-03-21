import pandas as pd
import numpy as np
import collections
import polars as pl
import re

class PandasColumn:
    def __init__(self, column):
        self._value = column
    
    def __dlpack__(self):
        return self._value.__dlpack__()
    
    def isnull(self):
        return PandasColumn(self.value.isna())

    def notnull(self):
        return PandasColumn(self.value.notna())
    
    def any(self):
        return self.value.any()

    def all(self):
        return self.value.all()
    
    def __len__(self):
        return len(self.value)

class PolarsColumn:
    def __init__(self, column):
        self.value = column
    
    def isnull(self):
        return PolarsColumn(self.value.isna())

    def notnull(self):
        return PolarsColumn(self.value.notna())
    
    def any(self):
        return self.value.any()

    def all(self):
        return self.value.all()
    
    def __len__(self):
        return len(self.value)


class PandasGroupBy:
    def __init__(self, df, keys):
        self.df = df
        self.keys = keys

    def _validate_result(self, result):
        failed_columns = self.df.columns.difference(result.columns)
        if len(failed_columns) > 0:
            raise RuntimeError(
                "Groupby operation could not be performed on columns "
                f"{failed_columns}. Please drop them before calling groupby."
            )

    def any(self, skipna: bool = True):
        result = self.df.groupby(self.keys, as_index=False).any()
        if not (self.df.drop(columns=self.keys).dtypes == 'bool').all():
            raise ValueError('Expected boolean types')
        self._validate_result(result)
        return PandasDataFrame(result)

    def all(self, skipna: bool = True):
        result = self.df.groupby(self.keys, as_index=False).all()
        if not (self.df.drop(columns=self.keys).dtypes == 'bool').all():
            raise ValueError('Expected boolean types')
        self._validate_result(result)
        return PandasDataFrame(result)

    def min(self, skipna: bool = True):
        result = self.df.groupby(self.keys, as_index=False).min()
        self._validate_result(result)
        return PandasDataFrame(result)

    def max(self, skipna: bool = True):
        result = self.df.groupby(self.keys, as_index=False).max()
        self._validate_result(result)
        return PandasDataFrame(result)

    def sum(self, skipna: bool = True):
        result = self.df.groupby(self.keys, as_index=False).sum()
        self._validate_result(result)
        return PandasDataFrame(result)

    def prod(self, skipna: bool = True):
        result = self.df.groupby(self.keys, as_index=False).prod()
        self._validate_result(result)
        return PandasDataFrame(result)

    def median(self, skipna: bool = True):
        result = self.df.groupby(self.keys, as_index=False).median()
        self._validate_result(result)
        return PandasDataFrame(result)

    def mean(self, skipna: bool = True):
        result = self.df.groupby(self.keys, as_index=False).mean()
        self._validate_result(result)
        return PandasDataFrame(result)

    def std(self, skipna: bool = True):
        result = self.df.groupby(self.keys, as_index=False).std()
        self._validate_result(result)
        return PandasDataFrame(result)

    def var(self, skipna: bool = True):
        result = self.df.groupby(self.keys, as_index=False).var()
        self._validate_result(result)
        return PandasDataFrame(result)

class PolarsGroupBy:
    def __init__(self, df, keys):
        self.df = df
        self.keys = keys

    def any(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col('*').any())
        return PolarsDataFrame(result)

    def all(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col('*').all())
        return PolarsDataFrame(result)

    def min(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col('*').min())
        return PolarsDataFrame(result)

    def max(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col('*').max())
        return PolarsDataFrame(result)

    def sum(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col('*').sum())
        return PolarsDataFrame(result)

    def prod(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col('*').product())
        return PolarsDataFrame(result)

    def median(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col('*').median())
        return PolarsDataFrame(result)

    def mean(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col('*').mean())
        return PolarsDataFrame(result)

    def std(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col('*').std())
        return PolarsDataFrame(result)

    def var(self, skipna: bool = True):
        result = self.df.groupby(self.keys).agg(pl.col('*').var())
        return PolarsDataFrame(result)


class PandasDataFrame:

    # Not technicall part of the standard

    def __init__(self, df):
        self._validate_columns(df.columns) 
        self.df = df

    def _validate_columns(self, columns):
        counter = collections.Counter(columns)
        for col, count in counter.items():
            if count > 1:
                raise ValueError(f'Expected unique column names, got {col} {count} time(s)')
        for col in columns:
            if not isinstance(col, str):
                raise TypeError(f'Expected column names to be of type str, got {col} of type {type(col)}')
    
    # In the standard
    
    def groupby(self, keys):
        if not isinstance(keys, collections.Sequence):
            raise TypeError(f'Expected sequence of strings, got: {type(keys)}')
        for key in keys:
            if key not in self.get_column_names():
                raise KeyError(f'key {key} not present in DataFrame\'s columns')
        return PandasGroupBy(self.df, keys)

    def get_column_by_name(self, name):
        return PandasColumn(self.df.loc[:, name])
    
    def get_columns_by_name(self, names):
        return PandasDataFrame(self.df.loc[:, names])

    def get_rows(self, indices):
        if not isinstance(indices, collections.Sequence):
            raise TypeError(f'Expected Sequence of int, got {type(indices)}')
        return PandasDataFrame(self.df.iloc[indices, :])
    
    def slice_rows(self, start, stop, step):
        return PandasDataFrame(self.df.iloc[start:stop:step])

    def get_rows_by_mask(self, mask):
        mask_array = np.asarray(mask)
        if not mask_array.dtype == 'bool':
            raise TypeError(f'Expected boolean array, got {type(mask_array)}')
        return PandasDataFrame(self.df.loc[mask_array, :])

    def insert(self, loc, label, value):
        value_array = np.asarray(value)
        before = self.df.iloc[:, :loc]
        after = self.df.iloc[:, loc+1:]
        to_insert = pd.Series(value_array, index=self.df.index)
        return pd.concat([before, to_insert, after], axis=1)

    def drop_column(self, label):
        if not isinstance(label, str):
            raise TypeError(f'Expected str, got: {type(label)}')
        return PandasDataFrame(self.df.drop(label, axis=1))

    def __iter__(self):
        raise NotImplementedError()
    
    def __eq__(self, other):
        assert len(other) == self.shape[0]
        return PandasDataFrame((self.df == other).to_frame())
        
    @property
    def shape(self):
        return self.df.shape
    

    

    
    def set_column(self, label, value):
        columns = self.df.columns
        if label in columns:
            idx = self.df.columns.index(idx)
            return self.drop_column(label).insert(idx, label, value)
        return PandasDataFrame(pd.concat([self.df, pd.Series(value, name=label)], axis=1))
    
    def rename(self, mapping):
        return PandasDataFrame(self.df.rename(columns=mapping))
    
    def get_column_names(self):
        return self.df.columns

class PolarsDataFrame:
    def __init__(self, df):
        # columns already have to be strings, and duplicates aren't
        # allowed, so no validation required
        self.df = df

    @property
    def shape(self):
        return self.df.shape

    def groupby(self, keys):
        return PolarsGroupBy(self.df, keys)

    def __iter__(self):
        yield from self.column_names
    
    def get_column_by_name(self, name):
        return PolarsColumn(self.df[name])

    def get_columns_by_name(self, names):
        return PolarsDataFrame(self.df.select(names))
    
    def get_rows(self, indices):
        return PolarsDataFrame(self.df.with_row_count('idx').filter(pl.col('idx').is_in(indices)).drop('idx'))

    def slice_rows(self, start, stop, step):
        return PolarsDataFrame(self.df.with_row_count('idx').filter(pl.col('idx').is_in(range(start, stop, step))).drop('idx'))
    
    def get_rows_by_mask(self, mask):
        return PolarsDataFrame(self.df.filter(mask))

    def insert(self, loc, label, value):
        df = self.df.clone()
        df.insert_at_idx(loc, pl.Series(label, value.value))
        return PolarsDataFrame(df)

    def drop_column(self, label):
        return PolarsDataFrame(self.df.drop(label))
    
    def set_column(self, label, value):
        columns = self.df.columns
        if label in columns:
            idx = self.df.columns.index(idx)
            return self.drop_column(label).insert(idx, label, value)
        return PolarsDataFrame(self.df.with_columns(label=pl.Series(value)))
    
    def rename(self, mapping):
        return PolarsDataFrame(self.df.rename(mapping))
    
    @property
    def column_names(self):
        return self.df.columns

def dataframe_standard(df):
    if isinstance(df, pd.DataFrame):
        return PandasDataFrame(df)
    elif isinstance(df, pl.DataFrame):
        return PolarsDataFrame(df)
