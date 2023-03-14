import pandas as pd
import collections
import polars as pl
import re

class Column:
    def __init__(self, column):
        self.value = column
    
    def not_null(self):
        return Column(self.value.notna())

    def is_null(self):
        return Column(self.value.isna())
    
    def any(self):
        return self.value.any()

    def all(self):
        return self.value.all()
    
    def __iter__(self):
        return self.value.__iter__()
    
    def __len__(self):
        return len(self.value)

class PandasDataFrame:
    def __init__(self, df):
        self._validate_columns(df.columns) 
        self.df = df
    
    def __iter__(self):
        yield from self.column_names
    
    def __eq__(self, other):
        assert len(other) == self.shape[0]
        return PandasDataFrame((self.df == other).to_frame())
        
    @property
    def shape(self):
        return self.df.shape
    
    def _validate_columns(self, columns):
        counter = collections.Counter(columns)
        for col, count in counter.items():
            if count > 1:
                raise ValueError(f'Expected unique column names, got {col} {count} time(s)')
        for col in columns:
            if not isinstance(col, str):
                raise TypeError(f'Expected column names to be of type str, got {col} of type {type(col)}')
    
    def get_column_by_name(self, name):
        return Column(self.df.loc[:, name])

    def get_columns_by_name(self, names):
        return PandasDataFrame(self.df.loc[:, names])
    
    def get_rows(self, indices):
        return PandasDataFrame(self.df.iloc[indices, :])

    def slice_rows(self, start, stop, step):
        return PandasDataFrame(self.df.iloc[start:stop:step])
    
    def get_rows_by_mask(self, mask):
        return PandasDataFrame(self.df.loc[mask, :])

    def insert(self, loc, label, value):
        # todo turn off index alignment
        df = self.df.copy()
        df.insert(loc, label, value.value)
        return PandasDataFrame(df)

    def drop_column(self, label):
        return PandasDataFrame(self.df.drop(label, axis=1))
    
    def set_column(self, label, value):
        columns = self.df.columns
        if label in columns:
            idx = self.df.columns.index(idx)
            return self.drop_column(label).insert(idx, label, value)
        return PandasDataFrame(pd.concat([self.df, pd.Series(value, name=label)], axis=1))
    
    def rename(self, mapping):
        return PandasDataFrame(self.df.rename(columns=mapping))
    
    @property
    def column_names(self):
        return self.df.columns.to_list()

class PolarsDataFrame:
    def __init__(self, df):
        # columns already have to be strings, and duplicates aren't
        # allowed, so no validation required
        self.df = df

    @property
    def shape(self):
        return self.df.shape

    def __iter__(self):
        yield from self.column_names
    
    def get_column_by_name(self, name):
        return Column(self.df[name])

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
