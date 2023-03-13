import pandas as pd
import collections
import polars as pl
import re
import janitor
from janitor.functions.clean_names import (
    _change_case,
    _normalize_1,
    _remove_special,
    _strip_accents,
    _strip_underscores_func,
)

class Column:
    def __init__(self, column):
        self.value = column

class PandasDataFrame:
    def __init__(self, df):
        self._validate_columns(df.columns) 
        self.df = df
    
    def _validate_columns(self, columns):
        counter = collections.Counter(columns)
        for col, count in counter.items():
            if count > 1:
                raise ValueError(f'Expected unique column names, got {col} {count} time(s)')
        if not all(isinstance(col, str) for col in columns):
            raise TypeError('Expected column names to be of type str, got {col} of type {type(col)}')
    
    def get_column_by_name(self, name):
        return Column(self.df.loc[:, name])

    def get_columns_by_name(self, names):
        return [self.get_column_by_name(name) for name in names]
    
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
        idx = self.df.columns.index(idx)
        return self.drop_column(label).insert(idx, label, value)
    
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
    
    def get_column_by_name(self, name):
        return Column(self.df[name])

    def get_columns_by_name(self, names):
        return [self.get_column_by_name(name) for name in names]
    
    def get_rows(self, indices):
        return PolarsDataFrame(self.dt.with_row_count('idx').filter(pl.col('idx')==indices))

    def slice_rows(self, start, stop, step):
        ...
    
    def get_rows_by_mask(self, mask):
        ...

    def insert(self, loc, label, value):
        ...

    def drop_column(self, label):
        ...
    
    def set_column(self, label, value):
        ...
    
    def rename(self, mapping):
        return PolarsDataFrame(self.df.rename(mapping))
    
    @property
    def column_names(self):
        return self.df.columns

def clean_names_with_standard(
    df,
    strip_underscores=None,
    case_type='lower',
    remove_special=False,
    strip_accents=True,
    preserve_original_columns=True,
    enforce_string=True,
    truncate_limit=None,
):
    if isinstance(df, pd.DataFrame):
        df = PandasDataFrame(df)
    elif isinstance(df, pl.DataFrame):
        df = PolarsDataFrame(df)

    if enforce_string:
        # not necessary, as the Standard already
        # imposes string-only names
        pass

    mapping = {}
    for old_label in df.column_names:
        new_label = _change_case(old_label, case_type)
        new_label = _normalize_1(new_label)
        if remove_special:
            new_label = _remove_special(new_label)
        if strip_accents:
            new_label = _strip_accents(new_label)
        new_label = re.sub("_+", "_", new_label)
        new_label = _strip_underscores_func(new_label, strip_underscores)
        new_label = new_label[:truncate_limit]
        mapping[old_label] = new_label
    df = df.rename(mapping)

    # Store the original column names, if enabled by user
    if preserve_original_columns:
        # This writes to df.__dict__ - is that allowed?
        pass
    return df.df

pd_df = pd.DataFrame(
    {
        "Aloha": range(3),
        "Bell Chart": range(3),
        "Animals@#$%^": range(3)
    }
)
pl_df = pl.from_pandas(pd_df)
print(pd_df)
# print(janitor.clean_names(df))
print(clean_names_with_standard(pd_df))
print(clean_names_with_standard(pl_df))

# ok, pretty cool. let's do this for polars as well, and then see what happens