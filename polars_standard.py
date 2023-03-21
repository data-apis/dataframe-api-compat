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