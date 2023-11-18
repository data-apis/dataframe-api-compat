# Tutorial

We're going to write a dataframe-agnostic "Standard Scaler". This class will have
`fit` and `transform` methods (like `scikit-learn` transformers), and will work
agnosstically for pandas and Polars.

We'll need to write two methods:
- `fit`: find the mean and standard deviation for each column from a given training set;
- `transform`: scale a given dataset with the mean and standard deviations calculated
  during `fit`.

The `fit` method is a bit complicated, so let's start with `transform`.
Suppose we've already calculated the mean and standard deviation of each column, and have
stored them in attributes `self.means` and `self.std_devs`.

## Transform method

The general strategy will be:
1. opt-in to the DataFrame API Standard by calling `__dataframe_consortium_standard__`;
2. calculate new columns using methods from the DataFrame API Standard:
   - `DataFrame.col`
   - `DataFrame.assign`
3. return the original (non-standard-compliant) dataframe to the user by calling
   `DataFrame.dataframe`

```python
class StandardScalar:
    def transform(self, df):
        df = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
        new_columns = []
        for col_name in df.column_names:
            new_columns.append((df.col(col_name) - self.means[col_name])/self.std_devs[col_name])
        df = df.assign(*new_columns)
        return df.dataframe
```

Note that all the calculations here can stay lazy if the underlying library permits it.
For Polars, the return value is a `polars.LazyFrame` - it is the caller's responsibility to
call `.collect()` on the result if they want to materialise its values.

## Fit method

Unlike the `transform` method, `fit` cannot stay lazy, as we need to compute concrete values
for the means and standard deviations.

We will need to use the `persist` method here. `DataFrame.persist` is a no-op in pandas, and
evaluates to `.collect().lazy` in Polars. Think of it as "materialise the dataframe up until
this point, then go back to being lazy". We need to call it in order to be able to extract
concrete values in `Column.get_value`.

```python
class StandardScalar:
    def fit(self, df):
        df = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
        new_columns = []
        for col_name in df.column_names:
            new_columns.append(df.col(col_name).mean().rename(f'{col_name}_mean')
            new_columns.append(df.col(col_name).std().rename(f'{col_name}_std')
        df = df.assign(*new_columns)
        df = df.persist()
        means = {}
        std_devs = {}
        for col_name in df.column_names:
            means[col_name] = float(df_means.col(col_name).get_value(0))
            std_devs[col_name] = float(df_std_devs.col(col_name).get_value(0))
        self._means = means
        self._std_devs = std_devs
```

## Putting it all together

Here is our dataframe-agnostic standard scaler:
```python
class StandardScalar:
    def fit(self, df):
        df = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
        new_columns = []
        for col_name in df.column_names:
            new_columns.append(df.col(col_name).mean().rename(f'{col_name}_mean')
            new_columns.append(df.col(col_name).std().rename(f'{col_name}_std')
        df = df.assign(*new_columns)
        df = df.persist()
        means = {}
        std_devs = {}
        for col_name in df.column_names:
            means[col_name] = float(df_means.col(col_name).get_value(0))
            std_devs[col_name] = float(df_std_devs.col(col_name).get_value(0))
        self._means = means
        self._std_devs = std_devs

    def transform(self, df):
        df = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
        new_columns = []
        for col_name in df.column_names:
            new_columns.append((df.col(col_name) - self.means[col_name])/self.std_devs[col_name])
        df = df.assign(*new_columns)
        return df.dataframe

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
scaler = StandardScaler()
scaler.fit(df)
print(scaler.transform(df))
```