# Complete example

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
1. opt-in to the Dataframe API Standard by calling `__dataframe_consortium_standard__`;
2. calculate new columns using methods from the Dataframe API Standard:
   - `DataFrame.col`
   - `DataFrame.assign`
3. return the original (non-standard-compliant) dataframe to the user by calling
   `DataFrame.dataframe`

```python
class StandardScalar:
    def transform(self, df):
        df = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
        new_columns = [(col - self.means[col.name])/self.std_devs[col_name] for col in df.columns_iter()]
        df = df.assign(*new_columns)
        return df.dataframe
```

Note that all the calculations here can stay lazy if the underlying library permits it.
For Polars, the return value is a `polars.LazyFrame` - it is the caller's responsibility to
call `.collect()` on the result if they want to materialise its values.

## Fit method

Unlike the `transform` method, `fit` cannot stay lazy, as we need to compute concrete values
for the means and standard deviations.

We will need to use the `persist` method here, see [persist](persist.md).
We need to call it in order to be able to extract concrete values in `Column.get_value`.

```python
class StandardScalar:
    def fit(self, df):
        df = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
        ns = df.__dataframe_namespace__()

        means = [col.mean() for col in df.columns_iter()]
        std_devs = [col.std() for col in df.columns_iter()]
        df_means = df.assign(*means)
        df_std_devs = df.assign(*std_devs)
        df = ns.concat([means, std_devs])
        df = df.persist()
        means = {col.name: float(col.get_value(0)) for col in df.columns_iter()}
        std_devs = {col.name: float(col.get_value(1)) for col in df.columns_iter()}
        std_devs = {}
        self._means = means
        self._std_devs = std_devs
```

## Putting it all together

Here is our dataframe-agnostic standard scaler:
```python exec="1" source="above" session="tute-ex1"
class StandardScaler:
    def fit(self, df):
        df = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
        ns = df.__dataframe_namespace__()

        df = ns.concat([df.mean(), df.std()])
        df = df.persist()
        means = {col.name: float(col.get_value(0)) for col in df.columns_iter()}
        std_devs = {col.name: float(col.get_value(1)) for col in df.columns_iter()}
        self._means = means
        self._std_devs = std_devs

    def transform(self, df):
        df = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
        new_columns = [(col - self._means[col.name])/self._std_devs[col.name] for col in df.columns_iter()]
        df = df.assign(*new_columns)
        return df.dataframe
```

Next, let's try running it. Note that `StandardScaler.transform` is completely lazy (it contains no `persist`)
calls, so the output for Polars is a `polars.LazyFrame`. So, to see the output, we need to call `.collect`:

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="tute-ex1"
    import pandas as pd

    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 7]})
    scaler = StandardScaler()
    scaler.fit(df)
    print(scaler.transform(df))
    ```

=== "Polars"
    ```python exec="true" source="material-block" result="python" session="tute-ex1"
    import polars as pl

    df = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 7]})
    scaler = StandardScaler()
    scaler.fit(df)
    print(scaler.transform(df).collect())
    ```
