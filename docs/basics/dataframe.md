# DataFrame

To write a dataframe-agnostic function, the steps you'll want to follow are:

1. Opt-in to the Dataframe API by calling `__dataframe_consortium_standard__` on your dataframe.
2. Express your logic using methods from the [Dataframe API](https://data-apis.org/dataframe-api/draft/API_specification/index.html)
   You may want to look at the [official examples](https://github.com/data-apis/dataframe-api/tree/main/spec/API_specification/examples)
   for inspiration.
3. If you need to return a dataframe to the user in its original library, call `DataFrame.dataframe`.

Let's try writing a simple example.

## Example 1: group-by and mean

Make a Python file `t.py` with the following content:
```python exec="1" source="above" session="df_ex1"
def func(df):
    # 1. Opt-in to the API Standard
    df_s = df.__dataframe_consortium_standard__(api_version='2023.11-beta')
    # 2. Use methods from the API Standard spec
    df_s = df_s.group_by('a').mean()
    # 3. Return a library from the user's original library
    return df_s.dataframe
```
Let's try it out:

=== "pandas"
    ```python exec="true" source="material-block" result="python" session="df_ex1"
    import pandas as pd

    df = pd.DataFrame({'a': [1, 1, 2], 'b': [4, 5, 6]})
    print(func(df))
    ```

=== "Polars"
    ```python exec="true" source="material-block" result="python" session="df_ex1"
    import polars as pl

    df = pl.DataFrame({'a': [1, 1, 2], 'b': [4, 5, 6]})
    print(func(df))
    ```


If you look at the two outputs, you'll see that:

- For pandas, the output is a `pandas.DataFrame`.
- But for Polars, the output is a `polars.LazyFrame`.

This is because the Dataframe API only has a single `DataFrame` class - so for Polars,
all operations are done lazily in order to make full use of Polars' query engine.
If you want to convert that to a `polars.DataFrame`, it is the caller's responsibility
to call `.collect`. Check the modified example below:


=== "pandas"
    ```python exec="true" source="material-block" result="python" session="df_ex1"
    import pandas as pd

    df = pd.DataFrame({'a': [1, 1, 2], 'b': [4, 5, 6]})
    print(func(df))
    ```

=== "Polars"
    ```python exec="true" source="material-block" result="python" session="df_ex1"
    import polars as pl

    df = pl.DataFrame({'a': [1, 1, 2], 'b': [4, 5, 6]})
    print(func(df).collect())
    ```
