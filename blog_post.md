# DataFrame Standard: Enabling cross-DataFrame code.

<h1 align="center">
	<img
		width="400"
		alt="standard-compliant DataFrame"
		src="https://github.com/data-apis/dataframe-api-compat/assets/33491632/fb4bc907-2b85-4ad7-8d13-c2b9912b97f5">
</h1>

Tired of getting lost in if-then statements when dealing with API differences
between DataFrame libraries? Would you like to be able to write your code
once, have it work with all major DataFrame libraries, and be done?
Let's learn about an initiative which will enable you to write
cross-DataFrame code - no special-casing nor data conversions required!

## Why would I want this anyway?

Say you want to write a function which selects rows of a DataFrame based
on the [z-score](https://en.wikipedia.org/wiki/Standard_score) of a given
column, and you want it to work with any DataFrame library. How might
you write that?

### Solution 1

Here's a typical solution:
```python
def remove_outliers(df: object, column: str) -> pd.DataFrame:
    if isinstance(df, pandas.DataFrame):
        z_score = (df[column] - df[column].mean())/df[column].std()
        return df[z_score.between(-3, 3)]
    if isinstance(df, polars.DataFrame):
        z_score = ((pl.col(column) - pl.col(column).mean()) / pl.col(column).std())
        return df.filter(z_score.is_between(-3, 3))
    if isinstance(df, some_other_library.DataFrame):
        ...
```
This quickly gets unwieldy. Libraries like `cudf` and `modin` _might_ work
in the `isinstance(df, pandas.DataFrame)` arm, but there's no guarantee -
their APIs are similar, but subtly different. Furthermore, there may be
new pandas-like library which you're not familiar with, which would happen
to work perfectly well in the pandas arm above, but which your function
wouldn't work on because you didn't hard-code support for it.

Can we do better?

### Solution 2: Interchange Protocol

An alternative, which wouldn't involve special-casing, could be to
leverage the [DataFrame interchange protocol](https://data-apis.org/dataframe-protocol/latest/index.html):
```python
def remove_outliers(df: object, column: str) -> pd.DataFrame:
    df_pd = pd.api.interchange.from_dataframe(df)
    z_score = (df_pd[column] - df_pd[column].mean())/df_pd[column].std()
    return df_pd[z_score.between(-3, 3)]
```
We got out of having to write if-then statements, but there's still a
couple of issues here:
1. we had to convert to pandas: this might be expensive if your data was
   originally stored on GPU;
2. the return value is a `pandas.DataFrame`, rather than an object of your
   original DataFrame library.

Can we do better, and avoid these issues? Can we really have it all?

### Solution 3: Introducing the DataFrame Standard

Yes, we really can have it all. To write cross-DataFrame code, we'll take these steps:
1. enable the Standard using ``.__dataframe_standard__``. This will return
   a Standard-compliant DataFrame;
2. write your code, using the [DataFrame Standard specification](https://data-apis.org/dataframe-api/draft/API_specification/index.html)
3. (optional) return a DataFrame from your original library by calling `.dataframe`.

Let's see how this would look like for our ``remove_outliers`` example function:
```python
def remove_outliers(df, column):
    # Get a Standard-compliant DataFrame.
    # NOTE: this has not yet been upstreamed, so won't work out-of-the-box!
    # See 'resources' below for how to try it out.
    df_standard = df.__dataframe_standard__()
    # Use methods from the Standard specification.
    col = df_standard.get_column_by_name(column)
    z_score = (col - col.mean()) / col.std()
    df_standard_filtered = df_standard.filter((z_score > -3) & (z_score < 3))
    # Return the result as a DataFrame from the original library.
    return df_standard_filtered.dataframe
```
This will work, as if by magic, on any DataFrame with a Standard-compliant implementation.
But it's not magic, of course, it's the power of standardisation!

## Standard Philosophy - will all DataFrame libraries have the same API one day?

Let's start with what this isn't: the Standard isn't an attempt to force all DataFrame
libraries to have the same API and do the same things. It also isn't a way to convert
between DataFrames: the [Interchange Protocol](https://data-apis.org/dataframe-protocol/latest/index.html),
whose adoption is increasing, already does that. It also doesn't aim to standardise
domain or industry specific functionality.

Rather, it is minimal set of essential DataFrame functionality which will work
the same way across libraries, typically accessible via separate namespaces. It
will behave in a strict and predictable manner across DataFrame libraries. Library
authours trying to write DataFrame-agnostic code are expected to greatly benefit from
this, as are their users.

## Who's this for? Do I need to learn yet another API?

Probably not. The DataFrame Standard is mainly targeted towards library developers,
who wish to target multiple DataFrame types. For example, a feature engineering
library could well want to support multiple DataFrame libraries. Given that people
doing machine learning are probably working with huge datasets, the overhead of
converting to pandas might be quite high for them - so, allowing the library to
write DataFrame-agnostic code would be a win for non-pandas users.

## Are we there yet? What lies ahead?

No, not yet. This is just a first draft, and a request for comments.
Have you ever tried writing DataFrame-agnostic-code? If so, we'd love to hear from you.
Would the DataFrame Standard help solve your problems?

Future plans include:
- increasing the scope of the Standard (currently, the spec is very minimal);
- creating implementations of the Standard for several major DataFrame libraries;
- creating a cross-DataFrame test-suite;
- aiming to ensure each major DataFrame library has a `__dataframe_standard__` method.

## Conclusion

We've introduced the DataFrame Standard, which allows you to write cross-DataFrame code.
We learned about its philosophy, as well as what it doesn't aim to be. Finally, we saw
what plans lie ahead - the Standard is in active development, so please watch this space!

## Resources

- Read more on the [official website](https://data-apis.org/dataframe-api/)
- Try out the [proof-of-concept implementation for pandas and polars](https://github.com/data-apis/dataframe-api-compat)!
