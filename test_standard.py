# todo: test that errors are appropriately raised when calls violate standard
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import pandas as pd
import polars as pl
from pandas_standard import PandasColumn, PandasDataFrame
import pandas_standard  # noqa
import polars_standard  # noqa


def pytest_generate_tests(metafunc: Any) -> None:
    if "library" in metafunc.fixturenames:
        metafunc.parametrize("library", ["pandas", "polars"])


def integer_series_1(library: str) -> Any:
    df: Any
    if library == "pandas":
        df = pd.DataFrame({"a": [1, 2, 3]})
        return df.__dataframe_standard__().get_column_by_name("a")
    if library == "polars":
        df = pl.DataFrame({"a": [1, 2, 3]})
        return df.__dataframe_standard__().get_column_by_name("a")
    raise AssertionError(f"Got unexpected library: {library}")


def integer_series_2(library: str) -> object:
    df: Any
    if library == "pandas":
        df = pd.DataFrame({"a": [4, 5, 6]})
        return df.__dataframe_standard__().get_column_by_name("a")
    if library == "polars":
        df = pl.DataFrame({"a": [4, 5, 6]})
        return df.__dataframe_standard__().get_column_by_name("a")
    raise AssertionError(f"Got unexpected library: {library}")


def integer_series_3(library: str) -> object:
    df: Any
    if library == "pandas":
        df = pd.DataFrame({"a": [1, 2, 4]})
        return df.__dataframe_standard__().get_column_by_name("a")
    if library == "polars":
        df = pl.DataFrame({"a": [1, 2, 4]})
        return df.__dataframe_standard__().get_column_by_name("a")
    raise AssertionError(f"Got unexpected library: {library}")


def integer_dataframe_1(library: str) -> Any:
    df: Any
    if library == "pandas":
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        return df.__dataframe_standard__()
    if library == "polars":
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        return df.__dataframe_standard__()
    raise AssertionError(f"Got unexpected library: {library}")


def integer_dataframe_2(library: str) -> Any:
    df: Any
    if library == "pandas":
        df = pd.DataFrame({"a": [1, 2, 4], "b": [4, 2, 6]})
        return df.__dataframe_standard__()
    if library == "polars":
        df = pl.DataFrame({"a": [1, 2, 4], "b": [4, 2, 6]})
        return df.__dataframe_standard__()
    raise AssertionError(f"Got unexpected library: {library}")


def integer_dataframe_3(library: str) -> Any:
    df: Any
    if library == "pandas":
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7], "b": [7, 6, 5, 4, 3, 2, 1]})
        return df.__dataframe_standard__()
    if library == "polars":
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7], "b": [7, 6, 5, 4, 3, 2, 1]})
        return df.__dataframe_standard__()
    raise AssertionError(f"Got unexpected library: {library}")


def bool_dataframe_1(library: str) -> object:
    df: Any
    if library == "pandas":
        df = pd.DataFrame({"a": [True, True, False], "b": [True, True, True]})
        return df.__dataframe_standard__()
    if library == "polars":
        df = pl.DataFrame({"a": [True, True, False], "b": [True, True, True]})
        return df.__dataframe_standard__()
    raise AssertionError(f"Got unexpected library: {library}")


@pytest.mark.parametrize(
    ("reduction", "expected_data"),
    [
        ("any", {"a": [True], "b": [True]}),
        ("all", {"a": [False], "b": [True]}),
    ],
)
def test_reductions(
    library: str, reduction: str, expected_data: dict[str, object]
) -> None:
    df = bool_dataframe_1(library)
    result = getattr(df, reduction)()
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    expected = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result_pd, expected)


@pytest.mark.parametrize(
    ("comparison", "expected_data"),
    [
        ("__eq__", {"a": [True, True, False], "b": [True, False, True]}),
        ("__ne__", {"a": [False, False, True], "b": [False, True, False]}),
        ("__ge__", {"a": [True, True, False], "b": [True, True, True]}),
        ("__gt__", {"a": [False, False, False], "b": [False, True, False]}),
        ("__le__", {"a": [True, True, True], "b": [True, False, True]}),
        ("__lt__", {"a": [False, False, True], "b": [False, False, False]}),
        ("__add__", {"a": [2, 4, 7], "b": [8, 7, 12]}),
        ("__sub__", {"a": [0, 0, -1], "b": [0, 3, 0]}),
        ("__mul__", {"a": [1, 4, 12], "b": [16, 10, 36]}),
        ("__truediv__", {"a": [1, 1, 0.75], "b": [1, 2.5, 1]}),
        ("__floordiv__", {"a": [1, 1, 0], "b": [1, 2, 1]}),
        ("__pow__", {"a": [1, 4, 81], "b": [256, 25, 46656]}),
        ("__mod__", {"a": [0, 0, 3], "b": [0, 1, 0]}),
    ],
)
def test_comparisons(
    library: str, comparison: str, expected_data: dict[str, object]
) -> None:
    df = integer_dataframe_1(library)
    other = integer_dataframe_2(library)
    result = getattr(df, comparison)(other).dataframe
    if library == "polars" and comparison == "__pow__":
        # Is this right? Might need fixing upstream.
        result = result.select(pl.col("*").cast(pl.Int64))
    result_pd = pd.api.interchange.from_dataframe(result)
    expected = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result_pd, expected)


@pytest.mark.parametrize(
    ("comparison", "expected_data"),
    [
        ("__eq__", [True, True, False]),
        ("__ne__", [False, False, True]),
        ("__ge__", [True, True, False]),
        ("__gt__", [False, False, False]),
        ("__le__", [True, True, True]),
        ("__lt__", [False, False, True]),
        ("__add__", [2, 4, 7]),
        ("__sub__", [0, 0, -1]),
        ("__mul__", [1, 4, 12]),
        ("__truediv__", [1, 1, 0.75]),
        ("__floordiv__", [1, 1, 0]),
        ("__pow__", [1, 4, 81]),
        ("__mod__", [0, 0, 3]),
    ],
)
def test_column_comparisons(
    library: str, comparison: str, expected_data: list[object]
) -> None:
    ser: Any
    ser = integer_series_1(library)
    other = integer_series_3(library)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": getattr(ser, comparison)(other)})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    expected = pd.Series(expected_data, name="result")
    if library == "polars" and comparison == "__pow__":
        # todo: what should the type be?
        result_pd = result_pd.astype("int64")
    pd.testing.assert_series_equal(result_pd, expected)


@pytest.mark.parametrize(
    ("comparison", "expected_data"),
    [
        ("__eq__", [False, False, True]),
        ("__ne__", [True, True, False]),
        ("__ge__", [False, False, True]),
        ("__gt__", [False, False, False]),
        ("__le__", [True, True, True]),
        ("__lt__", [True, True, False]),
        ("__add__", [4, 5, 6]),
        ("__sub__", [-2, -1, 0]),
        ("__mul__", [3, 6, 9]),
        ("__truediv__", [1 / 3, 2 / 3, 1]),
        ("__floordiv__", [0, 0, 1]),
        ("__pow__", [1, 8, 27]),
        ("__mod__", [1, 2, 0]),
    ],
)
def test_column_comparisons_scalar(
    library: str, comparison: str, expected_data: list[object]
) -> None:
    ser: Any
    ser = integer_series_1(library)
    other = 3
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": getattr(ser, comparison)(other)})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    expected = pd.Series(expected_data, name="result")
    if library == "polars" and comparison == "__pow__":
        # todo: what should the type be?
        result_pd = result_pd.astype("int64")
    pd.testing.assert_series_equal(result_pd, expected)


def test_divmod(library: str) -> None:
    df = integer_dataframe_1(library)
    other = integer_dataframe_2(library)
    result_quotient, result_remainder = df.__divmod__(other)
    result_quotient_pd = pd.api.interchange.from_dataframe(result_quotient.dataframe)
    result_remainder_pd = pd.api.interchange.from_dataframe(result_remainder.dataframe)
    expected_quotient = pd.DataFrame({"a": [1, 1, 0], "b": [1, 2, 1]})
    expected_remainder = pd.DataFrame({"a": [0, 0, 3], "b": [0, 1, 0]})
    pd.testing.assert_frame_equal(result_quotient_pd, expected_quotient)
    pd.testing.assert_frame_equal(result_remainder_pd, expected_remainder)


def test_get_column_by_name(library: str) -> None:
    df = integer_dataframe_1(library)
    result = df.get_column_by_name("a")
    namespace = df.__dataframe_namespace__()
    result = namespace.dataframe_from_dict({"result": result})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    expected = pd.Series([1, 2, 3], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_get_column_by_name_invalid(library: str) -> None:
    df = integer_dataframe_1(library)
    with pytest.raises(ValueError):
        df.get_column_by_name([True, False])


def test_get_columns_by_name(library: str) -> None:
    df = integer_dataframe_1(library)
    result = df.get_columns_by_name(["b"]).dataframe
    result_pd = pd.api.interchange.from_dataframe(result)
    expected = pd.DataFrame({"b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_get_columns_by_name_invalid(library: str) -> None:
    df = integer_dataframe_1(library)
    with pytest.raises(TypeError, match=r"Expected sequence of str, got <class \'str\'>"):
        df.get_columns_by_name("b")


def test_get_rows(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    indices = namespace.column_from_sequence([0, 2, 1], dtype="int64")
    result = df.get_rows(indices).dataframe
    result_pd = pd.api.interchange.from_dataframe(result)
    expected = pd.DataFrame({"a": [1, 3, 2], "b": [4, 6, 5]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_column_get_rows(library: str) -> None:
    ser = integer_series_1(library)
    namespace = ser.__column_namespace__()
    indices = namespace.column_from_sequence([0, 2, 1], dtype="int64")
    result = namespace.dataframe_from_dict({"result": ser.get_rows(indices)})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    expected = pd.Series([1, 3, 2], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_slice_rows(library: str) -> None:
    df = integer_dataframe_3(library)
    result = df.slice_rows(2, 7, 2)
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    expected = pd.DataFrame({"a": [3, 5, 7], "b": [5, 3, 1]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_get_rows_by_mask() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    mask = PandasColumn(pd.Series([True, False, True]))
    result = df.get_rows_by_mask(mask).dataframe
    expected = pd.DataFrame({"a": [1, 3], "b": [4, 6]})
    pd.testing.assert_frame_equal(result, expected)


def test_insert() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    new_col = PandasColumn(pd.Series([7, 8, 9]))
    result = df.insert(1, "c", new_col).dataframe
    expected = pd.DataFrame({"a": [1, 2, 3], "c": [7, 8, 9], "b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result, expected)


def test_drop_column() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = df.drop_column("a").dataframe
    expected = pd.DataFrame({"b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result, expected)


def test_drop_column_invalid() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    with pytest.raises(TypeError, match="Expected str, got: <class 'list'>"):
        df.drop_column(["a"])  # type: ignore[arg-type]


def test_rename_columns() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = df.rename_columns({"a": "c", "b": "e"}).dataframe
    expected = pd.DataFrame({"c": [1, 2, 3], "e": [4, 5, 6]})
    pd.testing.assert_frame_equal(result, expected)


def test_rename_columns_invalid() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    with pytest.raises(TypeError, match="Expected Mapping, got: <class 'function'>"):
        df.rename_columns(lambda x: x.upper())  # type: ignore[arg-type]


def test_get_column_names() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = df.get_column_names()
    assert [name for name in result] == ["a", "b"]


@pytest.mark.parametrize(
    ("aggregation", "expected_b", "expected_c"),
    [
        ("min", [1, 3], [4, 6]),
        ("max", [2, 4], [5, 7]),
        ("sum", [3, 7], [9, 13]),
        ("prod", [2, 12], [20, 42]),
        ("median", [1.5, 3.5], [4.5, 6.5]),
        ("mean", [1.5, 3.5], [4.5, 6.5]),
        (
            "std",
            [0.7071067811865476, 0.7071067811865476],
            [0.7071067811865476, 0.7071067811865476],
        ),
        ("var", [0.5, 0.5], [0.5, 0.5]),
    ],
)
def test_groupby_boolean(
    aggregation: str, expected_b: list[float], expected_c: list[float]
) -> None:
    df = PandasDataFrame(
        pd.DataFrame({"key": [1, 1, 2, 2], "b": [1, 2, 3, 4], "c": [4, 5, 6, 7]})
    )
    result = getattr(df.groupby(["key"]), aggregation)().dataframe
    expected = pd.DataFrame({"key": [1, 2], "b": expected_b, "c": expected_c})
    pd.testing.assert_frame_equal(result, expected)


def test_groupby_size() -> None:
    df = PandasDataFrame(
        pd.DataFrame({"key": [1, 1, 2, 2], "b": [1, 2, 3, 4], "c": [4, 5, 6, 7]})
    )
    result = df.groupby(["key"]).size().dataframe
    expected = pd.DataFrame({"key": [1, 2], "size": [2, 2]})
    pd.testing.assert_frame_equal(result, expected)


def test_groupby_invalid_any_all() -> None:
    df = PandasDataFrame(
        pd.DataFrame({"key": [1, 1, 2, 2], "b": [1, 2, 3, 4], "c": [4, 5, 6, 7]})
    )
    with pytest.raises(ValueError, match="Expected boolean types"):
        df.groupby(["key"]).any()
    with pytest.raises(ValueError, match="Expected boolean types"):
        df.groupby(["key"]).all()


@pytest.mark.parametrize(
    ("aggregation", "expected_b", "expected_c"),
    [
        ("any", [True, True], [True, False]),
        ("all", [False, True], [False, False]),
    ],
)
def test_groupby_numeric(
    aggregation: str, expected_b: list[bool], expected_c: list[bool]
) -> None:
    df = PandasDataFrame(
        pd.DataFrame(
            {
                "key": [1, 1, 2, 2],
                "b": [False, True, True, True],
                "c": [True, False, False, False],
            }
        )
    )
    result = getattr(df.groupby(["key"]), aggregation)().dataframe
    expected = pd.DataFrame({"key": [1, 2], "b": expected_b, "c": expected_c})
    pd.testing.assert_frame_equal(result, expected)


def test_isnan_nan() -> None:
    df = pd.DataFrame({"a": [1, 2, np.nan]})
    df_std = df.__dataframe_standard__()  # type: ignore[operator]
    result = df_std.isnan().dataframe
    expected = pd.DataFrame({"a": [False, False, True]})
    pd.testing.assert_frame_equal(result, expected)


def test_column_isnan() -> None:
    ser = pd.Series([1, 2, np.nan])
    ser_std = PandasColumn(ser)
    result = ser_std.isnan()._series
    expected = pd.Series([False, False, True])
    pd.testing.assert_series_equal(result, expected)


def test_any() -> None:
    df = pd.DataFrame({"a": [False, False], "b": [False, True], "c": [True, True]})
    df_std = df.__dataframe_standard__()  # type: ignore[operator]
    result = df_std.any().dataframe
    expected = pd.DataFrame({"a": [False], "b": [True], "c": [True]})
    pd.testing.assert_frame_equal(result, expected)


def test_all() -> None:
    df = pd.DataFrame({"a": [False, False], "b": [False, True], "c": [True, True]})
    df_std = df.__dataframe_standard__()  # type: ignore[operator]
    result = df_std.all().dataframe
    expected = pd.DataFrame({"a": [False], "b": [False], "c": [True]})
    pd.testing.assert_frame_equal(result, expected)


def test_column_any() -> None:
    ser = PandasColumn(pd.Series([False, False]))
    result = ser.any()
    assert not result


def test_column_all() -> None:
    ser = PandasColumn(pd.Series([False, False]))
    result = ser.all()
    assert not result


@pytest.mark.parametrize(
    ("dtype", "expected_values"),
    [("float64", [False, False, False]), ("Float64", [False, False, True])],
)
def test_isnull(dtype: str, expected_values: list[bool]) -> None:
    df = pd.DataFrame({"a": [0.0, 1.0, np.nan]}, dtype=dtype)
    df = df / df
    df_std = df.__dataframe_standard__()  # type: ignore[operator]
    result = df_std.isnull().dataframe
    expected = pd.DataFrame({"a": expected_values})
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("dtype", "expected_values"),
    [("float64", [False, False, False]), ("Float64", [False, False, True])],
)
def test_column_isnull(dtype: str, expected_values: list[bool]) -> None:
    ser = pd.Series([0.0, 1.0, np.nan], dtype=dtype)
    ser = ser / ser
    ser_std = PandasColumn(ser)
    result = ser_std.isnull()._series
    expected = pd.Series(expected_values)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("dtype", "expected_values"),
    [("float64", [True, False, True]), ("Float64", [True, False, False])],
)
def test_isnan(dtype: str, expected_values: list[bool]) -> None:
    df = pd.DataFrame({"a": [0.0, 1.0, np.nan]}, dtype=dtype)
    df = df / df
    df_std = df.__dataframe_standard__()  # type: ignore[operator]
    result = df_std.isnan().dataframe
    expected = pd.DataFrame({"a": expected_values})
    pd.testing.assert_frame_equal(result, expected)


def test_concat() -> None:
    df1 = PandasDataFrame(pd.DataFrame({"a": [1, 2]}))
    df2 = PandasDataFrame(pd.DataFrame({"a": [3, 4]}))
    namespace = df1.__dataframe_namespace__()
    result = namespace.concat([df1, df2]).dataframe
    expected = pd.DataFrame({"a": [1, 2, 3, 4]})
    pd.testing.assert_frame_equal(result, expected)


def test_concat_mismatch() -> None:
    df1 = PandasDataFrame(pd.DataFrame({"a": [1, 2]}))
    df2 = PandasDataFrame(pd.DataFrame({"b": [3, 4]}))
    namespace = df1.__dataframe_namespace__()
    with pytest.raises(ValueError, match="Expected matching columns"):
        namespace.concat([df1, df2]).dataframe


@pytest.mark.parametrize(
    ("ser_values", "other", "expected_values"),
    [
        ([2.0, 3.0], [1, np.nan], [False, False]),
        ([2.0, 1.0], [1, np.nan], [False, True]),
        ([np.nan, 2], [1, np.nan], [True, False]),
    ],
)
def test_is_in(
    ser_values: list[object], other: list[object], expected_values: list[bool]
) -> None:
    values = PandasColumn(pd.Series(other))
    ser = PandasColumn(pd.Series(ser_values))
    expected = pd.Series(expected_values)
    result = ser.is_in(values)._series
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("ser_values", "other", "expected_values"),
    [
        ([2.0, 3.0], [1, np.nan], [False, False]),
        ([2.0, 1.0], [1, np.nan], [False, True]),
        ([None, 2], [pd.NA], [True, False]),
    ],
)
def test_is_in_nullable_float(
    ser_values: list[object], other: list[object], expected_values: list[bool]
) -> None:
    values = PandasColumn(pd.Series(other, dtype="Float64"))
    ser = PandasColumn(pd.Series(ser_values, dtype="Float64"))
    expected = pd.Series(expected_values, dtype="boolean")
    result = ser.is_in(values)._series
    pd.testing.assert_series_equal(result, expected)


def test_is_in_nullable_float_nan() -> None:
    values = PandasColumn(pd.Series([None], dtype="Float64"))
    ser = pd.Series([0, None], dtype="Float64")
    ser /= ser
    expected = pd.Series([False, True], dtype="boolean")
    result = PandasColumn(ser).is_in(values)._series
    pd.testing.assert_series_equal(result, expected)


def test_is_in_raises() -> None:
    values = PandasColumn(pd.Series([1], dtype="int64"))
    ser = PandasColumn(pd.Series([0, None], dtype="Float64"))
    with pytest.raises(ValueError, match="`value` has dtype int64, expected Float64"):
        ser.is_in(values)._series


def test_len() -> None:
    result = len(PandasDataFrame(pd.DataFrame({"a": [1, 2]})))
    assert result == 2


def test_column_len() -> None:
    result = len(PandasColumn(pd.Series([1, 2])))
    assert result == 2


def test_getitem() -> None:
    result = PandasColumn(pd.Series([1, 999]))[1]
    assert result == 999


def test_unique() -> None:
    result = PandasColumn(pd.Series([1, 1, 2])).unique()._series
    expected = pd.Series([1, 2])
    pd.testing.assert_series_equal(result, expected)


def test_mean() -> None:
    result = PandasColumn(pd.Series([1, 1, 4])).mean()
    assert result == 2


def test_sorted_indices() -> None:
    result = (
        PandasDataFrame(pd.DataFrame({"a": [1, 1], "b": [4, 3]}))
        .sorted_indices(keys=["a", "b"])
        ._series
    )
    expected = pd.Series([1, 0])
    pd.testing.assert_series_equal(result, expected)


def test_column_sorted_indices() -> None:
    result = PandasColumn(pd.Series([1, 3, 2])).sorted_indices()._series
    expected = pd.Series([0, 2, 1])
    pd.testing.assert_series_equal(result, expected)


def test_column_invert() -> None:
    result = (~PandasColumn(pd.Series([True, False])))._series
    expected = pd.Series([False, True])
    pd.testing.assert_series_equal(result, expected)


def test_column_max() -> None:
    result = PandasColumn(pd.Series([1, 3, 2])).max()
    assert result == 3


def test_repeated_columns() -> None:
    df = pd.DataFrame({"a": [1, 2]}, index=["b", "b"]).T
    with pytest.raises(
        ValueError, match=r"Expected unique column names, got b 2 time\(s\)"
    ):
        PandasDataFrame(df)


def test_non_str_columns() -> None:
    df = pd.DataFrame({0: [1, 2]})
    with pytest.raises(
        TypeError,
        match=r"Expected column names to be of type str, got 0 of type <class 'int'>",
    ):
        PandasDataFrame(df)


def test_comparison_invalid() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [1, 2, 3]}))
    other = PandasDataFrame(pd.DataFrame({"b": [1, 2, 3]}))
    with pytest.raises(
        ValueError,
        match="Expected DataFrame with same length, matching "
        "columns, and matching index.",
    ):
        df > other


def test_groupby_invalid() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(
        TypeError, match=r"Expected sequence of strings, got: <class \'int\'>"
    ):
        df.groupby(0)  # type: ignore
    with pytest.raises(TypeError, match=r"Expected sequence of strings, got: str"):
        df.groupby("0")
    with pytest.raises(KeyError, match=r"key b not present in DataFrame\'s columns"):
        df.groupby(["b"])


def test_any_rowwise() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [True, False], "b": [False, False]}))
    result = df.any_rowwise()._series
    expected = pd.Series([True, False])
    pd.testing.assert_series_equal(result, expected)


def test_all_rowwise() -> None:
    df = PandasDataFrame(pd.DataFrame({"a": [True, False], "b": [False, False]}))
    result = df.all_rowwise()._series
    expected = pd.Series([False, False])
    pd.testing.assert_series_equal(result, expected)


def test_fill_nan() -> None:
    df_pd = pd.DataFrame({"a": [0.0, 1.0, np.nan], "b": [1, 2, 3]})
    df_pd["c"] = pd.Series([1, 2, pd.NA], dtype="Int64")
    df = PandasDataFrame(df_pd)
    result = df.fill_nan(-1).dataframe
    expected = pd.DataFrame({"a": [0.0, 1.0, -1.0], "b": [1, 2, 3]})
    expected["c"] = pd.Series([1, 2, pd.NA], dtype="Int64")
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (
            pd.Series([1, 2, pd.NA], dtype="Int64"),
            pd.Series([1, 2, pd.NA], dtype="Int64"),
        ),
        (
            pd.Series([1.0, 2.0, np.nan], dtype="float64"),
            pd.Series([1.0, 2.0, -1.0], dtype="float64"),
        ),
    ],
)
def test_column_fill_nan(
    series: "pd.Series[float]", expected: "pd.Series[float]"
) -> None:
    col = PandasColumn(series)
    result = col.fill_nan(-1)._series
    pd.testing.assert_series_equal(result, expected)
