# todo: test that errors are appropriately raised when calls violate standard
from __future__ import annotations

from typing import Any, Callable

import pytest
import pandas as pd
import polars as pl

from tests.utils import (
    bool_dataframe_1,
    bool_dataframe_2,
    bool_dataframe_3,
    convert_series_to_pandas_numpy,
    convert_dataframe_to_pandas_numpy,
    convert_to_standard_compliant_column,
    convert_to_standard_compliant_dataframe,
)


def integer_series_1(library: str) -> Any:
    ser: Any
    if library == "pandas-numpy":
        ser = pd.Series([1, 2, 3])
        return convert_to_standard_compliant_column(ser)
    if library == "pandas-nullable":
        ser = pd.Series([1, 2, 3], dtype="Int64")
        return convert_to_standard_compliant_column(ser)
    if library == "polars":
        ser = pl.Series([1, 2, 3])
        return convert_to_standard_compliant_column(ser)
    raise AssertionError(f"Got unexpected library: {library}")


def integer_series_3(library: str) -> object:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [1, 2, 4]}, dtype="int64")
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [1, 2, 4]}, dtype="Int64")
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    if library == "polars":
        df = pl.DataFrame({"a": [1, 2, 4]})
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    raise AssertionError(f"Got unexpected library: {library}")


def integer_series_5(library: str) -> Any:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [1, 1, 4]}, dtype="int64")
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [1, 1, 4]}, dtype="Int64")
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    if library == "polars":
        df = pl.DataFrame({"a": [1, 1, 4]})
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    raise AssertionError(f"Got unexpected library: {library}")


def integer_series_6(library: str) -> Any:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [1, 3, 2]}, dtype="int64")
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [1, 3, 2]}, dtype="Int64")
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    if library == "polars":
        df = pl.DataFrame({"a": [1, 3, 2]})
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    raise AssertionError(f"Got unexpected library: {library}")


def float_series_1(library: str) -> Any:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [2.0, 3.0]}, dtype="float64")
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [2.0, 3.0]}, dtype="Float64")
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    if library == "polars":
        df = pl.DataFrame({"a": [2.0, 3.0]})
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    raise AssertionError(f"Got unexpected library: {library}")


def float_series_2(library: str) -> Any:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [2.0, 1.0]}, dtype="float64")
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [2.0, 1.0]}, dtype="Float64")
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    if library == "polars":
        df = pl.DataFrame({"a": [2.0, 1.0]})
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    raise AssertionError(f"Got unexpected library: {library}")


def float_series_3(library: str) -> object:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [float("nan"), 2.0]}, dtype="float64")
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [0.0, 2.0]}, dtype="Float64")
        other = pd.DataFrame({"a": [0.0, 1.0]}, dtype="Float64")
        return convert_to_standard_compliant_dataframe(df / other).get_column_by_name("a")
    if library == "polars":
        df = pl.DataFrame({"a": [float("nan"), 2.0]})
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    raise AssertionError(f"Got unexpected library: {library}")


def float_series_4(library: str) -> object:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [1.0, float("nan")]}, dtype="float64")
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [1.0, 0.0]}, dtype="Float64")
        other = pd.DataFrame({"a": [1.0, 0.0]}, dtype="Float64")
        return convert_to_standard_compliant_dataframe(df / other).get_column_by_name("a")
    if library == "polars":
        df = pl.DataFrame({"a": [1.0, float("nan")]})
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    raise AssertionError(f"Got unexpected library: {library}")


def bool_series_1(library: str) -> Any:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [True, False, True]}, dtype="bool")
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [True, False, True]}, dtype="boolean")
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    if library == "polars":
        df = pl.DataFrame({"a": [True, False, True]})
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    raise AssertionError(f"Got unexpected library: {library}")


def bool_series_2(library: str) -> Any:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [True, False, False]}, dtype="bool")
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [True, False, False]}, dtype="boolean")
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    if library == "polars":
        df = pl.DataFrame({"a": [True, False, False]})
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    raise AssertionError(f"Got unexpected library: {library}")


def nan_series_1(library: str) -> Any:
    df: Any
    if library == "pandas-numpy":
        df = pd.DataFrame({"a": [0.0, 1.0, float("nan")]}, dtype="float64")
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [0.0, 1.0, 0.0]}, dtype="Float64")
        other = pd.DataFrame({"a": [1.0, 1.0, 0.0]}, dtype="Float64")
        return convert_to_standard_compliant_dataframe(df / other).get_column_by_name("a")
    if library == "polars":
        df = pl.DataFrame({"a": [0.0, 1.0, float("nan")]})
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    raise AssertionError(f"Got unexpected library: {library}")


def null_series_1(library: str, request: pytest.FixtureRequest) -> Any:
    df: Any
    if library == "pandas-numpy":
        mark = pytest.mark.xfail()
        request.node.add_marker(mark)
    if library == "pandas-nullable":
        df = pd.DataFrame({"a": [1.0, 2.0, pd.NA]}, dtype="Float64")
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    if library == "polars":
        df = pl.DataFrame({"a": [1.0, 2.0, None]})
        return convert_to_standard_compliant_dataframe(df).get_column_by_name("a")
    raise AssertionError(f"Got unexpected library: {library}")


def test_float_powers_column(library: str) -> None:
    ser = integer_series_1(library)
    other = integer_series_1(library) * 1.0
    result = ser.__pow__(other)
    namespace = ser.__column_namespace__()
    result_pd = pd.api.interchange.from_dataframe(
        namespace.dataframe_from_dict({"result": result}).dataframe
    )["result"]
    expected = pd.Series([1.0, 4.0, 27.0], name="result")
    result_pd = convert_series_to_pandas_numpy(result_pd)
    pd.testing.assert_series_equal(result_pd, expected)


def test_float_powers_scalar_column(library: str) -> None:
    ser = integer_series_1(library)
    other = 1.0
    result = ser.__pow__(other)
    namespace = ser.__column_namespace__()
    result_pd = pd.api.interchange.from_dataframe(
        namespace.dataframe_from_dict({"result": result}).dataframe
    )["result"]
    expected = pd.Series([1.0, 2.0, 3.0], name="result")
    result_pd = convert_series_to_pandas_numpy(result_pd)
    pd.testing.assert_series_equal(result_pd, expected)


def test_negative_powers_column(library: str) -> None:
    ser = integer_series_1(library)
    other = integer_series_1(library) * -1
    with pytest.raises(ValueError):
        ser.__pow__(-1)
    with pytest.raises(ValueError):
        ser.__pow__(other)


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
    result_pd = convert_series_to_pandas_numpy(result_pd)
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
        # todo: fix
        result_pd = result_pd.astype("int64")
    result_pd = convert_series_to_pandas_numpy(result_pd)
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_divmod(library: str) -> None:
    ser = integer_series_1(library)
    other = integer_series_3(library)
    namespace = ser.__column_namespace__()
    result_quotient, result_remainder = ser.__divmod__(other)
    result_quotient_pd = pd.api.interchange.from_dataframe(
        namespace.dataframe_from_dict({"result": result_quotient}).dataframe
    )["result"]
    result_remainder_pd = pd.api.interchange.from_dataframe(
        namespace.dataframe_from_dict({"result": result_remainder}).dataframe
    )["result"]
    expected_quotient = pd.Series([1, 1, 0], name="result")
    expected_remainder = pd.Series([0, 0, 3], name="result")
    result_quotient_pd = convert_series_to_pandas_numpy(result_quotient_pd)
    result_remainder_pd = convert_series_to_pandas_numpy(result_remainder_pd)
    pd.testing.assert_series_equal(result_quotient_pd, expected_quotient)
    pd.testing.assert_series_equal(result_remainder_pd, expected_remainder)


def test_column_divmod_with_scalar(library: str) -> None:
    ser = integer_series_1(library)
    other = 2
    namespace = ser.__column_namespace__()
    result_quotient, result_remainder = ser.__divmod__(other)
    result_quotient_pd = pd.api.interchange.from_dataframe(
        namespace.dataframe_from_dict({"result": result_quotient}).dataframe
    )["result"]
    result_remainder_pd = pd.api.interchange.from_dataframe(
        namespace.dataframe_from_dict({"result": result_remainder}).dataframe
    )["result"]
    expected_quotient = pd.Series([0, 1, 1], name="result")
    expected_remainder = pd.Series([1, 0, 1], name="result")
    result_quotient_pd = convert_series_to_pandas_numpy(result_quotient_pd)
    result_remainder_pd = convert_series_to_pandas_numpy(result_remainder_pd)
    pd.testing.assert_series_equal(result_quotient_pd, expected_quotient)
    pd.testing.assert_series_equal(result_remainder_pd, expected_remainder)


def test_column_get_rows(library: str) -> None:
    ser = integer_series_1(library)
    namespace = ser.__column_namespace__()
    indices = namespace.column_from_sequence([0, 2, 1], dtype=namespace.Int64())
    result = namespace.dataframe_from_dict({"result": ser.get_rows(indices)})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([1, 3, 2], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_get_rows_by_mask(library: str) -> None:
    ser = integer_series_1(library)
    namespace = ser.__column_namespace__()
    mask = namespace.column_from_sequence([True, False, True], dtype=namespace.Bool())
    result = ser.get_rows_by_mask(mask)
    result_pd = pd.api.interchange.from_dataframe(
        namespace.dataframe_from_dict({"result": result}).dataframe
    )["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([1, 3], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


@pytest.mark.parametrize(
    ("aggregation", "expected_b", "expected_c"),
    [
        ("any", [True, True], [True, False]),
        ("all", [False, True], [False, False]),
    ],
)
def test_groupby_boolean(
    library: str, aggregation: str, expected_b: list[bool], expected_c: list[bool]
) -> None:
    df = bool_dataframe_2(library)
    result = getattr(df.groupby(["key"]), aggregation)()
    # need to sort
    idx = result.sorted_indices(["key"])
    result = result.get_rows(idx)
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"key": [1, 2], "b": expected_b, "c": expected_c})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_any(library: str) -> None:
    df = bool_dataframe_3(library)
    result = df.any()
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    expected = pd.DataFrame({"a": [False], "b": [True], "c": [True]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_all(library: str) -> None:
    df = bool_dataframe_3(library)
    result = df.all()
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    expected = pd.DataFrame({"a": [False], "b": [False], "c": [True]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_column_any(library: str) -> None:
    ser = bool_series_1(library)
    result = ser.any()
    assert result


def test_column_all(library: str) -> None:
    ser = bool_series_1(library)
    result = ser.all()
    assert not result


def test_column_is_nan(library: str) -> None:
    ser = nan_series_1(library)
    result = ser.is_nan()
    namespace = ser.__column_namespace__()
    result_df = namespace.dataframe_from_dict({"result": result})
    result_pd = pd.api.interchange.from_dataframe(result_df.dataframe)["result"]
    expected = pd.Series([False, False, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_is_null_1(library: str) -> None:
    ser = nan_series_1(library)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": ser.is_null()})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    expected = pd.Series([False, False, False], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_is_null_2(library: str, request: pytest.FixtureRequest) -> None:
    ser = null_series_1(library, request)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": ser.is_null()})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    expected = pd.Series([False, False, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


@pytest.mark.parametrize(
    ("ser_factory", "other_factory", "expected_values"),
    [
        (float_series_1, float_series_4, [False, False]),
        (float_series_2, float_series_4, [False, True]),
        (float_series_3, float_series_4, [True, False]),
    ],
)
def test_is_in(
    library: str,
    ser_factory: Callable[[str], Any],
    other_factory: Callable[[str], Any],
    expected_values: list[bool],
) -> None:
    other = other_factory(library)
    ser = ser_factory(library)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": ser.is_in(other)})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series(expected_values, name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_is_in_raises(library: str) -> None:
    ser = float_series_1(library)
    other = integer_series_1(library)
    with pytest.raises(ValueError):
        ser.is_in(other)


def test_column_len(library: str) -> None:
    result = len(integer_series_1(library))
    assert result == 3


def test_get_value(library: str) -> None:
    result = integer_series_1(library).get_value(0)
    assert result == 1


def test_unique_indices_column(library: str) -> None:
    ser = integer_series_5(library)
    namespace = ser.__column_namespace__()
    ser = ser.get_rows(ser.unique_indices())
    ser = ser.get_rows(ser.sorted_indices())
    result = namespace.dataframe_from_dict({"result": ser})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([1, 4], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_mean(library: str) -> None:
    result = integer_series_5(library).mean()
    assert result == 2.0


def test_std(library: str) -> None:
    result = integer_series_5(library).std()
    assert abs(result - 1.7320508075688772) < 1e-8


@pytest.mark.parametrize(
    ("ascending", "expected_data"),
    [
        (True, [0, 2, 1]),
        (False, [1, 2, 0]),
    ],
)
def test_column_sorted_indices(
    library: str, ascending: bool, expected_data: list[int]
) -> None:
    ser = integer_series_6(library)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict(
        {"result": ser.sorted_indices(ascending=ascending)}
    )
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    # TODO standardise return type?
    result_pd = result_pd.astype("int64")
    expected = pd.Series(expected_data, name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_invert(library: str) -> None:
    ser = bool_series_1(library)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": ~ser})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([False, True, False], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_and(library: str) -> None:
    ser = bool_series_1(library)
    other = bool_series_2(library)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": ser & other})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([True, False, False], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_or(library: str) -> None:
    ser = bool_series_1(library)
    other = bool_series_2(library)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": ser | other})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([True, False, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_and_with_scalar(library: str) -> None:
    ser = bool_series_1(library)
    other = True
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": ser & other})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([True, False, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_or_with_scalar(library: str) -> None:
    ser = bool_series_1(library)
    other = True
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": ser | other})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([True, True, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_max(library: str) -> None:
    result = integer_series_1(library).max()
    assert result == 3


def test_repeated_columns() -> None:
    df = pd.DataFrame({"a": [1, 2]}, index=["b", "b"]).T
    with pytest.raises(
        ValueError, match=r"Expected unique column names, got b 2 time\(s\)"
    ):
        convert_to_standard_compliant_dataframe(df)


def test_non_str_columns() -> None:
    df = pd.DataFrame({0: [1, 2]})
    with pytest.raises(
        TypeError,
        match=r"Expected column names to be of type str, got 0 of type <class 'int'>",
    ):
        convert_to_standard_compliant_dataframe(df)


def test_all_rowwise(library: str) -> None:
    df = bool_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    result = namespace.dataframe_from_dict({"result": df.all_rowwise()})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    expected = pd.Series([True, True, False], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_fill_nan(library: str) -> None:
    # todo: test with nullable pandas
    ser = nan_series_1(library)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": ser.fill_nan(-1.0)})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([0.0, 1.0, -1.0], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        ("min", 1),
        ("max", 3),
        ("sum", 6),
        ("prod", 6),
        ("median", 2.0),
        ("mean", 2.0),
        ("std", 1.0),
        ("var", 1.0),
    ],
)
def test_column_reductions(library: str, reduction: str, expected: float) -> None:
    ser = integer_series_1(library)
    result = getattr(ser, reduction)()
    assert result == expected


def test_column_column() -> None:
    result = (
        convert_to_standard_compliant_dataframe(pl.DataFrame({"a": [1, 2, 3]}))
        .get_column_by_name("a")
        .column
    )
    pd.testing.assert_series_equal(result.to_pandas(), pd.Series([1, 2, 3], name="a"))
    result = (
        convert_to_standard_compliant_dataframe(pd.DataFrame({"a": [1, 2, 3]}))
        .get_column_by_name("a")
        .column
    )
    pd.testing.assert_series_equal(result, pd.Series([1, 2, 3], name="a"))


@pytest.mark.parametrize(
    ("values", "dtype", "expected"),
    [
        ([1, 2, 3], "Int64", pd.Series([1, 2, 3], dtype="int64", name="result")),
        ([1, 2, 3], "Int32", pd.Series([1, 2, 3], dtype="int32", name="result")),
        (
            [1.0, 2.0, 3.0],
            "Float64",
            pd.Series([1, 2, 3], dtype="float64", name="result"),
        ),
        (
            [1.0, 2.0, 3.0],
            "Float32",
            pd.Series([1, 2, 3], dtype="float32", name="result"),
        ),
        (
            [True, False, True],
            "Bool",
            pd.Series([True, False, True], dtype=bool, name="result"),
        ),
    ],
)
def test_column_from_sequence(
    library: str, values: list[Any], dtype: str, expected: pd.Series[Any]
) -> None:
    ser = integer_series_1(library)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict(
        {
            "result": namespace.column_from_sequence(
                values, dtype=getattr(namespace, dtype)()
            )
        }
    )
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_names(library: str) -> None:
    # nameless column
    ser = integer_series_1(library)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": ser})
    assert result.get_column_names() == ["result"]

    # named column
    ser = namespace.column_from_sequence(
        [1, 2, 3], dtype=namespace.Float64(), name="result"
    )
    result = namespace.dataframe_from_dict({"result": ser})
    assert result.get_column_names() == ["result"]

    # named column (different name)
    ser = namespace.column_from_sequence(
        [1, 2, 3], dtype=namespace.Float64(), name="result2"
    )
    with pytest.raises(ValueError):
        namespace.dataframe_from_dict({"result": ser})


def test_fill_null_noop_column(library: str) -> None:
    ser = nan_series_1(library)
    result = ser.fill_null(0)
    # nan should not have changed!
    assert result.column[2] != result.column[2]


@pytest.mark.parametrize(
    ("func", "expected_data"),
    [
        ("cumulative_sum", [1, 3, 6]),
        ("cumulative_prod", [1, 2, 6]),
        ("cumulative_max", [1, 2, 3]),
        ("cumulative_min", [1, 1, 1]),
    ],
)
def test_cumulative_functions_column(
    library: str, func: str, expected_data: list[float]
) -> None:
    ser = integer_series_1(library)
    namespace = ser.__column_namespace__()
    expected = pd.Series(expected_data, name="result")
    result = namespace.dataframe_from_dict({"result": getattr(ser, func)()})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    pd.testing.assert_series_equal(result_pd, expected)
