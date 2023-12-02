from __future__ import annotations

import collections
import secrets
import warnings
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterator
from typing import Literal
from typing import NoReturn

import polars as pl

import dataframe_api_compat
from dataframe_api_compat.utils import validate_comparand

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from dataframe_api import DataFrame as DataFrameT
    from dataframe_api.typing import AnyScalar
    from dataframe_api.typing import Column
    from dataframe_api.typing import DType
    from dataframe_api.typing import Namespace
    from dataframe_api.typing import NullType
    from dataframe_api.typing import Scalar

    from dataframe_api_compat.polars_standard.group_by_object import GroupBy

else:
    DataFrameT = object

POLARS_VERSION = tuple(int(v) for v in pl.__version__.split("."))


def generate_random_token(column_names: list[str]) -> str:
    token = secrets.token_hex(8)
    attempts = 0
    while token in column_names and attempts < 100:  # pragma: no cover
        token = secrets.token_hex(8)
        attempts += 1
        if attempts >= 100:
            msg = "Could not generate unique token, please report an issue"
            raise RuntimeError(msg)
    return token


class DataFrame(DataFrameT):
    def __init__(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        *,
        api_version: str,
        is_persisted: bool = False,
    ) -> None:
        self._df = df
        self._api_version = api_version
        self._is_persisted = is_persisted
        assert is_persisted ^ isinstance(df, pl.LazyFrame)

    # Validation helper methods

    def _validate_is_persisted(self) -> pl.DataFrame:
        if not self._is_persisted:
            msg = "Method  requires you to call `.persist` first on the parent dataframe.\n\nNote: `.persist` forces materialisation in lazy libraries and so should be called as late as possible in your pipeline, and only once per dataframe."
            raise ValueError(
                msg,
            )
        return self.dataframe  # type: ignore[return-value]

    def __repr__(self) -> str:  # pragma: no cover
        header = f" Standard DataFrame (api_version={self._api_version}) "
        length = len(header)
        return (
            "┌"
            + "─" * length
            + "┐\n"
            + f"|{header}|\n"
            + "| Add `.dataframe` to see native output         |\n"
            + "└"
            + "─" * length
            + "┘\n"
        )

    def _validate_booleanness(self) -> None:
        if not all(v == pl.Boolean for v in self.dataframe.schema.values()):
            msg = "'any' can only be called on DataFrame where all dtypes are 'bool'"
            raise TypeError(
                msg,
            )

    def _from_dataframe(self, df: pl.LazyFrame | pl.DataFrame) -> DataFrame:
        return DataFrame(
            df,
            api_version=self._api_version,
            is_persisted=self._is_persisted,
        )

    # Properties
    @property
    def schema(self) -> dict[str, DType]:
        return {
            column_name: dataframe_api_compat.polars_standard.map_polars_dtype_to_standard_dtype(
                dtype,
            )
            for column_name, dtype in self.dataframe.schema.items()
        }

    @property
    def column_names(self) -> list[str]:
        return self.dataframe.columns

    @property
    def dataframe(self) -> pl.LazyFrame | pl.DataFrame:
        return self._df

    # In the Standard

    def __dataframe_namespace__(self) -> Namespace:
        return dataframe_api_compat.polars_standard.Namespace(
            api_version=self._api_version,
        )

    def columns_iter(self) -> Iterator[Column]:
        return (self.col(col_name) for col_name in self.column_names)

    def col(self, value: str) -> Column:
        from dataframe_api_compat.polars_standard.column_object import Column

        if isinstance(self.dataframe, pl.DataFrame):
            return Column(
                self.dataframe.get_column(value),
                df=None,
                api_version=self._api_version,
                is_persisted=True,
            )
        return Column(
            pl.col(value),
            df=self,
            api_version=self._api_version,
            is_persisted=False,
        )

    def shape(self) -> tuple[int, int]:
        df = self._validate_is_persisted()
        return df.shape

    def group_by(self, *keys: str) -> GroupBy:
        from dataframe_api_compat.polars_standard.group_by_object import GroupBy

        return GroupBy(self, list(keys), api_version=self._api_version)

    def select(self, *columns: str) -> DataFrame:
        cols = list(columns)
        if cols and not isinstance(cols[0], str):
            msg = f"Expected iterable of str, but the first element is: {type(cols[0])}"
            raise TypeError(msg)
        return self._from_dataframe(
            self._df.select(cols),
        )

    def get_rows(self, indices: Column) -> DataFrame:
        _indices = validate_comparand(self, indices)
        if POLARS_VERSION < (0, 19, 14):
            return self._from_dataframe(
                self.dataframe.select(pl.all().take(_indices)),
            )
        return self._from_dataframe(
            self.dataframe.select(pl.all().gather(_indices)),
        )

    def slice_rows(
        self,
        start: int | None,
        stop: int | None,
        step: int | None,
    ) -> DataFrame:
        return self._from_dataframe(self._df[start:stop:step])

    def filter(self, mask: Column) -> DataFrame:
        _mask = validate_comparand(self, mask)
        return self._from_dataframe(self._df.filter(_mask))

    def assign(self, *columns: Column) -> DataFrame:
        from dataframe_api_compat.polars_standard.column_object import Column

        new_columns: list[pl.Expr] = []
        for col in columns:
            if not isinstance(col, Column):
                msg = (
                    f"Expected iterable of Column, but the first element is: {type(col)}"
                )
                raise TypeError(msg)
            _expr = validate_comparand(self, col)
            new_columns.append(_expr)
        df = self.dataframe.with_columns(new_columns)
        return self._from_dataframe(df)

    def drop_columns(self, *labels: str) -> DataFrame:
        return self._from_dataframe(self.dataframe.drop(labels))

    def rename_columns(self, mapping: Mapping[str, str]) -> DataFrame:
        if not isinstance(mapping, collections.abc.Mapping):
            msg = f"Expected Mapping, got: {type(mapping)}"
            raise TypeError(msg)
        return self._from_dataframe(
            self.dataframe.rename(dict(mapping)),
        )

    def get_column_names(self) -> list[str]:  # pragma: no cover
        # DO NOT REMOVE
        # This one is used in upstream tests - even if deprecated,
        # just leave it in for backwards compatibility
        return self.dataframe.columns

    def sort(
        self,
        *keys: str,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> DataFrame:
        if not keys:
            keys = tuple(self.dataframe.columns)
        # TODO: what if there's multiple `ascending`?
        return self._from_dataframe(
            self.dataframe.sort(list(keys), descending=not ascending),
        )

    # Binary operations

    def __eq__(  # type: ignore[override]
        self,
        other: AnyScalar,
    ) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.with_columns(pl.col("*").__eq__(other)),
        )

    def __ne__(  # type: ignore[override]
        self,
        other: AnyScalar,
    ) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.with_columns(pl.col("*").__ne__(other)),
        )

    def __ge__(self, other: AnyScalar) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.with_columns(pl.col("*").__ge__(other)),
        )

    def __gt__(self, other: AnyScalar) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.with_columns(pl.col("*").__gt__(other)),
        )

    def __le__(self, other: AnyScalar) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.with_columns(pl.col("*").__le__(other)),
        )

    def __lt__(self, other: AnyScalar) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.with_columns(pl.col("*").__lt__(other)),
        )

    def __and__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.with_columns(pl.col("*") & _other),
        )

    def __rand__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self.__and__(_other)

    def __or__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.with_columns(
                (pl.col(col) | _other).alias(col) for col in self.dataframe.columns
            ),
        )

    def __ror__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self.__or__(_other)

    def __add__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.with_columns(pl.col("*").__add__(_other)),
        )

    def __radd__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self.__add__(_other)

    def __sub__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.with_columns(pl.col("*").__sub__(_other)),
        )

    def __rsub__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return -1 * self.__sub__(_other)

    def __mul__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.with_columns(pl.col("*").__mul__(_other)),
        )

    def __rmul__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self.__mul__(_other)

    def __truediv__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.with_columns(pl.col("*").__truediv__(_other)),
        )

    def __rtruediv__(self, other: AnyScalar) -> DataFrame:  # pragma: no cover
        _other = validate_comparand(self, other)
        raise NotImplementedError

    def __floordiv__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.with_columns(pl.col("*").__floordiv__(_other)),
        )

    def __rfloordiv__(self, other: AnyScalar) -> DataFrame:  # pragma: no cover
        _other = validate_comparand(self, other)
        raise NotImplementedError

    def __pow__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        original_type = self.dataframe.schema
        ret = self.dataframe.select(
            [pl.col(col).pow(_other) for col in self.column_names],
        )
        for column in self.dataframe.columns:
            ret = ret.with_columns(pl.col(column).cast(original_type[column]))
        return self._from_dataframe(ret)

    def __rpow__(self, other: AnyScalar) -> DataFrame:  # pragma: no cover
        _other = validate_comparand(self, other)
        raise NotImplementedError

    def __mod__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.with_columns(pl.col("*") % _other),
        )

    def __rmod__(self, other: AnyScalar) -> DataFrame:  # type: ignore[misc]  # pragma: no cover
        _other = validate_comparand(self, other)
        raise NotImplementedError

    def __divmod__(
        self,
        other: DataFrame | AnyScalar,
    ) -> tuple[DataFrame, DataFrame]:
        _other = validate_comparand(self, other)
        quotient_df = self.dataframe.with_columns(pl.col("*") // _other)
        remainder_df = self.dataframe.with_columns(
            pl.col("*") - (pl.col("*") // _other) * _other,
        )
        return self._from_dataframe(
            quotient_df,
        ), self._from_dataframe(remainder_df)

    # Unary

    def __invert__(self) -> DataFrame:
        self._validate_booleanness()
        return self._from_dataframe(
            self.dataframe.select(~pl.col("*")),
        )

    def __iter__(self) -> NoReturn:
        raise NotImplementedError

    # Reductions

    def any(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.select(pl.col("*").any()),
        )

    def all(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.select(pl.col("*").all()),
        )

    def min(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.select(pl.col("*").min()),
        )

    def max(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.select(pl.col("*").max()),
        )

    def sum(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.select(pl.col("*").sum()),
        )

    def prod(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.select(pl.col("*").product()),
        )

    def mean(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.select(pl.col("*").mean()),
        )

    def median(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.select(pl.col("*").median()),
        )

    def std(
        self,
        *,
        correction: float | Scalar | NullType = 1.0,
        skip_nulls: bool | Scalar = True,
    ) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.select(pl.col("*").std()),
        )

    def var(
        self,
        *,
        correction: float | Scalar | NullType = 1.0,
        skip_nulls: bool | Scalar = True,
    ) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.select(pl.col("*").var()),
        )

    # Transformations

    def is_null(self) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.with_columns(pl.col("*").is_null()),
        )

    def is_nan(self) -> DataFrame:
        df = self.dataframe.with_columns(pl.col("*").is_nan())
        return self._from_dataframe(df)

    def fill_nan(
        self,
        value: float | NullType | Scalar,
    ) -> DataFrame:
        _value = validate_comparand(self, value)
        if isinstance(_value, self.__dataframe_namespace__().NullType):
            return self._from_dataframe(
                self.dataframe.fill_nan(pl.lit(None)),
            )
        return self._from_dataframe(
            self.dataframe.fill_nan(_value),
        )

    def fill_null(
        self,
        value: AnyScalar,
        *,
        column_names: list[str] | None = None,
    ) -> DataFrame:
        if column_names is None:
            column_names = self.dataframe.columns
        df = self.dataframe.with_columns(
            pl.col(col).fill_null(value) for col in column_names
        )
        return self._from_dataframe(df)

    def drop_nulls(
        self,
        *,
        column_names: list[str] | None = None,
    ) -> DataFrame:
        namespace = self.__dataframe_namespace__()
        mask = ~namespace.any_horizontal(
            *[
                self.col(col_name).is_null()
                for col_name in column_names or self.column_names
            ],
        )
        return self.filter(mask)

    # Other

    def join(
        self,
        other: DataFrame,
        *,
        how: Literal["left", "inner", "outer"],
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> DataFrame:
        if how not in ["left", "inner", "outer"]:
            msg = f"Expected 'left', 'inner', 'outer', got: {how}"
            raise ValueError(msg)

        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]

        if overlap := (set(self.column_names) - set(left_on)).intersection(
            set(other.column_names) - set(right_on),
        ):
            msg = f"Found overlapping columns in join: {overlap}. Please rename columns to avoid this."
            raise ValueError(msg)

        # workaround for https://github.com/pola-rs/polars/issues/9335
        extra_right_keys = set(right_on).difference(left_on)
        other_df = other.dataframe
        token = generate_random_token(self.column_names + other.column_names)
        other_df = other_df.with_columns(
            [pl.col(i).alias(f"{i}_{token}") for i in extra_right_keys],
        )

        result = self.dataframe.join(
            other_df,  # type: ignore[arg-type]
            left_on=left_on,
            right_on=right_on,
            how=how,
        )
        result = result.rename({f"{i}_{token}": i for i in extra_right_keys})

        return self._from_dataframe(result)

    def persist(self) -> DataFrame:
        if isinstance(self.dataframe, pl.DataFrame):
            warnings.warn(
                "Calling `.persist` on DataFrame that was already persisted",
                UserWarning,
                stacklevel=2,
            )
            df = self.dataframe
        else:
            df = self.dataframe.collect()
        return DataFrame(
            df,
            api_version=self._api_version,
            is_persisted=True,
        )

    # Conversion

    def to_array(self, dtype: DType | None = None) -> Any:
        df = self._validate_is_persisted()
        return df.to_numpy()
