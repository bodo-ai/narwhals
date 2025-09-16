from __future__ import annotations

import abc
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol, TypeVar, overload

import bodo.pandas as bd
import numpy as np

from narwhals._duration import Interval
from narwhals._utils import (
    Implementation,
    Version,
    _DeferredIterable,
    _StoresCompliant,
    _StoresNative,
    deep_getattr,
    isinstance_or_issubclass,
)
from narwhals.exceptions import (
    ColumnNotFoundError,
    ComputeError,
    DuplicateError,
    InvalidOperationError,
    NarwhalsError,
    ShapeError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from typing_extensions import TypeIs

    from narwhals._bodo.dataframe import Method
    from narwhals._bodo.expr import BodoExpr
    from narwhals._bodo.series import BodoSeries
    from narwhals._bodo.typing import NativeAccessor
    from narwhals.dtypes import DType
    from narwhals.typing import IntoDType

    T = TypeVar("T")
    NativeT = TypeVar(
        "NativeT", bound="bd.DataFrame | bd.LazyFrame | bd.Series | bd.Expr"
    )

NativeT_co = TypeVar("NativeT_co", "bd.Series", "bd.Expr", covariant=True)
CompliantT_co = TypeVar("CompliantT_co", "BodoSeries", "BodoExpr", covariant=True)
CompliantT = TypeVar("CompliantT", "BodoSeries", "BodoExpr")

BACKEND_VERSION = Implementation.BODO._backend_version()
"""Static backend version for `bodo`."""

SERIES_RESPECTS_DTYPE: Final[bool] = BACKEND_VERSION >= (0, 20, 26)
"""`bd.Series(dtype=...)` fixed in https://github.com/pola-rs/bodo/pull/15962

Includes `SERIES_ACCEPTS_PD_INDEX`.
"""

SERIES_ACCEPTS_PD_INDEX: Final[bool] = BACKEND_VERSION >= (0, 20, 7)
"""`bd.Series(values: pd.Index)` fixed in https://github.com/pola-rs/bodo/pull/14087"""


@overload
def extract_native(obj: _StoresNative[NativeT]) -> NativeT: ...
@overload
def extract_native(obj: T) -> T: ...
def extract_native(obj: _StoresNative[NativeT] | T) -> NativeT | T:
    return obj.native if _is_compliant_bodo(obj) else obj


def _is_compliant_bodo(
    obj: _StoresNative[NativeT] | Any,
) -> TypeIs[_StoresNative[NativeT]]:
    from narwhals._bodo.dataframe import BodoDataFrame, BodoLazyFrame
    from narwhals._bodo.expr import BodoExpr
    from narwhals._bodo.series import BodoSeries

    return isinstance(obj, (BodoDataFrame, BodoLazyFrame, BodoSeries, BodoExpr))


def extract_args_kwargs(
    args: Iterable[Any], kwds: Mapping[str, Any], /
) -> tuple[Iterator[Any], dict[str, Any]]:
    it_args = (extract_native(arg) for arg in args)
    return it_args, {k: extract_native(v) for k, v in kwds.items()}


@lru_cache(maxsize=16)
def native_to_narwhals_dtype(  # noqa: C901, PLR0912
    dtype: bd.DataType, version: Version
) -> DType:
    dtypes = version.dtypes
    if dtype == bd.Float64:
        return dtypes.Float64()
    if dtype == bd.Float32:
        return dtypes.Float32()
    if hasattr(pl, "Int128") and dtype == bd.Int128:  # pragma: no cover
        # Not available for Bodo pre 1.8.0
        return dtypes.Int128()
    if dtype == bd.Int64:
        return dtypes.Int64()
    if dtype == bd.Int32:
        return dtypes.Int32()
    if dtype == bd.Int16:
        return dtypes.Int16()
    if dtype == bd.Int8:
        return dtypes.Int8()
    if hasattr(pl, "UInt128") and dtype == bd.UInt128:  # pragma: no cover
        # Not available for Bodo pre 1.8.0
        return dtypes.UInt128()
    if dtype == bd.UInt64:
        return dtypes.UInt64()
    if dtype == bd.UInt32:
        return dtypes.UInt32()
    if dtype == bd.UInt16:
        return dtypes.UInt16()
    if dtype == bd.UInt8:
        return dtypes.UInt8()
    if dtype == bd.String:
        return dtypes.String()
    if dtype == bd.Boolean:
        return dtypes.Boolean()
    if dtype == bd.Object:
        return dtypes.Object()
    if dtype == bd.Categorical:
        return dtypes.Categorical()
    if isinstance_or_issubclass(dtype, bd.Enum):
        if version is Version.V1:
            return dtypes.Enum()  # type: ignore[call-arg]
        categories = _DeferredIterable(dtype.categories.to_list)
        return dtypes.Enum(categories)
    if dtype == bd.Date:
        return dtypes.Date()
    if isinstance_or_issubclass(dtype, bd.Datetime):
        return (
            dtypes.Datetime()
            if dtype is bd.Datetime
            else dtypes.Datetime(dtype.time_unit, dtype.time_zone)
        )
    if isinstance_or_issubclass(dtype, bd.Duration):
        return (
            dtypes.Duration()
            if dtype is bd.Duration
            else dtypes.Duration(dtype.time_unit)
        )
    if isinstance_or_issubclass(dtype, bd.Struct):
        fields = [
            dtypes.Field(name, native_to_narwhals_dtype(tp, version))
            for name, tp in dtype
        ]
        return dtypes.Struct(fields)
    if isinstance_or_issubclass(dtype, bd.List):
        return dtypes.List(native_to_narwhals_dtype(dtype.inner, version))
    if isinstance_or_issubclass(dtype, bd.Array):
        outer_shape = dtype.width if BACKEND_VERSION < (0, 20, 30) else dtype.size
        return dtypes.Array(native_to_narwhals_dtype(dtype.inner, version), outer_shape)
    if dtype == bd.Decimal:
        return dtypes.Decimal()
    if dtype == bd.Time:
        return dtypes.Time()
    if dtype == bd.Binary:
        return dtypes.Binary()
    return dtypes.Unknown()


dtypes = Version.MAIN.dtypes
NW_TO_PL_DTYPES: Mapping[type[DType], bd.DataType] = {
    dtypes.Float64: bd.Float64(),
    dtypes.Float32: bd.Float32(),
    dtypes.Binary: bd.Binary(),
    dtypes.String: bd.String(),
    dtypes.Boolean: bd.Boolean(),
    dtypes.Categorical: bd.Categorical(),
    dtypes.Date: bd.Date(),
    dtypes.Time: bd.Time(),
    dtypes.Int8: bd.Int8(),
    dtypes.Int16: bd.Int16(),
    dtypes.Int32: bd.Int32(),
    dtypes.Int64: bd.Int64(),
    dtypes.UInt8: bd.UInt8(),
    dtypes.UInt16: bd.UInt16(),
    dtypes.UInt32: bd.UInt32(),
    dtypes.UInt64: bd.UInt64(),
    dtypes.Decimal: bd.Decimal(),
}
UNSUPPORTED_DTYPES = (dtypes.Object, dtypes.Unknown)


def narwhals_to_native_dtype(  # noqa: C901
    dtype: IntoDType, version: Version
) -> bd.DataType:
    dtypes = version.dtypes
    base_type = dtype.base_type()
    if pl_type := NW_TO_PL_DTYPES.get(base_type):
        return pl_type
    if dtype == dtypes.Int128 and hasattr(pl, "Int128"):
        # Not available for Bodo pre 1.8.0
        return bd.Int128()
    if isinstance_or_issubclass(dtype, dtypes.Enum):
        if version is Version.V1:
            msg = "Converting to Enum is not supported in narwhals.stable.v1"
            raise NotImplementedError(msg)
        if isinstance(dtype, dtypes.Enum):
            return bd.Enum(dtype.categories)
        msg = "Can not cast / initialize Enum without categories present"
        raise ValueError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Datetime):
        return bd.Datetime(dtype.time_unit, dtype.time_zone)  # type: ignore[arg-type]
    if isinstance_or_issubclass(dtype, dtypes.Duration):
        return bd.Duration(dtype.time_unit)  # type: ignore[arg-type]
    if isinstance_or_issubclass(dtype, dtypes.List):
        return bd.List(narwhals_to_native_dtype(dtype.inner, version))
    if isinstance_or_issubclass(dtype, dtypes.Struct):
        fields = [
            bd.Field(field.name, narwhals_to_native_dtype(field.dtype, version))
            for field in dtype.fields
        ]
        return bd.Struct(fields)
    if isinstance_or_issubclass(dtype, dtypes.Array):  # pragma: no cover
        size = dtype.size
        kwargs = {"width": size} if BACKEND_VERSION < (0, 20, 30) else {"shape": size}
        return bd.Array(narwhals_to_native_dtype(dtype.inner, version), **kwargs)
    if issubclass(base_type, UNSUPPORTED_DTYPES):
        msg = f"Converting to {base_type.__name__} dtype is not supported for Bodo."
        raise NotImplementedError(msg)
    return bd.Unknown()  # pragma: no cover


def _is_bodo_exception(exception: Exception) -> bool:
    if BACKEND_VERSION >= (1,):
        # Old versions of Bodo didn't have BodoError.
        return isinstance(exception, bd.exceptions.BodoError)
    # Last attempt, for old Bodo versions.
    return "bodo.exceptions" in str(type(exception))  # pragma: no cover


def _is_cudf_exception(exception: Exception) -> bool:
    # These exceptions are raised when running bodo on GPUs via cuDF
    return str(exception).startswith("CUDF failure")


def catch_bodo_exception(exception: Exception) -> NarwhalsError | Exception:
    if isinstance(exception, bd.exceptions.ColumnNotFoundError):
        return ColumnNotFoundError(str(exception))
    if isinstance(exception, bd.exceptions.ShapeError):
        return ShapeError(str(exception))
    if isinstance(exception, bd.exceptions.InvalidOperationError):
        return InvalidOperationError(str(exception))
    if isinstance(exception, bd.exceptions.DuplicateError):
        return DuplicateError(str(exception))
    if isinstance(exception, bd.exceptions.ComputeError):
        return ComputeError(str(exception))
    if _is_bodo_exception(exception) or _is_cudf_exception(exception):
        return NarwhalsError(str(exception))  # pragma: no cover
    # Just return exception as-is.
    return exception


class BodoAnyNamespace(
    _StoresCompliant[CompliantT_co],
    _StoresNative[NativeT_co],
    Protocol[CompliantT_co, NativeT_co],
):
    _accessor: ClassVar[NativeAccessor]

    def __getattr__(self, attr: str) -> Callable[..., CompliantT_co]:
        def func(*args: Any, **kwargs: Any) -> CompliantT_co:
            pos, kwds = extract_args_kwargs(args, kwargs)
            method = deep_getattr(self.native, self._accessor, attr)
            return self.compliant._with_native(method(*pos, **kwds))

        return func


class BodoDateTimeNamespace(BodoAnyNamespace[CompliantT, NativeT_co]):
    _accessor: ClassVar[NativeAccessor] = "dt"

    def truncate(self, every: str) -> CompliantT:
        # Ensure consistent error message is raised.
        Interval.parse(every)
        return self.__getattr__("truncate")(every)

    def offset_by(self, by: str) -> CompliantT:
        # Ensure consistent error message is raised.
        Interval.parse_no_constraints(by)
        return self.__getattr__("offset_by")(by)

    to_string: Method[CompliantT]
    replace_time_zone: Method[CompliantT]
    convert_time_zone: Method[CompliantT]
    timestamp: Method[CompliantT]
    date: Method[CompliantT]
    year: Method[CompliantT]
    month: Method[CompliantT]
    day: Method[CompliantT]
    hour: Method[CompliantT]
    minute: Method[CompliantT]
    second: Method[CompliantT]
    millisecond: Method[CompliantT]
    microsecond: Method[CompliantT]
    nanosecond: Method[CompliantT]
    ordinal_day: Method[CompliantT]
    weekday: Method[CompliantT]
    total_minutes: Method[CompliantT]
    total_seconds: Method[CompliantT]
    total_milliseconds: Method[CompliantT]
    total_microseconds: Method[CompliantT]
    total_nanoseconds: Method[CompliantT]


class BodoStringNamespace(BodoAnyNamespace[CompliantT, NativeT_co]):
    _accessor: ClassVar[NativeAccessor] = "str"

    # NOTE: Use `abstractmethod` if we have defs to implement, but also `Method` usage
    @abc.abstractmethod
    def zfill(self, width: int) -> CompliantT: ...

    len_chars: Method[CompliantT]
    replace: Method[CompliantT]
    replace_all: Method[CompliantT]
    strip_chars: Method[CompliantT]
    starts_with: Method[CompliantT]
    ends_with: Method[CompliantT]
    contains: Method[CompliantT]
    slice: Method[CompliantT]
    split: Method[CompliantT]
    to_date: Method[CompliantT]
    to_datetime: Method[CompliantT]
    to_lowercase: Method[CompliantT]
    to_uppercase: Method[CompliantT]


class BodoCatNamespace(BodoAnyNamespace[CompliantT, NativeT_co]):
    _accessor: ClassVar[NativeAccessor] = "cat"
    get_categories: Method[CompliantT]


class BodoListNamespace(BodoAnyNamespace[CompliantT, NativeT_co]):
    _accessor: ClassVar[NativeAccessor] = "list"

    @abc.abstractmethod
    def len(self) -> CompliantT: ...

    get: Method[CompliantT]

    unique: Method[CompliantT]


class BodoStructNamespace(BodoAnyNamespace[CompliantT, NativeT_co]):
    _accessor: ClassVar[NativeAccessor] = "struct"
    field: Method[CompliantT]
