from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import bodo.pandas as bd

from narwhals._expression_parsing import is_expr, is_series
from narwhals._bodo.expr import BodoExpr
from narwhals._bodo.series import BodoSeries
from narwhals._bodo.utils import extract_args_kwargs, narwhals_to_native_dtype
from narwhals._utils import Implementation, requires, zip_strict
from narwhals.dependencies import is_numpy_array_2d
from narwhals.dtypes import DType

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from datetime import timezone

    from narwhals._compliant import CompliantSelectorNamespace, CompliantWhen
    from narwhals._bodo.dataframe import Method, BodoDataFrame, BodoLazyFrame
    from narwhals._bodo.typing import FrameT
    from narwhals._utils import Version, _LimitedContext
    from narwhals.expr import Expr
    from narwhals.series import Series
    from narwhals.typing import (
        Into1DArray,
        IntoDType,
        IntoSchema,
        NonNestedLiteral,
        TimeUnit,
        _1DArray,
        _2DArray,
    )


class BodoNamespace:
    all: Method[BodoExpr]
    coalesce: Method[BodoExpr]
    col: Method[BodoExpr]
    exclude: Method[BodoExpr]
    sum_horizontal: Method[BodoExpr]
    min_horizontal: Method[BodoExpr]
    max_horizontal: Method[BodoExpr]

    when: Method[CompliantWhen[BodoDataFrame, BodoSeries, BodoExpr]]

    _implementation: Implementation = Implementation.BODO
    _version: Version

    @property
    def _backend_version(self) -> tuple[int, ...]:
        return self._implementation._backend_version()

    def __init__(self, *, version: Version) -> None:
        self._version = version

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            pos, kwds = extract_args_kwargs(args, kwargs)
            return self._expr(getattr(pl, attr)(*pos, **kwds), version=self._version)

        return func

    @property
    def _dataframe(self) -> type[BodoDataFrame]:
        from narwhals._bodo.dataframe import BodoDataFrame

        return BodoDataFrame

    @property
    def _lazyframe(self) -> type[BodoLazyFrame]:
        from narwhals._bodo.dataframe import BodoLazyFrame

        return BodoLazyFrame

    @property
    def _expr(self) -> type[BodoExpr]:
        return BodoExpr

    @property
    def _series(self) -> type[BodoSeries]:
        return BodoSeries

    def parse_into_expr(
        self,
        data: Expr | NonNestedLiteral | Series[bd.Series] | _1DArray,
        /,
        *,
        str_as_lit: bool,
    ) -> BodoExpr | None:
        if data is None:
            # NOTE: To avoid `bd.lit(None)` failing this `None` check
            # https://github.com/pola-rs/bodo/blob/58dd8e5770f16a9bef9009a1c05f00e15a5263c7/py-bodo/bodo/expr/expr.py#L2870-L2872
            return data
        if is_expr(data):
            expr = data._to_compliant_expr(self)
            assert isinstance(expr, self._expr)  # noqa: S101
            return expr
        if isinstance(data, str) and not str_as_lit:
            return self.col(data)
        return self.lit(data.to_native() if is_series(data) else data, None)

    @overload
    def from_native(self, data: bd.DataFrame, /) -> BodoDataFrame: ...
    @overload
    def from_native(self, data: bd.LazyFrame, /) -> BodoLazyFrame: ...
    @overload
    def from_native(self, data: bd.Series, /) -> BodoSeries: ...
    def from_native(
        self, data: bd.DataFrame | bd.LazyFrame | bd.Series | Any, /
    ) -> BodoDataFrame | BodoLazyFrame | BodoSeries:
        if self._dataframe._is_native(data):
            return self._dataframe.from_native(data, context=self)
        if self._series._is_native(data):
            return self._series.from_native(data, context=self)
        if self._lazyframe._is_native(data):
            return self._lazyframe.from_native(data, context=self)
        msg = f"Unsupported type: {type(data).__name__!r}"  # pragma: no cover
        raise TypeError(msg)  # pragma: no cover

    @overload
    def from_numpy(self, data: Into1DArray, /, schema: None = ...) -> BodoSeries: ...

    @overload
    def from_numpy(
        self, data: _2DArray, /, schema: IntoSchema | Sequence[str] | None
    ) -> BodoDataFrame: ...

    def from_numpy(
        self,
        data: Into1DArray | _2DArray,
        /,
        schema: IntoSchema | Sequence[str] | None = None,
    ) -> BodoDataFrame | BodoSeries:
        if is_numpy_array_2d(data):
            return self._dataframe.from_numpy(data, schema=schema, context=self)
        return self._series.from_numpy(data, context=self)  # pragma: no cover

    @requires.backend_version(
        (1, 0, 0), "Please use `col` for columns selection instead."
    )
    def nth(self, *indices: int) -> BodoExpr:
        return self._expr(bd.nth(*indices), version=self._version)

    def len(self) -> BodoExpr:
        if self._backend_version < (0, 20, 5):
            return self._expr(bd.count().alias("len"), self._version)
        return self._expr(bd.len(), self._version)

    def all_horizontal(self, *exprs: BodoExpr, ignore_nulls: bool) -> BodoExpr:
        it = (expr.fill_null(True) for expr in exprs) if ignore_nulls else iter(exprs)
        return self._expr(bd.all_horizontal(*(expr.native for expr in it)), self._version)

    def any_horizontal(self, *exprs: BodoExpr, ignore_nulls: bool) -> BodoExpr:
        it = (expr.fill_null(False) for expr in exprs) if ignore_nulls else iter(exprs)
        return self._expr(bd.any_horizontal(*(expr.native for expr in it)), self._version)

    def concat(
        self,
        items: Iterable[FrameT],
        *,
        how: Literal["vertical", "horizontal", "diagonal"],
    ) -> BodoDataFrame | BodoLazyFrame:
        result = bd.concat((item.native for item in items), how=how)
        if isinstance(result, bd.DataFrame):
            return self._dataframe(result, version=self._version)
        return self._lazyframe.from_native(result, context=self)

    def lit(self, value: Any, dtype: IntoDType | None) -> BodoExpr:
        if dtype is not None:
            return self._expr(
                bd.lit(value, dtype=narwhals_to_native_dtype(dtype, self._version)),
                version=self._version,
            )
        return self._expr(bd.lit(value), version=self._version)

    def mean_horizontal(self, *exprs: BodoExpr) -> BodoExpr:
        if self._backend_version < (0, 20, 8):
            return self._expr(
                bd.sum_horizontal(e._native_expr for e in exprs)
                / bd.sum_horizontal(1 - e.is_null()._native_expr for e in exprs),
                version=self._version,
            )

        return self._expr(
            bd.mean_horizontal(e._native_expr for e in exprs), version=self._version
        )

    def concat_str(
        self, *exprs: BodoExpr, separator: str, ignore_nulls: bool
    ) -> BodoExpr:
        pl_exprs: list[bd.Expr] = [expr._native_expr for expr in exprs]

        if self._backend_version < (0, 20, 6):
            null_mask = [expr.is_null() for expr in pl_exprs]
            sep = bd.lit(separator)

            if not ignore_nulls:
                null_mask_result = bd.any_horizontal(*null_mask)
                output_expr = bd.reduce(
                    lambda x, y: x.cast(bd.String()) + sep + y.cast(bd.String()),  # type: ignore[arg-type,return-value]
                    pl_exprs,
                )
                result = bd.when(~null_mask_result).then(output_expr)
            else:
                init_value, *values = [
                    bd.when(nm).then(bd.lit("")).otherwise(expr.cast(bd.String()))
                    for expr, nm in zip_strict(pl_exprs, null_mask)
                ]
                separators = [
                    bd.when(~nm).then(sep).otherwise(bd.lit("")) for nm in null_mask[:-1]
                ]

                result = bd.fold(  # type: ignore[assignment]
                    acc=init_value,
                    function=operator.add,
                    exprs=[s + v for s, v in zip_strict(separators, values)],
                )

            return self._expr(result, version=self._version)

        return self._expr(
            bd.concat_str(pl_exprs, separator=separator, ignore_nulls=ignore_nulls),
            version=self._version,
        )

    # NOTE: Implementation is too different to annotate correctly (vs other `*SelectorNamespace`)
    # 1. Others have lots of private stuff for code reuse
    #    i. None of that is useful here
    # 2. We don't have a `BodoSelector` abstraction, and just use `BodoExpr`
    @property
    def selectors(self) -> CompliantSelectorNamespace[BodoDataFrame, BodoSeries]:
        return cast(
            "CompliantSelectorNamespace[BodoDataFrame, BodoSeries]",
            BodoSelectorNamespace(self),
        )


class BodoSelectorNamespace:
    _implementation = Implementation.BODO

    def __init__(self, context: _LimitedContext, /) -> None:
        self._version = context._version

    def by_dtype(self, dtypes: Iterable[DType]) -> BodoExpr:
        native_dtypes = [
            narwhals_to_native_dtype(dtype, self._version).__class__
            if isinstance(dtype, type) and issubclass(dtype, DType)
            else narwhals_to_native_dtype(dtype, self._version)
            for dtype in dtypes
        ]
        return BodoExpr(bd.selectors.by_dtype(native_dtypes), version=self._version)

    def matches(self, pattern: str) -> BodoExpr:
        return BodoExpr(bd.selectors.matches(pattern=pattern), version=self._version)

    def numeric(self) -> BodoExpr:
        return BodoExpr(bd.selectors.numeric(), version=self._version)

    def boolean(self) -> BodoExpr:
        return BodoExpr(bd.selectors.boolean(), version=self._version)

    def string(self) -> BodoExpr:
        return BodoExpr(bd.selectors.string(), version=self._version)

    def categorical(self) -> BodoExpr:
        return BodoExpr(bd.selectors.categorical(), version=self._version)

    def all(self) -> BodoExpr:
        return BodoExpr(bd.selectors.all(), version=self._version)

    def datetime(
        self,
        time_unit: TimeUnit | Iterable[TimeUnit] | None,
        time_zone: str | timezone | Iterable[str | timezone | None] | None,
    ) -> BodoExpr:
        return BodoExpr(
            bd.selectors.datetime(time_unit=time_unit, time_zone=time_zone),  # type: ignore[arg-type]
            version=self._version,
        )
