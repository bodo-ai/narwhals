from __future__ import annotations

from typing import TYPE_CHECKING, cast

from narwhals._utils import is_sequence_of

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from bodo.dataframe.group_by import GroupBy as NativeGroupBy
    from bodo.lazyframe.group_by import LazyGroupBy as NativeLazyGroupBy

    from narwhals._bodo.dataframe import BodoDataFrame, BodoLazyFrame
    from narwhals._bodo.expr import BodoExpr


class BodoGroupBy:
    _compliant_frame: BodoDataFrame
    _grouped: NativeGroupBy

    @property
    def compliant(self) -> BodoDataFrame:
        return self._compliant_frame

    def __init__(
        self,
        df: BodoDataFrame,
        keys: Sequence[BodoExpr] | Sequence[str],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        self._keys = list(keys)
        self._compliant_frame = df.drop_nulls(keys) if drop_null_keys else df
        self._grouped = (
            self.compliant.native.group_by(keys)
            if is_sequence_of(keys, str)
            else self.compliant.native.group_by(arg.native for arg in keys)
        )

    def agg(self, *aggs: BodoExpr) -> BodoDataFrame:
        agg_result = self._grouped.agg(arg.native for arg in aggs)
        return self.compliant._with_native(agg_result)

    def __iter__(self) -> Iterator[tuple[tuple[str, ...], BodoDataFrame]]:
        for key, df in self._grouped:
            yield tuple(cast("str", key)), self.compliant._with_native(df)


class BodoLazyGroupBy:
    _compliant_frame: BodoLazyFrame
    _grouped: NativeLazyGroupBy

    @property
    def compliant(self) -> BodoLazyFrame:
        return self._compliant_frame

    def __init__(
        self,
        df: BodoLazyFrame,
        keys: Sequence[BodoExpr] | Sequence[str],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        self._keys = list(keys)
        self._compliant_frame = df.drop_nulls(keys) if drop_null_keys else df
        self._grouped = (
            self.compliant.native.group_by(keys)
            if is_sequence_of(keys, str)
            else self.compliant.native.group_by(arg.native for arg in keys)
        )

    def agg(self, *aggs: BodoExpr) -> BodoLazyFrame:
        agg_result = self._grouped.agg(arg.native for arg in aggs)
        return self.compliant._with_native(agg_result)
