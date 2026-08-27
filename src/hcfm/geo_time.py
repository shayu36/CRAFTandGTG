"""显式 CRS 与城市本地时间/DST 处理。"""

from __future__ import annotations

from typing import Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd


def localize_city_timestamps(
    values: Iterable[object], timezone: str, *, ambiguous: str | bool = "raise",
    nonexistent: str = "raise",
) -> pd.DatetimeIndex:
    """把朴素本地时间显式本地化；默认对 DST 歧义/不存在时刻严格报错。"""

    if not timezone:
        raise ValueError("严格模式: manifest 必须声明 IANA timezone")
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"未知 IANA timezone={timezone!r}") from error
    timestamps = pd.DatetimeIndex(pd.to_datetime(list(values), errors="raise"))
    if timestamps.tz is not None:
        raise ValueError("输入已带 timezone；适配器要求先声明源时区语义，禁止重复 localize")
    return timestamps.tz_localize(timezone, ambiguous=ambiguous, nonexistent=nonexistent)


def transform_wkt_geometries(wkts, source_crs: str, target_crs: str):
    """解析并变换 WKT，严格拒绝空/无效几何与缺失 CRS。"""

    if not source_crs or not target_crs:
        raise ValueError("严格模式: source_crs/target_crs 均必须声明")
    import geopandas as gpd
    from shapely import wkt
    geometries = gpd.GeoSeries([wkt.loads(value) for value in wkts], crs=source_crs)
    if geometries.is_empty.any() or (~geometries.is_valid).any():
        raise ValueError("严格模式: WKT 几何为空或无效")
    return geometries.to_crs(target_crs)

