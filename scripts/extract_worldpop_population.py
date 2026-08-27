#!/usr/bin/env python3
"""Extract city population points from the 2026 WorldPop BigTIFF.

The input is a tiled BigTIFF which is too large to decode as one Pillow image.
This script reads one libtiff tile at a time through ``ctypes`` and writes only
valid pixels whose centers fall inside a city's full study grid.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import math
import os
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from shapely import contains_xy
from shapely.ops import unary_union


TIFFTAG_IMAGEWIDTH = 256
TIFFTAG_IMAGELENGTH = 257
TIFFTAG_BITSPERSAMPLE = 258
TIFFTAG_COMPRESSION = 259
TIFFTAG_SAMPLESPERPIXEL = 277
TIFFTAG_PLANARCONFIG = 284
TIFFTAG_SAMPLEFORMAT = 339
TIFFTAG_TILEWIDTH = 322
TIFFTAG_TILELENGTH = 323
TIFFTAG_MODELPIXELSCALE = 33550
TIFFTAG_MODELTIEPOINT = 33922
TIFFTAG_GEOKEYDIRECTORY = 34735
TIFFTAG_GDAL_NODATA = 42113
TIFFTAG_IMAGEDESCRIPTION = 270

EXPECTED_DESCRIPTION = "CHN population 2026 [WorldPop R2025A v1]"
EXPECTED_PIXEL_SIZE = 1.0 / 1200.0


def _read_pillow_metadata(path: Path):
    """Read tags without decoding image pixels."""
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        tags = image.tag_v2
        scales = tags.get(TIFFTAG_MODELPIXELSCALE)
        ties = tags.get(TIFFTAG_MODELTIEPOINT)
        if scales is None or ties is None or len(scales) < 2 or len(ties) < 5:
            raise ValueError("严格模式: GeoTIFF 缺少地理参考标签")
        description = str(tags.get(TIFFTAG_IMAGEDESCRIPTION, ""))
        nodata = str(tags.get(TIFFTAG_GDAL_NODATA, ""))
        geokeys = tuple(tags.get(TIFFTAG_GEOKEYDIRECTORY, ()))
        epsg = None
        if len(geokeys) >= 4:
            for offset in range(4, len(geokeys), 4):
                key_id, location, count, value = geokeys[offset : offset + 4]
                if key_id == 2048 and location == 0 and count == 1:
                    epsg = int(value)
                    break
        return (
            float(scales[0]),
            float(scales[1]),
            float(ties[3]),
            float(ties[4]),
            nodata,
            description,
            epsg,
        )


class _WorldPopReader:
    """Small ctypes wrapper around tiled libtiff reads."""

    def __init__(self, path: Path):
        name = ctypes.util.find_library("tiff")
        if not name:
            raise RuntimeError("严格模式: 找不到 libtiff，无法读取 BigTIFF")
        self.lib = ctypes.CDLL(name)
        self.lib.TIFFOpen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.TIFFOpen.restype = ctypes.c_void_p
        self.lib.TIFFClose.argtypes = [ctypes.c_void_p]
        # All calls below request one scalar/pointer output; declaring the
        # three fixed arguments avoids ctypes' unsafe variadic conversion.
        self.lib.TIFFGetField.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        self.lib.TIFFGetField.restype = ctypes.c_int
        self.lib.TIFFReadEncodedTile.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.TIFFReadEncodedTile.restype = ctypes.c_ssize_t
        self.lib.TIFFComputeTile.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint16]
        self.lib.TIFFComputeTile.restype = ctypes.c_uint32
        self.lib.TIFFTileSize.argtypes = [ctypes.c_void_p]
        self.lib.TIFFTileSize.restype = ctypes.c_size_t
        self.lib.TIFFNumberOfTiles.argtypes = [ctypes.c_void_p]
        self.lib.TIFFNumberOfTiles.restype = ctypes.c_uint64
        self.path = path
        self.handle = self.lib.TIFFOpen(os.fsencode(path), b"r")
        if not self.handle:
            raise OSError(f"严格模式: libtiff 无法打开 {path}")
        self.width = self._field_u32(TIFFTAG_IMAGEWIDTH)
        self.height = self._field_u32(TIFFTAG_IMAGELENGTH)
        self.bits = self._field_u16(TIFFTAG_BITSPERSAMPLE)
        self.samples = self._field_u16(TIFFTAG_SAMPLESPERPIXEL)
        self.planar = self._field_u16(TIFFTAG_PLANARCONFIG)
        self.sample_format = self._field_u16(TIFFTAG_SAMPLEFORMAT)
        self.tile_width = self._field_u32(TIFFTAG_TILEWIDTH)
        self.tile_length = self._field_u32(TIFFTAG_TILELENGTH)
        self.tile_size = int(self.lib.TIFFTileSize(self.handle))
        self.num_tiles = int(self.lib.TIFFNumberOfTiles(self.handle))
        (
            self.xres,
            self.yres,
            self.origin_x,
            self.origin_y,
            self.nodata,
            self.description,
            self.epsg,
        ) = _read_pillow_metadata(path)
        self._validate()

    def _field_u32(self, tag):
        value = ctypes.c_uint32()
        if not self.lib.TIFFGetField(self.handle, tag, ctypes.byref(value)):
            raise ValueError(f"严格模式: GeoTIFF 缺少 TIFF tag {tag}")
        return int(value.value)

    def _field_u16(self, tag):
        value = ctypes.c_uint16()
        if not self.lib.TIFFGetField(self.handle, tag, ctypes.byref(value)):
            raise ValueError(f"严格模式: GeoTIFF 缺少 TIFF tag {tag}")
        return int(value.value)

    def _validate(self):
        if self.width <= 0 or self.height <= 0 or self.tile_width <= 0 or self.tile_length <= 0:
            raise ValueError("严格模式: WorldPop 栅格尺寸/瓦片尺寸无效")
        if (self.bits, self.samples, self.planar, self.sample_format) != (32, 1, 1, 3):
            raise ValueError("严格模式: WorldPop 必须为单波段 float32 contiguous 栅格")
        if not math.isclose(self.xres, EXPECTED_PIXEL_SIZE, rel_tol=0, abs_tol=2e-8) or not math.isclose(
            self.yres, EXPECTED_PIXEL_SIZE, rel_tol=0, abs_tol=2e-8
        ):
            raise ValueError(f"严格模式: WorldPop 分辨率异常 {(self.xres, self.yres)}")
        if self.description != EXPECTED_DESCRIPTION:
            raise ValueError(f"严格模式: WorldPop 产品描述异常 {self.description!r}")
        if self.epsg != 4326:
            raise ValueError(f"严格模式: WorldPop CRS 必须为 EPSG:4326，实际 {self.epsg}")
        if not self.nodata:
            raise ValueError("严格模式: WorldPop 缺少 GDAL NoData 标签")
        try:
            self.nodata = float(self.nodata)
        except ValueError as exc:
            raise ValueError(f"严格模式: WorldPop NoData 不是数值 {self.nodata!r}") from exc
        if not math.isfinite(self.nodata):
            raise ValueError("严格模式: WorldPop NoData 非有限")
        if self.num_tiles <= 0 or self.tile_size <= 0:
            raise ValueError("严格模式: WorldPop 没有可读瓦片")

    def read_tile(self, tile_x: int, tile_y: int) -> np.ndarray:
        x = tile_x * self.tile_width
        y = tile_y * self.tile_length
        tile = int(self.lib.TIFFComputeTile(self.handle, x, y, 0, 0))
        buf = (ctypes.c_ubyte * self.tile_size)()
        got = int(self.lib.TIFFReadEncodedTile(self.handle, tile, buf, self.tile_size))
        if got <= 0:
            raise OSError(f"严格模式: 无法读取 WorldPop tile ({tile_x},{tile_y})")
        rows = min(self.tile_length, self.height - y)
        cols = min(self.tile_width, self.width - x)
        expected = rows * cols * 4
        if got < expected:
            raise OSError(f"严格模式: WorldPop tile ({tile_x},{tile_y}) 解码字节数不足 {got}<{expected}")
        return np.frombuffer(buf, dtype="<f4", count=rows * self.tile_width).reshape(rows, self.tile_width)[:, :cols]

    def close(self):
        if self.handle:
            self.lib.TIFFClose(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def _load_grid(path: Path):
    grid = gpd.read_file(path)
    if grid.crs is None or grid.crs.to_epsg() != 4326:
        raise ValueError(f"严格模式: {path} 必须为 EPSG:4326")
    if grid.empty or grid.geometry.is_empty.any() or grid.geometry.isna().any() or (~grid.geometry.is_valid).any():
        raise ValueError(f"严格模式: {path} 含空/无效 geometry")
    return grid


def extract_city(tif_path: Path, grid_path: Path, output_path: Path) -> dict:
    grid = _load_grid(grid_path)
    study_union = unary_union(grid.geometry.tolist())
    min_lon, min_lat, max_lon, max_lat = map(float, study_union.bounds)
    rows_out = []
    valid_pixels = 0
    zero_pixels = 0
    with _WorldPopReader(tif_path) as reader:
        # Pixel centers use the north-up convention of the GeoTIFF tie point.
        c0 = max(0, int(math.floor((min_lon - reader.origin_x) / reader.xres - 0.5)))
        c1 = min(reader.width - 1, int(math.ceil((max_lon - reader.origin_x) / reader.xres - 0.5)))
        r0 = max(0, int(math.floor((reader.origin_y - max_lat) / reader.yres - 0.5)))
        r1 = min(reader.height - 1, int(math.ceil((reader.origin_y - min_lat) / reader.yres - 0.5)))
        if c0 > c1 or r0 > r1:
            raise ValueError("严格模式: study area 与 WorldPop 栅格没有像元重叠")
        tx0, tx1 = c0 // reader.tile_width, c1 // reader.tile_width
        ty0, ty1 = r0 // reader.tile_length, r1 // reader.tile_length
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                arr = reader.read_tile(tx, ty)
                x_start, y_start = tx * reader.tile_width, ty * reader.tile_length
                cc0, cc1 = max(c0, x_start), min(c1, x_start + arr.shape[1] - 1)
                rr0, rr1 = max(r0, y_start), min(r1, y_start + arr.shape[0] - 1)
                if cc0 > cc1 or rr0 > rr1:
                    continue
                sub = arr[rr0 - y_start : rr1 - y_start + 1, cc0 - x_start : cc1 - x_start + 1]
                yy, xx = np.indices(sub.shape)
                cols = xx + cc0
                rows = yy + rr0
                lon = reader.origin_x + (cols + 0.5) * reader.xres
                lat = reader.origin_y - (rows + 0.5) * reader.yres
                valid = np.isfinite(sub) & (sub != reader.nodata) & (sub >= 0)
                if np.any(valid):
                    inside = contains_xy(study_union, lon, lat) & valid
                    if np.any(inside):
                        vals = sub[inside].astype(float)
                        if not np.isfinite(vals).all() or (vals < 0).any():
                            raise ValueError("严格模式: WorldPop 有效像元含负值/NaN/Inf")
                        rows_out.extend(zip(lon[inside].tolist(), lat[inside].tolist(), vals.tolist()))
                        valid_pixels += int(vals.size)
                        zero_pixels += int(np.count_nonzero(vals == 0))
    if not rows_out:
        raise ValueError("严格模式: study area 内没有有效 WorldPop 像元")
    frame = pd.DataFrame(rows_out, columns=["lon", "lat", "population"])
    if frame.duplicated(["lon", "lat"]).any():
        raise ValueError("严格模式: 输出人口点存在重复像元坐标")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        raise FileExistsError(f"严格模式: 拒绝覆盖已有非空输出 {output_path}")
    fd, tmp_name = tempfile.mkstemp(prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        frame.to_csv(tmp_path, index=False)
        check = pd.read_csv(tmp_path)
        if list(check.columns) != ["lon", "lat", "population"] or len(check) != len(frame):
            raise ValueError("严格模式: 临时人口 CSV 校验失败")
        tmp_path.replace(output_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return {
        "output": str(output_path),
        "input_points": int(valid_pixels),
        "zero_population_points": int(zero_pixels),
        "bbox": [min_lon, min_lat, max_lon, max_lat],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worldpop", required=True)
    ap.add_argument("--study-area-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--cities", nargs="+", default=["beijing", "chengdushi", "xianshi"])
    args = ap.parse_args()
    reports = []
    for city in args.cities:
        print(f"[extract_worldpop_population] processing {city} ...")
        reports.append(
            extract_city(
                Path(args.worldpop),
                Path(args.study_area_root) / city / "study_area" / "full_grid.geojson",
                Path(args.output_root) / city / "population_2026.csv",
            )
        )
    print(json.dumps({"cities": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
