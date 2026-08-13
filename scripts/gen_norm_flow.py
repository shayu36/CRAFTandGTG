"""生成 CRAFT 缺失的 norm_{phase}_len_{seq_length}.csv 归一化流量文件。

背景（详见 docs/STAGE1_AUDIT.md 第 4.5 节）:
  CRAFT 数据集随附 slide_bike_flow_{train,test}.csv (原始流量, 最大可达 335),
  但**未**随附 norm_*.csv, 亦无生成脚本。evaluate.py 全程在归一化空间比较,
  norm_*.csv 只需 [0,1] 归一化流量, 无需存储反归一化参数。

归一化规则 (严格无泄漏):
  - norm_mode='global' (默认): 逐城市, 对训练集全体 in/out flow 值取 min/max, 应用到 train 和 test。
  - norm_mode='region':        逐区域 min-max, train 拟合; 未出现在 train 的 test 区域回退到该城市训练全局统计 (计数记录)。
  归一化参数**只用训练集拟合**, 再应用到 test, 严禁用 test 统计。
  test 值超出 train 范围 → 裁剪到 [0,1] 并计数记录 (不静默)。

只读: CRAFT/cleared_data/{city}/slide_bike_flow_{train,test}.csv
写出: Paper/data/norm_flow/{city}/norm_{phase}_len_{seq_length}.csv
"""
import argparse
import ast
import json
import os
import sys
from os.path import join

import numpy as np
import pandas as pd

CITIES = ["chi", "dc", "toronto", "ny"]


def _parse_flow(series):
    return series.apply(lambda x: np.asarray(ast.literal_eval(x), dtype=np.float64))


def _stack(arrs):
    return np.stack(arrs.tolist(), axis=0)  # (num_rows, seq_len)


def fit_global(train_df):
    """对训练集全体 in/out flow 取 min/max。"""
    vals = np.concatenate([
        np.concatenate(train_df["in_flow"].tolist()),
        np.concatenate(train_df["out_flow"].tolist()),
    ])
    return float(vals.min()), float(vals.max())


def fit_per_region(train_df):
    """逐区域 min/max, 返回 {region_id: (min,max)}。"""
    stats = {}
    for rid, grp in train_df.groupby("region_id"):
        vals = np.concatenate([
            np.concatenate(grp["in_flow"].tolist()),
            np.concatenate(grp["out_flow"].tolist()),
        ])
        stats[int(rid)] = (float(vals.min()), float(vals.max()))
    return stats


def apply_norm(df, lo, hi):
    """min-max 到 [0,1], 统计裁剪数量。返回 (df, clip_count, total_count)。"""
    span = hi - lo
    if span <= 0:
        raise ValueError(f"退化归一化区间 lo={lo} hi={hi}, 无法归一化 (严格模式报错)")
    clip = 0
    total = 0
    for col in ["in_flow", "out_flow"]:
        out = []
        for arr in df[col]:
            v = (arr - lo) / span
            total += v.size
            clip += int(np.sum((v < 0) | (v > 1)))
            out.append(np.clip(v, 0.0, 1.0).tolist())
        df[col] = out
    return df, clip, total


def process_city(city, craft_root, out_root, seq_length, norm_mode):
    cdir = join(craft_root, city)
    tr_pth = join(cdir, "slide_bike_flow_train.csv")
    te_pth = join(cdir, "slide_bike_flow_test.csv")
    for p in (tr_pth, te_pth):
        if not os.path.exists(p):
            raise FileNotFoundError(f"严格模式: 缺失流量文件 {p}")

    train_df = pd.read_csv(tr_pth)
    test_df = pd.read_csv(te_pth)
    for df in (train_df, test_df):
        df["in_flow"] = _parse_flow(df["in_flow"])
        df["out_flow"] = _parse_flow(df["out_flow"])

    # 校验序列长度
    for name, df in [("train", train_df), ("test", test_df)]:
        lens = df["in_flow"].apply(len).unique().tolist()
        if lens != [seq_length]:
            raise ValueError(f"严格模式: {city} {name} in_flow 长度 {lens} != [{seq_length}]")

    report = {"city": city, "norm_mode": norm_mode}

    if norm_mode == "global":
        lo, hi = fit_global(train_df)
        report.update({"global_min": lo, "global_max": hi})
        outputs = {}
        for phase, df in [("train", train_df.copy()), ("test", test_df.copy())]:
            df, clip, total = apply_norm(df, lo, hi)
            outputs[phase] = df
            report[f"{phase}_clipped"] = clip
            report[f"{phase}_total_vals"] = total
    elif norm_mode == "region":
        stats = fit_per_region(train_df)
        g_lo, g_hi = fit_global(train_df)  # 未见区域回退
        report.update({"num_train_regions": len(stats), "fallback_global": [g_lo, g_hi]})
        outputs = {}
        for phase, df in [("train", train_df.copy()), ("test", test_df.copy())]:
            fallback_regions = set()
            clip_total = 0
            val_total = 0
            parts = []
            for rid, grp in df.groupby("region_id"):
                lo, hi = stats.get(int(rid), (g_lo, g_hi))
                if int(rid) not in stats:
                    fallback_regions.add(int(rid))
                grp, clip, total = apply_norm(grp.copy(), lo, hi)
                clip_total += clip
                val_total += total
                parts.append(grp)
            outputs[phase] = pd.concat(parts).sort_index()
            report[f"{phase}_clipped"] = clip_total
            report[f"{phase}_total_vals"] = val_total
            report[f"{phase}_fallback_regions"] = sorted(fallback_regions)
    else:
        raise ValueError(f"未知 norm_mode={norm_mode}")

    # 写出
    city_out = join(out_root, city)
    os.makedirs(city_out, exist_ok=True)
    for phase, df in outputs.items():
        df = df[["region_id", "date", "weekday", "start_hour", "in_flow", "out_flow", "month"]].copy()
        # in_flow/out_flow 转回字符串 (load_norm_flow 用 ast.literal_eval 解析)
        df["in_flow"] = df["in_flow"].apply(lambda x: str(list(map(float, x))))
        df["out_flow"] = df["out_flow"].apply(lambda x: str(list(map(float, x))))
        out_pth = join(city_out, f"norm_{phase}_len_{seq_length}.csv")
        df.to_csv(out_pth, index=False)
        report[f"{phase}_rows"] = len(df)
        report[f"{phase}_path"] = out_pth
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--craft_root", default="/root/autodl-tmp/projects/CRAFT/cleared_data")
    ap.add_argument("--out_root", default="/root/autodl-tmp/projects/Paper/data/norm_flow")
    ap.add_argument("--seq_length", type=int, default=24)
    ap.add_argument("--norm_mode", default="global", choices=["global", "region"])
    ap.add_argument("--cities", nargs="*", default=CITIES)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    reports = []
    for city in args.cities:
        print(f"[gen_norm_flow] processing {city} (mode={args.norm_mode}) ...", file=sys.stderr)
        rep = process_city(city, args.craft_root, args.out_root, args.seq_length, args.norm_mode)
        reports.append(rep)
        print(f"  -> train_rows={rep.get('train_rows')} test_rows={rep.get('test_rows')} "
              f"train_clip={rep.get('train_clipped')} test_clip={rep.get('test_clipped')}", file=sys.stderr)

    meta_pth = join(args.out_root, "norm_flow_meta.json")
    with open(meta_pth, "w") as f:
        json.dump({"seq_length": args.seq_length, "norm_mode": args.norm_mode, "cities": reports}, f, indent=2)
    print(f"[gen_norm_flow] done. meta -> {meta_pth}", file=sys.stderr)


if __name__ == "__main__":
    main()
