"""norm 流量文件测试: 存在性、值域 [0,1]、序列长度、无泄漏(train 拟合)。"""
import ast
import os

import numpy as np
import pandas as pd
import pytest

from conftest import NORM_DIR, CITIES


def _load(city, phase):
    pth = os.path.join(NORM_DIR, city, f"norm_{phase}_len_24.csv")
    assert os.path.exists(pth), f"缺失 {pth}"
    df = pd.read_csv(pth)
    df["in_flow"] = df["in_flow"].apply(lambda x: np.array(ast.literal_eval(x)))
    df["out_flow"] = df["out_flow"].apply(lambda x: np.array(ast.literal_eval(x)))
    return df


@pytest.mark.parametrize("city", CITIES)
def test_columns_and_seq_len(city):
    df = _load(city, "train")
    assert set(["region_id", "date", "weekday", "start_hour", "in_flow", "out_flow", "month"]).issubset(df.columns)
    assert (df["in_flow"].apply(len) == 24).all()
    assert (df["out_flow"].apply(len) == 24).all()


@pytest.mark.parametrize("city", CITIES)
def test_value_range_0_1(city):
    for phase in ["train", "test"]:
        df = _load(city, phase)
        vals = np.concatenate([
            np.concatenate(df["in_flow"].tolist()),
            np.concatenate(df["out_flow"].tolist()),
        ])
        assert vals.min() >= 0.0 - 1e-9, f"{city} {phase} 最小值 < 0"
        assert vals.max() <= 1.0 + 1e-9, f"{city} {phase} 最大值 > 1"


@pytest.mark.parametrize("city", CITIES)
def test_train_fit_reaches_bounds(city):
    # 训练集(全局 min-max 在 train 上拟合)应恰好触及 0 和 1
    df = _load(city, "train")
    vals = np.concatenate([
        np.concatenate(df["in_flow"].tolist()),
        np.concatenate(df["out_flow"].tolist()),
    ])
    assert abs(vals.min() - 0.0) < 1e-6
    assert abs(vals.max() - 1.0) < 1e-6
