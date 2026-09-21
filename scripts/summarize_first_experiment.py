#!/usr/bin/env python3
"""Export the first Stage-4 experiment report, metrics CSV, and loss figure."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT / "outputs" / "stage4_three_layer_diffusion" / "history.json"
REPORT_DIR = ROOT / "docs" / "experiments"
REPORT_PATH = REPORT_DIR / "第一次实验.md"
FIGURE_PATH = REPORT_DIR / "第一次实验_loss趋势.png"
CSV_PATH = REPORT_DIR / "第一次实验_metrics.csv"
LOG_DIR = REPORT_DIR / "logs"
LOG_NAMES = (
    "train_5gpu.log",
    "retry_epoch9.log",
    "train_5gpu_tmux_history.log",
    "train_5gpu_live.log",
)


METRIC_COLUMNS = (
    "epoch",
    "train_loss",
    "validation_normalized_noise_loss",
    "validation_region_normalized_noise_loss",
    "validation_syntax_normalized_noise_loss",
    "validation_road_normalized_noise_loss",
    "validation_rag_mean_actual_top_k",
    "validation_rag_mean_top1_weight",
    "generation_weighted_mae",
    "generation_weighted_rmse",
)


def percent_drop(first: float, last: float) -> float:
    return (first - last) / first * 100.0


def write_csv(history: list[dict]) -> None:
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in history:
            generation = row.get("physical_generation", {})
            writer.writerow({
                "epoch": row["epoch"],
                "train_loss": row["train_loss"],
                "validation_normalized_noise_loss": row["validation_normalized_noise_loss"],
                "validation_region_normalized_noise_loss": row["validation_region_normalized_noise_loss"],
                "validation_syntax_normalized_noise_loss": row["validation_syntax_normalized_noise_loss"],
                "validation_road_normalized_noise_loss": row["validation_road_normalized_noise_loss"],
                "validation_rag_mean_actual_top_k": row["validation_rag_mean_actual_top_k"],
                "validation_rag_mean_top1_weight": row["validation_rag_mean_top1_weight"],
                "generation_weighted_mae": generation.get("weighted_mae", ""),
                "generation_weighted_rmse": generation.get("weighted_rmse", ""),
            })


def write_figure(history: list[dict]) -> None:
    epochs = [row["epoch"] for row in history]
    generation_rows = [row for row in history if "physical_generation" in row]
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(3, 1, figsize=(13, 13), constrained_layout=True)

    axes[0].plot(epochs, [row["train_loss"] for row in history], color="#1565c0", linewidth=2)
    axes[0].scatter([0], [history[0]["train_loss"]], color="#d32f2f", zorder=3, label="epoch 0 smoke")
    axes[0].set_title("Stage-4 Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Joint noise loss")
    axes[0].set_yscale("log")
    axes[0].legend()

    validation_series = (
        ("validation_normalized_noise_loss", "Total", "#212121"),
        ("validation_region_normalized_noise_loss", "Region", "#ef6c00"),
        ("validation_syntax_normalized_noise_loss", "Syntax", "#2e7d32"),
        ("validation_road_normalized_noise_loss", "Road", "#6a1b9a"),
    )
    for key, label, color in validation_series:
        axes[1].plot(epochs, [row[key] for row in history], label=label, color=color, linewidth=2)
    axes[1].set_title("Validation Normalized Noise Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(ncol=4)

    generation_epochs = [row["epoch"] for row in generation_rows]
    generation_mae = [row["physical_generation"]["weighted_mae"] / 1e6 for row in generation_rows]
    generation_rmse = [row["physical_generation"]["weighted_rmse"] / 1e6 for row in generation_rows]
    axes[2].plot(generation_epochs, generation_mae, marker="o", label="Weighted MAE", color="#00838f")
    axes[2].plot(generation_epochs, generation_rmse, marker="s", label="Weighted RMSE", color="#c62828")
    axes[2].set_title("Physical Generation Metrics (every 10 epochs)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Metric (million units)")
    axes[2].legend()

    figure.suptitle("First Experiment: Three-layer RAG + Diffusion", fontsize=18)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def write_report(history: list[dict]) -> None:
    formal_first = history[1]
    final = history[-1]
    best_validation = min(history, key=lambda row: row["validation_normalized_noise_loss"])
    generation_rows = [row for row in history if "physical_generation" in row]
    final_generation = generation_rows[-1]["physical_generation"]
    log_files = [
        ROOT / "outputs" / "stage4_three_layer_diffusion" / name
        for name in LOG_NAMES
    ]
    log_bytes = sum(path.stat().st_size for path in log_files)

    generation_table = "\n".join(
        f'| {row["epoch"]} | {row["physical_generation"]["weighted_mae"]:,.2f} | '
        f'{row["physical_generation"]["weighted_rmse"]:,.2f} |'
        for row in generation_rows
    )
    report = f"""# 第一次实验：三层 RAG + Conditional Diffusion

记录日期：2026-09-21  
状态：已完成 50 个记录 epoch（`0–49`），最佳 checkpoint 位于 epoch {best_validation['epoch']}。

> 注意：epoch 0 是单 snapshot smoke test，不是完整训练 epoch。正式三城训练从 epoch 1 开始，因此收敛率以 epoch 1 和 epoch 49 比较。

## 1. 实验配置

| 项目 | 设置 |
|---|---|
| Source cities | Beijing、Chengdushi、Xianshi |
| Train snapshots | 69（每城 23） |
| Eval snapshots | 18（每城 validation 3、test 3） |
| GPU | 5 × NVIDIA RTX 2080 Ti（约 11 GiB/卡） |
| Train diffusion steps | 500 |
| Validation sampling | DDIM 100 steps，eta=0 |
| Learning rate | 5e-6 |
| Node chunk size | 64 |
| Memory strategy | gradient checkpointing、CPU source cache、detached source keys |
| Distributed strategy | 5 rank city-bucket 分发，显式 gradient all-reduce/average |

## 2. Loss 结果

| 指标 | epoch 1 | epoch 49 | 降幅 | 最佳 epoch |
|---|---:|---:|---:|---:|
| Train joint noise loss | {formal_first['train_loss']:.6f} | {final['train_loss']:.6f} | {percent_drop(formal_first['train_loss'], final['train_loss']):.2f}% | {min(history[1:], key=lambda row: row['train_loss'])['epoch']} |
| Validation total noise loss | {formal_first['validation_normalized_noise_loss']:.6f} | {final['validation_normalized_noise_loss']:.6f} | {percent_drop(formal_first['validation_normalized_noise_loss'], final['validation_normalized_noise_loss']):.2f}% | {best_validation['epoch']} |
| Validation Region loss | {formal_first['validation_region_normalized_noise_loss']:.6f} | {final['validation_region_normalized_noise_loss']:.6f} | {percent_drop(formal_first['validation_region_normalized_noise_loss'], final['validation_region_normalized_noise_loss']):.2f}% | {min(history, key=lambda row: row['validation_region_normalized_noise_loss'])['epoch']} |
| Validation Syntax loss | {formal_first['validation_syntax_normalized_noise_loss']:.6f} | {final['validation_syntax_normalized_noise_loss']:.6f} | {percent_drop(formal_first['validation_syntax_normalized_noise_loss'], final['validation_syntax_normalized_noise_loss']):.2f}% | {min(history, key=lambda row: row['validation_syntax_normalized_noise_loss'])['epoch']} |
| Validation Road loss | {formal_first['validation_road_normalized_noise_loss']:.6f} | {final['validation_road_normalized_noise_loss']:.6f} | {percent_drop(formal_first['validation_road_normalized_noise_loss'], final['validation_road_normalized_noise_loss']):.2f}% | {min(history, key=lambda row: row['validation_road_normalized_noise_loss'])['epoch']} |

![第一次实验 Loss 趋势](第一次实验_loss趋势.png)

完整逐 epoch 数据见 [第一次实验_metrics.csv](第一次实验_metrics.csv)。

## 3. 物理生成指标

| Epoch | Weighted MAE | Weighted RMSE |
|---:|---:|---:|
{generation_table}

epoch 49 分层结果：

| 层 | MAE | RMSE |
|---|---:|---:|
| Region | {final_generation['layers']['region']['mae']:,.2f} | {final_generation['layers']['region']['rmse']:,.2f} |
| Syntax | {final_generation['layers']['syntax']['mae']:,.2f} | {final_generation['layers']['syntax']['rmse']:,.2f} |
| Road | {final_generation['layers']['road']['mae']:,.2f} | {final_generation['layers']['road']['rmse']:,.2f} |
| 三层加权 | {final_generation['weighted_mae']:,.2f} | {final_generation['weighted_rmse']:,.2f} |

## 4. 关键观察

- 正式训练 loss 从 {formal_first['train_loss']:.4f} 降至 {final['train_loss']:.4f}，下降 {percent_drop(formal_first['train_loss'], final['train_loss']):.2f}%，优化过程有效。
- Validation total loss 从 {formal_first['validation_normalized_noise_loss']:.4f} 降至 {final['validation_normalized_noise_loss']:.4f}，epoch 49 仍在改善，尚未出现明确平台期。
- Train/validation loss 差距较大，可能存在过拟合、跨城市泛化差距或训练/验证噪声任务难度差异。
- `top_k=5` 时最终 RAG top-1 weight 为 {final['validation_rag_mean_top1_weight']:.6f}，接近均匀权重 0.2，说明当前检索区分度偏弱。
- 物理空间指标在 epoch 19/29 曾反弹，epoch 49 达到本轮最低加权 MAE/RMSE，但绝对值仍很高，不能作为成熟生成结果。
- 继续增加 epoch 前，建议优先审计 Region normalizer 反变换、三层指标量纲和 RAG 检索权重；确认无尺度问题后再扩展到 75/100 epoch。

## 5. 运行事件与修复

- 初始 Road U-Net 在 11 GiB GPU 上 OOM；通过 `node_chunk_size=64`、gradient checkpointing、CPU source cache 修复。
- 第 10 个 epoch 的 500-step DDPM 物理生成曾出现非有限 epsilon；验证采样改为 DDIM 100 steps，并增加 sampling-only `x0` clip=8.0。
- 训练 loss 的 NaN/Inf 严格检查保持不变；物理生成异常会记录而不再终止训练。
- 修复后 epoch 9、19、29、39、49 的物理生成均成功完成。

## 6. GitHub 实验产物

原始 `.log` 合计仅 {log_bytes / 1024:.1f} KiB，因此与清洁后的分析结果一并提交：

- 本报告；
- Loss/生成指标趋势图；
- 完整逐 epoch CSV；
- [完整结构化 history](logs/stage4_history.json)；
- [epoch 1–8 与首次生成失败日志](logs/train_5gpu.log)；
- [epoch 9 修复验证日志](logs/retry_epoch9.log)；
- [tmux 历史输出](logs/train_5gpu_tmux_history.log)；
- [epoch 12–49 实时日志](logs/train_5gpu_live.log)。

`best.pt` 和 `last.pt` 各约 279 MB，继续由 `.gitignore` 排除，不上传 GitHub。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    if len(history) != 50 or history[0]["epoch"] != 0 or history[-1]["epoch"] != 49:
        raise ValueError("第一次实验 history 必须完整覆盖 epoch 0–49")
    write_csv(history)
    write_figure(history)
    write_report(history)
    for name in LOG_NAMES:
        source = ROOT / "outputs" / "stage4_three_layer_diffusion" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, LOG_DIR / name)
    shutil.copy2(HISTORY_PATH, LOG_DIR / "stage4_history.json")
    print(REPORT_PATH)
    print(FIGURE_PATH)
    print(CSV_PATH)


if __name__ == "__main__":
    main()
