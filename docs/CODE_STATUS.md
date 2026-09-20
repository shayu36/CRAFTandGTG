# 当前代码状态

更新时间：2026-09-20。本文只描述当前有效路线和当前工作区状态。

## 当前主路线

```text
Static hierarchy
  → Unified three-layer GraphGPS + joint LapPE
  → H_joint_low / H_joint_high
  → Hierarchical RAG
  → Region → Syntax → Road Conditional Diffusion
```

旧 HCFM/FM 代码不是当前训练入口。

## 当前产物

训练城市固定为 `beijing chengdushi xianshi`。

```text
outputs/stage2_three_layer_graphgps_lappe/best.pt
outputs/stage2_three_layer_graphgps_lappe/spectral_features/
outputs/stage3_three_layer_rag/train_snapshots.pt
outputs/stage3_three_layer_rag/eval_snapshots.pt
outputs/stage3_three_layer_rag/train_snapshots.normalizer.json
outputs/stage3_three_layer_rag/rag_memory_v3.pt
outputs/stage4_three_layer_diffusion/best.pt
```

Stage 3 为 train 69 个、eval 18 个；每个城市 train 23 个、validation 3 个、test 3 个。RAG memory 使用三城 train split 的 69 个 snapshot。

## Stage 2

三城 100 epoch 训练已完成。当前 `metrics.json` 的最优结果为：

```text
best epoch = 97
best valid_city_macro_rmse = 0.06179773683349291
```

三城 `H_joint`、`H_joint_low`、`H_joint_high` 及三层对应特征已导出，并完成 shape、节点范围、graph identity 和频率重构校验。

## Stage 3

已完成 history/value 动态 snapshots、source-train normalizer、validation/test eval bundle 和 `rag_memory_v3.pt`。RAG 启用 source city、calendar、value separation 和 graph identity 泄漏防护。

## Stage 4 当前实现

`configs/stage4_three_layer_diffusion.yaml` 的低显存配置为：

```text
node_chunk_size = 64
gradient_checkpointing = true
train_source_keys = false
source_cache_device = cpu
```

Road/Syntax/Region 的 flattened node batch 分块计算；训练激活使用 gradient checkpointing；source key 不保留反向图；source cache 保存在 CPU，检索时按 chunk 搬运到当前 GPU。RAG query、Diffusion 和 EMA 仍参与训练。

Stage 4 单卡 1 epoch smoke 已完成并生成 `best.pt/last.pt/history.json`。训练器已支持五卡 `torchrun`：按 city bucket 分发 batch，各 rank 对参数梯度显式 all-reduce/average，rank 0 负责验证、checkpoint 和 history。

## 当前剩余验证

- 三城 Stage 4 长训练的最终 loss 和生成指标；
- 生成结果到 GTG 轨迹解码的闭环。

最近一次定向测试为 36 passed，五卡 Stage 4 smoke 已以 `EXIT_CODE=0` 完成。测试覆盖 Diffusion、RAG、retriever 严格契约、节点分块、CPU cache 和 gradient checkpointing。环境中的 `graph_tool` 与 PyTorch 自带 `libgomp` 存在 `GOMP_5.0` ABI 风险。
