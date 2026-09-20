# CRAFTandGTG

本仓库研究跨城市交通表征与生成，当前重点路线是把 Road、Syntax/Locality、
Region 组成统一三层异构层次图，使用单一 GraphGPS 编码器联合建模，再以低频
表征驱动 Hierarchical RAG、以高频表征作为 Hierarchical Conditional Diffusion
的城市特异条件，并按 Region → Syntax → Road 生成三层未来动态。

## 当前状态

截至 2026-09-19：

| 阶段 | 状态 |
|---|---|
| 三层静态层级图 | 已完成 |
| 单一联合 GraphGPS | 已完成 |
| 联合 LapPE 与 low/high 分解 | 已完成 |
| Beijing/Chengdushi/Xianshi weighted-v3 Stage 2 训练与导出 | 代码已完成，必须重新训练/导出 |
| Road/Syntax/Region 真实动态序列构建 | 代码已完成，weighted-v3 产物待重建 |
| Hierarchical RAG 数据契约、模型和 memory | 代码已完成，weighted-v3 memory 待重建 |
| Train/val/test RAG snapshots | 构建流程已完成，weighted-v3 bundle 待重建 |
| RAG + Hierarchical Conditional Diffusion 代码与 CLI | 已完成（单元/synthetic smoke） |
| Stage 4 真实三城训练与生成指标 | 尚未运行 |
| 后续 GTG 轨迹解码闭环 | 尚未接入 |

详细状态与真实运行结果分别见：

- [当前代码状态](docs/CODE_STATUS.md)
- [当前运行结果](docs/RUN_RESULTS.md)

## 模型路线

```text
Road nodes        Syntax/Locality nodes        Region nodes
     \                    |                       /
      └──────── 统一三层异构层次图 ─────────────┘
                         ↓
                 Single GraphGPSStack
                         ↓
              H_joint_low / H_joint_high
                         ↓
        ┌────────────────┴────────────────┐
        ↓                                 ↓
三层 low + 三层 history + calendar       三层 high
        ↓                                 │
Hierarchical RAG                          │
        ↓                                 │
R_region / R_syntax / R_road ─────────────┘
                         ↓
       Region → Syntax → Road Conditional Diffusion
```

联合图节点编号固定为：

```text
Road:   [0, M)
Syntax: [M, M+K)
Region: [M+K, M+K+N)
```

不存在 Road→Region 直接层次边。GraphGPS 消息图保持有向，LapPE 使用单独的
联合无向化副本。

## 仓库结构

```text
src/static_hierarchy/       三层静态图、跨层算子和缓存契约
src/three_layer_graphgps/   联合 GraphGPS、LapPE、频率分解与 Stage 2 engine
src/three_layer_rag/        三层 RAG、动态序列、memory、I/O 和训练工具
src/three_layer_diffusion/  三层条件 Diffusion、U-Net、EMA、数据与指标
src/hcfm/                   legacy HCFM/FM 独立实验路线（非当前主路线）

scripts/build_static_hierarchy.py
scripts/train_three_layer_graphgps.py
scripts/build_three_layer_rag_snapshots.py
scripts/build_rag_memory.py
scripts/query_three_layer_rag.py
scripts/train_three_layer_diffusion.py
scripts/generate_three_layer_diffusion.py

configs/stage1_three_layer_static.yaml
configs/stage2_three_layer_graphgps_lappe.yaml
configs/stage3_three_layer_rag.yaml
configs/stage4_three_layer_diffusion.yaml
```

## 三层特征契约

### Stage 2 静态输入

```text
Road raw feature:   [M,33]
Syntax raw feature: [K,5]
Region raw feature: [N,45]
```

三类输入先分别投影到相同 hidden dimension，再加入 node type embedding 和
联合 LapPE，随后只调用一次 GraphGPS。

### RAG 动态输入

```text
Road:   [M,3,24] = passage_count/speed/travel_time
Syntax: [K,3,24] = Road 动态经 Road→Syntax 权重聚合
Region: [N,2,24] = in_flow/out_flow
```

每个 snapshot 分开保存历史 Query/Key 序列和未来检索 Value：

```text
history = [t-24,t)
value   = [t,t+24)
```

## 本地产物与版本迁移

真实数据、cache、checkpoint 和大体积 tensor 不提交到 Git。此前本地运行生成过：

```text
outputs/stage2_three_layer_graphgps_lappe/best.pt
outputs/stage2_three_layer_graphgps_lappe/spectral_features/
outputs/stage3_three_layer_rag/train_snapshots.pt
outputs/stage3_three_layer_rag/eval_snapshots.pt
outputs/stage3_three_layer_rag/rag_memory_v2.pt  # 历史旧产物
```

这些结果使用旧无权 LapPE / spectral-features-v2 语义，仅作为历史运行记录，不能被
当前 weighted-v3 代码继续训练或导出。当前契约会显式拒绝旧 checkpoint、旧 spectral
feature、旧 snapshot 和旧 memory。必须按 Stage 2 重训与导出 → train/eval snapshots
→ RAG memory 的顺序全部重建；不能只覆盖下游某一个文件。对应的历史指标、节点数量、
shape 和 fingerprint 记录在 [当前运行结果](docs/RUN_RESULTS.md)。

## 主要运行入口

### Stage 2 训练

```bash
python scripts/train_three_layer_graphgps.py \
  --config configs/stage2_three_layer_graphgps_lappe.yaml \
  --action train \
  --source_cities beijing chengdushi xianshi \
  --device cpu
```

### Stage 2 特征导出

```bash
python scripts/train_three_layer_graphgps.py \
  --config configs/stage2_three_layer_graphgps_lappe.yaml \
  --action export_features \
  --source_cities beijing chengdushi xianshi \
  --checkpoint outputs/stage2_three_layer_graphgps_lappe/best.pt \
  --feature_output_dir outputs/stage2_three_layer_graphgps_lappe/spectral_features \
  --device cpu
```

### RAG train snapshots

```bash
python scripts/build_three_layer_rag_snapshots.py \
  --cities beijing chengdushi xianshi \
  --hierarchy-cache-dir cache/static_hierarchy_start_v2 \
  --spectral-feature-dir outputs/stage2_three_layer_graphgps_lappe/spectral_features \
  --gtg-data-root data \
  --region-flow-root data/gtg_craft \
  --region-flow-file-name hourly_boundary_flow_raw.csv \
  --output outputs/stage3_three_layer_rag/train_snapshots.pt \
  --splits train \
  --history-length 24 \
  --value-length 24 \
  --snapshot-stride-hours 24 \
  --overwrite
```

随后使用新生成的 source-train normalizer 重建 `val/test` snapshots；完整命令见
[Stage 3 文档](docs/STAGE3_THREE_LAYER_RAG.md)。

### RAG memory

```bash
python scripts/build_rag_memory.py \
  --input outputs/stage3_three_layer_rag/train_snapshots.pt \
  --output outputs/stage3_three_layer_rag/rag_memory_v3.pt \
  --source-cities beijing chengdushi xianshi \
  --require-graph-identity
```

### Stage 4 联合训练

```bash
python scripts/train_three_layer_diffusion.py \
  --config configs/stage4_three_layer_diffusion.yaml \
  --device cuda:0
```

### Stage 4 分层生成

```bash
python scripts/generate_three_layer_diffusion.py \
  --config configs/stage4_three_layer_diffusion.yaml \
  --checkpoint outputs/stage4_three_layer_diffusion/best.pt \
  --input outputs/stage3_three_layer_rag/eval_snapshots.pt \
  --output outputs/stage4_three_layer_diffusion/generated \
  --device cuda:0
```

Stage 4 训练时用真实归一化父层 future 作 teacher forcing；推理时严格先生成
Region，再把生成 Region 稀疏广播给 Syntax，最后把生成 Syntax 广播给 Road。
生成条件不读取真实 `value_temporal_features`，该字段仅可在采样完成后计算离线指标。

## 文档索引

- [当前代码状态](docs/CODE_STATUS.md)
- [当前运行结果](docs/RUN_RESULTS.md)
- [Stage 1 三层静态图实现](docs/STAGE1_THREE_LAYER_STATIC_IMPLEMENTATION.md)
- [Stage 1 三层静态图建模](docs/STAGE1_THREE_LAYER_STATIC_MODELING.md)
- [Stage 1 START Road 接入审计](docs/STAGE1_START_ROAD_INTEGRATION_AUDIT.md)
- [Stage 1 START Road 接入报告](docs/STAGE1_START_ROAD_INTEGRATION_REPORT.md)
- [Stage 1 测试报告](docs/STAGE1_THREE_LAYER_STATIC_TEST_REPORT.md)
- [Stage 2 联合 GraphGPS + LapPE](docs/STAGE2_THREE_LAYER_GRAPHGPS_LAPPE.md)
- [Stage 3 Hierarchical RAG](docs/STAGE3_THREE_LAYER_RAG.md)
- [Stage 4 Hierarchical Conditional Diffusion](docs/STAGE4_THREE_LAYER_DIFFUSION.md)

## 下一步

Stage 4 的代码、配置、CLI 和 synthetic 测试已经落地，下一步是执行真实三城训练：

```text
RAG references + Stage 2 high-frequency features + calendar + parent dynamics
                         ↓
Region → Syntax → Road Hierarchical Conditional Diffusion
                         ↓
joint diffusion noise objective / EMA checkpoint / physical-space validation
```

当前没有运行真实 Stage 4 重训练，因此仓库尚无训练后的 `best.pt` 或真实三城生成
指标。`scripts/query_three_layer_rag.py` 仍只是独立调试入口；正式生成必须使用联合
训练 checkpoint 和 `scripts/generate_three_layer_diffusion.py`。
