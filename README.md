# CRAFTandGTG

本仓库研究跨城市交通表征与生成，当前重点路线是把 Road、Syntax/Locality、
Region 组成统一三层异构层次图，使用单一 GraphGPS 编码器联合建模，再以低频
表征驱动 Hierarchical RAG、以高频表征作为后续 Hierarchical Flow Matching 的
城市特异条件。

## 当前状态

截至 2026-09-19：

| 阶段 | 状态 |
|---|---|
| 三层静态层级图 | 已完成 |
| 单一联合 GraphGPS | 已完成 |
| 联合 LapPE 与 low/high 分解 | 已完成 |
| Beijing/Chengdushi/Xianshi Stage 2 训练与导出 | 已完成 |
| Road/Syntax/Region 真实动态序列构建 | 已完成 |
| Hierarchical RAG 数据契约、模型和 memory | 已完成 |
| Train/val/test RAG snapshots | 已完成 |
| RAG + Hierarchical Flow Matching 联合训练 | 尚未完成 |
| 新三层端到端生成闭环 | 尚未完成 |

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
           Region → Syntax → Road Flow Matching
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
src/hcfm/                   旧 HCFM 独立路线

scripts/build_static_hierarchy.py
scripts/train_three_layer_graphgps.py
scripts/build_three_layer_rag_snapshots.py
scripts/build_rag_memory.py
scripts/query_three_layer_rag.py

configs/stage1_three_layer_static.yaml
configs/stage2_three_layer_graphgps_lappe.yaml
configs/stage3_three_layer_rag.yaml
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

## 已生成的本地产物

真实数据、cache、checkpoint 和大体积 tensor 不提交到 Git。当前本地运行已生成：

```text
outputs/stage2_three_layer_graphgps_lappe/best.pt
outputs/stage2_three_layer_graphgps_lappe/spectral_features/
outputs/stage3_three_layer_rag/train_snapshots.pt
outputs/stage3_three_layer_rag/eval_snapshots.pt
outputs/stage3_three_layer_rag/rag_memory_v2.pt
```

对应的实际指标、节点数量、shape、fingerprint、文件大小和测试结果记录在
[当前运行结果](docs/RUN_RESULTS.md)。

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
  --snapshot-stride-hours 24
```

### RAG memory

```bash
python scripts/build_rag_memory.py \
  --input outputs/stage3_three_layer_rag/train_snapshots.pt \
  --output outputs/stage3_three_layer_rag/rag_memory_v2.pt \
  --source-cities beijing chengdushi xianshi \
  --require-graph-identity
```

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

## 下一步

下一阶段不是继续生成同类数据，而是补齐正式联合训练入口：

```text
RAG references + Stage 2 high-frequency features + calendar
                         ↓
Region → Syntax → Road Hierarchical Flow Matching
                         ↓
joint FM objective / checkpoint / validation / test
```

在该入口完成以前，`scripts/query_three_layer_rag.py` 不应被当作正式推理命令，
因为它当前会创建尚未训练的 RAG 参数。

