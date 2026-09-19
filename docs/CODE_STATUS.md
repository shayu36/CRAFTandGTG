# 当前代码状态

> 状态日期：2026-09-19  
> 当前基线提交：`b7c52a3`（本地仍有本轮三层动态序列构建相关修改待提交）

## 1. 总体架构

当前仓库同时保留两条彼此独立的路线：

1. 新三层路线：`Static hierarchy → Unified GraphGPS → Hierarchical RAG → Hierarchical Flow Matching`；
2. 旧 HCFM 路线：`src/hcfm/` 与 `scripts/run_stage2.py`。

本轮工作只扩展新三层路线，没有把联合 GraphGPS 或新 RAG 静默接入旧 HCFM。

新三层路线的目标信息流是：

```text
Road / Syntax / Region 静态图
        ↓
统一三层异构图 + 单一 GraphGPS
        ↓
H_*_low                              H_*_high
   ↓                                     ↓
三层历史动态 + 时间条件                  保留为城市特异条件
   ↓                                     ↓
Hierarchical RAG ── R_road/R_syntax/R_region
                    ↓
      Region → Syntax → Road Flow Matching
```

## 2. Stage 1：三层静态层级图

状态：**已实现并已生成三城缓存**。

核心目录：

```text
src/static_hierarchy/
scripts/build_static_hierarchy.py
configs/stage1_three_layer_static.yaml
```

固定节点层级：

```text
Road → Syntax/Locality → Region
```

静态契约包括：

- Road 有向对偶图；
- Syntax 有向层内图；
- Region 层内图；
- Road→Syntax 均值池化算子；
- Syntax→Region UTM 几何长度归一化算子；
- 稳定 Road 顺序和严格 metadata 校验。

当前使用的三城缓存位于：

```text
cache/static_hierarchy_start_v2/
```

该目录是本地派生产物，不提交真实 cache 到 Git。

## 3. Stage 2：统一三层 GraphGPS + LapPE

状态：**已实现、已训练、已导出真实三城特征**。

核心代码：

```text
src/three_layer_graphgps/data.py
src/three_layer_graphgps/model.py
src/three_layer_graphgps/posenc.py
src/three_layer_graphgps/spectral_lap_pe.py
src/three_layer_graphgps/frequency.py
src/three_layer_graphgps/engine.py
scripts/train_three_layer_graphgps.py
configs/stage2_three_layer_graphgps_lappe.yaml
```

### 3.1 联合节点编号

设 Road、Syntax、Region 数量分别为 `M/K/N`：

```text
Road:   [0, M)
Syntax: [M, M+K)
Region: [M+K, M+K+N)
```

三层分别使用输入投影和节点类型 embedding，但只存在一个联合
`GraphGPSStack`。`forward()` 对 `[M+K+N, hidden_dim]` 调用一次
`self.graphgps(...)`，不是三套分层 GraphGPS。

### 3.2 联合边与消息传播

联合消息图保留以下五种有向关系：

```text
road_intra
syntax_intra
region_intra
road_to_syntax
syntax_to_region
```

跨层权重没有被丢弃，也没有增加 Road→Region 直接边。local branch 使用关系类型、
边方向和边权；global attention 覆盖整个联合节点集合。

### 3.3 LapPE 与频率分解

- 消息传播使用原始有向联合图；
- LapPE 单独使用联合图的无向化副本；
- 使用 sparse normalized Laplacian 与 `scipy.sparse.linalg.eigsh`；
- 在联合 `H_joint` 上进行一次低频投影；
- 再按稳定节点区间切分 Road/Syntax/Region 的 low/high 特征。

当前 Stage 2 特征版本：

```text
three-layer-joint-graphgps-spectral-features-v2
```

## 4. Stage 3：三层 Hierarchical RAG

状态：**输入契约、模型、真实动态构建、memory 和基础 checkpoint 工具已实现；完整训练调用链尚未完成**。

核心代码：

```text
src/three_layer_rag/contracts.py
src/three_layer_rag/model.py
src/three_layer_rag/io.py
src/three_layer_rag/dynamics.py
src/three_layer_rag/training.py
scripts/build_three_layer_rag_snapshots.py
scripts/build_rag_memory.py
scripts/build_three_layer_rag_input.py
scripts/query_three_layer_rag.py
configs/stage3_three_layer_rag.yaml
```

### 4.1 已实现的 RAG 输入

```text
H_road_low / H_syntax_low / H_region_low
history_Y_road / history_Y_syntax / history_Y_region
month / weekday / start_hour / holiday / time_of_day
```

高频特征在公开接口处被拒绝，不能进入 RAG。source memory 还必须提供与 history
分开的 `value_temporal_features`，避免把未来 Value 复用为 Query/Key 历史输入。

### 4.2 三层检索结构

内部包含 Region、Syntax、Road 三个检索分支，顺序为：

```text
Region → Syntax → Road
```

Syntax 使用 Region 上下文；Road 使用 Syntax 和 Region 上下文。跨层条件优先使用
静态层级缓存中的加权稀疏算子，尤其不会把 Syntax→Region 多对多几何映射压成
一对一 ID。

### 4.3 真实动态序列构建

`src/three_layer_rag/dynamics.py` 和
`scripts/build_three_layer_rag_snapshots.py` 已实现：

- Road：由带时间戳 GTG 轨迹生成 `passage_count/speed_kmh/travel_time_seconds`；
- Syntax：用原始 Road→Syntax 均值权重聚合 Road 动态；
- Region：默认读取 `hourly_boundary_flow_raw.csv` 的 `in_flow/out_flow`；
- history 使用 `[t-24,t)`；
- Value 使用独立 `[t,t+24)`；
- 跨越 train/val 边界的窗口被丢弃；
- `log1p_zscore` 只在 source-train 时间范围拟合；
- `valid` CLI 别名会规范化为 RAG 契约中的 `val`。

现有 `train_label.csv/valid_label.csv` 只有 `time_index=0..23` 的小时-of-day 汇总，
没有日期。默认不会把它们广播成历史序列；只有显式启用
`--label-fill-mode hour_of_day_prior` 才作为稀疏先验使用，并记录潜在泄漏风险。

## 5. 尚未完成的核心代码

以下部分不能写成已完成：

1. 没有正式的 RAG + 三层 Flow Matching 联合训练 CLI；
2. `RAGTrainer` 只是接受下游 objective 的训练工具，不是完整训练程序；
3. `query_three_layer_rag.py` 当前会新建 RAG 模型，没有加载已训练 RAG checkpoint；
4. `R_road/R_syntax/R_region` 尚未接入新的 Region→Syntax→Road Flow Matching；
5. Stage 2 高频特征与 RAG reference 的联合条件接口尚未闭合；
6. 尚无联合 RAG/FM checkpoint、验证指标和生成结果；
7. 旧 `src/hcfm/` 仍是独立路线，不能当作新三层 FM 已完成的证据。

因此，当前正确的代码完成度是：

| 模块 | 状态 |
|---|---|
| 三层静态层级图 | 已完成 |
| 单一联合 GraphGPS | 已完成 |
| 联合 LapPE 与 low/high 分解 | 已完成 |
| 三层真实动态序列构建 | 已完成 |
| Hierarchical RAG 输入契约与模型 | 已完成 |
| source-train RAG memory | 已完成 |
| val/test RAG snapshots | 已完成 |
| RAG 独立正式训练入口 | 未完成 |
| RAG → 三层 Flow Matching | 未完成 |
| 新三层端到端生成闭环 | 未完成 |

## 6. 下一阶段接口要求

后续实现必须使用以下本地产物：

```text
outputs/stage2_three_layer_graphgps_lappe/spectral_features/
outputs/stage3_three_layer_rag/train_snapshots.pt
outputs/stage3_three_layer_rag/eval_snapshots.pt
outputs/stage3_three_layer_rag/rag_memory_v2.pt
```

联合训练应满足：

```text
三层 low + 三层 history + calendar → RAG
RAG references + 三层 high + calendar → Hierarchical Flow Matching
FM objective → 同时更新 RAG 与 FM
```

训练 checkpoint 至少应绑定：

- GraphGPS checkpoint fingerprint；
- 每个城市的 joint graph hash；
- spectral feature version；
- RAG memory version 和 graph identity；
- dynamic normalizer fingerprint；
- RAG/FM 配置与训练步数。

## 7. 已知环境风险

当前环境导入 PyG 时会报告 `torch-scatter`、`torch-cluster`、
`torch-spline-conv` 和 `torch-sparse` 的二进制符号不匹配。这些扩展在当前联合
GraphGPS、动态构建和 RAG memory 路径中被 PyG 禁用后仍能运行，但后续新 FM 若
直接依赖这些扩展，需要安装与当前 PyTorch/CUDA 完全匹配的 wheel。

