# Stage 3 三层 Hierarchical RAG

本模块位于联合三层 GraphGPS 之后、Flow Matching 之前。它只接收
GraphGPS 的三层低频表征、历史可观测的三层动态时间序列和日历条件；高频表征保留给
后续 Flow Matching。

## 输入契约

```text
H_road_low    [B,M,D]
H_syntax_low  [B,K,D]
H_region_low  [B,N,D]

history_Y_road        [B,M,C_road,T]
history_Y_syntax      [B,K,C_syntax,T]
history_Y_region      [B,N,C_region,T]

calendar = {month, weekday, start_hour, holiday?}
```

`ThreeLayerRAGInputs.from_graphgps_output()` 可把 Stage 2 输出转换为契约。
它只读取三层 `H_*_low` 字段；若显式向模型传入 `high_features`，模型会严格
报错，避免把高频条件静默带入检索。

Source memory 还必须提供独立的 `value_temporal_features`。它们是被检索的
动态 Value，不能与用于 Query/Key 的历史序列复用，以区分历史可观测条件和
未来/被引用的动态 Value，避免时间标签泄漏。

动态序列按道路、Syntax/Locality、Region 三个节点集合分别排列。节点数可以
跨城市不同，检索按层级逐节点执行，不把三个城市的节点矩阵直接拼接。父层
关系优先通过加权稀疏算子提供：

```text
parent_operator[name].edge_index: row=parent, col=child
parent_operator[name].weight:     几何/池化权重
```

Syntax→Region 会保留第一阶段的 UTM 几何长度权重，不再强制压缩成一对一
`parent_index`。只有没有稀疏算子时，才允许使用一对一 assignment；两者
都缺失时才使用可审计的父层均值 fallback。

## 三个检索分支

`HierarchicalThreeLayerRAG` 内部有三个独立的 temporal encoder、Query/Key
投影和检索分支：

```text
Region: H_region_low + E_region_time + E_calendar
Syntax: H_syntax_low + E_syntax_time + Region context + E_calendar
Road:   H_road_low + E_road_time + Syntax context + Region context + E_calendar
```

Region 先检索，得到 `R_region`；Syntax query 使用对应的
`R_region_time`；Road query 再使用 `R_syntax_time` 和 `R_region_time`，因此
满足 coarse-to-fine 的 `Region → Syntax → Road` 条件结构。

输出为：

```text
R_region  [B,N,C_region,T]
R_syntax  [B,K,C_syntax,T]
R_road    [B,M,C_road,T]
```

同时返回三层检索 latent、calendar embedding、top-k 权重和诊断索引。原始
Road/Syntax/Region ID 不进入任何 Q/K/V 投影；`city_id` 只用于 source-train
memory 的泄漏检查和目标城市排除。

## Source memory 与本地数据

memory 必须由 source cities 的 `train` snapshot 构建：

```bash
python scripts/build_rag_memory.py \
  --input /local/path/three_layer_train_snapshots.pt \
  --output outputs/stage3_three_layer_rag/rag_memory_v2.pt \
  --source-cities beijing chengdushi xianshi \
  --require-graph-identity
```

真实 CSV、`cache/`、谱特征、memory 和 checkpoint 不提交到 Git。输入 bundle
的格式是 `ThreeLayerRAGInputs` 字段；它可以由本地 Stage 2 输出、道路动态
聚合和 Region flow loader 共同构成。Stage 2 v2 文件可以通过
`load_stage2_low_features()` 读取；`build_rag_inputs_from_local_artifacts()`
会同时接入静态 hierarchy 的两个稀疏加权算子。

也可以直接用本地 Stage-2 导出组装一个 snapshot：

```bash
python scripts/build_three_layer_rag_input.py \
  --spectral-feature outputs/stage2_three_layer_graphgps_lappe/spectral_features/beijing.pt \
  --hierarchy-cache-dir cache/static_hierarchy_start_v2 \
  --city beijing \
  --history /local/path/beijing_history_temporal.pt \
  --value /local/path/beijing_value_temporal.pt \
  --calendar /local/path/beijing_calendar.json \
  --split train \
  --output /local/path/beijing_train_snapshot.pt
```

运行单个 Query：

```bash
python scripts/query_three_layer_rag.py \
  --config configs/stage3_three_layer_rag.yaml \
  --memory outputs/stage3_three_layer_rag/rag_memory_v2.pt \
  --query /local/path/target_query.pt \
  --output outputs/stage3_three_layer_rag/target_reference.pt
```

memory 会保存并校验每个 source city 的：

```text
joint_graph_hash
checkpoint_fingerprint
static_feature_version
spectral_feature_version
road/syntax/region node range
```

不同城市允许拥有不同 graph hash，但同一城市的 snapshots 不能混用不同
GraphGPS checkpoint 或节点顺序。

当前三城本地数据已经具备：带 `start_time` 的道路序列 `rid_list`、逐道路
`dur_list`，以及 `train_label/valid_label` 的 `time_index/rid/dur_mean/speed_mean`。
因此 Road 动态可由这些文件构建；Syntax 动态应通过已有 Road→Syntax 权重
聚合，不能用 Region flow 代替。当前仓库中 `norm_flow` 主要是 Region 级
`in_flow/out_flow`，不是现成的 Road/Syntax 序列。

## 与 Flow Matching 的边界

RAG 输出 `R_road/R_syntax/R_region` 与三层高频特征分别交给后续层：

```text
R_road/R_syntax/R_region + calendar ─┐
H_road_high/H_syntax_high/H_region_high ─┼─> Hierarchical Flow Matching
```

本模块不实现 Flow Matching，也不修改旧 HCFM 的 Region-only 入口；
`src/hcfm/rag.py` 仅导出新类以便后续接入，同时保留旧兼容检索器。
