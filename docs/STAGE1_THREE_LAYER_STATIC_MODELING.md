# 第一阶段三层城市静态图建模

## 目标与边界

本阶段只学习城市静态结构的多尺度 Region 表征，Road 层采用 START 风格静态道路特征，唯一的消息路径是：

```text
Road 静态路网结构层 → Spatial Syntax 空间句法层 → CRAFT Grid Region 语义层
```

不使用或实现 CoSpec、START `trans_prob`、GTG 专属损失、域对抗、GRL、Road cost、Rank、RAG、Diffusion、Flow Matching、HCFM 或动态 Road 流量。
START 参考实现：`https://github.com/aptx1231/START`；本阶段只采用静态 Road 属性，不接入轨迹转移概率。
CRAFT 原有 TFA/self-similarity 与 CCA/Wasserstein 只作为可选的原始静态表征训练机制，数学定义和权重不变。

## 三层含义

### Road 层（START 风格静态路网结构层）

每条稳定排序的有向道路段是一个节点，Road 边沿用 `gtg_features.dual_graph` 的定义：
`road_i.to_node_id == road_j.from_node_id` 时建立 `road_i → road_j`。正式 v2 输入为固定 33 维：

```text
road_type one-hot       8
length_log_minmax       1
lanes one-hot           6
maxspeed one-hot        6
indegree one-hot        6
outdegree one-hot       6
```

`road_type` 使用项目既有 `ROAD_TYPES/ROAD_TYPE_TO_ID`；长度使用 UTM 米制几何长度的 `log1p` 后单城市 min-max；`lanes` 和 `maxspeed` 缺失或无法解析时进入显式 `unknown`。当前数据中的无单位纯数字 `maxspeed` 按已确认的 `km/h` 解释，带 `mph`/`m/s` 后缀时先转换为 `km/h`。

Road 层不输入流量、轨迹频率、POI、人口、GTG 五维 Syntax、Region 45 维、Road ID embedding、geometry 坐标 embedding、START `trans_prob` 或 CoSpec 特征。v1 的 `[bias, in_degree, out_degree, total_degree]` 仅作为旧 cache 兼容模式保留。

### Syntax 层

使用现有 Metis 分区得到 K 个 Syntax 节点，并保留 `road_to_syntax_assignment[M]`。Road 级空间句法指标按固定顺序计算/读取：

```text
connectivity, total_depth, integration, choice, mean_depth
```

每个 Syntax 节点对其 Road 成员取均值，形成 `syntax_x[K,5]`。跨分区的 Road 对偶边聚合为有向 Syntax 边，重复边 coalesce；同分区边不产生自环。

### Region 层

Region 继续使用 CRAFT 的 `region_x[N,45]` 及 `grid_region_rel.csv` 的区域邻接。45 维顺序保持原实现：

```text
population, population_density, dist_to_center, road_num, road_length,
poi_num_0..11, poi_score_0..11, road_num_0..7, road_length_0..7
```

Region 只接收自身语义特征和 Syntax 聚合结果，不直接接收 Road 节点或 Road 邻接。

## 两个跨层算子

`P^{syntax←road}[K,M]` 使用 `row=syntax_id, column=road_id`。Road 属于 Syntax `s` 时权重为 `1/|R_s|`，每个 Syntax 行权重和为 1。

`P^{region←syntax}[N,K]` 使用 `row=region_id, column=syntax_id`。先计算真实 Road 与 Region 的 UTM 相交长度，再按 Syntax 归属累加：

```text
L[i,s] = sum(length(road ∩ region_i) for road assigned to s)
P[i,s] = L[i,s] / sum_q L[i,q]
```

无道路 Region 保留空行并记录 `region_has_syntax=False`；异常高的空映射比例按配置直接报错。模型路径始终经过 Syntax，不存在可学习的 Road→Region 直连池化。

## 编码器

```text
road_x [M,33]
  → RoadStaticEncoder(GATv2 on directed road_edge_index)
  → road_h [M,rep_dim]
  → P^{syntax←road}
  → syntax_x [K,5] + pooled road_h
  → SyntaxEncoder(GATv2 on syntax_edge_index)
  → syntax_h [K,rep_dim]
  → P^{region←syntax}
  → CRAFT RegionInit(region_x [N,45]) + syntax projection
  → original CRAFT GraphTransformer on region_edge_index
  → region_rep [N,rep_dim]
```

## 多 Source 静态表征训练协议

三层模式下，Source 城市的 TFA 和 CCA 使用不同的 Region 集合：

- TFA 对每座 Source 单独取 `reps[value_region_ids]` 与对应动态 value，调用原有 `self_sim_loss`，再对 Source 城市的 loss 做等权平均；不会产生跨城市 TFA 两两配对。
- CCA 使用每座 Source 的完整静态 `reps`，包括没有训练流量窗口的 Region。Source OT 边际为每个城市总质量 `1/S`，即城市 `c` 的每个 Region 质量为 `1/(S*N_c)`。
- 三层模式的 CCA 代价固定为 `1-cosine`，由 `cca_metric: cosine` 显式声明；不复用 RAG 的 `retrieve_metric`。

Target 仍只提供完整静态表征，Target OT 边际保持总质量为 1 的均匀分布，不读取 target `norm_train`。三个 Source 城市和一个 Target 城市共享同一个三层编码器参数。

source 城市需要动态 Region value 供原 CRAFT TFA 使用；target 城市只加载静态三层图，不读取 train flow、不伪造零 value。

## 严格模式

缓存和加载均校验文件存在、Region/Road ID 顺序、shape、索引范围、有限值、非负权重、Road 唯一 Syntax 归属、非空 Syntax 分区和非空 Region 行归一化。v1 cache 使用 `three-layer-static-v1` 与 `road_topo_x`；START v2 cache 使用 `three-layer-start-road-v2` 与 `road_x`，两者不会静默混用。`road_feature_mode: cospec` 只保留显式接口，当前抛出：

```text
NotImplementedError: CoSpec road features are not implemented in Stage 1
```
