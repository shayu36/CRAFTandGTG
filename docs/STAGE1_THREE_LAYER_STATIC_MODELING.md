# 第一阶段三层城市静态图建模

## 目标与边界

本阶段只学习城市静态结构的多尺度 Region 表征，唯一的消息路径是：

```text
Road 纯拓扑层 → Spatial Syntax 空间句法层 → CRAFT Grid Region 语义层
```

不实现 CoSpec、GTG 专属损失、域对抗、GRL、Road cost、Rank、RAG、Diffusion、Flow Matching、HCFM 或动态 Road 流量。
CRAFT 原有 TFA/self-similarity 与 CCA/Wasserstein 只作为可选的原始静态表征训练机制，数学定义和权重不变。

## 三层含义

### Road 层

每条稳定排序的有向道路段是一个节点。道路边沿用 `gtg_features.dual_graph` 的定义：
`road_i.to_node_id == road_j.from_node_id` 时建立 `road_i → road_j`。

Road 输入只有四个拓扑量：

```text
[bias, in_degree, out_degree, total_degree]
```

不输入流量、Road 类型、长度、车道、限速、轨迹频率、POI、人口、边几何属性、Road ID embedding 或 CoSpec。

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
road_topo_x [M,4]
  → RoadTopologyEncoder(GATv2 on road_edge_index)
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

四个城市使用同一个 `ThreeLayerStaticEncoder` 参数实例。source 城市需要动态 Region value 供原 CRAFT TFA 使用；target 城市只加载静态三层图，不读取 train flow、不伪造零 value。

## 严格模式

缓存和加载均校验文件存在、Region/Road ID 顺序、shape、索引范围、有限值、非负权重、Road 唯一 Syntax 归属、非空 Syntax 分区和非空 Region 行归一化。`road_feature_mode: cospec` 只保留显式接口，当前抛出：

```text
NotImplementedError: CoSpec road features are not implemented in Stage 1
```
