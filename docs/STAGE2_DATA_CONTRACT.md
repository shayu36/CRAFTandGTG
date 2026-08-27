# HCFM 统一数据契约

## 1. 快照主键与防泄漏

联合样本唯一主键是 `(city_id, date, start_hour)`。构造时必须同时满足：

```text
macro.city_id == micro.city_id
macro.time_window == micro.time_window
macro.split == micro.split
```

`split` 至少为 `train|val|test`。归一化器与 RAG 库的 `fitted_cities` 必须是源城市，`fitted_split` 必须是 `train`。目标城市静态图可用于领域判别；目标城市动态标签不得进入零样本训练、归一化拟合或 RAG 库。

## 2. 静态城市图

### Region 图

- `city_id: str`
- `region_id: int64 [N]`，顺序固定且唯一
- `region_geometry`，源 CRS 与 `region_crs`
- `region_x: float32 [N,45]`
- `region_edge_index: int64 [2,E_macro]`

### 有向 Road 图

- `directed_road_id: str [M]`，稳定顺序
- `parent_source_road_id`；只有源数据真实提供时才写 `parent_osm_way_id`，禁止伪造 OSM way id
- `source_node/target_node`
- `road_geometry`、`road_crs`、`road_type`、`road_length`、`direction`
- `road_x: float32 [M,d_road]`，静态属性与真实空间句法特征顺序写入 manifest
- `road_edge_index: int64 [2,E_micro]`，有向道路 `j->k` 当且仅当 `target_node[j]==source_node[k]`

CRAFT `road.csv` 的 `oneway=False` 行被可靠展开成 forward/reverse 两条逻辑有向道路；`oneway=True` 仅保留 forward。源表未提供 `osm_way_id`，因此只能保存 `parent_source_road_id=road_id`，`parent_osm_way_id=null`。

### Region--Road 关系

`P_struct [N,M]` 使用 COO/CSR 稀疏表示：

```text
P_struct[i,j] = intersection_length(road_j, region_i)
                / sum_k intersection_length(road_k, region_i)
```

非空 Region 行和为 1；空 Region 保持空行并在 manifest 中列出，禁止补均匀权重。另保存 `road_to_region_edge_index/weight` 与反向边。

`B_in/B_out [N,M]` 是动态边界流算子，和 `P_struct` 分开。Road 方向由 geometry 起终点与 `source_node/target_node` 一致性校验。跨 Region 有向道路从 A 到 B 时：`B_out[A,j]=1`、`B_in[B,j]=1`；Region 内道路不计边界流。城市外端点用 region id `-1` 表示，从外到 B 只计 `B_in[B,j]`，A 到外只计 `B_out[A,j]`。

## 3. 动态数据

宏观目标：`macro_flow [B,N,2,T]`，通道顺序固定为 `in_flow,out_flow`。

微观主目标：`micro_flow [B,M,1,T]`，唯一守恒通道为 `road_passage_count`。`speed/travel_time/occupancy` 只能作为辅助监督，不能进入 `B_in/B_out` 守恒聚合。

整城样本字段：

```python
{
    "city_id": str,
    "date": str,
    "start_hour": int,
    "region_x": FloatTensor[N, 45],
    "region_edge_index": LongTensor[2, E_macro],
    "macro_flow": FloatTensor[N, 2, T],
    "road_x": FloatTensor[M, d_road],
    "road_edge_index": LongTensor[2, E_micro],
    "micro_flow": FloatTensor[M, 1, T],
    "p_struct": sparse[N, M],
    "b_in": sparse[N, M],
    "b_out": sparse[N, M],
    "region_mask": BoolTensor[N],
    "road_mask": BoolTensor[M],
    "time_features": {"month": int, "weekday": int, "start_hour": int},
    "split": str,
}
```

当前选择“一批一个城市快照”，避免不同城市 Road 节点 padding 造成显存浪费。`B=1` 仍保留 batch 维。

## 4. 动态文件适配器

支持 manifest 映射原始字段名，不要求重命名源文件。道路动态至少提供以下一种：

1. 已聚合长表：`city_id,directed_road_id,timestamp,road_passage_count`；
2. 已 map-matching：`trajectory_id,city_id,directed_road_id,enter_time,leave_time`；
3. 原始 GPS 点：`trajectory_id,city_id,timestamp,longitude,latitude`，且必须配置具备方向和路径连续性检查的 map matcher。

只有 OD 起终点不满足本契约。缺失动态文件、id 未覆盖、同一窗口错城/错 split、CRS 不可转换、NaN/Inf 均立即报错。

`assemble_joint_samples` 还强制调用方提供 IANA `timezone` 与 `dst_policy=raise|first|second`，并真实 localize 每个 `(date,start_hour)`；DST 不存在时刻一律报错，歧义时刻必须由数据说明选择 first/second，不能静默猜测。

## 5. 归一化与 checkpoint 元数据

分别保存 `macro_normalizer`、`micro_count_normalizer`、可选 `micro_speed_normalizer`、`micro_time_normalizer`、`static_feature_normalizer`。每个统计对象包含：`method`、`center/scale`、`feature_order`、`fitted_cities`、`fitted_split=train`、`data_version`。加载 checkpoint 时逐项验证。

跨尺度状态损失先分别反归一化，再在物理计数单位中经 `B_in/B_out` 聚合；禁止直接比较两套归一化张量。
