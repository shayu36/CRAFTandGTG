# Stage 1 START Road 静态层实施报告

## 1. 实施结论

Road 静态特征设计参考 `https://github.com/aptx1231/START`，但本阶段只采用其静态道路结构信息，不接入轨迹转移概率。

北京、成都、西安的三层静态数据均已完成 START Road v2 构建与契约校验。Road 层现在使用固定 33 维静态特征，唯一模型消息路径仍为：

```text
Road → Syntax → Region
```

三个城市共享同一个 `ThreeLayerStaticEncoder` 参数实例。target 城市仍必须由后续数据上传后通过配置或命令行显式指定；本报告没有使用伪造 target 流量。

## 2. Road 特征

固定顺序为：

```text
road_type_residential, road_type_trunk, road_type_primary, road_type_secondary,
road_type_tertiary, road_type_motorway, road_type_living_street, road_type_unclassified,
length_log_minmax,
lanes_unknown, lanes_1, lanes_2, lanes_3, lanes_4, lanes_5_plus,
maxspeed_unknown, maxspeed_le_30, maxspeed_31_50, maxspeed_51_70,
maxspeed_71_90, maxspeed_gt_90,
indegree_0, indegree_1, indegree_2, indegree_3, indegree_4, indegree_5_plus,
outdegree_0, outdegree_1, outdegree_2, outdegree_3, outdegree_4, outdegree_5_plus
```

规则：

- `road_type` 使用既有 `ROAD_TYPES/ROAD_TYPE_TO_ID`；
- `length` 使用 UTM 投影几何的米制长度，执行 `log1p` 后在单城市内 min-max 到 `[0,1]`；
- `lanes` 缺失/无法解析进入 `unknown`；
- 无单位纯数字 `maxspeed` 按 `km/h` 解释，带 `mph` 或 `m/s` 后缀时转换为 `km/h` 后分桶；缺失/无法解析进入 `unknown`；
- `indegree/outdegree` 在原始有向 Road 对偶图、添加任何 self-loop 之前计算；
- 不使用 START `trans_prob`、轨迹频率、动态 Road flow、CoSpec 或 GTG Syntax 特征。

## 3. 三层数据规模

| 城市 | city_id | N Region | M Road | E Road | K Syntax | E Syntax | Syntax→Region links | lanes 缺失率 | maxspeed 缺失率 | 空 Region 比例 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 北京 | `beijing` | 81 | 14685 | 28262 | 293 | 1293 | 817 | 89.31% | 98.48% | 0 |
| 成都 | `chengdushi` | 23 | 3514 | 7703 | 70 | 304 | 214 | 78.14% | 97.84% | 0 |
| 西安 | `xianshi` | 26 | 4147 | 8794 | 82 | 390 | 219 | 64.05% | 97.42% | 0 |

Road 属性缺失率来自真实 `gtg_road.csv`，不代表 Road 节点缺失。所有道路仍保留，通过显式 `unknown` 类别进入共享 Road 编码器。

## 4. 缓存和兼容性

- 旧 v1 cache：`cache/static_hierarchy/`，版本 `three-layer-static-v1`，保存 `road_topo_x[M,4]`；
- 新 v2 cache：`cache/static_hierarchy_start_v2/`，版本 `three-layer-start-road-v2`，保存 `road_x[M,33]`；
- loader 通过 `expected_feature_version` 严格拒绝 v1/v2 混读；
- `CityStaticHierarchy.road_topo_x` 仅为旧调用方的只读兼容别名，内部只保存一份 `road_x`；
- 未覆盖或删除原始 CRAFT/GTG 数据及旧 cache。

## 5. 模型和入口

`ThreeLayerStaticEncoder` 在 `start_static` 模式中使用 `RoadStaticEncoder(33, rep_dim)`，随后执行 Road→Syntax 稀疏均值聚合、Syntax GATv2、Syntax→Region 几何长度归一化、CRAFT 45 维 Region 初始化和原 CRAFT GraphTransformer。

正式配置已切换为：

```yaml
static_structure_mode: three_layer
road_feature_mode: start_static
road_feature_dim: 33
maxspeed_unit: km/h
static_hierarchy_cache_dir: cache/static_hierarchy_start_v2
```

入口仍为 `scripts/run_stage1_static.py`；没有接入 RAG、Retriever、Diffusion、Flow Matching、HCFM 或 Stage 2。

## 6. 实际验证

- v2 预处理：北京、成都、西安均成功生成 `[M,33]` Road cache；
- v2 加载：三座城市均通过 `three-layer-start-road-v2` contract；
- CPU 前向：`road_h → road_to_syntax_h → syntax_h → syntax_to_region_h → region_rep` 全部 finite；
- 反向：Road encoder、Syntax encoder、Region init、Region fusion、Region GraphTransformer 均收到非空有限梯度；
- 自动化测试：`25 passed, 4 warnings`；
- 语法和空白检查：`compileall`、`git diff --check` 成功。

尚未执行完整研究训练，也未执行 target 城市的正式 source/target 预训练，因为 target 数据尚未上传或显式指定。
