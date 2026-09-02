# Stage 1 START Road 层接入审计报告

## 1. 审计结论

本次审计在 START Road 模型修改之前完成，随后按门禁结果实施 v2。结论为：

- CRAFT Region 层已经进入真实三层模型和静态预训练调用链，不是仅存在于文档或测试中。
- Locality / Spatial Syntax 层已经实现，Road→Syntax→Region 两个跨层算子均被实际调用。
- 北京（`beijing`）、成都（`chengdushi`）和西安（仓库实际 ID 为 `xianshi`）的现有 v1 三层静态 cache 可通过契约校验；北京已完成真实前向、反向和 `GTAggregator.calc_loss()` 验证。
- START Road v2 已实现，正式 Road 输入为固定 `road_x[M,33]`；旧 `road_topo_x[M,4]` 仅保留兼容路径。
- 当前正式配置中的三个 source city 正是 `beijing/chengdushi/xianshi`，与实际数据目录一致。
- 仓库中未发现 `chongqing`、`chongqingshi` 或 `xian` 目录；本任务城市为成都，不能将 `chengdushi` 误写为重庆。

因此门禁结论为：

> **B. 三座城市审计和 START Road v2 代码实现已完成；目标城市仍需由后续数据上传并显式指定，当前不能启动正式跨城市预训练。**

实现阶段没有修改原始 CRAFT/GTG 目录或原始数据；仅新增独立 v2 派生 cache 和模型代码。

## 2. 版本和环境快照

审计命令：

```bash
pwd
git status --short
git branch --show-current
git rev-parse HEAD
git log -1 --oneline
```

结果：

- 工作目录：`/root/autodl-tmp/projects/Paper`
- 分支：`main`
- commit：`5e7ca415ca38ad53336a0cb4a675882e0f73b2e5`
- 提交说明：`5e7ca41 Add Stage 1 three-layer static hierarchy and training protocol test`
- 审计开始时工作树干净；此前用户已有修改未被覆盖。
- 仓库中未找到可读取的 `AGENTS.md` 文件，本报告遵循用户提供的写入和安全边界。

环境：

- Python `3.10.20`
- PyTorch `2.0.1+cu118`
- geopandas `1.1.4`
- shapely `2.0.6`
- numpy `1.26.4`
- pandas `2.3.3`
- PyG 原生扩展导入会出现 ABI warning，仓库 `pyg_compat` 后备实现可用。
- `graph_tool` 不可用：`ImportError: ... libgomp-a34b3233.so.1: version 'GOMP_5.0' not found`。

数据和缓存根目录：

- CRAFT 兼容静态数据：`data/gtg_craft/`
- GTG cache：`cache/gtg/`
- 当前三层 cache：`cache/static_hierarchy/`
- 现有真实城市：`beijing`、`chengdushi`、`xianshi`
- 未发现重庆目录；`chengdushi` 是成都目录，不是重庆。

## 3. Region 层审计

### 3.1 真实文件契约

对 `beijing`、`chengdushi` 和 `xianshi` 检查了：

```text
grid_region.csv
grid_region_feature.csv
grid_region_rel.csv
data_feature.json
```

结果：

| city_id | Region 数量 | 两个 Region 文件顺序 | CRAFT 45 维 | 有效 Region 边 | 重复边 | 自环 | 孤立 Region | 几何 | `utm_epsg` |
|---|---:|---|---|---:|---:|---:|---:|---|---:|
| `beijing` | 81 | `0..80`，一致 | 全有限 | 494 | 0 | 0 | 0 | 有效 Polygon | 32650 |
| `chengdushi` | 23 | `0..22`，一致 | 全有限 | 130 | 0 | 0 | 0 | 有效 Polygon | 32648 |
| `xianshi` | 26 | `0..25`，一致 | 全有限 | 150 | 0 | 0 | 0 | 有效 Polygon | 32649 |

45 维严格顺序为：

```text
population
population_density
dist_to_center
road_num
road_length
poi_num_0..11
poi_score_0..11
road_num_0..7
road_length_0..7
```

`grid_region_rel.csv` 的 `ori/des` 均在合法范围内，`is_adj == 1` 形成非空 Region 图。没有静默删除重复边、自环或孤立节点；本次真实数据统计中三类异常均为零。

三座目标城市对应的四个文件均存在；没有发现 Region 数据硬阻塞。

### 3.2 Region 实际模型和训练路径

代码调用链为：

```text
scripts/build_static_hierarchy.py --action pretrain
→ data_loaders.get_graph_datasets()
→ load_region_graph(require_flow_labels=...)
→ GTAggregator.calc_loss()
→ ThreeLayerStaticEncoder.forward()
→ FeatureInitLayer(region_x[N,45])
→ region_fusion
→ original CRAFT GraphTransformer(region_edge_index)
→ region_rep[N,rep_dim]
```

`ThreeLayerStaticEncoder.forward()` 实际调用 `craft_integrated.rep_model.FeatureInitLayer`，并将 Region 语义和 Syntax→Region 聚合结果送入 Region fusion，再送入 CRAFT `GraphTransformer`。

真实北京 cache 的前向结果：

```text
road_h              [14685,16]
road_to_syntax_h    [293,16]
syntax_h            [293,16]
syntax_to_region_h  [81,16]
region_semantic_h   [81,16]
region_rep          [81,16]
```

Region init、Region fusion、Region GraphTransformer、Syntax GATv2 均被 forward hook 观察到实际调用；输出全为 finite。对 `region_rep.sum()` 反向传播后，Road encoder、Syntax encoder、Region init、Region fusion 和 Region GraphTransformer 均获得非空且有限梯度。扰动北京 Road 拓扑输入会改变最终 Region 表征。

真实 `GTAggregator.calc_loss()` 也已执行一次（仅使用现有 `beijing` source 和仓库已有 `chi` 静态 target 做调用链验证）：

- loss 为 finite；
- source 使用动态 `value`；
- target `value is None`；
- 对 target 的 `load_norm_flow` 进行了保护，未读取 target flow；
- Road/Syntax/Region 梯度均存在。

`chi` 仅用于验证已有 target 静态加载，不代表本任务的目标城市；target 仍需由配置或命令行显式指定。

## 4. Syntax / Locality 层审计

### 4.1 Road 级 GTG cache

三座城市均优先使用 `gtg_road.csv`，而不是含 OSM 补充道路的合并 `road.csv`。

| city_id | GTG Road 文件 | Road 行数 | `road_feat` shape | Road ID 顺序 | finite | 前五维顺序 |
|---|---|---:|---|---|---|---|
| `beijing` | `cache/gtg/beijing_gtg_road.npz` | 14685 | `[14685,9]` | 完全一致 | 是 | 正确 |
| `chengdushi` | `cache/gtg/chengdushi_gtg_road.npz` | 3514 | `[3514,9]` | 完全一致 | 是 | 正确 |
| `xianshi` | `cache/gtg/xianshi_gtg_road.npz` | 4147 | `[4147,9]` | 完全一致 | 是 | 正确 |

前五维严格为：

```text
connectivity
total_depth
integration
choice
mean_depth
```

后四维为旧 GTG 分区上下文特征，仅作为现有离线 Syntax cache 的历史字段；当前三层模型使用的中层节点 `syntax_x` 仍严格为五维，不把 Region 级 9 维 cache 当作 Road 节点输入。

cache 缺失时，`src/static_hierarchy/preprocessing.py` 会懒加载 `graph_tool` 重算；但本环境的 `graph_tool` 因 `GOMP_5.0` ABI 问题不可用。因此已有 cache 可验证，缺失 cache 的城市不能在本环境中重算或伪造。

### 4.2 Metis 和 Road→Syntax

实际配置 `local_size=50`，Metis assignment 与稳定 Road 行顺序一致：

| city_id | M Road | K Syntax | assignment shape | 最小分区 | 最大分区 | 平均分区 | Road→Syntax links |
|---|---:|---:|---|---:|---:|---:|---:|
| `beijing` | 14685 | 293 | `[14685]` | 48 | 51 | 50.119 | 14685 |
| `chengdushi` | 3514 | 70 | `[3514]` | 49 | 51 | 50.200 | 3514 |
| `xianshi` | 4147 | 82 | `[4147]` | 49 | 52 | 50.573 | 4147 |

契约检查确认：

- 每条 Road 恰好有一个 Syntax 归属；
- assignment 没有越界或空分区；
- `P^{syntax←road}` shape 分别为 `[293,14685]` 和 `[82,4147]`；
- 每个 Road→Syntax 权重为所属分区大小的倒数；
- 每个 Syntax 行权重和为 1；
- 模型通过 `torch.sparse.mm` 实际执行 Road→Syntax 池化。

### 4.3 Syntax 图和 Syntax→Region

Road 对偶图跨分区边被聚合为有向 Syntax 边，同分区边不生成自环，重复边已 coalesce：

| city_id | Road 对偶边 E | Syntax 边 E | 自环 | 重复 Syntax 边 |
|---|---:|---:|---:|---:|
| `beijing` | 28262 | 1293 | 0 | 0 |
| `chengdushi` | 7703 | 304 | 0 | 0 |
| `xianshi` | 8794 | 390 | 0 | 0 |

`syntax_edge_index` 实际进入 `SyntaxEncoder` 的 GATv2。Syntax 五维特征和 Road 池化表征均参与 Syntax fusion，并共同影响最终 Region 输出。

`P^{region←syntax}` 按真实 UTM 投影几何相交长度构造：

```text
L[i,s] = sum(length(road ∩ region_i) for road assigned to s)
P[i,s] = L[i,s] / sum_q L[i,q]
```

当前真实 cache：

- `beijing` shape `[81,293]`，空 Region 比例 `0.0`；
- `chengdushi` shape `[23,70]`，空 Region 比例 `0.0`；
- `xianshi` shape `[26,82]`，空 Region 比例 `0.0`；
- 所有权重有限、非负，非空 Region 行和为 1；
- 空 Region 会保留空行并由 `region_has_syntax=False` 标识，比例超过阈值时严格报错；
- 模型不存在可学习的 Road→Region 直连。

## 5. Road 数据和 START 输入审计

### 5.1 Road 文件选择和字段

三座城市的 `gtg_road.csv` 均包含：

```text
road_id
from_node_id
to_node_id
road_type
road_type_id
length
geometry
oneway
lanes
maxspeed
```

Road ID 连续、唯一，`from_node_id/to_node_id` 无缺失，geometry 全部为有效非空 `LineString`，`length` 全部有限且为正。声明长度与 UTM 几何长度的相对误差接近机器精度。

`road.csv` 比 `gtg_road.csv` 多出 OSM `residential/living_street` 补充道路，并使用负节点 ID 与 GTG 拓扑隔离：

| city_id | `gtg_road.csv` 行数 | `road.csv` 行数 | 额外道路类型 | 负节点行数 |
|---|---:|---:|---|---:|
| `beijing` | 14685 | 22037 | `residential` 7014，`living_street` 338 | 7352 |
| `chengdushi` | 3514 | 4653 | `residential` 1055，`living_street` 84 | 1139 |
| `xianshi` | 4147 | 5442 | `residential` 1092，`living_street` 203 | 1295 |

因此未来 START Road/Syntax 层必须继续使用 `gtg_road.csv`，不能把合并 `road.csv` 的 OSM 补充道路直接接入有向 Road 对偶图。

### 5.2 START 字段统计

| city_id | `road_type_id` 分布 | length min/median/p95/max (m) | lanes 缺失率 | lanes 可解析率 | maxspeed 缺失率 | maxspeed 可解析率 | maxspeed 单位 |
|---|---|---|---:|---:|---:|---:|---|
| `beijing` | 1:1583, 2:1422, 3:2182, 4:6970, 5:314, 7:2214 | 0.544 / 109.303 / 499.430 / 4073.586 | 89.31% | 10.69% | 98.48% | 1.52% | 223 个纯数值，无 mph/km/h 文本 |
| `chengdushi` | 1:89, 2:842, 3:763, 4:1137, 5:6, 7:677 | 0.547 / 139.258 / 464.506 / 2901.442 | 78.14% | 21.86% | 97.84% | 2.16% | 76 个纯数值，无 mph/km/h 文本 |
| `xianshi` | 1:239, 2:745, 3:1102, 4:947, 7:1114 | 2.299 / 148.254 / 525.664 / 1429.640 | 64.05% | 35.95% | 97.42% | 2.58% | 107 个纯数值，无 mph/km/h 文本 |

缺失 `lanes` 和 `maxspeed` 按任务要求进入显式 `unknown` bucket，并在 START v2 metadata 中保留计数和比例。当前正式 v2 Road 层使用这些字段；旧 v1 `road_topo_x` 兼容路径不使用它们。

补充单位审计：三座城市的 `maxspeed`/源 `free_speed` 值均为无单位纯数值；取值范围为北京 `20..120`、成都 `20..80`、西安 `50..70`。结合国内城市道路限速语义，按用户确认采用 `km/h`；若解释为 `m/s` 将对应 `72..432 km/h`，不符合数据语义。该选择已写入 v2 metadata 的 `maxspeed_unit: km/h`。

### 5.3 三城市兼容性矩阵

| 城市 | 实际 city_id | 实际目录 | Region 文件 | Road 文件 | GTG Road cache | N | M | E | K | Road ID 对齐 | Region 45维 | Syntax 5维 | Road→Syntax | Syntax→Region | lanes 缺失率 | maxspeed 缺失率 | START 改造状态 | 阻塞原因 |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---|---|
| 北京 | `beijing` | `data/gtg_craft/beijing` | 完整 | `gtg_road.csv` | 有且对齐 | 81 | 14685 | 28262 | 293 | 是 | 是 | 是 | 是 | 是 | 89.31% | 98.48% | 可作为后续改造输入 | 无数据硬阻塞 |
| 西安 | `xianshi` | `data/gtg_craft/xianshi` | 完整 | `gtg_road.csv` | 有且对齐 | 26 | 4147 | 8794 | 82 | 是 | 是 | 是 | 是 | 是 | 64.05% | 97.42% | 可作为后续改造输入 | 无数据硬阻塞 |
| 成都 | `chengdushi` | `data/gtg_craft/chengdushi` | 完整 | `gtg_road.csv` | 有且对齐 | 23 | 3514 | 7703 | 70 | 是 | 是 | 是 | 是 | 是 | 78.14% | 97.84% | 可作为后续改造输入 | 无数据硬阻塞 |

### 5.4 当前代码与 START v2 的兼容性问题

审计发现并已修复的代码兼容性问题：

1. `CityStaticHierarchy`、NPZ/JSON loader 和 contract 现同时支持 v1/v2，并拒绝版本错配。
2. 新增 `RoadStaticEncoder(33, rep_dim)`，保留 `RoadTopologyEncoder` 作为 v1 兼容路径。
3. `configs/stage1_three_layer_static.yaml` 已切换为 `road_feature_mode: start_static`、`road_feature_dim: 33` 和 `cache/static_hierarchy_start_v2`。
4. `data_loaders.py`、`scripts/build_static_hierarchy.py` 和模型前向已接入 `start_static`/v2 cache 路由。
5. HCFM 完整 `road_x` 未复用；START 特征构造独立位于 `static_hierarchy.preprocessing`。
6. 第一阶段不使用任何 CoSpec 特征或损失，也不使用 START `trans_prob`。

## 6. 真实调用链检查命令

已执行的关键命令和结果：

```bash
python - <<'PY' ...  # Region/Road/GTG cache 字段、ID、几何、缺失率统计
PY
```

结果：北京、成都、西安的真实字段、v1 cache 和新生成的 v2 cache 均完成对齐校验；未发现重庆目录，但重庆不在本任务城市范围内。

```bash
PYTHONPATH=src:src/craft_integrated python - <<'PY' ...  # 北京真实 cache forward/backward 和 hook
PY
```

结果：所有三层中间表示 finite，Road/Syntax/Region 梯度均非空，扰动 Road 会改变 Region 表征。

```bash
PYTHONPATH=src:src/craft_integrated python - <<'PY' ...  # GTAggregator.calc_loss，target flow guard
PY
```

结果：`beijing` source + `chi` static target 的 loss finite，target `value=None`，target flow guard 未触发，梯度存在。

已对北京、成都、西安执行 v2 preprocess/load/validate/forward/backward smoke；完整 `run_stage1_static.py --action pretrain` 仍需等待后续显式 target 城市 cache。

## 7. 门禁和后续要求

### 当前阻塞

- 目标城市尚未上传或显式指定，不能启动正式 source/target 跨城市预训练。
- 当前环境无法用 `graph_tool` 重算缺失空间句法 cache；三座 source 城市本次均使用已存在且已对齐的 GTG Road cache，因此不影响本次 v2 构建。

### 非阻塞问题

- 北京、成都、西安 `lanes`/`maxspeed` 缺失率较高，但可通过显式 unknown bucket 兼容，前提是记录 metadata。
- v2 Road 输入已升级为独立 `three-layer-start-road-v2` cache 和固定 33 维 schema。
- 正式配置已经使用真实成都 ID `chengdushi`；后续不得将其改写为重庆。

### 允许进入 START 实施的条件

1. 上传或指定 target 城市的真实静态三层数据和 v2 cache 输入。
2. 运行 target 的 v2 preprocess/load/validate/forward smoke。
3. 在 target 就绪后，再运行正式多 Source TFA/CCA 静态预训练。

## 8. 范围确认

本次审计没有：

- 修改或实现 START `trans_prob`；
- 接入轨迹频率、动态 Road flow、POI、人口、Region 或 Syntax 特征到 Road 层；
- 修改 GTG/Craft 损失、域对抗、GRL、RAG、Retriever、Diffusion、Flow Matching 或 HCFM；
- 修改原始 CRAFT、GTG 目录或原始数据；
- 生成假 Road、假 Region、假 Syntax、假轨迹或假映射；
- 执行 `git push`、创建 PR 或破坏性 Git 命令。
