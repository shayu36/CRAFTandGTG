# 第一阶段三层静态图实现说明

## 新增模块

| 文件 | 作用 |
|---|---|
| `src/static_hierarchy/contracts.py` | `CityStaticHierarchy` 数据对象和严格契约校验 |
| `src/static_hierarchy/operators.py` | 稀疏算子、边 coalesce、Syntax→Region 几何映射 |
| `src/static_hierarchy/preprocessing.py` | 从 Road/Region CSV 与既有 GTG Road cache 构建三层图 |
| `src/static_hierarchy/data.py` | 独立 NPZ/JSON cache 保存与加载 |
| `src/static_hierarchy/model.py` | START `RoadStaticEncoder`、兼容 v1 `RoadTopologyEncoder`、Syntax、Region 三层编码器 |
| `src/static_hierarchy/__init__.py` | 公共 API |
| `configs/stage1_three_layer_static.yaml` | 静态阶段配置，target 不设默认值 |
| `scripts/build_static_hierarchy.py` | `validate/preprocess/smoke/pretrain` 实现入口 |
| `scripts/run_stage1_static.py` | 面向用户的第一阶段静态入口包装 |

旧的 `cache/gtg/{city}_gtg_region.npz`、`{city}_gtg_road.npz`、v1 三层 cache 和旧 `GTAggregator` 路径均保留。START v2 新缓存写入独立目录：

```text
cache/static_hierarchy_start_v2/{city}_static_hierarchy.npz
cache/static_hierarchy_start_v2/{city}_static_hierarchy_meta.json
```

v2 cache 的 Road 真值字段为 `road_x[M,33]`，版本为 `three-layer-start-road-v2`；旧 v1 cache 的 `road_topo_x[M,4]` 只在 `topology_only` 兼容模式中读取，加载器会拒绝版本错配。

## 模式路由

`src/craft_integrated/rep_model.py` 的 `GTAggregator` 支持：

```text
craft_only       原 CRAFT 45 维路径
flat_gtg_region  旧 45+9 Region 扁平路径
three_layer      新 Road→Syntax→Region 路径
```

`three_layer` 与 `use_gtg_topology: true` 同时出现时直接报配置错误。三层模式下不读取旧 9 维 Region cache，不实例化旧 `GTGTopoBranch`。

正式配置使用 `road_feature_mode: start_static`。Road 33 维由固定 8 类 Road type、单城市米制长度 `log1p`-min-max、6 类 `lanes`、6 类 `maxspeed`、6 类有向入度和 6 类有向出度组成。无属性道路进入显式 `unknown`；不使用 CoSpec 或 START `trans_prob`。当前三座城市无单位纯数字速度按已确认的 `km/h` 解释。

三层模式的 `GTAggregator.calc_loss()` 已按多 Source 协议分离 TFA/CCA：TFA 为每座 Source 城市内部计算并等权平均；CCA 使用所有 Source 的完整静态 Region，并将每座城市的 OT 总质量固定为 `1/S`，代价固定为 `1-cosine`。

`src/craft_integrated/data_loaders.py` 增加了 `require_flow_labels` 边界：

```text
source: load_region_graph(..., require_flow_labels=True)
target: load_region_graph(..., require_flow_labels=False)
```

target 静态图不会调用 `load_norm_flow`，也不会创建伪造 value。

## 运行方式

先显式指定目标城市构建 cache：

```bash
python scripts/run_stage1_static.py \
  --config configs/stage1_three_layer_static.yaml \
  --action preprocess \
  --source_cities beijing chengdushi xianshi \
  --target_city TARGET_CITY
```

静态前向和梯度连通性检查：

```bash
python scripts/build_static_hierarchy.py \
  --config configs/stage1_three_layer_static.yaml \
  --action smoke \
  --source_cities beijing chengdushi xianshi \
  --target_city TARGET_CITY
```

如需原 CRAFT TFA/CCA 静态预训练：

```bash
python scripts/build_static_hierarchy.py \
  --config configs/stage1_three_layer_static.yaml \
  --action pretrain \
  --source_cities beijing chengdushi xianshi \
  --target_city TARGET_CITY
```

该入口只保存 `*_region_rep.npy` 和 `static_encoder.pth` 后退出，不调用 Retriever、Diffusion、Flow Matching、HCFM 或生成入口。

## 已知环境约束

`graph_tool` 仅在既有 GTG Road cache 缺失、需要离线重算空间句法时懒加载；普通静态加载、模型前向和测试不会导入它。若确需重算，按仓库约定使用 `LD_PRELOAD` 预加载 conda 的 `libstdc++/libgomp`。
