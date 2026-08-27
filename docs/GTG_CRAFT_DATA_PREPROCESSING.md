# GTG 三城转 CRAFT 兼容车辆流量

## 1. 已确认口径

- 目标是生成与本仓库 CRAFT 读取代码兼容的 GTG 车辆数据，不宣称复现共享单车语义。
- GTG 原 `traj/train.csv` 与 `traj/test.csv` 是同一月份随机轨迹划分：先合并，再聚合宏观流量。
- `start_time` 是 Unix 秒表示的 UTC 时刻，统一转换成 `Asia/Shanghai` 本地日期、星期和小时。
- 流量统计车辆穿越栅格边界：`A -> B` 时，A 的 `out_flow += 1`、B 的 `in_flow += 1`。
- 单条道路内的穿界时间按“投影几何累计长度占比 × dur_list 对应耗时”估算，假设路段内匀速。
- 栅格为约 2 km UTM 正方形，完整矩形格网覆盖被合并轨迹使用的道路；仅保留与这些道路有正长度相交的格子。
- CRAFT 筛选规则：每个样本的 24 个 `in_flow` 与 24 个 `out_flow` 共 48 值必须全部 `> 0`。
- CRAFT 道路聚合：道路与格子相交即计数，并将整条道路长度计入该格；跨格道路重复计入。
- GTG 缺失的 `residential`、`living_street` 必须由真实 OSM 2026 数据补充；POI、人口也不能用全零占位。
- 学长确认不设置 GTG 源域验证集：合并后的全部 GTG 正值窗口用于训练，CRAFT 城市仅用于最终测试。
- 训练固定运行 `diff_train_epoch`，关闭验证早停和验证损失驱动的学习率调度，保存最后一个 EMA 检查点。

## 2. 仍属证据化推断

CRAFT 未公开原始预处理脚本，因此以下规则有意写入 `preprocess_meta.json` 的 `inferred`，不能表述成作者原始实现：

- 线性插值：每格的 `in_flow/out_flow` 分别处理，将左右均有正值的内部 0 视作稀疏缺测；不做首尾外推。
- 归一化：逐 GTG 城市，以全部正值训练窗口的 in/out 值共同拟合全局 Min-Max。
- 未预分类的 OSM 多标签 POI：按 CRAFT 12 类固定顺序选择一个类别。若希望避免该推断，应在外部文件中明确提供 `poi_type_id`。

## 3. 先导出 OSM/WorldPop 研究范围

该步骤只读 `Paper/data/{city}` 中的 GTG 文件，输出仍在 `Paper/` 内：

```bash
cd /root/autodl-tmp/projects/Paper
python3 scripts/export_gtg_study_area.py --city beijing
python3 scripts/export_gtg_study_area.py --city chengdushi
python3 scripts/export_gtg_study_area.py --city xianshi
```

每城得到：

```text
data/external_2026/{city}/study_area/
├── full_grid.geojson
├── selected_grid.geojson
└── study_area_meta.json
```

`study_area_meta.json` 的 `overpass_bbox_order` 已按 Overpass 所需的 `south,west,north,east` 顺序给出。

## 4. 从 OSM 获取数据

OSM 没有“年度数据包”这一字段。2026 年执行提取时，应保存实际提取日期、查询文本和原始导出；标准 Overpass 查询返回当时数据库现状。大范围查询必须按 `full_grid.geojson` 分块，避免公共 Overpass 实例超时。

### 4.1 12 类 POI

在 Overpass Turbo 中把 `SOUTH,WEST,NORTH,EAST` 替换为单个分块 bbox：

```overpass
[out:json][timeout:900][bbox:SOUTH,WEST,NORTH,EAST];
(
  nwr["amenity"~"^(bicycle_rental|fast_food|restaurant|bicycle_parking|cafe)$"];
  nwr["public_transport"];
  nwr["shop"];
  nwr["tourism"];
  nwr["leisure"];
  nwr["office"];
  nwr["historic"];
  nwr["sport"];
);
out center tags;
```

导出 GeoJSON 后合并分块并按 `OSM element type + osmid` 去重。保留下列标签列：

```text
name, amenity, public_transport, shop, tourism, leisure, office, historic, sport, geometry
```

当前实际文件：

```text
data/external_2026/{city}/osm_poi_2026.geojson  # EPSG:4326
```

CRAFT 类别顺序固定为：

```text
0 bicycle_rental_amenity
1 fast_food_amenity
2 restaurant_amenity
3 bicycle_parking_amenity
4 cafe_amenity
5 public_transport
6 shop
7 tourism
8 leisure
9 office
10 historic
11 sport
```

### 4.2 补充道路

只提取 GTG 缺失的两类：

```overpass
[out:json][timeout:900][bbox:SOUTH,WEST,NORTH,EAST];
way["highway"~"^(residential|living_street)$"];
out geom tags;
```

合并分块并去重，保存：

```text
data/external_2026/{city}/osm_roads_2026.geojson  # EPSG:4326
```

至少保留 `highway,oneway,lanes,maxspeed,geometry`。流水线只把它们用于 CRAFT 静态 8 类道路统计，不会伪造它们与 GTG 原路网的拓扑连接。

## 5. 获取 2026 人口

人口不来自 OSM。应使用 WorldPop `Global 2015–2030` 的中国 2026、约 100 m、total population、constrained 产品。该栅格像元值是“每格人口数”；按区域应求和，而不是把人口密度再次求和。

中国全国 GeoTIFF 很大，建议在 QGIS 中：

1. 用 `full_grid.geojson` 对 WorldPop 2026 GeoTIFF 执行 `Clip Raster by Mask Layer`。
2. 用 `Raster pixels to points` 转为点；删除 NoData，但保留真实的 0 人口像元。
3. 转为 EPSG:4326，添加 `lon`、`lat` 字段，把像元值字段重命名为 `population`。
4. 每城保存：

```text
data/external_2026/{city}/population_2026.csv
```

CSV 必须严格包含：

```text
lon,lat,population
```

## 6. 完整构建

取得外部数据后，检查 `configs/gtg_craft_preprocess.yaml`：

- `osm_snapshot_date` 必须是 GeoJSON 中记录的真实 2026 提取日期；
- `use_source_validation: false`，`validation_start: null`；
- 当前使用 GeoJSON，相应 `*_layer` 保持 `null`。

然后运行：

```bash
python3 scripts/build_gtg_craft_data.py --config configs/gtg_craft_preprocess.yaml
```

每城静态/流量输出位于 `data/gtg_craft/{city}/`，只生成训练归一化文件
`data/norm_flow/{city}/norm_train_len_24.csv`。不生成 GTG `test/norm_test`。流水线不会覆盖非空目录。

## 7. 与融合训练读取器衔接

训练配置中保留 CRAFT 测试城市的只读根，并为 GTG 源城市指定覆盖目录：

```yaml
craft_data_root: /root/autodl-tmp/projects/CRAFT/cleared_data
norm_flow_root: /root/autodl-tmp/projects/Paper/data/norm_flow
city_data_dirs:
  beijing: /root/autodl-tmp/projects/Paper/data/gtg_craft/beijing
  chengdushi: /root/autodl-tmp/projects/Paper/data/gtg_craft/chengdushi
  xianshi: /root/autodl-tmp/projects/Paper/data/gtg_craft/xianshi
```

GTG 拓扑缓存构建：

```bash
python3 scripts/build_gtg_features.py \
  --craft_root /root/autodl-tmp/projects/Paper/data/gtg_craft \
  --cities beijing chengdushi xianshi
```

拓扑管线会优先读取每城的 `gtg_road.csv`；混入 OSM 补类的 `road.csv` 只服务于 CRAFT 静态特征。

## 8. 跨月检索协议（已确认）

GTG 三城只有 11 月，CRAFT 测试流量是 9–12 月，因此训练配置明确使用：

```yaml
retrieve_match_month: false
```

检索只按 `weekday + start_hour` 匹配，不再要求月份相同。若一个区域跨月存在多条记录，
`top_k` 先在不同区域之间选择，再汇总选中区域的所有候选月份，避免同一区域重复占位。
