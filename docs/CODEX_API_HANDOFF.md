# Codex API 交接：GTG 三城转 CRAFT 兼容车辆流量

更新时间：2026-08-22

## 0. 接手代理的第一条指令

先完整阅读 `/root/autodl-tmp/projects/Paper/AGENTS.md`，再完整阅读本文件。以真实代码、数据和测试结果为准继续工作，不重做已完成事项，不清理工作树，不修改 GTG/CRAFT 原始目录，不下载新的外部数据。

当前目标是完成 **北京、成都、西安 GTG 车辆轨迹到 CRAFT 兼容数据的预处理与严格验证**。当前不授权启动昂贵训练实验；先把数据管线、静态特征、归一化文件和读取器严格性做完并验证。

## 1. 不可违反的边界

- 所有写入只允许位于 `/root/autodl-tmp/projects/Paper/`。
- `/root/autodl-tmp/projects/GTG-main`、`/root/autodl-tmp/projects/CRAFT` 及其原始数据只读。
- 不执行 `git push`、`git reset --hard`、`git clean`、删除原始资产等操作。
- 当前工作树存在大量用户已有的 Stage 2 和其他未提交改动；不得回滚、覆盖或“顺手整理”。
- 严格模式：缺文件、坐标异常、区域覆盖异常、NaN/Inf、ID/形状不一致必须报错；不得用占位 0 静默通过。
- 归一化只用训练数据拟合，再应用于验证/测试，禁止目标测试泄漏。
- `graph_tool` 和 `torch` 不要放在同一进程。若运行 `graph_tool`，按 `AGENTS.md` 设置 `LD_PRELOAD`。

## 2. 已经由用户/学长确认的研究口径

1. 只生成与 `Paper/src/craft_integrated` 兼容的 **GTG 车辆流量**；兼容文件虽沿用 `slide_bike_flow*.csv` 名称，但不能称为共享单车数据。
2. 流量定义为车辆穿越栅格边界：`A -> B` 时，A 的 `out_flow += 1`，B 的 `in_flow += 1`。
3. GTG 原 `traj/train.csv` 与 `traj/test.csv` 是同一时段随机轨迹划分；流量聚合前合并。
4. `start_time` 是 Unix 秒表示的 UTC 时刻，统一转换为 `Asia/Shanghai` 本地时间。
5. GTG 不设源域验证集。所有通过 CRAFT 正值筛选的 GTG 窗口用于训练；不开验证早停，不生成 GTG `test/norm_test`。
6. 最终测试集使用 CRAFT 城市，不做 GTG 三城留一城测试。
7. GTG 只有 11 月，而 CRAFT 测试含 9–12 月；检索取消月份条件，仅匹配 `weekday + start_hour`。
8. 24 小时滑窗按 CRAFT 行为：24 个 `in_flow` 与 24 个 `out_flow` 共 48 个值必须全部 `> 0`。
9. 约 2 km 栅格无需复刻 CRAFT 城市划分；以实现简单且覆盖 GTG 实际轨迹/OSM 为准。
10. 道路长度按 CRAFT 样例行为：道路与格相交即计数，并把整条道路长度计入该格；跨格可重复计入。
11. GTG 缺失的 `residential`、`living_street` 从真实 2026 OSM 补充；OSM 补路只用于 CRAFT 静态道路统计，不伪造 GTG 拓扑连接。
12. POI 和人口使用真实 2026 外部数据；缺失人口不得用 0 占位。

CRAFT 未公开原始预处理脚本，以下仍是证据化推断，输出元数据中必须继续标为 `inferred`：

- 内部 0 段线性插值：左右均有正值才插值，不做首尾外推。
- 当前归一化重建为逐城市、训练窗口全部 in/out 共同拟合全局 Min-Max。
- 未预分类的 OSM 多标签 POI 按 CRAFT 12 类固定顺序选择单一类别。
- WorldPop 像元归属 2 km 栅格拟采用“像元中心点落格”规则；必须在实现和元数据中明确其为复现选择。

## 3. 已完成的实现

### 3.1 GTG→CRAFT 预处理

核心文件：

- `src/gtg_preprocessing/contracts.py`
- `src/gtg_preprocessing/flow.py`
- `src/gtg_preprocessing/static.py`
- `src/gtg_preprocessing/pipeline.py`
- `scripts/export_gtg_study_area.py`
- `scripts/build_gtg_craft_data.py`
- `configs/gtg_craft_preprocess.yaml`
- `docs/GTG_CRAFT_DATA_PREPROCESSING.md`
- `tests/test_gtg_preprocessing.py`

已实现：

- 合并 GTG 原随机 train/test。
- `start_time` 转 `Asia/Shanghai`。
- 用 `rid_list + dur_list + GTG road geometry` 估算逐路段时间，并按投影长度比例估算路段内部穿界时间。
- 统计逐小时栅格车辆边界流入/流出。
- 2 km UTM 格网、邻接关系、GTG/OSM 道路转换、12 类 POI、8 类道路、45 维 CRAFT 静态特征。
- 内部 0 线性插值、24 小时窗口、48 值全正筛选、训练集 Min-Max。
- `use_source_validation: false`：固定 epoch、关闭验证早停/验证调度、保存最后 EMA。
- `retrieve_match_month: false`：检索只匹配 weekday/hour，并避免同一区域跨月重复占据 top-k。
- GTG 拓扑缓存构建优先读取 `gtg_road.csv`，不让 OSM 补路污染 GTG 拓扑。

CRAFT 固定类别：

- 12 类 POI：`bicycle_rental_amenity, fast_food_amenity, restaurant_amenity, bicycle_parking_amenity, cafe_amenity, public_transport, shop, tourism, leisure, office, historic, sport`。
- 8 类道路：`residential, trunk, primary, secondary, tertiary, motorway, living_street, unclassified`。
- 静态特征固定 45 维，顺序由 `craft_feature_columns()` 唯一定义。

### 3.2 研究区和 OSM

三城研究区已经生成在 `data/external_2026/{city}/study_area/`：

| 城市 | UTM | 完整格 | 保留格 | 合并轨迹数 | 使用 GTG 道路数 |
|---|---:|---:|---:|---:|---:|
| beijing | EPSG:32650 | 144 | 81 | 2,344,762 | 13,573 |
| chengdushi | EPSG:32648 | 30 | 23 | 1,252,233 | 3,142 |
| xianshi | EPSG:32649 | 30 | 26 | 804,302 | 3,881 |

六个 OSM GeoJSON 已放在：

- `data/external_2026/{city}/osm_poi_2026.geojson`
- `data/external_2026/{city}/osm_roads_2026.geojson`

前次真实审计结果：三城均为 EPSG:4326、几何有效、无空几何；道路文件只含 `residential/living_street`，没有规范化几何重复。POI 输入数北京 30,690、成都 5,167、西安 7,528；北京只有两条 `public_transport=no` 不分类，其余可分类。某些类别真实计数为 0，不是占位：北京/成都 `bicycle_rental=0`，成都/西安 `sport=0`。

### 3.3 WorldPop 2026

完整文件：

`/root/autodl-tmp/projects/Paper/data/external_2026/worldpop/chn_pop_2026_CN_100m_R2025A_v1.tif`

已验证：

- 精确大小 `920795037` bytes。
- BigTIFF，单波段 float32，LZW。
- EPSG:4326。
- 像元大小约 `0.000833°`（约 100 m）。
- NoData=`-99999`。
- 描述为 `CHN population 2026 [WorldPop R2025A v1]`。
- `tiffinfo -D` 完整遍历像元成功，退出码 0。

残缺下载保留为同目录下 `.tif.partial`；未经用户明确同意不要删除。

## 4. 当前真实状态与剩余风险

### 4.1 WorldPop、norm 与三城数据已完成

已新增 `scripts/extract_worldpop_population.py`，通过 `libtiff` 单 512×512 tile 流式读取 BigTIFF，并严格校验产品描述、2026、EPSG:4326、分辨率、NoData、单波段 float32。正式输出已生成：

- `data/external_2026/beijing/population_2026.csv`
- `data/external_2026/chengdushi/population_2026.csv`
- `data/external_2026/xianshi/population_2026.csv`

| 城市 | 有效像元点 | 真实零人口点 | selected 格 | 训练窗口 |
|---|---:|---:|---:|---:|
| beijing | 86,147 | 283 | 81 | 51,844 |
| chengdushi | 16,178 | 52 | 23 | 15,751 |
| xianshi | 16,863 | 0 | 26 | 18,058 |

输出字段严格为 `lon,lat,population`；未输出 NoData、NaN、Inf 或负值。三城每个 selected region 均有有效像元覆盖，真实零人口与无覆盖已区分。

三城 GTG→CRAFT 目录和训练 norm 已生成，未生成 GTG test/validation 文件。CRAFT 四城 norm 仅用各自 train 拟合，test 越界裁剪计数写入 `norm_flow_meta.json`。

环境现状：没有 `rasterio`、没有 Python `tifffile`，但有 Pillow 10.3.0 和 `/root/miniconda3/bin/tiffinfo`。不得为了方便把依赖装到 Paper 之外。可优先用 Pillow 对 BigTIFF 做按城市小窗口读取；如确需新依赖，只能放入 Paper 内独立环境，并记录原因。

### 4.2 已修复的严格性缺口

1. `prepare_population_features()` 目前先以全 0 数组承接聚合结果，对未匹配人口像元的区域只记录 `zero_population_region_ids`，没有区分：
   - 有有效 WorldPop 像元但人口和真实为 0；
   - 完全无有效像元覆盖，属于缺失。

   已统计每个 selected region 的有效像元覆盖数；覆盖数为 0 直接报错。真实有效像元和为 0 单独记录在 `zero_population_region_ids`。

2. `src/craft_integrated/data_loaders.py::load_region_graph()` 使用：

   ```python
   region_values.get(region_id, np.zeros(48))
   ```

   已删除静默补 0。现在完整区域图全部保留，`value_region_ids/value_mask` 标记有训练窗口的 active 区域；只有 active 区域参与 source 的流量-表征损失，无监督区域仍参与 GNN 消息传播但不伪造零流量。对存在的区域使用全部训练窗口均值，而不是硬编码 `start_hour=0`。北京、成都、西安及 CRAFT 四城均已完成 graph loader 冒烟。

3. 已在三个配置中加入绝对路径 mapping，并保留 `test_city_data_dirs_override_only_selected_city` 回归测试。

### 4.3 当前剩余工作

本轮定义的代码、数据、缓存和训练前 graph/loss 冒烟均已完成。后续可在明确 `src_cities/trg_cities` 后启动正式训练；不应重新下载 SDK 或请求 API。

### 4.4 当前测试结果

2026-08-22 实际运行：

```bash
python3 -m pytest -q \
  tests/test_gtg_preprocessing.py \
  tests/test_retriever_strict.py \
  tests/test_data_loaders.py \
  tests/test_model_integration.py
```

专项结果：`21 passed, 5 warnings`（含 graph mask 回归）；此前按项目要求设置 `LD_PRELOAD` 的全量结果为 `104 passed, 14 warnings`。mask 改动后另有 CPU GFA loss 冒烟通过：`loss=0.2269935`，finite。

警告为现有 PyG 可选 CUDA 扩展 ABI 不匹配、pyproj deprecation、TypedStorage deprecation；不能声称消除。

## 5. 接手后的推荐执行顺序

### 步骤 A：实现并测试 WorldPop→三城 CSV（已完成）

新增一个 Paper 内脚本和测试，建议路径：

`scripts/extract_worldpop_population.py`

要求：

- 流式/窗口读取全国 BigTIFF，不把全国 73,530×45,337 全部载入内存。
- 从 GeoTIFF 标签严格验证产品名、年份 2026、EPSG:4326、分辨率、NoData、单波段 float32。
- 按三城 `full_grid.geojson` bbox 读取，再用像元中心点筛到 full grid 范围。
- 删除 NoData；保留 WorldPop 中真实的 0；人口值不得为负、NaN、Inf。
- 输出字段严格为 `lon,lat,population`。
- 输出前检查 selected region 每格有效像元覆盖数均大于 0；分别记录真实零人口格。
- 不覆盖现有非空输出；写临时文件并校验后再切换。
- 为像元坐标计算、窗口边界、NoData、真实 0 与无覆盖差异写单元测试。

### 步骤 B：修复严格性缺口（已完成）

- 强化 `prepare_population_features()` 的覆盖校验。
- 去掉 `load_region_graph()` 的 `np.zeros(48)` 静默回退并补回归测试。
- 生成数据后给三个训练配置加入：

```yaml
city_data_dirs:
  beijing: /root/autodl-tmp/projects/Paper/data/gtg_craft/beijing
  chengdushi: /root/autodl-tmp/projects/Paper/data/gtg_craft/chengdushi
  xianshi: /root/autodl-tmp/projects/Paper/data/gtg_craft/xianshi
```

### 步骤 C：生成 GTG 三城 CRAFT 兼容数据（已完成）

先逐城运行，检查过滤损失和异常，再全量运行。流水线拒绝覆盖非空目录；不要删除已有输出绕过检查。

```bash
python3 scripts/build_gtg_craft_data.py \
  --config configs/gtg_craft_preprocess.yaml \
  --cities beijing

python3 scripts/build_gtg_craft_data.py \
  --config configs/gtg_craft_preprocess.yaml \
  --cities chengdushi

python3 scripts/build_gtg_craft_data.py \
  --config configs/gtg_craft_preprocess.yaml \
  --cities xianshi
```

每城应产生：

```text
data/gtg_craft/{city}/
├── grid_region.csv
├── grid_region_feature.csv
├── grid_region_rel.csv
├── poi.csv
├── population.csv
├── road.csv
├── gtg_road.csv
├── slide_bike_flow.csv
├── slide_bike_flow_train.csv
├── hourly_boundary_flow_raw.csv
├── hourly_boundary_flow_interpolated.csv
├── data_feature.json
├── road_type_mapping.json
└── preprocess_meta.json

data/norm_flow/{city}/
├── norm_train_len_24.csv
└── normalization_meta.json
```

因 `use_source_validation: false`，不得生成 GTG `slide_bike_flow_test.csv` 或 `norm_test_len_24.csv`。

重点审计 `preprocess_meta.json`：

- 三城 region 数必须分别为 81、23、26。
- `bad_list_lengths=0`。
- 边界穿越歧义率不超过配置阈值。
- 每个 selected region 必须有有效人口覆盖、道路/POI字段 finite。
- 48 值全正筛选后的区域覆盖不能静默缺失；若丢失区域，先报错并分析，不允许图构建补 0。
- 45 维静态特征顺序、ID 连续性、邻接关系、流量列表长度必须严格验证。

### 步骤 D：生成 CRAFT 四城 norm 并完成读取测试（已完成）

先修改/确认 `scripts/gen_norm_flow.py`：输出目录非空时不得覆盖；元数据明确 `inferred` 和 train-only 拟合。然后运行：

```bash
python3 scripts/gen_norm_flow.py \
  --craft_root /root/autodl-tmp/projects/CRAFT/cleared_data \
  --out_root /root/autodl-tmp/projects/Paper/data/norm_flow \
  --norm_mode global \
  --cities chi dc toronto ny
```

这一步只读 CRAFT，写 Paper。随后重跑相关测试；不得把当前 `chi` 缺文件失败描述为代码回归。

### 步骤 E：GTG 拓扑缓存和最终验收（已完成）

已运行：

```bash
python3 scripts/build_gtg_features.py \
  --craft_root /root/autodl-tmp/projects/Paper/data/gtg_craft \
  --cities beijing chengdushi xianshi
```

三城缓存均读取 `gtg_road.csv`，道路映射覆盖率 100%、空区域 0、region 特征 finite 且维度为 9；配置注入后的三城静态读取形状分别为 `(81,54)`、`(23,54)`、`(26,54)`。全量测试在规定 `LD_PRELOAD` 下为 `104 passed, 14 warnings`。

## 6. 完成定义

只有同时满足以下条件才可报告完成：

- 三城 `population_2026.csv` 来自已校验的 WorldPop 2026，不含 NoData/NaN/Inf/负值，无区域覆盖缺失。
- 三城 CRAFT 兼容静态、原始/插值流量、训练滑窗和训练归一化文件全部生成。
- 不生成 GTG 验证/测试流量文件。
- CRAFT 四城 norm 文件仅用各自 train 拟合并生成，测试读取器不再因文件缺失失败。
- 读取器不再对人口或缺失区域流量静默补 0。
- 三个训练配置显式指向 GTG 三城静态目录，CRAFT 测试数据仍只读原目录。
- GTG 拓扑缓存成功生成，ID、shape、finite、region 数全部严格一致；输入明确为 `gtg_road.csv`。
- 实际测试结果和剩余警告如实记录。
- 未修改 GTG/CRAFT 原始资产，未清理或覆盖用户的无关工作树改动。

## 7. 当前 Git/工作树提醒

当前存在多个 modified/untracked 文件，包括 Stage 2 的 `src/hcfm/`、`configs/stage2_*`、`docs/STAGE2_*`、`data_backup/` 等。它们不属于本轮人口/GTG 预处理收尾范围。接手代理必须先查看 `git status --short` 和相关 diff，只做最小局部修改，绝对不能 reset/clean。

## 8. 交给 Codex API/新代理的方式

本轮按用户要求只保存交接文档，不安装 Codex SDK、不发起 API 请求。把具备本机 `Paper/` 工作区权限的新代理工作目录设为：

`/root/autodl-tmp/projects/Paper`

并发送下面这一条消息即可：

```text
请完整阅读 AGENTS.md 和 docs/CODEX_API_HANDOFF.md，严格遵守写入范围与只读原始资产约束。当前步骤 A–E 已完成；先核对交接文档中的真实产物和测试结果，再根据用户新指令继续，不要重做已完成工作，不要清理或回滚现有工作树，不要启动昂贵训练。
```

普通远程 Responses API 请求不会自动获得服务器本地文件系统；若所谓“Codex API”没有挂载该工作区，必须把本文件内容作为输入并另外提供受控的本地文件/命令工具，否则它只能给出建议，不能真正续做仓库任务。
