# STAGE1 审计报告：GTG 拓扑特征融入 CRAFT（第一阶段）

> 目标：以 CRAFT 为主框架，保留其 Population/POI/Road 特征、区域/网格划分、空间编码器、GFA（含 TFA/CCA）、
> 检索增强条件、Diffusion 训练/采样/验证/测试与原始数据划分/损失/指标；**新增** GTG 的数据处理与拓扑表征
> （Space Syntax、道路拓扑特征、GTG 拓扑聚合、Metis），并在 **CRAFT GFA 之前** 完成融合。
> 本阶段**不含** GTG 的强化学习/路径决策/出行偏好/成本预测。

本报告为**只读审计**结论，所有真实路径、字段、张量形状均经实机核对。

---

## 1. 真实路径

| 名称 | 路径 | 权限 |
|---|---|---|
| CRAFT 代码（只读） | `/root/autodl-tmp/projects/CRAFT/*.py` | 只读 |
| CRAFT 数据（只读） | `/root/autodl-tmp/projects/CRAFT/cleared_data/{city}/` | 只读 |
| GTG 代码（只读参考） | `/root/autodl-tmp/projects/GTG-main/*.py` | 只读 |
| 本阶段全部产出 | `/root/autodl-tmp/projects/Paper/` | 可写 |
| CRAFT 源码副本（改动在此） | `/root/autodl-tmp/projects/Paper/src/craft_integrated/*.py` | 可写 |

**硬边界**：GTG、CRAFT 原始目录与数据集只读；所有新增/修改代码、配置、文档、测试、中间结果只允许写入 `Paper/`。

---

## 2. CRAFT 原始数据流与关键文件

```
grid_region_feature.csv (N_region, 45维 CRAFT 特征)  ──┐
grid_region_rel.csv (is_adj → edge_index)          ──┤→ load_region_graph → Data(x,edge_index,value,city)
norm_train_len_24.csv (逐区域流量, value)           ──┘         │
                                                               ▼
                                              rep_model.GTAggregator (= GFA / 空间编码器)
                                              init_proj(45→128) + GraphTransformer(depth3,heads4)
                                              训练损失: self_sim_loss(TFA) + wasserstein_loss(CCA)
                                                               │  → {city}_rep.npy  (N_region,128)
                                                               ▼
                                              data_loaders: 逐样本取 feature=rep[region_id]
                                              + 检索增强(reference) → FlowDataset
                                                               ▼
                                              craft.CRAFTModel.get_cond:
                                              concat(feature128, hour64, weekday64, reference256)=512 → cond_mlp→256
                                                               ▼
                                              diffusion.GaussianDiffusion1D + unet.Unet1D
                                              x:(B,2,24) 训练/采样/验证
                                                               ▼
                                              generate.py → evaluate.py (cpc / min_max_mae / min_max_rmse)
```

### 关键张量形状
- CRAFT 区域特征：`node_feature (N_region, 45)`。45 维 = `population, population_density, dist_to_center, road_num, road_length`(5) + `poi_num_0..11`(12) + `poi_score_0..11`(12) + `road_num_0..7`(8) + `road_length_0..7`(8)。
- 区域邻接：`edge_index (2, E)`，来自 `grid_region_rel.csv` 中 `is_adj==1` 的 `(ori,des)`。
- GFA 输入 `GTAggregator.forward(nodes(N,45), edge_index)` → 内部构造 `adj_mat(N,N)` → `init_proj`→(N,128) → `GraphTransformer`→(N,128)。
- 表征输出 `{city}_rep.npy (N_region,128)`；扩散输入 `x (B,2,24)`；条件 `cond (B,256)`。

### 各城市规模（实测）
| city | region 数 | road 数 | utm_epsg | 训练流量行 | 测试流量行 | seq_len | train 覆盖区域 | test 覆盖区域 | test 缺席于 train |
|---|---|---|---|---|---|---|---|---|---|
| chi | 73 | 31093 | 32616 | 71836 | 35949 | 24 | 41 | 39 | 无 |
| dc | 81 | 39563 | 32618 | 41416 | 22916 | 24 | 31 | 27 | 无 |
| toronto | 60 | 20343 | 32617 | 26208 | 14915 | 24 | 29 | 22 | {37,42} |
| ny | 95 | 45289 | 32618 | 231652 | 126574 | 24 | 52 | 54 | {79,89} |

> `region_id` 均为 0..N-1 连续（满足 `load_region_feature` 的断言）。流量仅覆盖部分区域（如 chi 41/73）；无流量区域的 `value` 由 `load_region_graph` 填零（**仅 value，不影响 45 维特征**）。`utm_epsg` 取自各城市 `data_feature.json`，用于空间句法与 road→region 的度量投影，**不再猜测**。
> 注意：toronto/ny 存在 **test 区域未出现在 train** 的情形，直接影响逐区域归一化的可行性（见 4.5）。

---

## 3. GTG 侧被移植的真实实现（来源逐一核对）

| 能力 | GTG 源 | 移植方式 |
|---|---|---|
| 道路对偶图（角度/距离/长度/介数） | `GTG-main/prepare.py: gen_edge_data / calc_angle` | 忠实移植：邻接为 `from.to_node_id == this.from_node_id`；`length=均值`、`dist=形心距(投影CRS)`、`angle=端点方位角差`、`bet=betweenness` |
| 介数中心性 = 空间句法 Choice | `prepare.py` 用 `graph_tool.betweenness` | **`graph_tool` 实机可用**，直接复用其 `betweenness`（无需退化到 networkx） |
| Metis 图分区 | `GTG-main/dataloader.py: metis_cluster` (`pymetis.part_graph`) | 忠实移植：无向化邻接 + `num_clusters = num_nodes/local_size` |
| 连续特征 z-score 标准化 | `dataloader.py: get_node_feature` | 复用其“连续列 z-score、离散列保留、fillna(0)”策略，但**只用训练城市统计拟合**以防泄漏 |
| 拓扑聚合 GATv2（用户所称 “SAGAT”） | `GTG-main/models.py: TopoAggregator` | 忠实移植其结构：`proj(Linear) + N×GATv2Conv(concat=False,heads) + 残差 + relu` |

> 说明：GTG 代码库中并**不存在**名为 `SAGAT` 的模块；用户所指为 `TopoAggregator`（GATv2 拓扑聚合器），本阶段按此忠实实现。
> GTG 的 `GTGModel/DisentEncoder/ObsCostPredictor/DomClassifier`（解耦编码、成本预测、域判别、GRL 对抗）属于强化学习/成本预测/出行偏好范畴，**本阶段不引入**。

### 空间句法四项核心指标（在 CRAFT 道路对偶图上用 graph_tool 计算）
- **Connectivity（连接度）**：对偶图节点度。
- **Total Depth（总深度 TD）**：节点到所有可达节点的最短路径距离之和。
- **Integration（整合度）**：由平均深度 `MD=TD/(n-1)` 得相对不对称 `RA=2(MD-1)/(n-2)`，`Integration=1/RA`（对孤立/退化节点严格报错或按分量处理，不静默补零）。
- **Choice（选择度）**：= betweenness 介数中心性（与 GTG `gen_edge_data` 的 `bet` 同源）。

---

## 4. 关键发现与设计决策

### 4.1 CRAFT 与 GTG **零城市重叠**
- CRAFT：chi/dc/toronto/ny（美国/加拿大）；GTG：beijing/chengdushi/xianshi（中国）。
- 结论：**不能**直接使用 GTG 的道路数据/预训练权重；必须把 GTG 的算法作用到 CRAFT 自己的 `road.csv` 上。
- GTG 预训练权重（beijing→xianshi）与本任务在维度与城市上均不兼容 → GTG 拓扑聚合分支**端到端训练**，不冻结、不加载 GTG 权重。

### 4.2 空间句法在 GTG 代码中由 QGIS 计算，仓库内无实现
- GTG 的 `road.csv` 已含 `CONN/TD/INT/CH/...`（QGIS Space Syntax Toolkit 产物），代码只读取。
- CRAFT `road.csv` **无**这些列 → 必须在 CRAFT 道路对偶图上**自行计算**四项核心指标（graph_tool），这是本阶段新增数据处理的核心。

### 4.3 融合注入点：**CRAFT GFA 之前**（表征级）
数据流：
```
CRAFT road.csv → GTG 对偶图 → 空间句法/拓扑(road级) → Metis 分区聚合
  → road→region 长度加权映射 → region 级 GTG 特征 gtg_region(N,K) [离线缓存]
  → [可学习] GTG GATv2 分支(TopoAggregator 结构, 作用于 region 邻接) → h_gtg(N,128)
  → 与 h_craft = init_proj(craft45) 融合 fusion_proj(concat) → fused(N,128)
  → CRAFT GraphTransformer(GFA) → {city}_rep.npy → 检索增强 → Diffusion（全部不变）
```
- **可学习聚合放在 region 级**（73 节点，代价低、可 CPU 冒烟、与 ckpt 兼容），空间句法/介数/Metis 等 road 级重计算放在**离线缓存**。
- 备选方案（road 级可学习 GNN 后再池化，31k 节点）代价高且威胁 ckpt 兼容，本阶段不采用，记为后续可选项。此选择落在任务规定的“road 级特征映射到 region 后于 GFA 前融合”数据流之内，非研究性分叉。

### 4.4 checkpoint 兼容设计
- `GTAggregator` 保持 `init_proj`(45→128) 与 `gnn`(GraphTransformer) 键名不变；GTG 分支 `gtg_encoder` 与 `fusion_proj` 为**新增**键。
- 基线（`use_gtg_topology=false`）：结构与原 CRAFT **逐字节等价**，旧 `craft.pth`/`{city}_rep.npy` 完全可加载，数值可复现。
- 融合（`use_gtg_topology=true`）：加载旧 ckpt 时仅 `gtg_encoder/fusion_proj` 为 missing keys，其余严格命中。
- 融合模式下区域特征在数据加载期拼接为 `(N, 45+K)`；`forward` 按 `raw_feature_dim=45` 切分，前 45 维走 `init_proj`，后 K 维走 GTG 分支。`raw_feature_dim` 始终为 45。

### 4.5 缺失的 `norm_{phase}_len_24.csv` —— 重建（非数据缺失）
- 数据集**存在** `slide_bike_flow_{train,test}.csv`（原始流量，最大值达 335），但 CRAFT 未随附 `norm_*.csv`，也无生成脚本。
- `evaluate.py` 全程在归一化空间比较（`min_max_mae/rmse`），`norm_*.csv` 只需 [0,1] 归一化流量，**无需存储反归一化参数**。
- **重建方案**：**逐城市全局 min-max 归一化**，**仅用该城市训练集拟合**（对 in/out flow 全体训练值取 min/max），同时应用到 train/test，生成到 `Paper/data/norm_flow/{city}/`。默认 `norm_mode: global`——始终有定义、无泄漏、保留跨区域量级（利于 `cpc`）、且规避 toronto/ny 的“test 区域未见于 train”边界问题。
- test 值可能超出 train min/max → 归一化后裁剪到 [0,1]，并**记录裁剪数量**（不静默）。
- 备选 `norm_mode: region`（逐区域 min-max，train 拟合）保留可用；对未出现在 train 的 test 区域，回退到该城市训练全局统计并**计数记录**（仍仅用训练统计，无泄漏，非静默补零）。
- 此为在无原始配方情况下的可复现、无泄漏重建，已在脚本与本报告显式标注为假设。

### 4.6 严格模式（默认启用）
缺失城市 / 坐标系不一致 / 区域数量不一致 / 特征含 NaN/Inf / 映射覆盖率异常 → **直接报明确错误**，不静默补零。归一化参数只用训练城市/训练数据拟合再应用到验证/测试。

---

## 5. 环境依赖（实机验证）
| 依赖 | 状态 |
|---|---|
| `graph_tool` | ✅ 可用（介数/最短路径用其 C++ 实现） |
| `pymetis` | ✅ `part_graph` 正常 |
| `torch_geometric.GATv2Conv` | ✅ CPU 前向正常 |
| `geopandas / shapely` | 用于 WKT 解析与投影（road→region 映射、形心距） |

---

## 6. 计划改动位置（仅在 Paper/ 内）
| 文件 | 改动 |
|---|---|
| `Paper/src/gtg_features/` (新增) | 空间句法/对偶图/Metis/road→region 映射/归一化/缓存 |
| `Paper/src/craft_integrated/rep_model.py` | `GTAggregator` 加 GTG 分支 + 融合层（`use_gtg_topology` 开关） |
| `Paper/src/craft_integrated/data_loaders.py` | 可配置数据根路径；融合模式拼接 GTG 区域特征 |
| `Paper/configs/{baseline,fusion}.yaml` | 两套配置 |
| `Paper/scripts/` | norm 生成、GTG 特征预处理入口 |
| `Paper/tests/` | 单元/基线回归/特征质量/模型 smoke |

---

## 7. 已识别风险
1. **空间句法全对全最短路**：31k 节点对偶图 all-pairs 深度计算约 O(n·E)，用 graph_tool 单源 BFS 累加，预计每城市数分钟；结果缓存。必要时加半径上限并显式 `log` 丢弃范围（不静默截断）。
2. **road→region 覆盖率**：部分道路可能落在任何区域之外，或某些区域无道路。严格模式下记录空区域比例/未映射比例；空区域的 GTG 特征处理策略需显式（报错阈值或标记），写入覆盖报告。
3. **对偶图连通性**：多连通分量会使 TD/Integration 在跨分量间无定义；按分量内计算并显式标注，不跨分量补零。
4. **norm 重建假设**：逐区域 min-max 为重建选择，影响学习分布但不影响管线正确性；已配置化并显式标注。
5. **oneway 方向**：CRAFT `road.csv` 有 `oneway` 字段；GTG `gen_edge_data` 默认按 `from/to_node_id` 有向连边，本阶段沿用 GTG 有向构造，不额外引入方向假设。
