# 第一阶段实现说明（GTG 拓扑特征融入 CRAFT）

本阶段以 **CRAFT 为主框架**，在其区域表征进入 GFA（GraphTransformer）**之前**，融入
**GTG 的空间句法 / 路网拓扑特征**。保留 CRAFT 的 Population/POI/Road 特征、区域网格划分、
空间编码器、GFA（TFA/CCA）、检索增强条件、Diffusion 训练/采样/验证/测试、原始数据划分、
损失与指标不变。本阶段**不含** GTG 的强化学习、路径规划、轨迹决策、出行偏好、成本预测，
**不替换** CRAFT 的 Diffusion，不引入额外创新模块 / 对比目标 / 新损失。

## 0. 边界与只读约束

- 所有新增/修改代码、配置、文档、测试、中间结果**只写在 `Paper/` 内**。
- `GTG/`、`CRAFT/` 原始目录、原始数据集目录**只读**，未修改/覆盖/移动/删除。
- 未执行 `git push`、强制重置、清理或删除。
- 严格模式默认开启：缺失城市、坐标不一致、区域数不一致、特征含 NaN/Inf、映射覆盖率异常
  → 直接报明确错误，**不静默补零**。
- 归一化参数**只在训练城市/训练数据上拟合**，再应用到验证/测试，杜绝信息泄漏。

## 1. 端到端数据流

```
GTG 路网 (road.csv)
  └─ 对偶图 (gtg_features/dual_graph.py, 角度分段, 移植 GTG gen_edge_data)
       └─ 空间句法 (gtg_features/space_syntax.py, graph_tool)
       │     Connectivity / Total Depth / Integration / Choice / mean_depth
       └─ Metis 分区 (gtg_features/partition.py, pymetis) → 4 个分区池化特征
            └─ 路段→区域映射 (gtg_features/road_to_region.py, 长度加权)
                 └─ 缓存 cache/gtg/{city}_gtg_region.npz  (9 维 / 区域, 原始未归一化)
                      │
CRAFT 区域特征 (grid_region_feature.csv, 45 维 Population/POI/Road)
                      ▼  data_loaders.load_region_feature (融合模式拼接)
             节点特征 [craft(45) | gtg(9)] = 54 维
                      ▼  rep_model.GTAggregator.forward (融合注入点)
     craft(45)─init_proj─┐
                         ├─ concat ─ fusion_proj ─► GraphTransformer(GFA) ─► 区域表征
     gtg(9) ─gtg_branch──┘        (逐图标准化)
                      ▼
        检索增强条件 → Gaussian Diffusion 1D + Unet1D → 训练/采样/验证/测试
```

**融合发生在 GFA 之前的表征层**，与原任务要求一致（不在 Diffusion 内、不替换 Backbone）。

## 2. 新增模块（`Paper/src/gtg_features/`）

| 文件 | 职责 | 关键点 |
|------|------|--------|
| `dual_graph.py` | 由 `road.csv` 构建路段对偶图 | 移植 GTG `gen_edge_data`；road_id 重编号 0..N-1；几何 WGS84→UTM（按城市 `utm_epsg`）；有向对偶边 `B.from==A.to`；边属性 length/dist/angle；`calc_angle` 用端点 atan2；groupby 字典避免 O(N²) |
| `space_syntax.py` | 空间句法四指标 | `import graph_tool.all as gt`；Choice=有向长度加权 betweenness；Connectivity/TotalDepth/Integration 在无向视图；Integration=1/RA, RA=2(MD-1)/(n-2)；n≤2 退化节点显式计数（不静默） |
| `partition.py` | Metis 分区 + 分区池化 | 移植 GTG `metis_cluster`（pymetis）；4 指标按簇均值后广播回路段（`part_` 前缀） |
| `road_to_region.py` | 路段特征→区域 | geopandas sjoin 路段×区域；长度加权均值；报告 unmapped/empty 比例，超阈值报错 |
| `pipeline.py` | 编排 + 缓存 | `build_city()` 产出 `{city}_gtg_region.npz`(region_feat, feat_names) 与 `{city}_gtg_meta.json`；`load_city_gtg()`；**缓存原始未归一化值**，归一化在模型分支内逐图完成 |

**特征顺序（9 维，固定）：**
`connectivity, total_depth, integration, choice, mean_depth, part_connectivity, part_total_depth, part_integration, part_choice`

## 3. 修改的 CRAFT 文件（`Paper/src/craft_integrated/`）

拷贝自 CRAFT（只读原始未动），在副本上做**最小侵入式**改动：

### 3.1 `data_loaders.py`
- 新增模块级配置 `_CRAFT_DATA_ROOT / _NORM_FLOW_ROOT / _USE_GTG_TOPOLOGY / _GTG_CACHE_DIR / _GTG_FEATURE_DIM` 与 `configure(cfg)`（启动注入，默认值保持原始 CRAFT 行为）。
- `load_norm_flow`：路径改为 `_NORM_FLOW_ROOT or _CRAFT_DATA_ROOT`（读 Paper 内生成的 norm 流量）。
- `load_region_feature`：读 `_CRAFT_DATA_ROOT` 的 45 维原始特征；`_USE_GTG_TOPOLOGY=True` 时经
  `_load_gtg_region_feature(city, num_regions)` 拼接 9 维 GTG 特征 → 54 维。
- `_load_gtg_region_feature`：**严格校验**——缺缓存 / 区域数不符 / 维度不符 / NaN·Inf 直接 raise。

### 3.2 `rep_model.py`
- `GTAggregator.__init__`：`use_gtg_topology=cfg.get(...)`；为真时构建 `gtg_branch=GTGTopoBranch(...)`
  与 `fusion_proj = Linear(2*rep_dim→rep_dim)+ReLU+Linear(rep_dim→rep_dim)`。
- `GTAggregator.forward`：融合时按 `raw_feature_dim=45` 切分，`craft` 走 `init_proj`，`gtg` 走
  `gtg_branch`，concat 后 `fusion_proj`，再进 `gnn`；**基线路径 `nodes=init_proj(nodes)` 逐字节不变**。
- 新增 `GTGTopoBranch`：`proj Linear` + N×`GATv2Conv(concat=False, heads)` 残差+relu；输入端
  **逐图标准化**（mean/var, eps=1e-5，与 CRAFT `FeatureInitLayer` 一致，无跨集合统计→无泄漏）。

### 3.3 `main.py`
- 新增 `--config`（默认 `configs/config.yaml`）；`config.update(非 None 命令行项)`；
  `setdefault('pretrain_epoch',20)`/`setdefault('diff_train_epoch',3)`；启动即 `data_loaders.configure(config)`。

## 4. 检查点兼容性

- **基线（`use_gtg_topology=false`）与原始 CRAFT 结构逐字节等价**：`state_dict` 键集完全相同，
  相同权重相同输入前向 `allclose(atol=1e-6)`（见测试 `test_baseline_numerically_equals_original_craft`）。
- **旧 ckpt 可直接加载进融合模型**：`init_proj/gnn` 键名不变；`gtg_branch/fusion_proj` 为新增
  → `load_state_dict(strict=False)` 的 `missing` 全部属于 GTG 新增，`unexpected==[]`
  （见 `test_old_ckpt_loads_into_fusion`）。

## 5. 缺失的 norm 流量文件（重建说明）

CRAFT 原始数据缺 `norm_{phase}_len_24.csv`（原仓库无生成脚本）。由
`scripts/gen_norm_flow.py` 从 `slide_bike_flow` 重建，写入 `Paper/data/norm_flow/{city}/`：

- 默认 `norm_mode=global`：**在 train 全部 in/out 流量值上拟合 per-city min-max**，应用到
  train+test，clip 到 [0,1] 并记录裁剪计数。选择 global 而非 per-region，因 toronto/ny 的测试
  区域含训练集不存在的区域（`{37,42}`、`{79,89}`），per-region 拟合会缺参；global 亦保留跨区域
  幅度供 cpc 指标。
- 训练拟合、无泄漏：验证/测试仅**应用** train 的 min/max，不参与统计。
- 裁剪计数：chi train=0/test=0；dc test=192；toronto test=96；ny test=168（均为 test 超出 train 极值的自然裁剪）。

## 6. 配置文件（`Paper/configs/`）

- `baseline.yaml`：`use_gtg_topology: false`，其余同原始 CRAFT。
- `fusion.yaml`：`use_gtg_topology: true` + `gtg_cache_dir`、`gtg_feature_dim: 9`、
  `gtg_gat_layers: 4`、`gtg_gat_heads: 8`、`gtg_dropout: 0.1`。
- 两者：`craft_data_root=/root/autodl-tmp/projects/CRAFT/cleared_data`、
  `norm_flow_root=/root/autodl-tmp/projects/Paper/data/norm_flow`、`raw_feature_dim: 45`。

## 7. 环境依赖与库冲突

- 离线特征构建阶段用 `graph_tool`（C++，betweenness/最短距离）、`pymetis`、`geopandas`、`shapely`、`pyproj`。
- 训练阶段用 `torch`、`torch_geometric`。
- `graph_tool` 与 `torch` 各自捆绑不兼容的 `libgomp`/`libstdc++`，**同进程冲突**；但真实流程二者
  从不共存（graph_tool 只在 `build_gtg_features.py`，torch 只在训练）。测试因此**分两次调用**
  （详见 `STAGE1_TEST_REPORT.md`）。

## 8. 复现步骤

```bash
# 1) 生成缺失的 norm 流量文件 (train 拟合, 无泄漏)
cd /root/autodl-tmp/projects/Paper
python3 scripts/gen_norm_flow.py

# 2) 离线构建 GTG 区域拓扑特征 (graph_tool 环境)
python3 scripts/build_gtg_features.py     # 产出 cache/gtg/{city}_gtg_region.npz

# 3) 训练 (torch 环境)
cd src/craft_integrated
python3 main.py --config ../../configs/fusion.yaml     # 融合
python3 main.py --config ../../configs/baseline.yaml   # 基线对照
```
