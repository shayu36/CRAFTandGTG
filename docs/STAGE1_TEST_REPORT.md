# 第一阶段测试报告

本报告记录**真实运行**的测试结果，未虚构任何通过项或指标。

## 运行环境与调用方式

- Python 3.10.20，pytest 9.1.1，平台 linux。
- 因 `graph_tool`（C++，捆绑较新 `libstdc++`/`libgomp`）与 `torch`（捆绑较旧
  `libgomp`，缺 `GOMP_5.0`）在同一 pytest 进程内加载时符号冲突，测试**分两次调用**。
  这与生产流程一致——二者从不共存（graph_tool 只在离线特征构建，torch 只在训练）。

### 调用 1：graph_tool 相关（对偶图 / 空间句法 / Metis）

```bash
cd /root/autodl-tmp/projects/Paper/tests
LD_PRELOAD="/root/miniconda3/lib/libstdc++.so.6 /root/miniconda3/lib/libgomp.so.1" \
  python3 -m pytest test_dual_graph.py -v
```

`LD_PRELOAD` 让 conda 的新版 `libstdc++`/`libgomp` 先于系统旧库进入进程，解决
`GLIBCXX_3.4.31 not found` 与 `GOMP_5.0 not found`。

### 调用 2：torch / 数据加载 / 特征质量 / norm

```bash
cd /root/autodl-tmp/projects/Paper/tests
python3 -m pytest test_model_integration.py test_data_loaders.py test_gtg_features.py test_norm_flow.py -v
```

（不加 `LD_PRELOAD`，走正常 torch 环境。）

## 结果汇总

| 调用 | 文件 | 用例数 | 结果 | 耗时 |
|------|------|--------|------|------|
| 1 | test_dual_graph.py | 4 | **4 passed** | 1.56s |
| 2 | test_model_integration.py + test_data_loaders.py + test_gtg_features.py + test_norm_flow.py | 44 | **44 passed** | 172.57s |
| — | **合计** | **48** | **48 passed, 0 failed** | — |

- 调用 1 原始尾行：`4 passed, 1 warning in 1.56s`
- 调用 2 原始尾行：`44 passed, 16 warnings in 172.57s (0:02:52)`
- warnings 均为第三方库弃用提示（`TypedStorage` deprecated、`np.find_common_type` deprecated、
  `GLib.unix_signal_add_full` deprecated），与被测逻辑无关。

## 各测试文件覆盖内容

### test_dual_graph.py（4）— 对偶图与空间句法（合成小路网，可手工核对）
- `test_calc_angle_horizontal`：水平线段角度为 0。
- `test_dual_graph_edges`：A→B→C→D 路径的对偶边为 `{(0,1),(1,2)}`，边长/距离形状与正负正确。
- `test_space_syntax_connectivity`：无向度 `[1,2,1]`；四指标全有限；中间节点 Choice（介数）最高。
- `test_metis_partition_small`：3 节点 / local_size=50 → 回退单簇（k=1），标签形状 (3,)。

### test_model_integration.py（6）— 模型集成与检查点兼容（CPU）
- `test_baseline_keys_no_gtg`：基线无 `gtg_branch/fusion_proj` 键。
- `test_fusion_adds_only_gtg_modules`：融合模型键集 ⊇ 基线键集，新增键**仅** GTG 相关。
- `test_old_ckpt_loads_into_fusion`：旧 ckpt 加载进融合模型，`unexpected==[]`，`missing` 全为 GTG 新增。
- `test_fusion_forward_smoke`：(N,45+9)→(N,128)，全有限。
- `test_baseline_forward_smoke`：(N,45)→(N,128)，全有限。
- `test_baseline_numerically_equals_original_craft`：加载原始 CRAFT `GTAggregator`，权重对齐后
  相同输入前向 `allclose(atol=1e-6)` —— **基线与原始 CRAFT 数值等价**。

### test_data_loaders.py（10）— 数据加载与融合维度
- `test_baseline_feature_dim[chi/dc/toronto/ny]`：基线区域特征 45 维，edge_index 形状 (2,·)。
- `test_fusion_feature_dim[chi/dc/toronto/ny]`：融合 45+9=54 维；全有限；前 45 维与基线一致。
- `test_fusion_missing_cache_dir_raises`：融合开启但未给缓存目录 → 严格报 `ValueError`。
- `test_region_graph_build_fusion`：融合模式下 `load_region_graph` 的 `x` 为 54 维，city 正确。

### test_gtg_features.py（16）— GTG 区域特征质量（4 城 × 4 检查）
- `test_cache_exists_and_shape`：缓存存在，特征顺序符合 FEATURE_ORDER，形状 (区域数, 9)。
- `test_no_nan_inf`：无 NaN/Inf。
- `test_coverage_reasonable`：未映射道路比例 ≤ 0.05，空区域比例 ≤ 0.2。
- `test_metrics_have_variation`：各特征非零值标准差 > 0（无退化常数列）。

### test_norm_flow.py（8）— 归一化流量（4 城 × 2 检查）
- `test_columns_and_seq_len`：列完整，in_flow/out_flow 序列长 24。
- `test_value_range_0_1`：train+test 值域 ⊆ [0,1]。
- `test_train_fit_reaches_bounds`：train（min-max 在 train 拟合）恰好触及 0 与 1。

## 结论

- **48 个测试用例全部通过，0 失败。**
- 关键验收点均由真实测试覆盖并通过：
  - 基线与原始 CRAFT **数值等价**（可回归、旧 ckpt 兼容）。
  - 融合仅新增 GTG 分支/融合层，前 45 维不变。
  - GTG 特征无 NaN/Inf、覆盖率合理、有区域区分度。
  - norm 流量值域正确、train 拟合无泄漏。
  - 严格模式在缺缓存/维度不符等情形正确报错。
