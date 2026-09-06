# 第二阶段：三层 GraphGPS + LapPE + 显式谱解耦

## 1. 范围与最终流程

第二阶段消费第一阶段 `three-layer-start-road-v2` 三层静态 cache：

```text
Road road_x[M,33] + directed road_edge_index
  → sparse Laplacian eigenpairs → LapPE → Road GraphGPS → H_road
  → P(syntax←road) mean pooling（使用 mixed H_road）
Syntax syntax_x[K,5] + syntax_edge_index
  → sparse Laplacian eigenpairs → LapPE + fusion → Syntax GraphGPS → H_syntax
  → P(region←syntax) UTM-length weighted pooling（使用 mixed H_syntax）
Region region_x[N,45] + region_edge_index
  → sparse Laplacian eigenpairs → LapPE + fusion → Region GraphGPS → H_region
  → Road/Syntax/Region 各层显式 low-rank spectral projection
  → 三层 low-frequency features + high-frequency residuals
```

唯一可学习跨层消息路径仍是 `Road → Syntax → Region`，不存在 Road→Region
聚合模块。Road 输入没有加入动态流量、轨迹、POI、人口、坐标、Road ID
embedding、START `trans_prob` 或 CoSpec 特征。

## 2. LapPE 与频率特征的区别

必须区分以下三个概念：

- LapPE 是 GraphGPS 的结构位置编码；
- GraphGPS hidden state `H_layer` 是混合频率表征；
- `H_layer_low` / `H_layer_high` 是在 GraphGPS 后显式投影得到的频率分量。

因此 LapPE 本身不等于低频输出，未经投影的 GraphGPS hidden state 也不能称为
已经解耦的高频或低频特征。

## 3. Sparse LapPE

`src/three_layer_graphgps/spectral_lap_pe.py` 对每层执行：

1. 复制并无向化该层 `edge_index`，去除自环并 coalesce；
2. 用 `scipy.sparse` 构造 symmetric normalized Laplacian；
3. 用 `scipy.sparse.linalg.eigsh` 求最小 eigenpairs；
4. 按 eigenvalue 升序排列，节点数不足或 ARPACK 部分收敛时定长 padding 并返回 mask；
5. 输出 `eigvals[V,k,1]`、`eigvecs[V,k]`、`mask[k]`；
6. 按城市、层名、图 hash、节点数、`k`、normalization、静态版本和 PE 版本缓存。

Road 的两个边张量保持分离：

```text
road_edge_index_msg = 第一阶段原始有向 road_edge_index
road_edge_index_pe  = 仅用于 Laplacian 的无向副本
```

Road 消息边没有被 PE 边覆盖。实现不调用 dense Laplacian、`toarray()`、
`np.linalg.eigh` 或 Road 最大 eigenpairs。

## 4. 显式谱特征分解

`src/three_layer_graphgps/frequency.py` 在每层 GraphGPS 输出后选择 mask 中有效、
eigenvalue 最小的前 `q_l` 个模式。零 eigenvalue（全局或连通分量模式）保留为低频：

$$
C^{(l)} = U_{low}^{(l)T} H^{(l)}
$$

$$
H_{low}^{(l)} = U_{low}^{(l)} C^{(l)}
$$

$$
H_{high}^{(l)} = H^{(l)} - H_{low}^{(l)}
$$

代码按 `U.T @ H`、`U @ coefficients` 的顺序计算，不构造节点×节点 projector。
复杂度为 `O(N_l q_l D)`。`H` 没有被 detach，低频、高频和预测损失均可回传到
三层 LapPE 输入模块与 GraphGPS。

运行时严格校验 eigenpair shape、dtype、device、finite、有效 mask、频率排序和
`U_low.T @ U_low ≈ I`。请求模式超过有效谱宽时发出 warning，并记录实际模式数；
不会把 padding 列参与投影。

## 5. GraphGPS 与注意力策略

每个 block 使用 local GATv2、global attention 和 residual FFN：

```text
h = h + LocalGATv2(LayerNorm(h), edge_index)
      + GlobalAttention(LayerNorm(h))
h = h + FFN(LayerNorm(h))
```

- Road 默认 `linear` global attention；不会默认构造 `M×M` attention。
- Road 显式选择 `full` 且节点数超过阈值时 warning 并 fallback 到 `linear`。
- Syntax 和 Region 默认标准 full attention。
- Road local message passing 始终使用原始有向图。
- 跨层 pooling 始终使用 mixed `H_road` 和 `H_syntax`，没有改变第一阶段算子定义。

## 6. 输出与预测头

模型保持原字段并新增平铺的稳定字段：

```text
H_road / H_road_low / H_road_high
road_low_coefficients
pooled_road_to_syntax

H_syntax / H_syntax_low / H_syntax_high
syntax_low_coefficients
pooled_syntax_to_region

H_region / H_region_low / H_region_high
region_low_coefficients
pred
```

默认 hidden dimension 为 128。预测头使用：

```text
concat(H_region_low, H_region_high)
→ frequency_fusion
→ prediction_head
→ pred[N,48]
```

`H_region` 继续作为 mixed representation 返回，用于兼容和重构检查。配置初始化时
强制 `model.output_dim == 2 * data.seq_length`，当前第一版显式只允许
`data.seq_length=24`；该字段已真实传给 flow loader，不再是无效 YAML 字段。

## 7. 第三阶段输入契约

导出版本为 `three-layer-spectral-features-v1`。预期语义和消费方向为：

```text
三层 H_low + source 时间序列特征
  → Stage 3 RAG retrieval
  → retrieved dynamic context
  → Hierarchical Flow Matching input/context

三层 H_high
  → Hierarchical Flow Matching target-specific condition
```

低频表示跨城市相对共享的平滑功能结构和宏观共性；高频保留城市局部结构、边界
变化和路网纹理。`build_low_frequency_transfer_inputs()` 仅提供静态低频迁移接口：
cost metric 固定为 cosine，多个 source 每城总质量为 `1/S`。它不计算新损失，
也不接受高频或 target 动态值。真实 target 到达后，可优先将 `H_region_low` 接入
原 CRAFT CCA/Wasserstein；当前没有宣称正式 source-target transfer 已完成。

## 8. 频率特征导出

导出必须显式提供训练完成的 v2 checkpoint：

```bash
python scripts/train_three_layer_graphgps.py \
  --config configs/stage2_three_layer_graphgps_lappe.yaml \
  --action export_features \
  --source_cities beijing chengdushi xianshi \
  --checkpoint outputs/stage2_three_layer_graphgps_lappe/best.pt
```

每个城市 `.pt` 保存三层 mixed/low/high、eigenvalues、实际低频模式数、Road 原始
ID、Syntax/Region 连续 ID、三层 graph hash、静态版本和 checkpoint SHA-256。
加载时 city、节点顺序、graph hash、静态版本或 checkpoint 任一不匹配都会拒绝。
target 导出走 `require_targets=False`，不读取 target flow。大型导出文件不应提交 Git。

## 9. 训练与验证命令

```bash
python scripts/train_three_layer_graphgps.py \
  --config configs/stage2_three_layer_graphgps_lappe.yaml \
  --action validate \
  --source_cities beijing chengdushi xianshi

python scripts/train_three_layer_graphgps.py \
  --config configs/stage2_three_layer_graphgps_lappe.yaml \
  --action smoke \
  --source_cities beijing chengdushi xianshi

python scripts/train_three_layer_graphgps.py \
  --config configs/stage2_three_layer_graphgps_lappe.yaml \
  --action train
```

监督 baseline 仍按城市内唯一 `date + start_hour` 做 chronological split，对每个
Region 聚合 24 步 in/out 为 48 维标签。无观测 Region 不伪造零标签。三个 source
共享同一个模型，target 只加载静态图和 LapPE。

## 10. 2026-09-06 真实三城验证

默认 `hidden_dim=128`、三层 `k=q=16` 的 CPU `validate` 已执行 forward/backward：

| city | Road low/high | Syntax low/high | Region low/high | 最大重构相对误差 | 最大正交残差 |
|---|---:|---:|---:|---:|---:|
| beijing | `[14685,128]` | `[293,128]` | `[81,128]` | `0.0` | `2.996e-7` |
| chengdushi | `[3514,128]` | `[70,128]` | `[23,128]` | `0.0` | `1.508e-7` |
| xianshi | `[4147,128]` | `[82,128]` | `[26,128]` | `0.0` | `1.476e-7` |

Road 分解耗时约为北京 `0.0092s`、成都 `0.0028s`、西安 `0.0028s`；投影中间张量
估算峰值分别为 `15,985,472`、`3,831,424`、`4,520,128` bytes。三城共享模型的
1 epoch CPU smoke 已完成 optimizer step、validation、test 和 v2 checkpoint 保存；
smoke 数值只证明流程连通，不作为正式训练效果。

真实 checkpoint 的三城 `export_features` 和严格回读也已运行通过。当前没有真实
target 静态数据，因此未导出或伪造 target。

## 11. 测试与环境限制

```text
pytest -q tests/test_stage2_graphgps_lappe.py tests/test_stage2_frequency_decoupling.py
→ 17 passed

pytest -q --ignore=tests/test_dual_graph.py
→ 143 passed

pytest -q
→ collection error: graph_tool requires GOMP_5.0, current libgomp ABI 不兼容
```

全量失败发生在未进入测试执行前的 `tests/test_dual_graph.py` import 阶段，与本次
频率代码无关。环境另有 PyG optional CUDA extensions ABI warning，PyG 自动禁用这些
扩展；本次 CPU 测试和真实验证仍完成。

## 12. 尚未实现

- SignNet 对照；
- 显式 high/low spectral filtering loss；
- high/low transfer loss；
- RAG；
- Hierarchical Flow Matching；
- 正式 target 训练；
- 完整 CoSpec dual-path/prototype 模块。

这些内容只保留消费接口或文档边界，没有被写成已完成模块。
