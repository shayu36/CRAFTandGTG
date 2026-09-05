# 第二阶段：三层 GraphGPS + LapPE

## 1. 范围

第二阶段第一版直接消费第一阶段 `three-layer-start-road-v2` cache：

```text
Road road_x[M,33] + directed road_edge_index
  → Road LapPE + Road GraphGPS
  → P(syntax←road) mean pooling
Syntax syntax_x[K,5] + syntax_edge_index
  → Syntax LapPE + fusion + Syntax GraphGPS
  → P(region←syntax) UTM-length normalized weighted pooling
Region region_x[N,45] + region_edge_index
  → Region LapPE + fusion + Region GraphGPS
  → H_region[N,128]
  → prediction head
  → pred[N,48]
```

唯一可学习消息路径为 `Road → Syntax → Region`。模型没有 Road→Region 聚合模块。
Road 输入仍严格是第一阶段固定 33 维 START-style 静态特征，不加入流量、轨迹、
POI、人口、坐标、Road ID embedding、START `trans_prob` 或 CoSpec 特征。

## 2. Sparse LapPE

`src/three_layer_graphgps/spectral_lap_pe.py` 对每层执行：

1. 复制并无向化该层 `edge_index`，去除自环、coalesce 重复边；
2. 用 `scipy.sparse` 构造 symmetric normalized Laplacian；
3. 用 `scipy.sparse.linalg.eigsh` 求最小 eigenpairs；
4. 按 eigenvalue 升序排列；节点数不足或 ARPACK 部分收敛时，定长 padding 并返回 mask；
5. 输出 `eigvals[V,k,1]`、`eigvecs[V,k]`、`mask[k]`；
6. 按城市、层名、节点数、无向图 hash、k、normalization、静态图版本和 PE 版本缓存。

Road 消息边与 PE 边是两个张量：

```text
road_edge_index_msg = 第一阶段原始有向 road_edge_index
road_edge_index_pe  = 仅用于 Laplacian 的无向副本
```

不会用 PE 边覆盖消息边。实现中没有 `L.toarray()` 或 `np.linalg.eigh`。

## 3. LapPE encoder

训练时，每个有效 eigenvector 独立随机乘 `+1/-1`。每个频率编码
`[eigenvector value, eigenvalue]`，使用 DeepSet MLP 后在频率维做 masked mean，
再与该层原始节点特征的独立投影拼接融合。

Road、Syntax、Region 分别拥有 `33→hidden_dim`、`5→hidden_dim`、
`45→hidden_dim` 投影和各自的 LapPE encoder。

## 4. GraphGPS 层

每个 GPS block 使用并行分支：

```text
h = h + GATv2(LayerNorm(h), directed_or_layer_edge_index)
      + GlobalAttention(LayerNorm(h))
h = h + FFN(LayerNorm(h))
```

- Road 默认 `linear` 全局注意力，避免 `M≈14k` 时的 `O(M²)` attention matrix。
- Road 支持 `none/local/linear/full`；`full` 只有显式配置才启用，超过
  `road_full_attn_max_nodes` 自动 warning 并 fallback 到 linear。
- Syntax 和 Region 默认标准 `MultiheadAttention` full attention。
- GATv2 不额外写入 self-loop；节点自身信息由 residual 路径保留，原有向边不被对称化。

## 5. 监督目标和数据边界

仓库当前北京、成都、西安只有 `norm_train_len_24.csv`，没有独立 source
validation/test 文件。入口按城市内唯一 `date + start_hour` 时间键做确定性的
`0.8/0.1/0.1` chronological split，在每个 split 内按 Region 求 24 步 in/out
均值并拼为 48 维：

```text
label[active_region, 48] = [mean_in_flow_24, mean_out_flow_24]
```

没有观测的 Region 不制造零标签，只通过 `region_ids` 从 loss/metric 中排除。
三个 source 共享同一个 GraphGPS 编码器。target 只允许加载静态三层图和 LapPE，
不读取 target 动态流量。

这是一版静态 Region profile prediction baseline，不是逐时间窗口预测器，也不是
HCFM、Diffusion 或 Flow Matching 生成模型。

## 6. 配置与运行

预计算三层 LapPE：

```bash
python scripts/train_three_layer_graphgps.py \
  --config configs/stage2_three_layer_graphgps_lappe.yaml \
  --action precompute \
  --source_cities beijing chengdushi xianshi
```

真实三城 forward/shape 校验：

```bash
python scripts/train_three_layer_graphgps.py \
  --config configs/stage2_three_layer_graphgps_lappe.yaml \
  --action validate \
  --source_cities beijing chengdushi xianshi
```

单 epoch 完整 smoke：

```bash
python scripts/train_three_layer_graphgps.py \
  --config configs/stage2_three_layer_graphgps_lappe.yaml \
  --action smoke \
  --source_cities beijing chengdushi xianshi
```

正式训练与 checkpoint 评估：

```bash
python scripts/train_three_layer_graphgps.py \
  --config configs/stage2_three_layer_graphgps_lappe.yaml \
  --action train

python scripts/train_three_layer_graphgps.py \
  --config configs/stage2_three_layer_graphgps_lappe.yaml \
  --action evaluate \
  --checkpoint outputs/stage2_three_layer_graphgps_lappe/best.pt
```

target 若已具备真实 START v2 静态 cache，可以在 `precompute/validate` 中显式提供
`--target_city CITY`。它不会参与 source 监督 loss。

## 7. 实际 smoke 结果

2026-09-05 在 CPU 上使用默认 `hidden_dim=128, k=16` 执行三城验证：

| city | Road | Syntax | Region | pred |
|---|---:|---:|---:|---:|
| beijing | `[14685,128]` | `[293,128]` | `[81,128]` | `[81,48]` |
| chengdushi | `[3514,128]` | `[70,128]` | `[23,128]` | `[23,48]` |
| xianshi | `[4147,128]` | `[82,128]` | `[26,128]` | `[26,48]` |

三城共享模型的 1 epoch CPU smoke 已完成 forward、backward、optimizer step、
validation、test、best/last checkpoint 和 metrics 写入。smoke 指标只证明流程连通，
不能作为正式训练效果结论。

## 8. 第一版明确未实现

- SignNet 对照；
- 显式 high/low spectral filtering；
- high/low transfer loss。

LapPE encoder 和层级模型各自独立，后续可在不改变两级稀疏算子方向的前提下扩展。
