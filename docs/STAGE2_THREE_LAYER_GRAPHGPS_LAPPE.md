# 第二阶段：三层 GraphGPS + LapPE + 显式谱解耦

## 1. 范围与最终流程（联合 v2）

第二阶段消费第一阶段 `three-layer-start-road-v2` 三层静态 cache：

```text
Road road_x[M,33] ─┐
Syntax syntax_x[K,5] ─┼→ 分层输入投影 + 节点类型编码 + 统一 LapPE
Region region_x[N,45] ─┘
        ↓ x_joint[M+K+N, hidden_dim]
统一有向异构 edge_index/edge_type/edge_weight
        ↓ 唯一 GraphGPSStack（只调用一次）
H_joint[M+K+N, hidden_dim]
        ↓ 统一谱低/高频分解
按固定节点区间切分 H_road/H_syntax/H_region 与 low/high
        ↓ 只将 Region low+high 接入 prediction_head
```

节点编号固定为 `Road=[0,M)`、`Syntax=[M,M+K)`、`Region=[M+K,M+K+N)`；
Road→Syntax 和 Syntax→Region 的第一阶段算子权重原样保存并用于 local message passing。

唯一可学习跨层消息路径仍是 `Road → Syntax → Region`，不存在 Road→Region
聚合模块。Road 输入没有加入动态流量、轨迹、POI、人口、坐标、Road ID
embedding、START `trans_prob` 或 CoSpec 特征。

## 2. LapPE 与频率特征的区别

必须区分以下三个概念：

- LapPE 是 GraphGPS 的结构位置编码；
- GraphGPS hidden state `H_joint` 是混合频率表征；
- `H_joint_low` / `H_joint_high` 是在 GraphGPS 后显式投影得到的频率分量。

因此 LapPE 本身不等于低频输出，未经投影的 GraphGPS hidden state 也不能称为
已经解耦的高频或低频特征。

## 3. Sparse LapPE

`src/three_layer_graphgps/spectral_lap_pe.py` 对联合消息图执行：

1. 复制并无向化完整 `edge_index_joint`，去除自环并 coalesce；
2. 用 `scipy.sparse` 构造 symmetric normalized Laplacian；
3. 用 `scipy.sparse.linalg.eigsh` 求最小 eigenpairs；
4. 按 eigenvalue 升序排列，节点数不足或 ARPACK 部分收敛时定长 padding 并返回 mask；
5. 内部使用 `eigvals[V,k,1]`、`eigvecs[V,k]`、`mask[k]`；共享谱在 v2 导出为
   `joint_eigvals[1,k,1]`、`joint_eigvecs[V,k]`、`joint_eigenpair_mask[k]`；
6. 按城市、联合图 hash、节点数、`k`、normalization、层次结构版本和 PE 版本缓存。

联合图的两个边张量保持分离：

```text
edge_index_joint_msg = 有向、带关系和权重的完整三层消息图
edge_index_joint_pe  = 仅用于 Laplacian 的无向副本
```

有向消息边没有被 PE 边覆盖。实现不调用 dense Laplacian、`toarray()`、
`np.linalg.eigh` 或按层重复求 eigenpairs。

## 4. 显式谱特征分解

`src/three_layer_graphgps/frequency.py` 在统一 GraphGPS 输出后选择 mask 中有效、
eigenvalue 最小的前 `q_l` 个模式。零 eigenvalue（全局或连通分量模式）保留为低频：

$$C_{joint} = U_{low,joint}^{T} H_{joint}$$

$$H_{joint,low} = U_{low,joint} C_{joint}$$

$$H_{joint,high} = H_{joint} - H_{joint,low}$$

代码按 `U.T @ H`、`U @ coefficients` 的顺序计算，不构造节点×节点 projector。
复杂度为 `O(N_l q_l D)`。`H` 没有被 detach，低频、高频和预测损失均可回传到
三层 LapPE 输入模块与 GraphGPS。

运行时严格校验 eigenpair shape、dtype、device、finite、有效 mask、频率排序和
`U_low.T @ U_low ≈ I`。请求模式超过有效谱宽时发出 warning，并记录实际模式数；
不会把 padding 列参与投影。

## 5. GraphGPS 与注意力策略

模型只有一个 `self.graphgps`。每个 block 使用关系感知加权 local message passing、
global attention 和 residual FFN：

```text
h = h + LocalRelationWeightedMP(LayerNorm(h), edge_index, edge_type, edge_weight)
      + GlobalAttention(LayerNorm(h))
h = h + FFN(LayerNorm(h))
```

- 默认 `attention.global_attn=linear`，global branch 一次接收 `[M+K+N,D]`。
- 显式选择 `full` 且联合节点数超过 `full_attention_max_nodes` 时 warning 并 fallback 到 linear；
  判断基于联合 `V`，不会分别判断三层。
- local branch 同时接收五种关系：`road_intra`、`syntax_intra`、`region_intra`、
  `road_to_syntax`、`syntax_to_region`。不新增 Road→Region 直连。

## 6. 输出与预测头

模型保持原字段并新增平铺的稳定字段：

```text
H_joint / H_joint_low / H_joint_high
joint_low_coefficients

H_road / H_road_low / H_road_high
pooled_road_to_syntax

H_syntax / H_syntax_low / H_syntax_high
pooled_syntax_to_region

H_region / H_region_low / H_region_high
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

导出版本升级为 `three-layer-joint-graphgps-spectral-features-v2`。旧 v1 只可作为
历史格式识别，不能与 v2 混合训练或导出。

预期语义和消费方向为：

```text
三层 H_low + source 时间序列特征
  → Stage 3 RAG retrieval
  → retrieved dynamic context
  → Hierarchical Conditional Diffusion retrieval context

三层 H_high
  → Hierarchical Conditional Diffusion target-specific condition
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

每个城市 `.pt` 保存联合 mixed/low/high、统一 eigenpairs、实际低频模式数、固定节点
区间、Road 原始 ID、Syntax/Region 连续 ID、联合 graph hash、静态版本和 checkpoint
SHA-256。加载时 city、节点顺序、联合边结构/hash、静态版本或 checkpoint 任一不匹配都会拒绝。
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

## 10. 验证状态

当前仓库已用 synthetic hierarchy 完成联合节点/偏移、五类关系边、单一 stack、
LapPE 消息边分离、low/high 重构、梯度有限性和 full-attention fallback 测试。
旧 v1 文档中的三城分层 GraphGPS 数值不代表当前 v2 语义；在重新生成联合三层
cache 和 v2 checkpoint 前，不宣称真实三城联合结果。

## 11. 测试与环境限制

```text
pytest -q tests/test_stage2_graphgps_lappe.py tests/test_stage2_frequency_decoupling.py
→ 18 passed

pytest -q --ignore=tests/test_dual_graph.py
→ 145 passed

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
- Hierarchical Conditional Diffusion（由 Stage 4 实现）；
- 正式 target 训练；
- 完整 CoSpec dual-path/prototype 模块。

这些内容只保留消费接口或文档边界，没有被写成已完成模块。
