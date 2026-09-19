# 当前运行结果

> 结果日期：2026-09-19  
> 数据范围：Beijing、Chengdushi、Xianshi 三个本地 GTG 城市数据集  
> 说明：以下仅记录真实执行过并已落盘/校验的结果，不包含尚未运行的 Stage 4 RAG/Diffusion 真实指标。

## 1. 输入数据和静态层级规模

| 城市 | Road `M` | Syntax `K` | Region `N` | 联合节点 `V` |
|---|---:|---:|---:|---:|
| Beijing | 14,685 | 293 | 81 | 15,059 |
| Chengdushi | 3,514 | 70 | 23 | 3,607 |
| Xianshi | 4,147 | 82 | 26 | 4,255 |

实际使用的本地输入包括：

```text
data/{city}/map/road.csv
data/{city}/traj/train.csv
data/{city}/traj/test.csv
data/{city}/traj/train_label.csv
data/{city}/traj/valid_label.csv
data/gtg_craft/{city}/hourly_boundary_flow_raw.csv
cache/static_hierarchy_start_v2/{city}_static_hierarchy.npz
```

## 2. Stage 2 训练结果

执行配置：

```text
configs/stage2_three_layer_graphgps_lappe.yaml
hidden_dim = 128
num_layers = 2
joint LapPE modes = 16
joint low-frequency modes = 16
global attention = linear
epochs = 100
device = cpu
```

最优 checkpoint：

```text
outputs/stage2_three_layer_graphgps_lappe/best.pt
```

最优验证 epoch：`95`。

总体指标：

| Split | City-macro MSE | City-macro MAE | City-macro RMSE |
|---|---:|---:|---:|
| Train | 0.00368909 | 0.04254492 | 0.05851540 |
| Validation | 0.00359057 | 0.04225291 | **0.05890447** |
| Test | 0.00406810 | 0.04397237 | **0.06246441** |

逐城市结果：

| Split | 城市 | MSE | MAE | RMSE | 有效 Region | Observations |
|---|---|---:|---:|---:|---:|---:|
| Train | Beijing | 0.00662761 | 0.05652520 | 0.08141013 | 75 | 41,447 |
| Train | Chengdushi | 0.00202172 | 0.03525804 | 0.04496349 | 23 | 12,588 |
| Train | Xianshi | 0.00241794 | 0.03585152 | 0.04917257 | 26 | 14,425 |
| Validation | Beijing | 0.00352273 | 0.04650646 | 0.05935261 | 75 | 5,146 |
| Validation | Chengdushi | 0.00204512 | 0.03394234 | 0.04522304 | 23 | 1,587 |
| Validation | Xianshi | 0.00520386 | 0.04630991 | 0.07213776 | 26 | 1,794 |
| Test | Beijing | 0.00418239 | 0.04933800 | 0.06467137 | 74 | 5,251 |
| Test | Chengdushi | 0.00208696 | 0.03408805 | 0.04568326 | 23 | 1,576 |
| Test | Xianshi | 0.00593495 | 0.04849106 | 0.07703861 | 26 | 1,839 |

完整训练历史保存在：

```text
outputs/stage2_three_layer_graphgps_lappe/metrics.json
```

Stage 2 始终对全部 Region 输出 prediction；表中的“有效 Region”是该 split 中实际
存在监督观测并参与指标计算的 Region 数量，因此可能小于静态图 Region 总数。

## 3. Stage 2 三城谱特征导出

checkpoint fingerprint：

```text
136955a82d989ee367b5d4575738f6272c86d2d0130bf257fcce04c0a6e46ce9
```

导出文件：

```text
outputs/stage2_three_layer_graphgps_lappe/spectral_features/beijing_spectral_features.pt
outputs/stage2_three_layer_graphgps_lappe/spectral_features/chengdushi_spectral_features.pt
outputs/stage2_three_layer_graphgps_lappe/spectral_features/xianshi_spectral_features.pt
```

三个城市均包含：

```text
H_joint / H_joint_low / H_joint_high
H_road / H_road_low / H_road_high
H_syntax / H_syntax_low / H_syntax_high
H_region / H_region_low / H_region_high
joint_graph_hash / checkpoint_fingerprint / stable node ranges
```

导出后执行了严格 round-trip 校验。

### 3.1 联合频率分解诊断

| 城市 | Low modes | Cutoff eigenvalue | Reconstruction relative error | Orthogonal residual relative error |
|---|---:|---:|---:|---:|
| Beijing | 16 | 0.01101603 | 0.0 | 1.0689e-7 |
| Chengdushi | 16 | 0.02849730 | 0.0 | 1.0502e-7 |
| Xianshi | 16 | 0.03118244 | 0.0 | 1.0659e-7 |

这些结果确认：

```text
H_joint_low + H_joint_high = H_joint
```

在当前浮点容差下重构误差为 0，low/high 正交残差约 `1e-7`。

## 4. 三层动态序列与 RAG snapshots

实际构建规则：

```text
Road:   passage_count / speed_kmh / travel_time_seconds
Syntax: Road 动态经 Road→Syntax 权重聚合
Region: raw in_flow / out_flow
History: [t-24, t)
Value:   [t, t+24)
Stride:  24 hours
```

归一化：

```text
mode = log1p_zscore
fit scope = Beijing/Chengdushi/Xianshi source-train hours only
```

normalizer SHA-256：

```text
7548317859d525d9db04f7393d37e6184e9b12242baa030c044754f440ee8c6e
```

Train bundle：

| 城市 | Train snapshots |
|---|---:|
| Beijing | 23 |
| Chengdushi | 23 |
| Xianshi | 23 |
| **总计** | **69** |

Eval bundle：

| 城市 | Val snapshots | Test snapshots |
|---|---:|---:|
| Beijing | 3 | 3 |
| Chengdushi | 3 | 3 |
| Xianshi | 3 | 3 |
| **总计** | **9** | **9** |

已确认 eval normalizer 与 train normalizer 字节级完全一致。

### 4.1 Snapshot 张量形状

| 城市 | Road history/value | Syntax history/value | Region history/value |
|---|---|---|---|
| Beijing | `[14685,3,24]` | `[293,3,24]` | `[81,2,24]` |
| Chengdushi | `[3514,3,24]` | `[70,3,24]` | `[23,2,24]` |
| Xianshi | `[4147,3,24]` | `[82,3,24]` | `[26,2,24]` |

## 5. RAG memory 验证

memory 文件：

```text
outputs/stage3_three_layer_rag/rag_memory_v2.pt
```

严格加载结果：

```text
version = three-layer-hierarchical-rag-v2
source_cities = [beijing, chengdushi, xianshi]
source_split = train
num_snapshots = 69
require_value_separation = true
require_graph_identity = true
```

三个城市绑定到同一个 Stage 2 checkpoint fingerprint，同时分别保存自己的
`joint_graph_hash`、节点范围和特征版本。history 和未来 Value 的三层 shape 均已
校验通过。

## 6. 本地产物大小

| 文件 | 大小 |
|---|---:|
| `best.pt` | 7,658,839 bytes |
| Beijing spectral features | 26,550,702 bytes |
| Chengdushi spectral features | 6,388,053 bytes |
| Xianshi spectral features | 7,526,638 bytes |
| `train_snapshots.pt` | 315,437,023 bytes |
| `eval_snapshots.pt` | 91,315,190 bytes |
| `rag_memory_v2.pt` | 584,104,017 bytes |

这些文件包含真实或派生城市数据，只保存在本地，不应提交到 GitHub。

## 7. 已运行测试

Stage 4 Conditional Diffusion 新增测试：

```text
pytest -q tests/test_three_layer_diffusion.py
→ 19 passed, 4 warnings
```

覆盖 CRAFT schedule、buffer/q_sample、三层通道、节点 reshape、RAG/high 条件、
频带隔离、稀疏父子广播、mask loss、层间 loss 尺度、LOCO、DDPM/DDIM、确定性
DDIM、self-conditioning、RAG 梯度、EMA/checkpoint、normalizer、无 Road 稠密注意力、
stable node order 和完整三层 synthetic generation。

三层动态、RAG 和 Stage 2 定向回归测试：

```text
31 passed, 4 warnings
```

除 `tests/test_dual_graph.py` 外的仓库全量测试：

```text
pytest -q --ignore=tests/test_dual_graph.py
→ 176 passed, 14 warnings
```

不跳过时，`pytest -q` 在收集 `tests/test_dual_graph.py` 阶段失败，准确错误为：

```text
ImportError: .../graph_tool/libgraph_tool_core.so:
version `GOMP_5.0' not found
```

这是当前 `graph_tool` 与 PyTorch 自带 `libgomp` 的 ABI 环境问题，测试尚未进入执行，
不是 Diffusion 代码断言失败。

此外已对真实本地产物执行只读契约检查：69 个 train snapshots、18 个 eval snapshots、
69 个 RAG memory snapshots、三城 Stage-2 high/low stable order、graph hash、GraphGPS
fingerprint 和 normalizer fingerprint 全部一致。该检查没有运行训练或生成。

四个 warning 均来自当前 PyTorch 与 PyG 可选二进制扩展的符号不匹配：

```text
torch-scatter
torch-cluster
torch-spline-conv
torch-sparse
```

PyG 已禁用这些扩展；当前 Stage 2 CPU 训练、特征导出、动态序列构建和 RAG
memory 验证均成功完成，因此这些 warning 不是本轮失败项。

## 8. 尚无运行结果的部分

以下结果当前不存在，不能宣称已经验证：

- 训练后的 Hierarchical RAG checkpoint；
- RAG 检索质量指标；
- RAG 与三层 Conditional Diffusion 联合训练指标；
- Region→Syntax→Road 生成指标；
- 新三层路线的最终端到端生成样本；
- RAG/Diffusion GPU 训练结果。

Stage 4 正式代码入口已经接通，但尚未执行耗时的真实三城训练；因此没有训练后的
checkpoint 或真实生成指标。这是“尚未运行”，不是本地真实数据缺失，也不是把
synthetic smoke test 当成城市生成结果。
