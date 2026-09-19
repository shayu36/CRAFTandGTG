# 当前代码状态

> 状态日期：2026-09-19  
> 本轮实现基线提交：`0153556c878bbc6d5d0bf77fe856b67409d920da`（`RAG`）
> 当前工作区包含尚未提交的 Stage 4 Conditional Diffusion 改造。

## 1. 总体架构

当前仓库同时保留两条彼此独立的路线：

1. 当前正式新三层路线：`Static hierarchy → Unified GraphGPS → Hierarchical RAG → Hierarchical Conditional Diffusion`；
2. legacy HCFM/FM 路线：`src/hcfm/` 与 `scripts/run_stage2.py`。

本轮工作只扩展新三层路线，没有把联合 GraphGPS 或新 RAG 静默接入旧 HCFM。

新三层路线的目标信息流是：

```text
Road / Syntax / Region 静态图
        ↓
统一三层异构图 + 单一 GraphGPS
        ↓
H_*_low                              H_*_high
   ↓                                     ↓
三层历史动态 + 时间条件                  保留为城市特异条件
   ↓                                     ↓
Hierarchical RAG ── R_road/R_syntax/R_region
                    ↓
  Region → Syntax → Road Conditional Diffusion
```

## 2. Stage 1：三层静态层级图

状态：**已实现并已生成三城缓存**。

核心目录：

```text
src/static_hierarchy/
scripts/build_static_hierarchy.py
configs/stage1_three_layer_static.yaml
```

固定节点层级：

```text
Road → Syntax/Locality → Region
```

静态契约包括：

- Road 有向对偶图；
- Syntax 有向层内图；
- Region 层内图；
- Road→Syntax 均值池化算子；
- Syntax→Region UTM 几何长度归一化算子；
- 稳定 Road 顺序和严格 metadata 校验。

当前使用的三城缓存位于：

```text
cache/static_hierarchy_start_v2/
```

该目录是本地派生产物，不提交真实 cache 到 Git。

## 3. Stage 2：统一三层 GraphGPS + LapPE

状态：**已实现、已训练、已导出真实三城特征**。

核心代码：

```text
src/three_layer_graphgps/data.py
src/three_layer_graphgps/model.py
src/three_layer_graphgps/posenc.py
src/three_layer_graphgps/spectral_lap_pe.py
src/three_layer_graphgps/frequency.py
src/three_layer_graphgps/engine.py
scripts/train_three_layer_graphgps.py
configs/stage2_three_layer_graphgps_lappe.yaml
```

### 3.1 联合节点编号

设 Road、Syntax、Region 数量分别为 `M/K/N`：

```text
Road:   [0, M)
Syntax: [M, M+K)
Region: [M+K, M+K+N)
```

三层分别使用输入投影和节点类型 embedding，但只存在一个联合
`GraphGPSStack`。`forward()` 对 `[M+K+N, hidden_dim]` 调用一次
`self.graphgps(...)`，不是三套分层 GraphGPS。

### 3.2 联合边与消息传播

联合消息图保留以下五种有向关系：

```text
road_intra
syntax_intra
region_intra
road_to_syntax
syntax_to_region
```

跨层权重没有被丢弃，也没有增加 Road→Region 直接边。local branch 使用关系类型、
边方向和边权；global attention 覆盖整个联合节点集合。

### 3.3 LapPE 与频率分解

- 消息传播使用原始有向联合图；
- LapPE 单独使用联合图的无向化副本；
- 使用 sparse normalized Laplacian 与 `scipy.sparse.linalg.eigsh`；
- 在联合 `H_joint` 上进行一次低频投影；
- 再按稳定节点区间切分 Road/Syntax/Region 的 low/high 特征。

当前 Stage 2 特征版本：

```text
three-layer-joint-graphgps-spectral-features-v2
```

## 4. Stage 3：三层 Hierarchical RAG

状态：**输入契约、模型、真实动态构建和 memory 已实现；RAG 已接入 Stage 4
Diffusion loss 的联合训练调用链。RAG 不使用独立的 Region-only surrogate objective。**

核心代码：

```text
src/three_layer_rag/contracts.py
src/three_layer_rag/model.py
src/three_layer_rag/io.py
src/three_layer_rag/dynamics.py
src/three_layer_rag/training.py
scripts/build_three_layer_rag_snapshots.py
scripts/build_rag_memory.py
scripts/build_three_layer_rag_input.py
scripts/query_three_layer_rag.py
configs/stage3_three_layer_rag.yaml
```

### 4.1 已实现的 RAG 输入

```text
H_road_low / H_syntax_low / H_region_low
history_Y_road / history_Y_syntax / history_Y_region
month / weekday / start_hour / holiday / time_of_day
```

高频特征在公开接口处被拒绝，不能进入 RAG。source memory 还必须提供与 history
分开的 `value_temporal_features`，避免把未来 Value 复用为 Query/Key 历史输入。

### 4.2 三层检索结构

内部包含 Region、Syntax、Road 三个检索分支，顺序为：

```text
Region → Syntax → Road
```

Syntax 使用 Region 上下文；Road 使用 Syntax 和 Region 上下文。跨层条件优先使用
静态层级缓存中的加权稀疏算子，尤其不会把 Syntax→Region 多对多几何映射压成
一对一 ID。

### 4.3 真实动态序列构建

`src/three_layer_rag/dynamics.py` 和
`scripts/build_three_layer_rag_snapshots.py` 已实现：

- Road：由带时间戳 GTG 轨迹生成 `passage_count/speed_kmh/travel_time_seconds`；
- Syntax：用原始 Road→Syntax 均值权重聚合 Road 动态；
- Region：默认读取 `hourly_boundary_flow_raw.csv` 的 `in_flow/out_flow`；
- history 使用 `[t-24,t)`；
- Value 使用独立 `[t,t+24)`；
- 跨越 train/val 边界的窗口被丢弃；
- `log1p_zscore` 只在 source-train 时间范围拟合；
- `valid` CLI 别名会规范化为 RAG 契约中的 `val`。

现有 `train_label.csv/valid_label.csv` 只有 `time_index=0..23` 的小时-of-day 汇总，
没有日期。默认不会把它们广播成历史序列；只有显式启用
`--label-fill-mode hour_of_day_prior` 才作为稀疏先验使用，并记录潜在泄漏风险。

## 5. Stage 4：Hierarchical Conditional Diffusion

状态：**代码、配置、训练/生成 CLI 和 synthetic 测试已实现；尚未执行真实三城训练。**

核心实现：

```text
src/three_layer_diffusion/gaussian_diffusion.py
src/three_layer_diffusion/unet.py
src/three_layer_diffusion/conditioning.py
src/three_layer_diffusion/model.py
src/three_layer_diffusion/data.py
src/three_layer_diffusion/training.py
src/three_layer_diffusion/metrics.py
configs/stage4_three_layer_diffusion.yaml
scripts/train_three_layer_diffusion.py
scripts/generate_three_layer_diffusion.py
```

实际接线为：

```text
三层 H_low + history + calendar → Hierarchical RAG
R_* + 三层 H_high + calendar + 父层动态 → Conditional Diffusion
diffusion epsilon loss → 同时更新 RAG query/key/temporal encoder 和三层 U-Net
```

三个节点共享的 1D 专家通道为 Region=2、Syntax=3、Road=3。节点维从
`[B,N,C,T]` reshape 为 `[B*N,C,T]`，U-Net 注意力只沿短时间轴 `T` 执行，不构建
Road `M×M` 矩阵。beta/alpha/posterior 均注册为 buffer；支持 DDPM、DDIM、
self-conditioning 和 EMA。默认 `clip_x0=false`，物理空间恢复使用现有
`log1p_zscore` normalizer，而不是 CRAFT 的 `(x+1)/2`。

层次条件严格使用静态稀疏算子的转置语义：

```text
syntax_to_region: Region future → Syntax condition
road_to_syntax:   Syntax future → Road condition
```

每个 child 的入边权重重新归一化。训练时 Syntax/Road 分别读取真实归一化父层
future（teacher forcing）；推理时只读取上一级完整生成结果，且生成入口会先移除
target snapshot 的 `value_temporal_features`。

checkpoint 保存 RAG、三层 Diffusion、EMA、optimizer/scheduler、epoch/step、完整
配置、随机种子，并绑定 GraphGPS fingerprint、三城 graph hash、feature version、
RAG memory version 和 normalizer fingerprint。

当前正确的完成度是：

| 模块 | 状态 |
|---|---|
| 三层静态层级图 | 已完成 |
| 单一联合 GraphGPS | 已完成 |
| 联合 LapPE 与 low/high 分解 | 已完成 |
| 三层真实动态序列构建 | 已完成 |
| Hierarchical RAG 输入契约与模型 | 已完成 |
| source-train RAG memory | 已完成 |
| val/test RAG snapshots | 已完成 |
| RAG + 三层 Diffusion 正式训练入口 | 已实现，未真实训练 |
| RAG → Region→Syntax→Road Diffusion | 已实现，synthetic smoke 通过 |
| DDPM/DDIM、self-conditioning、EMA/checkpoint | 已实现，单元测试通过 |
| 真实三城 Stage 4 checkpoint/生成指标 | 未运行 |
| Diffusion → GTG 轨迹解码 | 未接入 |

## 6. 正式训练与生成接口

后续实现必须使用以下本地产物：

```text
outputs/stage2_three_layer_graphgps_lappe/spectral_features/
outputs/stage3_three_layer_rag/train_snapshots.pt
outputs/stage3_three_layer_rag/eval_snapshots.pt
outputs/stage3_three_layer_rag/rag_memory_v2.pt
```

联合训练已经按以下契约实现：

```text
三层 low + 三层 history + calendar → RAG
RAG references + 三层 high + calendar + 父层动态 → Hierarchical Conditional Diffusion
三层 epsilon objective → 同时更新 RAG 与 Diffusion
```

训练 checkpoint 至少应绑定：

- GraphGPS checkpoint fingerprint；
- 每个城市的 joint graph hash；
- spectral feature version；
- RAG memory version 和 graph identity；
- dynamic normalizer fingerprint；
- Diffusion contract、完整配置与训练步数。

命令：

```bash
python scripts/train_three_layer_diffusion.py \
  --config configs/stage4_three_layer_diffusion.yaml \
  --device cuda:0

python scripts/generate_three_layer_diffusion.py \
  --config configs/stage4_three_layer_diffusion.yaml \
  --checkpoint outputs/stage4_three_layer_diffusion/best.pt \
  --input outputs/stage3_three_layer_rag/eval_snapshots.pt \
  --output outputs/stage4_three_layer_diffusion/generated \
  --device cuda:0
```

## 7. 已知环境风险

当前环境导入 PyG 时会报告 `torch-scatter`、`torch-cluster`、
`torch-spline-conv` 和 `torch-sparse` 的二进制符号不匹配。这些扩展在当前联合
GraphGPS、动态构建和 RAG memory 路径中被 PyG 禁用后仍能运行。Stage 4 的
Diffusion U-Net 本身不依赖这些扩展，但导入上游 RAG 包时仍会显示这些 warning。

另一个实际风险是现有 RAG source memory 按 snapshot/node 在线重算 key；三城正式
训练的吞吐和显存尚未基准测试。当前实现语义完整，但在开始长训练前应先执行少量
snapshot 的 GPU 性能烟雾测试，再决定是否增加无泄漏的 key cache/分块检索。
