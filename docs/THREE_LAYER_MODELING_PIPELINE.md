# 当前三层交通建模全流程

更新时间：2026-09-20

本文是当前项目唯一的流程说明，合并静态层级图、Stage 2 GraphGPS、Stage 3 RAG 和 Stage 4 Diffusion 的有效内容。

## 1. 总体路线

```text
Road + Syntax + Region
          ↓
统一三层异构层次图
          ↓
单一 GraphGPS + 联合 LapPE
          ↓
H_low / H_high 频率分解
       ↓             ↓
三层 Hierarchical RAG   高频条件
       ↓             ↓
R_region/R_syntax/R_road
          ↓
Region → Syntax → Road Diffusion
          ↓
未来 24 小时三层动态
```

当前训练城市为 `beijing`、`chengdushi`、`xianshi`；legacy HCFM/FM 不是当前训练入口。

## 2. 三层静态图

节点编号固定为 `Road=[0,M)`、`Syntax=[M,M+K)`、`Region=[M+K,M+K+N)`。静态输入维度为 Road `[M,33]`、Syntax `[K,5]`、Region `[N,45]`。

图关系包括 Road→Road、Road→Syntax、Syntax→Syntax 和 Syntax→Region，不存在 Road→Region 直接层次边。当前 cache 为 `cache/static_hierarchy_start_v2/`，加载时校验版本、节点顺序、shape、索引范围和图身份。

## 3. Stage 2：联合 GraphGPS 与频率分解

三类节点先分别投影到 hidden dimension 128，再加入 node type embedding 和联合 LapPE。三层节点拼接成一个联合图，只调用一个 GraphGPS 编码器；消息图保持有向，LapPE 使用联合无向副本计算。

GraphGPS 输出 `H_joint` 后显式分解为 `H_joint_low + H_joint_high`，再按固定节点范围切出三层 low/high 特征。当前 LapPE mode 为 16，global attention scope 为 `joint`。

三城 Stage 2 已完成 100 epoch 训练，最优 `valid_city_macro_rmse=0.06179773683349291`（epoch 97），并已导出谱特征。

有效产物：`outputs/stage2_three_layer_graphgps_lappe/best.pt`、`metrics.json` 和 `spectral_features/`。

## 4. Stage 3：动态 snapshots 与 RAG

每个动态 snapshot 使用 `history=[t-24,t)` 作为 RAG Query/Key，使用 `value=[t,t+24)` 作为 RAG Value 和 Diffusion target，窗口步长为 24 小时。

Road/Syntax 使用 `passage_count`、`speed_kmh`、`travel_time_seconds` 三个通道；Region 使用 `in_flow`、`out_flow` 两个通道。Syntax 动态由 Road 动态按 Road→Syntax 权重聚合；归一化统计量只从三城 source-train hours 拟合。

RAG 使用三层 low 特征、历史动态和 calendar 条件。memory 只来自三城 train split，并启用 source city、calendar、Leave-One-City-Out、value separation、graph identity 和分块 Top-K 检索约束。

当前数量为：train snapshots 69 个（每城 23 个），eval snapshots 18 个（每城 validation 3 个、test 3 个），`rag_memory_v3.pt` 包含 69 个 source-train snapshots。

有效产物：`outputs/stage3_three_layer_rag/train_snapshots.pt`、`eval_snapshots.pt`、`train_snapshots.normalizer.json` 和 `rag_memory_v3.pt`。

## 5. Stage 4：三层 Conditional Diffusion

low/high 分工为 `H_*_low → RAG`、`H_*_high → Diffusion`。三个 Diffusion expert 的通道为 Region 2、Syntax 3、Road 3。

Region 条件为 `H_region_high + R_region + calendar`；Syntax 额外接收 Region future；Road 额外接收 Syntax future。训练使用真实父层 future teacher forcing；推理严格按 Region → Syntax → Road 使用生成的父层 future。

每层对 normalized future 加 Gaussian noise，U-Net 预测噪声；三层 loss 分别按有效 mask 归一化后求和，避免 Road 节点数量支配其他层。模型使用 EMA、固定 validation timestep/noise bank 和 checkpoint identity 校验。

## 6. 当前五卡低显存策略

Stage 4 当前配置为 `node_chunk_size=64`、`gradient_checkpointing=true`、`train_source_keys=false`、`source_cache_device=cpu`。

训练时按 64 个节点分块计算，使用 gradient checkpointing；source key cache 保存在 CPU 且不保留 source key 反向图。五个 rank 按 city bucket 分发 batch，各 rank 反向后显式 all-reduce/average 梯度；仅 rank 0 验证、保存 checkpoint 和写日志。

五卡 smoke 已通过，`EXIT_CODE=0`；最近一次定向回归为 `36 passed`。

## 7. 当前产物与运行顺序

环境为 `/root/miniconda3` 的 `base`，PyTorch `2.0.1+cu118`，CUDA 可用。环境变量已写入 `/root/miniconda3/etc/conda/activate.d/paper_env.sh`。

正式训练命令为：`torchrun --standalone --nproc_per_node=5 scripts/train_three_layer_diffusion.py --config configs/stage4_three_layer_diffusion.yaml --device cuda:0 --resume outputs/stage4_three_layer_diffusion/last.pt`。

训练后使用 `python scripts/generate_three_layer_diffusion.py --config configs/stage4_three_layer_diffusion.yaml --checkpoint outputs/stage4_three_layer_diffusion/best.pt --input outputs/stage3_three_layer_rag/eval_snapshots.pt --output outputs/stage4_three_layer_diffusion/generated --device cuda:0` 生成并评估。

当前剩余工作是三城 Stage 4 长训练最终指标、完整生成评估和 Diffusion 到 GTG 轨迹解码闭环。
